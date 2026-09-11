# Design

SpoolCache is one external vLLM V1/HMA connector with rank-local persistent
token files. It uses public runtime contracts and leaves serving, orchestration
and model choice to vLLM and the deployment. Earlier snapshot/slot implementations
are retired; historical evidence is indexed in [Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md).

## Token-file storage

The default `spoolcache.vllm.connector.SpoolCacheConnector` stores one file per
complete runtime-aligned full-KV interval per rank, using token-chained keys inspired
by LMCache and bounded buffered transfers inspired by SparkCache. Keys include
earlier tokens, salt, media and deployment identity. There is no explicit tree,
separate physical packing unit, slot arena, reference graph or native I/O engine.

Let A be the least common multiple of every reusable group's logical page span
(including DCP sharding). One-page scratch capacity does not constrain A.
The file interval is `ceil(256 / A) * A`: the smallest aligned width of at least
256 tokens. For example, A=32 gives 256, A=192 gives 384, and A=1664 gives 1664.
The floor limits small-file overhead; it is not another alignment constraint.
There is no model-dependent chunk setting.
Files are distributed by the first two hex key characters across up to 256
on-demand directories. This filesystem sharding does not affect token intervals.

Mixed HMA additionally needs an independently keyed file containing complete
non-full state at the exact captured boundary. Its key binds the terminal prefix.
Missing earlier window/recurrent state cannot be reconstructed from a longer
prefix. Data/state files share publication, pin, scrub and GC contracts.

## Components and ownership

| Component | Responsibility |
| --- | --- |
| [Connector](../src/spoolcache/vllm/connector.py) | Public hooks, plans, complete-participant admission and inventory |
| [Compatibility](../src/spoolcache/vllm/compat.py), [HMA](../src/spoolcache/hma.py) | Runtime admission, semantic discovery and complete page coverage |
| [Token files](../src/spoolcache/token_files.py), [metadata](../src/spoolcache/manifest.py) | One file per key, chained lookup, authentication and LRU |
| [Rank store](../src/spoolcache/rank_store.py), [pins](../src/spoolcache/leases.py) | Directory ownership, durable generations, locks and withdrawal |
| [Token mover](../src/spoolcache/token_mover.py), [GPU primitives](../src/spoolcache/gpu.py), [buffers](../src/spoolcache/buffers.py) | Missing-key capture, opaque page copies and bounded credits |
| [Token scrub](../src/spoolcache/token_scrub.py), [scheduler](../src/spoolcache/maintenance.py) | Authenticated cursor progress, rate limits and bounded shutdown |
| Identity, prefix, topology and quorum | Model/runtime identity, chained keys and cross-rank agreement |
| Telemetry and event journal | Bounded observations and durable counters |

The scheduler owns logical keys; each worker captures/restores its physical
shard. There is no second backend, CPU L1, remote payload service or in-package
process supervisor.

## Runtime admission

User constraint reaffirmed on 2026-09-11: keep implementation and qualification
rules independent of model names. Derive cache geometry, state dependencies and
interface capabilities from runtime facts; do not add model defaults, profiles,
allowlists or exceptions. Model locators/revisions identify benchmark fixtures
and persistence namespaces only. Evidence tools must also derive page geometry
from recorded runtime data and record observed native-cache behavior instead of
assuming a repeated request must hit. Unknown or contradictory contracts still
fail closed.

Startup inspects the installed public V1 connector interface and proves that
SpoolCache overrides accept its call shapes. It also attests installed vLLM
package contents. Runtime helpers used by discovery belong to the checked
contract. Unknown signatures, cache semantics, topology or ownership fail closed.

The gate does not select a version, model, architecture or modality profile.
Passing it proves the inspected contracts, not a successful model round trip.
Runtime/live qualification remains separate; see [Compatibility](COMPATIBILITY.md).

The allocator gate rejects enabled `expandable_segments` settings. SpoolCache
must not patch an allocator, overwrite vLLM or insert model exceptions to make
an incompatible runtime appear supported.

## Cache semantics and HMA

Discover semantic kind and capabilities from public runtime facts. Do not branch
on cache-spec class names or infer universal safety from a few sampled lengths.
All HMA groups belong to one logical transaction: one missing group cannot be
advertised as a partial hit.

- Full-attention groups reuse the required prefix pages.
- Sliding/windowed groups follow runtime page-selection semantics and retain the
  exact pages needed at the reusable boundary.
- Stateful/recurrent groups require a proven snapshot boundary and ownership
  contract. An unsafe producer boundary is skipped rather than guessed.
- Non-prefix scratch state must prove non-participation, exactly one request
  block-table page and one physical page at the deployment's real bounds.
  Packed scratch declarations repeat that proof for the shared allocator/group.

`kv_cache_groups` defines block-table-owning layers. Final tensor registration
can include KV-sharing aliases. Omit an extra name only after proving it is the
same Torch storage view as an owner; missing owners or independent extra tensors
fail closed. Bind the verified alias map into rank ownership.

Logical scheduler layouts and physical worker page geometry are distinct.
Coordination binds ordered group/page-selection semantics. Rank-local identities
also bind tensor dtype, shape, strides, storage offsets and page-byte geometry.
This lets PP stages own different layers without inventing a global physical layout.

## Identity and exact prefixes

Cache identity includes model locator/revision, runtime
build, package version, topology, execution facts and discovered layout. Prefer
nonempty public `model_weights`, otherwise `model`, and bind `revision`.
Served aliases and configuration class names are not model identity.

SpoolCache authenticates its own KV objects. A model locator/revision is a
namespace, not an attestation of all checkpoint bytes. Operators keep model
artifacts immutable or change the model locator/revision or cache path. The connector does not
scan, copy or hash an entire model repository.

Prefix identity uses the `spoolcache-exact-prefix/v2` domain and includes exact
tokens and salt. The former operator namespace is no longer an input. For
multimodal requests it also binds qualified public media identifiers, modality and placeholder geometry.
Discover runtime-enabled input modalities through the public registry and
limits; a verified-model table must never become an admission allowlist.
Incomplete media identity, unsupported prompt embeddings and LoRA requests use
the existing bypass path rather than ambiguous persistent keys.

Only exact shared prefixes are reusable. Chunk alignment and the semantic reuse
boundary determine the stored span. A shorter producer reused by a longer
consumer must come from the same exact token/media prefix.

## Distributed agreement

Derive PP/TP/DCP coordinates from public process groups and cross-check public
parallel configuration. Values are literal non-boolean integers; coercing a
changed runtime value with `int()` is not a valid ownership proof. DCP is a TP
subdivision. Each global PP×TP participant owns its rank-local directory.

PP>1 requires the public PP-aware handshake. Stages can own different layer
sets; they must agree on coordination identity and reusable span. The scheduler
requires the complete participant set before it can advertise an entry.
Missing, extra, duplicated, mistyped or wrong-identity startup inventories fail
startup. Missing offers during operation yield a miss.

Worker inventories are bounded at every boundary: local catalog, report page,
combined delta, pending checkpoint and rank count. A bounded startup subset is
a safe false negative. Gaps and malformed reports withdraw the rank until a
complete checkpoint arrives.

Bulk additions are emitted over bounded deltas; existing synchronized entries
stay available while new keys wait in the held catalog. Removals take priority
in the next report. If those removals exceed one report, an intentional sequence
gap withdraws the rank until its replacement checkpoint completes. Checkpoints
bind the emitted inventory sequence rather than unreported additions.

## Internal protocol boundaries

The file schema `spoolcache-token-key-file/v3` and each embedded header bind the
deployment, physical rank, topology, HMA layout, data/state kind, key/parent,
span, derived chunk width, transfer credit, page segments, length, whole-payload
hash and bounded per-transfer hashes. The framed header has its own
checksum. Metadata must match runtime-derived expectations before payload I/O.

Layout, coordination and durable generation protocols remain independent:
scheduler logical layout differs from worker physical byte geometry, while
every worker must match the scheduler coordination identity. Per-step connector
metadata carries plans through vLLM without another schema negotiation layer.
Old cache roots are not reinterpreted or automatically migrated.

## Store and restore lifecycle

At the pre-forward ownership boundary, saves authenticate existing keys before
GPU capture. Only missing/corrupt keys need D2H capture. Each borrowed staging
credit remains owned until its copies finish and the consumer finishes writing.
Saves remain synchronous; no queued whole-prefix payload or background GPU read
can outlive the public page-ownership contract.

Each temporary file receives its complete header/payload, then file fsync.
Small files use one `writev`; larger files reserve the header, stream fixed-size
payload ranges, and rewrite the completed checksum header before that same
single file fsync. No transfer range becomes a separate file or durable key.
Publishing a hardlink and fsyncing the shard make the key durable before it is
advertised. Cleanup attempts both unlink and directory fsync without masking a
primary failure. A publication failure can leave earlier independent keys
durable; only confirmed keys enter inventory. Incomplete HMA is never a hit.

Admission requires every data key from the chain root and exact boundary state
across every participant. Restore pins all keys under the namespace lock before
releasing that lock. Payload reads fill bounded pinned credits, authenticate
each transfer range's SHA-256, then copy to selected GPU pages. A range can cross
page/layer boundaries or contain only part of a large opaque page. Full-file
authentication and the complete restore's CUDA drain finish before
releasing pins or reusing credits. Corruption returns a borrowed credit before
waiting on the quarantine lock, avoiding reader/writer lock inversion.

## Buffered I/O and durability

Readv/writev use ordinary file descriptors. OS page cache is separate from the
explicit process staging budget. No O_DIRECT, io_uring registration or compiled
extension is used. CUDA host registration remains necessary for pinned copies.
Durability comes from ordered file and directory fsyncs, not from I/O mode.

## Resource bounds

| Resource | Bound |
| --- | --- |
| Full-KV data file | Smallest multiple of common runtime alignment >=256 tokens |
| Data file or complete HMA state payload | At most 512 transfer ranges (32 GiB); also limited by the 64 KiB header |
| Transfer batch | Small files: at most 16 files in one 64 MiB credit; large files: consecutive ranges of at most 64 MiB |
| Explicit payload staging per rank | Two 64 MiB CUDA credits plus one 64 MiB CPU credit: 192 MiB |
| Chain | No separate token/key-count, aggregate header or descriptor-memory cap; shared layout views and incremental coverage checks |
| Inventory | 512 MiB conservative per-rank record allowance covering reporter/catalog/checkpoint/transport views; capacity derived from object sizes; 64 per periodic report |
| In-flight admission | Two restores, one store plan, through their complete lifecycle |
| GC | 80% trigger; 20%-of-key rounds, at most four candidates per batch |
| Scrub | 64 MiB/s, at most 64 keys per step; 60 s startup delay, six-hour cycle |

Token descriptors share the runtime layout and derive page slices as consumed.
HMA coverage uses one cursor per layer; neither path retains every file/layer
combination. File headers remain independently authenticated. The request token
limit comes from vLLM's runtime context, with no additional million-token ceiling.
Active-chain metadata grows with key and transfer-checksum counts; it has no
independent byte quota. This is separate from inventory and payload staging.

Payload memory does not grow with prefix length. Page cache, metadata and model
runtime allocations are outside the 192 MiB staging figure. Unsupported geometry
fails startup; there is no model-specific page size or capacity preset.

## Generation, withdrawal and repair

Exactly one worker holds the rank's lifetime inventory-owner lease. Durable,
strict generations and an initialization sentry prevent reuse after restart or
clock rollback. Missing initialized state fails closed. Namespace operations
retain their cross-process lock protocol; live restores additionally pin keys.

Corrupt entries receive a durable withdrawal marker before quarantine is attempted.
Metadata scans cannot clear it. Full replacement or complete authentication of
that exact key can release it; all-rank contiguous admission still applies.
Before sending inventory, the reporter streams actual withdrawal markers and
removes every marked key from its held image. This avoids probing every healthy
key when the marker namespace is empty. Marker acknowledgements use bounded,
cursor-driven pages, including offline markers outside the held image.
Absence is directory-fsynced before a marker is
acknowledged. Parent-directory receipts are retried even for existing state.

## Scrub and reclamation

Scrub persists one lexical key cursor only after complete payload authentication.
Concurrent state/target changes are rechecked under the rank lock. Pinned keys
are deferred; new keys behind the cursor are visited next cycle. There is no
SQLite work queue or shared-object reference scan. Unexpected scrub failure
withdraws inventory until reconciliation; shutdown retains resources until any
timed-out reader exits.

GC uses file mtime as its sole LRU source, including after process reopen.
Metadata inspection does not refresh it. Reverse batch touches favor earlier
prefix keys; equal-mtime candidates are ordered by key. Pins are skipped. There
is no fixed stop watermark and a 20% key quota does not imply 20% of bytes.
Directory rescans remain a scaling cost. Deleting an intermediate key makes
later chains miss; it does not fabricate missing HMA boundary state.

Startup recovery removes only recognized abandoned token temporaries under the
rank lock. Quarantine/control bytes are accounted separately and are not silently
deleted to satisfy a capacity target. See [Operations](OPERATIONS.md).

## Failure boundaries

| Boundary | Behavior |
| --- | --- |
| Unsupported startup contract/root | Fail initialization. |
| Ineligible prefix, missing quorum or corrupt pre-admission entry | Bypass or miss; never promise a partial hit. |
| Store failure | Do not publish the incomplete entry; inference may continue. |
| Post-admission restore failure | Fail-stop the affected worker with exit 70; deployment replaces the group. |
| Scrub shutdown timeout | Emit a bounded shutdown receipt and retain reader resources until finalization. |

SpoolCache does not provide remote process supervision or automatic group
reconstruction. A responsive head health endpoint is not proof that all PP
participants can serve.

## Configuration, compatibility and exclusions

`spoolcache config` emits the only connector entry point. Public settings are
path and capacity; request flags independently skip persistent reads/writes.
The old snapshot maintenance CLI and experimental connector alias are removed.
Model/runtime identity, topology and geometry are automatic; no backend selector
or model preset is accepted.

Python 3.11+ uses a pure-Python wheel. Serving still needs 64-bit Linux OFD locks,
CUDA and admitted vLLM contracts. Artifact authentication and installed-runtime
tests remain required. The [Gemma multimodal receipt](receipts/2026-09-11-token-files/README.md)
qualifies cache equivalence on `9061441`, TP=1/PP=1, with its mixed-recognition
limit. PP>1, maximum-context and sustained GC qualification remain separate
from host contracts and earlier model receipts.
