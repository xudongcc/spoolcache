# Design

SpoolCache is an external vLLM V1 KV connector with persistent, rank-local
storage. Its contracts are exact-prefix identity, complete-group restore,
bounded staging memory and authenticated durable publication. It uses public
vLLM hooks and leaves model serving, process supervision and deployment to vLLM
and the operator.

This document describes the current implementation. Historical alternatives,
measurements and review details are preserved in [the archive](archive/README.md)
and indexed by [Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md).

## Components and ownership

| Component | Responsibility |
| --- | --- |
| [Connector](../src/spoolcache/vllm/connector.py) | Scheduler/worker hooks, admission, metadata exchange and completion |
| [Compatibility gate](../src/spoolcache/vllm/compat.py) | Public-call compatibility and installed-runtime attestation |
| [HMA discovery](../src/spoolcache/hma.py) | Runtime cache semantics, groups, layer ownership and reuse boundaries |
| [Identity](../src/spoolcache/identity.py), [prefix](../src/spoolcache/prefix.py), [topology](../src/spoolcache/topology.py) | Model/runtime namespace, exact-prefix keys and participant coordinates |
| [Quorum](../src/spoolcache/quorum.py) | Bounded worker inventories and complete-participant offers |
| [GPU mover](../src/spoolcache/gpu.py) and [buffers](../src/spoolcache/buffers.py) | Fixed staging slots and transfer ownership |
| [Store](../src/spoolcache/store.py) and [manifests](../src/spoolcache/manifest.py) | Immutable objects, durable publication, lookup, quarantine and GC |
| [Maintenance](../src/spoolcache/maintenance.py) | Resumable scrub and targeted requests |
| [Telemetry](../src/spoolcache/telemetry.py) and [journal](../src/spoolcache/event_journal.py) | Bounded observations and durable counters |

The scheduler chooses logical prefix IDs. Workers capture and restore their
own physical shard. Disk payloads are not exchanged between hosts. There is no
standalone SpoolCache engine, CPU L1, remote backend or in-package supervisor.

## Runtime admission

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

## Internal protocol boundaries

| Boundary | Active contract |
| --- | --- |
| Scheduler to worker plans | `SpoolCacheMetadata` carries `loads` and `stores` through vLLM; workers check its Python type. There is no separately negotiated metadata schema. |
| Worker startup and inventory | Coordination digests bind runtime/layout/topology; bounded reports and complete PP×TP quorum control admission. |
| Persistent cache | Manifest/envelope schemas, layout protocol identity and deployment/rank digests authenticate stored data. |

`vllm-runtime-kv-v1` is the fixed internal layout protocol identifier stored in
`profile` fields. Manifest lookup and HMA coverage validate it; it is not a
removed operator profile setting. `StorePlanLike(Protocol)` is a Python typing
contract for store admission, not a network protocol. Model namespaces and
filesystem namespaces likewise serve identity and directory ownership; neither
reintroduces the removed `SPOOLCACHE_NAMESPACE` setting.

The former `SpoolCacheMetadata.schema` field had no reader or validation and has
been removed. Persistent schema versions and crash-recovery generation handling
remain active correctness checks.

## Store and restore lifecycle

A store captures the runtime-proven complete page set through fixed staging
slots. It writes immutable content-addressed objects, fsyncs data and publishes
an authenticated manifest only after every object has a durable receipt.
Objects, manifests and directory links follow the maintenance lock and durable
publication order. CUDA pointers, request IDs and allocator block IDs are not
serialized as persistent identities.

Before admission, lookup validates manifest identity and object metadata. Restore
streams payload through the fixed slots and computes SHA-256 over logical bytes,
checking stored length and zero padding. Each object is fully authenticated in
host staging before it is copied to request-private GPU blocks. Earlier objects
may already be on the GPU when a later object fails; completion is withheld
until the entire restore authenticates and CUDA completes. A later failure cannot
be treated as a successful partial restore or silently recomputed after admission.

The current store/restore path is synchronous at the model-runner boundary.
Separate slots and CUDA completion events preserve transfer ownership: a slot
cannot be overwritten while CUDA still consumes it. The production path keeps
separate pinned and aligned-I/O pools.

## Direct I/O and durability

Payload files always use `O_DIRECT`. Objects are padded to the implementation's
4096-byte alignment and staging buffers are aligned. Initialization probes the
cache root and rejects unsupported direct I/O. Short payload reads/writes fail
instead of retrying with potentially unaligned offsets or pointers.

Small manifests, SQLite state and control files use ordinary I/O. Payloads do
not use file-backed mmap; anonymous mmap backs the bounded aligned buffers.
`O_DIRECT` controls page-cache use, not durability. Payload fsync and parent
directory fsync remain part of the publication protocol.

Temporary cleanup attempts both unlink and directory fsync. A secondary cleanup
failure must not replace the primary publication error. Existing directories or
idempotent markers need a directory-fsync receipt; their existence alone does
not prove a preceding failed publication survived a crash.

## Resource bounds

These are implementation constants, not deployment tuning options:

| Bound | Current value |
| --- | --- |
| Pinned staging per rank | 2 × 64 MiB |
| Separate aligned-I/O staging per rank | 2 × 64 MiB |
| Pending restores | 2 across the complete admitted lifecycle |
| Pending stores | 1 |
| Cache chunk / minimum persistent span | 256 / 1,024 tokens |
| Internal maximum span | 1,048,576 tokens; not a model-context support claim |
| Startup inventory selection | 512 digests |
| Report batch | 64 items |
| Catalog bound | 100,000 entries |
| Capacity low watermark | 90% of the configured per-rank maximum |

The two staging pools total about 256 MiB per rank, plus bounded metadata and
runtime overhead. This is not a total host or model-memory limit. Capacity is
managed asynchronously and does not reserve filesystem space. The public
`spoolcache_max_size` setting uses GB as 1024³ bytes; its default of 200 retains
the previous 200 GiB target. Its derived `max_bytes` property supplies whole
bytes for the existing GC thresholds and accounting.

## Generation, withdrawal and repair

Each rank has one lifetime inventory-owner lease. Startup reserves a monotonic
epoch in `state/generation.json` before exposing inventory. A persistent sentry
distinguishes a new store from lost required state. Legacy raw-clock epochs
migrate into a disjoint high domain; the state is a correctness boundary.

A newer generation withdraws the prior rank image before accepting its complete
checkpoint. Lower reports are ignored only when their exact UUID/epoch is in
bounded observed history. Unknown lower or conflicting equal generations
withdraw the rank.

Known-bad entries receive durable withdrawal markers before quarantine renames.
A reporter consumes those markers under the maintenance lock before publishing
stats; an offline maintenance client does not compete for its lifetime lease.
Catalog scan/replace and manifest publication/reporter-add use their respective
single critical sections so stale observations cannot re-add withdrawn entries.

A corrupt content-address collision fences the digest before traversing its
references. Reference withdrawal is streamed, evidence retained, and replacement
published durably. A failed namespace operation keeps fences/tombstones active.
Repairing one object cannot clear every referring entry tombstone: another object
or incomplete operation may still make that entry unsafe. Release a tombstone
only after a complete replacement or final authentication of its full manifest.

Startup selection streams validation before choosing the bounded newest healthy
subset. Tombstones or corrupt files must not crowd healthy entries out of that
selection window. Normalize data-driven manifest decode failures into manifest
errors, while allowing resource exhaustion and unrelated implementation errors
to remain real failures.

## Scrub and reclamation

Scrub persists namespace work queues, live references and completed-object
progress in SQLite. Cycle start records state without scanning the whole tree.
Snapshot phases stream fixed batches, allow cancellation between items and use
idempotent inserts when an interrupted phase is rescanned. Partial hashes are
not trusted after restart.

Authentication binds the rank identity before payload reads and checks regular
file type, logical/stored length, SHA-256 and zero padding. Quarantine withdraws
all affected offers. An orphan is deleted only after a complete manifest pass
and a final live-reference recheck under the maintenance lock.

Capacity pressure first reclaims proven crash orphans, then healthy LRU manifests.
Maintenance treats unknown managed paths and links as quarantine candidates,
not destinations to follow. Inventory inspection does not refresh recency.
See [Operations](OPERATIONS.md) for scheduling, status and recovery procedures.

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

The two public settings are described in [Configuration](CONFIGURATION.md).
Configuring the connector enables both persistent reads and writes; no access
mode gates startup checks, worker registration, restore admission or store plans.
The request flag `spoolcache.skip_read=true` skips persistent restore admission;
`spoolcache.skip_write=true` suppresses store tracking. They are independent and
only literal JSON booleans activate them. Neither changes runtime compatibility,
background maintenance, or vLLM's own caches.
`spoolcache_path` defaults to the current user's `~/.cache/spoolcache`; old
root and I/O-mode settings are removed in the [unreleased migration](MIGRATION.md).

The persistent schema names remain `spoolcache-manifest/v1`,
`spoolcache-manifest-envelope/v1`, `spoolcache-deployment/v2` and
`spoolcache-coordination/v2`. A schema name does not promise cache hits across
changed model/runtime/package/topology identities. Keep complete rank state
for rollback; do not rename data into another identity.

No asynchronous store, layerwise restore, shared staging pool, compression,
encoder cache, cross-node replication, GDS/RDMA payload path, Redis/S3 backend
or model selector is shipped. A shared-pool probe is feasibility evidence only.
New optimizations require a measured need and one qualified ownership path;
see [Goals](TODO_GOALS.md).
