# SpoolCache

SpoolCache is an experimental, out-of-tree vLLM KV connector. Its first
qualification was the fixed DeepSeek-V4 Flash Vision deployment in
[`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).
It keeps each runtime-discovered PP x TP worker's cache shard on that machine's
local NVMe and does not modify the vLLM installation.

Implemented in the current alpha:

- patch-free `KVConnectorBase_V1`/`SupportsHMA` integration;
- exact text and multimodal-prefix identities (media content ID, modality, and
  placeholder geometry are part of the key);
- startup discovery of every model input modality enabled by vLLM's public
  multimodal registry and deployment limits, with no model, architecture, or
  modality allowlist in SpoolCache;
- complete capture and restore of every runtime-discovered cache group and
  block-table-owning layer, including EAGLE/MTP and auxiliary state when vLLM
  exposes it; registered cross-layer sharing aliases are omitted only after an
  exact Torch storage-view proof;
- rank-local `O_DIRECT` payload I/O, immutable content-addressed objects, and
  manifest-last atomic commits;
- resumable, rate-bounded deep scrubbing of manifest identity, object type and
  length, logical SHA-256, and zero padding, with corrupt entries removed from
  worker offers and moved to rank-local quarantine;
- crash-safe temporary/orphan cleanup plus capacity maintenance that reclaims
  incomplete publication objects before evicting healthy LRU manifests from a
  configured maximum to its internally derived 90% low watermark;
- fixed 128 MiB pinned GPU staging and fixed 128 MiB aligned I/O staging per
  rank (`2 x 64 MiB` internal bounded defaults);
- startup inventory, bounded incremental reports, and an all-PP-stage/all-rank
  quorum derived from vLLM's public process groups;
- fail-closed request bypass for incomplete media identities, prompt embeds,
  and LoRA requests;
- an explicit per-request `kv_transfer_params.spoolcache_bypass=true` control
  that skips both persistent restore and Store while leaving vLLM's in-process
  GPU prefix cache enabled;
- automatic public-contract compatibility checks, an installed-package content
  SHA-256 build attestation, and an allocator compatibility gate;
- unconditional discovery of cache groups, layer ownership, page sizes, block
  sizes, windows, DCP/EAGLE flags, and manager capacity from vLLM's startup
  `KVCacheConfig`, with no profile selector or model-size table;
- parameter-substitutability checks for the connector constructor, HMA hook,
  and all 18 connector callbacks before model allocation.

At startup, a multimodal model is queried through vLLM's
`MULTIMODAL_REGISTRY`. Every modality in `supported_mm_limits` whose configured
per-prompt limit is non-zero is enabled automatically. SpoolCache applies one
generic identity rule to all of them; it does not recognize model names or keep
a list such as `image`/`video`/`audio`. A request bypasses persistent reuse only
when vLLM did not enable its modality or did not provide a stable media ID and
placeholder offset/length. Thus an untested model is not rejected merely for
being absent from the qualification table below.

The verified-model table is evidence, not a runtime allowlist:

| Model/deployment | vLLM-declared modalities | Text KV | Image KV | Video KV | Audio KV |
|---|---|---|---|---|---|
| google/gemma-4-E2B-it, TP=1 development baseline | image, video, audio | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified |
| google/gemma-4-E2B-it, TP=1/PP=2 two-node G4 qualification | image, video, audio | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified |
| DeepSeek-V4-Flash-Vision-Uncensored, TP=2 | image | cross-restart + content verified | cross-restart + content verified | not runtime-enabled | not runtime-enabled |
| Qwen3.8-Flash-Next-NVFP4, TP=2 | image, video | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified | not runtime-enabled |
| GLM-5.3-Flash-EXL3, TP=2 | image, video | cross-restart + content verified | cross-restart + content verified | cross-restart + content verified | not runtime-enabled |

The current DeepSeek text maximum receipt restores an authenticated
130,048-token HMA prefix into an exact 196,608-token consumer after a complete
two-node restart; its full output hash matches the independent 196,608-token
cold control. The earlier real-image receipt restores a 12,288-token prefix,
and a different image produces a cold entry.
Qwen and GLM independently restored real COCO image and PyTorchVideo video
prefixes after local GPU/encoder/multimodal cache resets. Every listed hit has
an identical bypass/restore output hash and authenticated rank-local payloads.
Qwen was additionally qualified at a 260,800-token request length, just below
its declared 262,144-token limit: text, image, and video consumers each restored
an authenticated 160,000-token prefix on both ranks and matched the exact
bypass output hash. Two attempted 260,800-token stateful producers crossed the
runtime-proven boundary and were deliberately skipped as `unsafe_boundary`;
SpoolCache did not publish a guessed maximum-length snapshot.
The immutable media hashes and exact receipts are recorded in the performance
notes. None of the three compatibility deployments declared audio. The pinned
Gemma 4 E2B functional baseline has now restored authenticated text, image,
audio, video, and combined image+audio+video prefixes after an engine restart;
the bypass/restore output hashes match and the content oracles identify cats,
“Mary Had a Little Lamb”, and archery. The exact receipt is
[`2026-09-07-gemma4-text-multimodal-cross-restart-summary.json`](docs/receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json).
The separate G4 two-node PP=2 receipt repeats all five paths across both
pipeline stages, binds their different stage-local tensor ownership, and
requires the PP-aware vLLM startup handshake before any entry can reach quorum:
[`2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`](docs/receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json).
Any future runtime-declared modality is accepted by the same generic identity
path and is never gated by this table.

For model-backed development, the project uses two deliberately separate test
roles. Runtime compatibility development and qualification continue to use the
DeepSeek, Qwen, and GLM deployments above so that the automatic contract gate is
exercised against different vLLM builds and runtime-discovered layouts. Every
other model-backed feature, correctness, fault-injection, PP, performance, and
release test uses
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) as the
single development model. The reproducible starting revision is
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`; a future revision change must be
recorded with the resulting qualification receipt. CPU-only storage and
protocol tests remain model-independent. This test policy never participates
in runtime admission and must not introduce a model-name branch or allowlist.

This is still an alpha with a deliberately conservative scope. Stores and
restores are synchronous at the model-runner boundary. A post-admission restore
failure is fatal because the current public vLLM contract cannot safely roll back a
partially restored HMA transaction without a patch. Store failures only skip
publication and leave inference results unaffected.

The runtime package contains only production-connected cache behavior.
Unconnected `layerwise.py` and `publication.py` planning scaffolds and their
capability-looking tests were removed in G3c; the `StoreEconomics` calculator
now lives only in `benchmarks/`. The upstream layer-hook ordering experiment is
kept as an explicitly non-product qualification probe. Future performance work
will start from a measured benefit and the then-current public vLLM contract,
not revive either planner or add model-specific profiles.

## DSpark deployment

Version releases use GitHub Actions + python-semantic-release and PyPI Trusted
Publishing. See the [release workflow and first-publisher setup](docs/RELEASE.md).

Build one release wheel and install it into every deployment runtime using
[`docs/RELEASE.md`](docs/RELEASE.md). Select the resulting immutable image in the
deployment repository, then enable the connector:

```bash
SPOOLCACHE_ENABLE=1
SPOOLCACHE_ACCESS_MODE=read-write
SPOOLCACHE_HOST_ROOT=${HOME}/.cache/spoolcache
SPOOLCACHE_DIRECT_IO=required
```

The connector derives a stable model namespace automatically from vLLM's
public configuration: a non-empty `model_weights` locator takes precedence over
the possibly role-local `model` path, and `revision` is bound alongside it.
Served aliases and model configuration class names are irrelevant. The
launchers supply no SpoolCache model digest or compatibility profile; they only
preserve their ordinary vLLM locator/revision arguments and run the same
wheel installed in each immutable runtime image. SpoolCache does not copy or scan a model repository.
Production artifact immutability remains the responsibility of the image/model
publication process. Enabling SpoolCache also removes
`expandable_segments:True` before PyTorch starts; this is a launch configuration
change, not a vLLM patch.

### Runtime-discovered layouts

Layout and API compatibility discovery are always enabled and have no
environment switch or model/build selector. Startup hashes the bytes in the
installed vLLM package and checks every public callback shape against the
SpoolCache override before model allocation.

SpoolCache writes the internal `vllm-runtime-kv-v1` protocol identity after
inspecting vLLM's real cache groups. This is not a configurable profile and
contains no model-specific geometry. Unknown specs, callback drift, mixed reuse
semantics, or inconsistent registered tensors still fail startup. The model
locator/revision identity remains a cache-namespace boundary, not a compatibility
allowlist or proof of the underlying weight bytes.

Older deployments should delete `SPOOLCACHE_PROFILE` and
`SPOOLCACHE_EXPECTED_VLLM_VERSION`, `SPOOLCACHE_RUNTIME_COMPATIBILITY`, and the
old chunk/span/slot/pending/inventory/catalog/low-watermark tuning variables
and `SPOOLCACHE_CHECKPOINT_SHA256` from their environment. Remove the
corresponding lower-case fields plus `spoolcache_checkpoint_sha256`,
`spoolcache_profile`, `spoolcache_expected_vllm_version`, and
`spoolcache_qualified_gpu_mover` from hand-written connector JSON. Legacy
model-profile and `spoolcache-deployment/v1` entries remain isolated by their
old deployment identity and are not reused after the automatic namespace moves
new writes to `spoolcache-deployment/v2`. No startup path deletes them. Archive
or remove those old cache directories through normal operational cleanup when
they are no longer needed.

This mode reads the actual groups and tensors supplied by vLLM. For reusable
prefix state, SpoolCache asks vLLM's public `get_kv_cache_spec_kind()` resolver
for semantics instead of comparing concrete or inherited class names. Full
history, sliding-window, and align-mode recurrent semantics map to their
corresponding generic page-selection rules, so a new implementation class is
accepted automatically when vLLM assigns it an already supported semantic
kind. For a uniform per-layer group, SpoolCache first honors the aggregate
semantic kind returned by that same public resolver; only an aggregate
`unknown` falls back to the member declarations. This supports future
registered wrappers without naming their classes.

A request-owned scratch spec does not need a recognized concrete type or
semantic kind. It is accepted as one circular page only when a public boolean
capability explicitly excludes it from prefix sharing,
`max_num_blocks_per_req()` proves that its request block table owns exactly one
page, and `max_memory_usage_bytes()` equals exactly one physical page under the
current deployment config. If an admission-bound method is present, it must
also return exactly one block at the deployment's real maximum in-flight and
model-length bounds. For a packed group, the actual shared allocator/group must
independently prove the same one-page block-table and memory bounds, and its
prefix-sharing declaration must not contradict its members. Boolean results
are rejected as non-integers. Conflicting, missing, or failing ownership
evidence fails startup.

Public semantic kinds whose safe physical-page selection is not yet implemented
also fail closed. This is a cache-semantics boundary, not a model allowlist: no
model, architecture, modality, concrete cache class, or vLLM version is used to
grant access. Malformed page geometry, missing layers, and incomplete group
ownership likewise fail before persistent KV I/O is enabled. A different
runtime layout produces a different cache identity automatically.

Stateful recurrent/circular groups are published only while the request is at
the exact discovered boundary; an older or overshot state is skipped rather
than guessed. Producers may publish their complete aligned prompt, while a
consumer still leaves at least one token for local execution. This keeps the
rule independent of scheduler batch size and model identity. For an align-mode
recurrent group, vLLM copies the completed boundary into the next active state
page during input preparation, before the connector hook runs. SpoolCache
therefore captures and restores that active page derived from the runtime block
table, rather than the already-consumed historical page.

vLLM exposes a logical `KVCacheConfig` to the scheduler and a packed physical
page view to workers, so their byte counts may differ. SpoolCache gives those
views one shared logical layout digest for prefix keys and additionally binds
each rank's exact registered tensor dtype, shape, byte stride, storage offset,
and page geometry in its rank identity. Acceptance by the automatic gate is
not a model qualification claim; the target runtime, layout, topology and GPU
mover still require the round-trip and failure tests recorded below.

For a one-off request that must not read or write persistent KV, pass the JSON
boolean below in an OpenAI-compatible completion/chat request. This is mainly
useful for controlled experiments; omitting the field preserves normal cache
behaviour.

```json
{
  "kv_transfer_params": {
    "spoolcache_bypass": true
  }
}
```

In an isolated lab, starting vLLM with `VLLM_SERVER_DEV_MODE=1` enables a
GPU-only prefix reset without restarting the model:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/reset_gpu_prefix_cache.py
# For image/video/audio qualification, also clear vLLM's media/encoder caches:
PYTHONPATH=src .venv/bin/python benchmarks/reset_gpu_prefix_cache.py --multimodal
```

The helper explicitly keeps `reset_external=false`, so SpoolCache's NVMe
entries remain available for the next request. vLLM development mode also
exposes RPC/debug endpoints; leave it disabled outside development.

Maximum-context qualification is a host-safety exercise as well as a cache
test. Increase lengths gradually and monitor every rank host's SSH/API health,
`MemAvailable`, and `SwapFree` throughout each request. A request that bypasses
SpoolCache can still exhaust unified memory during model prefill. Do not replay
a length that previously required a host reboot merely because its connector
metadata contained no loads or stores.

## Operations

SpoolCache metrics are exported on vLLM's existing `/metrics` endpoint with a
`spoolcache_` prefix. The connector pre-registers a finite label vocabulary;
prompt data, token IDs, API keys, tenant salts and full entry IDs never become
metric labels. The surface includes lookup outcomes, authenticated hit tokens,
restore/store bytes and duration, skipped stores, post-admission failures,
quarantine events, scrub progress/failures, orphan and temporary-file cleanup,
incremental namespace items, bounded-shutdown failures,
live-cache/managed/quarantine disk usage, rank generations, quorum and readiness.

Each enabled worker (`restore-only` or `read-write`) starts a low-priority
rank-local deep scrubber. Its
production bounds are internal and model-independent: 64 MiB/s, at most 64
work items per step, a 60-second first-cycle delay for a new store, and a
six-hour cycle interval. Starting a cycle only records its cutoff and resets
the durable work tables; it does not scan the namespace. Separate snapshot
phases stream at most 64 raw paths per step under the maintenance lock and
commit them with idempotent SQLite inserts. A restart safely rescans the active
namespace phase from its beginning while retaining committed work, rather than
trusting a filesystem directory offset. Manifest/object progress, the live
reference set, and targeted-request acknowledgement remain durable in
`state/deep-scrub.sqlite3`.

Scheduler shutdown checks for cancellation between snapshot items and uses a
fixed five-second join timeout. Its `spoolcache-scrub-shutdown/v1` receipt
reports `stopped` or `timeout`; a timeout emits a synchronous structured log
receipt, then its daemon finalizer best-effort increments the persistent
`spoolcache_scrub_shutdown_failures_total` counter and defers closing the mover
and store until the scrub thread really exits. Neither a blocked counter fsync
nor a stuck scrub can make process exit unbounded, and storage is not closed
underneath an active reader. There are no scrub tuning environment variables
or per-model compatibility entries.

Before exposing startup inventory, each rank crash-consistently reserves a
strictly increasing generation epoch in `state/generation.json`. The first
upgraded worker enters a high epoch domain above legacy wall-clock values, and
`generation.required.json` makes later state loss ambiguous and therefore a
startup failure. A lower report is ignored only when its exact UUID/epoch was
previously observed; an unknown lower identity withdraws the rank. Each rank
root also has one lifetime inventory-owner lease, so overlapping worker
generations cannot retain independent reporter images.

If scrub finds known-bad content, it durably creates an entry-specific marker
under `state/inventory-withdrawn/` before attempting the quarantine rename.
Metadata rescans cannot re-admit that manifest after a rename failure. A
standalone maintenance process leaves the marker for the live owner to consume
in fixed-size, cursor-driven pages, including markers for manifests deleted
while no worker was online. Marker, managed-namespace, and shard creation
always re-fsync their parent, even on an idempotent retry. Only manifest
absence made durable by an exact-shard
fsync and locked recheck, a fully verified replacement commit, or full
deep-scrub authentication retires an entry marker. The marker intentionally
does not encode inferred provenance, so repairing one object never clears the
entry-level decision; successful full-manifest authentication schedules an
inventory rescan. Before traversing the potentially large set of
manifests which share a corrupt object, the store durably fences that digest
under `state/object-withdrawn/`. Lookup, startup scan, and bounded reporter
reconciliation all honor the fence, so termination after the first reference
cannot expose a later one. References are streamed rather than accumulated.
A collision preserves the bad inode as evidence, then atomically replaces the
live name with the already-fsynced object. The fence is released only after a
durable repair/quarantine receipt or a final locked re-hash. An unreferenced
fence is acknowledged only after every canonical manifest shard is fsynced and
the reference set is checked again. Manifest visibility and `reporter.add()`
share the same maintenance critical section, so a concurrent quarantine/remove
always wins after a published entry.

Operators can inspect progress or request immediate authentication of one
known entry without enabling a debug endpoint:

```bash
PYTHONPATH=src python3 -m spoolcache.maintenance status \
  --root /absolute/spoolcache/deployment-digest/rank-0000
PYTHONPATH=src python3 -m spoolcache.maintenance request \
  --root /absolute/spoolcache/deployment-digest/rank-0000 \
  --entry 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

The root must be one absolute, non-symlink rank store carrying SpoolCache's
ownership marker. Target status is one of `authenticated`, `absent`, or
`quarantined`. Treat the durable receipt as rank-local evidence only: vLLM
transports worker inventory changes on its connector stats channel after a
scheduler iteration. Before replaying the affected entry, wait until
`spoolcache_rank_quorum_entries` reflects the withdrawal. If the service is
idle, use an unrelated small `spoolcache_bypass=true` request as an explicit
report barrier; never use the affected prompt as that barrier.

The reference launchers expose deployment health checks (`./start.sh health`
for GLM and `./health.sh` for Qwen/DeepSeek), but SpoolCache itself does not run
a process or container supervisor. Following LMCache's connector boundary,
restart policy and complete TP/PP group recovery belong to the deployment's
external orchestrator. The connector exports bounded readiness, rank and fatal
metrics that an orchestrator may combine with API and rank liveness.

Any unrecoverable post-admission restore error still terminates the affected
worker with exit code 70 so partially restored KV is never used for inference.
Operators that require automatic recovery must configure their process manager
to replace the complete distributed group; persistent cache roots must not be
deleted as part of ordinary recovery.

## Development

Outstanding work is tracked in [`docs/TODO_GOALS.md`](docs/TODO_GOALS.md).
That checklist is the durable source for goal status, dependencies, acceptance
criteria, and completion receipts; update it when a goal is completed rather
than leaving the result only in a discussion or commit message.

Use `google/gemma-4-E2B-it` for every development or test path that needs a
real model, except compatibility work, which must retain the DeepSeek/Qwen/GLM
matrix. Pin the Hugging Face revision in each machine-readable receipt. Do not
substitute the development-model choice for vLLM runtime discovery.

Host-side tests use a lightweight `uv` development group. They deliberately do
not install vLLM, Torch, or CUDA into `.venv`:

```bash
uv sync --group dev
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Real-model development uses the root [`Dockerfile`](Dockerfile) and
[`compose.yaml`](compose.yaml). The image extends the official
`vllm/vllm-openai:v0.28.0` multi-architecture image at the pinned digest in the
Dockerfile, installs SpoolCache and the four optional audio decoder/resampler
packages without resolving or replacing vLLM's compiled
dependencies, and loads the connector through `kv_connector_module_path`.
The Compose baseline serves TP=1/PP=1 on port 8000 with a 16,384-token context;
long-context and distributed topology qualification use their dedicated safe
harnesses instead of silently changing this daily-development baseline. Since
this Compose file is development-only, it always sets
`VLLM_LOGGING_LEVEL=DEBUG` and `VLLM_SERVER_DEV_MODE=1`.
The latter exposes vLLM's development-only `POST /reset_prefix_cache`,
`POST /reset_mm_cache`, and `POST /reset_encoder_cache` routes; it is not a
SpoolCache production-management API.

```bash
.venv/bin/hf download google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
# First build/export the wheel path, SHA-256 and commit per docs/RELEASE.md.
docker compose build vllm
docker compose up -d vllm
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

Only one model service should use a DGX Spark GPU at a time. Stop the currently
active lab deployment before `docker compose up`; `docker compose down` stops
this development service but retains the named SpoolCache volume. The host
Hugging Face cache is mounted read-only and defaults to
`/root/.cache/huggingface`; set the standard `HF_HOME` Compose variable only
when the host cache lives elsewhere. Serving imports the wheel installed in
the image. Build and qualify a new release wheel/image to test code changes;
a restart never overlays a mutable source checkout.

For two-node PP development, use
[`scripts/gemma-pp2-dev.sh`](scripts/gemma-pp2-dev.sh). It fixes the qualified
Gemma revision, TP=1/PP=2 layout, vLLM development image and cache policy while
leaving host paths and CX-7 wiring in the optional ignored
`.env.gemma-pp2`. `start` first rejects occupied GPUs or a mismatched
image/model, uses the same wheel installed in the image on both hosts, starts the
worker stage before the head stage, and waits for `/health`. It is a launcher,
not a SpoolCache supervisor; failed startup removes the incomplete pair, while
ordinary `stop` retains both NVMe cache roots.

```bash
# Only needed when this node's wiring/paths differ from the checked-in defaults.
cp scripts/gemma-pp2-dev.env.example .env.gemma-pp2

# Build/download locally first. These explicit transfers use the worker's CX-7
# SSH address and are only needed when the worker is missing that exact input.
# First build/export the wheel path, SHA-256 and commit per docs/RELEASE.md.
docker compose build vllm
scripts/gemma-pp2-dev.sh image-sync
scripts/gemma-pp2-dev.sh model-sync

scripts/gemma-pp2-dev.sh preflight
scripts/gemma-pp2-dev.sh start
scripts/gemma-pp2-dev.sh status
scripts/gemma-pp2-dev.sh logs worker
scripts/gemma-pp2-dev.sh restart  # complete PP group; persistent roots remain
scripts/gemma-pp2-dev.sh stop
```

The `12,23` PP partition is a test-harness fact for this pinned Gemma/vLLM
pair, already validated without SpoolCache. It is not a model profile,
allowlist, or runtime default in `src/spoolcache`. Use DeepSeek/Qwen/GLM only
for the separate compatibility matrix through their own deployment launchers.

Development mode exposes vLLM's cache-reset endpoints. To prove an external
restore without deleting the named SpoolCache volume, reset only process-local
caches and explicitly keep the external cache:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/reset_gpu_prefix_cache.py \
  --api http://127.0.0.1:8000 --multimodal
```

Treat HTTP success only as proof that vLLM accepted the reset. A SpoolCache hit
still requires cached-token, all-rank entry/span, payload, and output-oracle
evidence.

Run the real-runtime contract subset without starting the API server:

```bash
docker compose run --rm --no-deps -w /opt/spoolcache --entrypoint python3 vllm \
  -m unittest -v tests.test_hma tests.test_vllm_contract \
  tests.test_vllm_cache_semantics_runtime \
  tests.test_vllm_model_namespace_runtime
```

CUDA mover tests run automatically when CUDA Torch is available and otherwise
skip. See [`docs/SPOOLCACHE_DESIGN.md`](docs/SPOOLCACHE_DESIGN.md) for the
format, failure semantics, qualification receipt, and remaining work.
Project-specific agent instructions live in
[`spoolcache-development`](.agents/skills/spoolcache-development/SKILL.md).

## Performance probes

The scripts in `benchmarks/` measure the authenticated direct-I/O path, the
CUDA staging sequence, per-object versus event-driven restore synchronization,
shared `O_DIRECT + cudaHostRegister` staging feasibility, and end-to-end
cold/store or cross-restart restore TTFT. The visual-prefix probe embeds a
deterministic PNG so a cross-process media-cache receipt is reproducible without
external files. They default to the qualified `2 x 64 MiB` staging
configuration and emit one machine-readable JSON object per invocation.

See [`docs/BENCHMARK_2026-09-04.md`](docs/BENCHMARK_2026-09-04.md) for the
matched enabled/disabled DSpark results. The incremental P1–P6 decisions,
failures, raw measurements, and post-optimization receipts are recorded in
[`docs/PERFORMANCE_IMPLEMENTATION_NOTES.md`](docs/PERFORMANCE_IMPLEMENTATION_NOTES.md).
