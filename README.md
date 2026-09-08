# SpoolCache

SpoolCache is an experimental, out-of-tree vLLM KV connector. Its first
qualification was the fixed DeepSeek-V4 Flash Vision deployment in
[`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).
It keeps each runtime-discovered PP x TP worker's cache shard on that machine's
local NVMe and does not modify the vLLM installation.

## Quick start

[Install](#installation) · [Use with vLLM](#use-with-vllm) ·
[Develop](#development) · [Operations](#operations) ·
[Release guide](docs/RELEASE.md)

### Installation

The published version is [SpoolCache 0.1.0](https://pypi.org/project/spoolcache/0.1.0/).
Install it **in the same Python environment or container as your vLLM server**:

```bash
python -m pip install spoolcache==0.1.0
python -c 'import spoolcache; print(spoolcache.__version__)'
spoolcache-maintenance --help
```

SpoolCache requires Python 3.10 or newer; CI covers 3.10, 3.11 and 3.12.
Serving requires Linux, an NVIDIA GPU with a working CUDA/vLLM runtime, and a
writable persistent cache directory on local storage (NVMe recommended).
SpoolCache does not install vLLM, PyTorch, CUDA, or model weights. Install and
verify your vLLM runtime first; for the reproducible development setup, use the
pinned vLLM 0.28.0 image in [Development](#development).
The runtime contract is checked at startup; an arbitrary vLLM version is not
supported merely because the Python package installs successfully.

For distributed serving, install the same wheel in every participant's image.
Use the immutable wheel and `release.json` from the
[GitHub release](https://github.com/xudongcc/spoolcache/releases/tag/v0.1.0)
with [`Dockerfile.release`](Dockerfile.release); the complete build and SHA-256
verification commands are in [the release guide](docs/RELEASE.md).

### Use with vLLM

SpoolCache runs inside vLLM as a KV connector. There is no separate cache server
to start. The following single-GPU example uses the qualified Gemma revision;
obtain access to the model and download its weights first if needed.

Run these commands in the environment where both vLLM and SpoolCache are installed:

```bash
# Choose an absolute, writable path on persistent local storage.
export SPOOLCACHE_CONTAINER_ROOT="$HOME/.cache/spoolcache"
export SPOOLCACHE_NAMESPACE="my-service"
export SPOOLCACHE_ACCESS_MODE="read-write"
export SPOOLCACHE_DIRECT_IO="required"
export SPOOLCACHE_MAX_BYTES=214748364800  # 200 GiB per rank
mkdir -p "$SPOOLCACHE_CONTAINER_ROOT"

# This baseline uses the normal allocator; expandable_segments:True is unsupported.
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
export VLLM_SERVER_DEV_MODE=0

KV_CONFIG=$(python -m spoolcache.vllm.config_json)
vllm serve google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 16384 --gpu-memory-utilization 0.80 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-transfer-config "$KV_CONFIG"
```

`spoolcache.vllm.config_json` converts the five environment settings below into
validated `--kv-transfer-config` JSON, including the external connector module,
`kv_role=kv_both`, and `kv_load_failure_policy=fail`. Setting environment
variables alone does not attach the connector; pass the generated JSON to vLLM.

| Environment setting | Meaning |
| --- | --- |
| `SPOOLCACHE_CONTAINER_ROOT` | Absolute cache path **as seen by the vLLM process**; mount persistent storage here when using Docker. |
| `SPOOLCACHE_NAMESPACE` | Operator-chosen deployment namespace; use a different value to isolate deployments. |
| `SPOOLCACHE_ACCESS_MODE` | `read-write`, `restore-only`, `store-only`, or `disabled`. |
| `SPOOLCACHE_DIRECT_IO` | `required` enforces `O_DIRECT`; the filesystem must support it. `best-effort` permits a buffered fallback; `disabled` uses buffered I/O. |
| `SPOOLCACHE_MAX_BYTES` | Managed storage limit in bytes for each rank; background GC targets 90% of this limit. |

SpoolCache creates deployment/rank subdirectories below the root. Keep those
roots across restarts. A container's writable layer is not persistent cache
storage; use a bind mount or named volume. See [DSpark deployment](#dspark-deployment)
for the existing multi-node launchers and their host-path settings.

From another terminal, send an ordinary OpenAI-compatible request:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E2B-it","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":32,"temperature":0}'

curl --fail http://127.0.0.1:8000/metrics
```

No SpoolCache-specific request field is needed for normal caching. Reuse requires
an exact shared prefix under compatible model, revision, namespace and runtime
identities; a stored prefix must also reach the internal minimum span and a
runtime-safe boundary. The short request above checks serving only: a nonzero cached-token
count can come from vLLM's GPU cache. Persistent-hit qualification also requires
SpoolCache hit/restore logs, all-rank agreement and output/payload verification;
see [the development checks](#verify-persistent-restores).

## Features and qualified runtimes

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

Download one published release wheel and install it into every deployment runtime using
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
uv run --locked python benchmarks/reset_gpu_prefix_cache.py
# For image/video/audio qualification, also clear vLLM's media/encoder caches:
uv run --locked python benchmarks/reset_gpu_prefix_cache.py --multimodal
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
spoolcache-maintenance status \
  --root /absolute/spoolcache/deployment-digest/rank-0000
spoolcache-maintenance request \
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

### Set up a checkout and run CPU tests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git and
Python 3.12 (uv can provision the interpreter), then work from the repository root:

```bash
git clone https://github.com/xudongcc/spoolcache.git
cd spoolcache
git switch -c dev/my-change
uv sync --python 3.12 --locked --group dev
uv run --locked pytest -q

# Run a focused suite while changing storage behavior:
uv run --locked pytest tests/test_manifest_store.py -q
```

`uv sync` installs the project in editable mode for local development. Changes
under `src/spoolcache` are visible to host tests without setting `PYTHONPATH`.
The dev group supplies pytest and the Hugging Face CLI; it does not install
vLLM, Torch or CUDA. Runtime/CUDA tests skip when those dependencies are absent.
CI also runs `python -m unittest discover -s tests -v` on Python 3.10–3.12.

Read [`docs/SPOOLCACHE_DESIGN.md`](docs/SPOOLCACHE_DESIGN.md) before changing
cache semantics, and use [`docs/TODO_GOALS.md`](docs/TODO_GOALS.md) for planned
work and acceptance criteria. Project-specific development rules are in
[the development skill](.agents/skills/spoolcache-development/SKILL.md).
Real-model feature, correctness and fault tests use `google/gemma-4-E2B-it` at
revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`; DeepSeek/Qwen/GLM are the
separate runtime-compatibility matrix. This test policy never enters connector
model selection or admission logic.

### Build a wheel and a GPU development image

GPU development additionally needs Docker Engine with Compose GPU support,
NVIDIA Container Toolkit, enough GPU memory for the model, and local persistent
storage. The root [`Dockerfile`](Dockerfile) pins the official vLLM 0.28.0 image
by digest and adds SpoolCache plus optional audio decoders without replacing the
vLLM/Torch/CUDA stack. Serving always imports the wheel installed in the image.

After editing and testing, commit your changes locally using a Conventional
Commit. The candidate builder requires a clean tree, including untracked files,
and an empty output directory. Use a fresh path for each build:

```bash
# Stage your intended changes and commit them before running the builder.
# Example commit message: feat: support the new runtime contract
CANDIDATE_DIR="dist/candidate-$(git rev-parse --short HEAD)"
uv run --locked python scripts/build-release.py --output "$CANDIDATE_DIR"

# Keep these exports in the same shell for subsequent Compose commands.
export SPOOLCACHE_WHEEL="$CANDIDATE_DIR/$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' "$CANDIDATE_DIR/release.json")"
export SPOOLCACHE_WHEEL_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel_sha256"])' "$CANDIDATE_DIR/release.json")"
export SPOOLCACHE_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$CANDIDATE_DIR/release.json")"

docker compose build vllm
```

The wheel path must be relative to the repository's Docker build context.
`release.json` binds its SHA-256 to the source commit. Local candidate artifacts
belong to isolated development caches; they do not replace an already published
version. python-semantic-release stamps the next version in CI before building
the public artifact.
To reproduce a published wheel, build from its exact release tag as described
in [the release guide](docs/RELEASE.md).

Rebuild the wheel and image after code changes, then recreate the container with
`docker compose up -d --force-recreate vllm`. A restart alone keeps the installed
package. Retain the three `SPOOLCACHE_*` build exports even for Compose `ps`,
`run` and `down`, because Compose validates build-argument interpolation.

### Run the single-GPU Gemma development server

Set `HF_HOME` to the host cache path before downloading. Compose mounts this
path read-only into the container and runs with `HF_HUB_OFFLINE=1`, so the pinned
snapshot must already be present. If model access requires authentication,
complete the model's access process and run `uv run --locked hf auth login` first.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv run --locked hf download google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7

docker compose up -d vllm
docker compose ps
docker compose logs -f vllm
# After startup completes, run in another terminal:
curl --fail http://127.0.0.1:8000/health

# Stop/remove containers while retaining the named cache volume:
docker compose down
```

Use Ctrl+C to stop following logs; the server keeps running. Run subsequent
Compose commands in the shell containing the wheel build exports.

This baseline uses TP=1/PP=1, port 8000 and a 16,384-token context. Use one model
service per lab GPU at a time. The Compose service sets debug logging and
`VLLM_SERVER_DEV_MODE=1`, exposing reset/RPC/debug endpoints on the host network;
run it only in an isolated development environment. Production launchers keep
these endpoints disabled. `docker compose down -v` would delete the cache
volume; omit `-v` for ordinary stops, restarts and recovery.

### Verify persistent restores

With the development server running, reset only process-local GPU/media caches:

```bash
uv run --locked python benchmarks/reset_gpu_prefix_cache.py \
  --api http://127.0.0.1:8000 --multimodal
```

The helper keeps `reset_external=false`, preserving SpoolCache's NVMe entries.
A reset or HTTP 200 is not a persistence test by itself. Follow the
[qualification procedure](.agents/skills/spoolcache-development/references/live-lab.md):
establish two disjoint cold bypass controls, store an exact shared prefix,
reset local caches, restore, then restart the whole group and repeat. Check
cached-token spans, the same entry on every rank, authenticated payloads and
complete output equality. The [G6 receipts](docs/receipts/2026-09-08-g6/README.md)
include tested examples and known model/runtime limitations.

To run runtime contracts without starting an API server, first stop the model
service so the test container has the GPU to itself:

```bash
docker compose stop vllm
docker compose run --rm --no-deps -w /opt/spoolcache --entrypoint python3 vllm \
  -m unittest -v tests.test_hma tests.test_vllm_contract \
  tests.test_vllm_cache_semantics_runtime \
  tests.test_vllm_model_namespace_runtime tests.test_vllm_pp_runtime
```

### Run the two-node PP development harness

[`scripts/gemma-pp2-dev.sh`](scripts/gemma-pp2-dev.sh) targets the two-DGX-Spark
CX-7 lab, with passwordless SSH, Docker/GPU access and RDMA devices on both
hosts. It is not a generic multi-node installer. Stop the single-node Compose
service first. The harness fixes Gemma's revision, TP=1/PP=2 and the qualified
`12,23` partition; only host paths and fabric settings are configurable.

```bash
# Optional: copy once and edit for your lab's paths, addresses and interfaces.
cp scripts/gemma-pp2-dev.env.example .env.gemma-pp2

# After the wheel/image build and model download above:
scripts/gemma-pp2-dev.sh image-sync
scripts/gemma-pp2-dev.sh model-sync
scripts/gemma-pp2-dev.sh preflight
scripts/gemma-pp2-dev.sh start
scripts/gemma-pp2-dev.sh status
scripts/gemma-pp2-dev.sh logs worker
scripts/gemma-pp2-dev.sh restart
scripts/gemma-pp2-dev.sh stop
```

Set `GEMMA_HEAD_HF_HOME` in `.env.gemma-pp2` to the download's `HF_HOME` when it
differs from `/root/.cache/huggingface`. Each participant needs the same image
ID and pinned model snapshot. The worker starts first; failed startup removes
the incomplete group and ordinary stops retain both cache roots. A remote PP
worker failure can leave the head `/health` responding: observe every rank and
recover the complete group. This launcher does not provide automatic supervision.

### Contribute and release

Use Conventional Commits: `feat:` for features, `fix:` for fixes, and `docs:` or
`test:` for documentation/tests. Push your branch and open a pull request;
include relevant tests and, for GPU/runtime changes, qualification receipts.
Keep vLLM unpatched and keep model-specific rules out of `src/spoolcache`.

After changes reach `main`, GitHub Actions runs the test matrix, and
python-semantic-release updates versions, `uv.lock`, changelog and tags when a
release is due. The workflow verifies reproducible builds and a fresh wheel
installation, then publishes the same wheel to GitHub Releases and PyPI through
Trusted Publishing. Documentation/test-only changes do not bump the version.
Do not hand-edit release versions or tags; see [the release guide](docs/RELEASE.md)
for release policy and retry procedures.

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
