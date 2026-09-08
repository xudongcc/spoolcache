# Development

Use CPU tests for configuration, identity, storage and protocol work. Use the
pinned development runtime for CUDA and real-model checks. Keep experiments
reproducible and separate measured results from implementation claims.

## CPU development

From a checkout containing the changes being tested:

```bash
uv sync --python 3.12 --locked --group dev
uv run --locked pytest -q
python3 -m compileall -q src tests benchmarks
```

This installs the project in editable mode and supplies pytest and the Hugging
Face CLI. It does not install vLLM, Torch or CUDA into the host environment.
Tests requiring those components report skips when they are absent.

Storage tests use real `O_DIRECT` even without a GPU. On Linux, point `TMPDIR`
at a supported filesystem if the default temporary directory is unsuitable.
A focused storage/configuration run is:

```bash
uv run --locked pytest tests/test_direct_io.py tests/test_manifest_store.py \
  tests/test_config_json.py tests/test_config_identity_prefix.py -q
```

CI additionally runs `python -m unittest discover -s tests -v` on Python
3.10–3.12. Use existing meaningful contract tests and failure injection rather
than replacing the storage path with a buffered test mode.

## Change workflow

Read [Design](SPOOLCACHE_DESIGN.md) before changing identity, storage, HMA or
failure behavior. Read [Goals](TODO_GOALS.md) before planned work and
[performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md) before optimization.
Agent-specific workflow instructions remain in the
[development skill](../.agents/skills/spoolcache-development/SKILL.md).

Define an observable behavior or measurable performance target, add the cheapest
relevant regression, make the smallest change, then validate the affected layer
and the full CPU suite. For CUDA, distributed or serving changes, qualify the
exact target image and preserve the model revision, topology and output evidence.
Do not modify vLLM to make the connector pass.

## Build the development image

The [Dockerfile](../Dockerfile) pins the official vLLM 0.28.0 image by digest.
It installs the supplied wheel and selected audio dependencies without resolving
a replacement Torch/CUDA stack. Tests and benchmarks are included as tools;
serving imports the wheel, not a bind-mounted `src/` directory.

The builder requires a clean committed candidate. Follow
[the release build and export commands](RELEASE.md#build-a-local-candidate), then:

```bash
docker compose build vllm
```

Keep `SPOOLCACHE_WHEEL`, `SPOOLCACHE_WHEEL_SHA256` and `SPOOLCACHE_COMMIT` exported
for subsequent Compose commands: interpolation is validated even for `ps` and
`down`. After code changes, build a new candidate/image and recreate the container;
restarting an old container does not install new code.

## Run the single-GPU development fixture

The repository fixture uses `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, TP=1/PP=1 and a 16,384-token context.
This is a test fixture, not a package model default.

Obtain model access and authenticate with `uv run --locked hf auth login` if
required. Set `HF_HOME` to the host model-cache directory, then download the
pinned weights:

```bash
: "${HF_HOME:?Set the host Hugging Face cache directory}"
uv run --locked hf download google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7

# Optional: export SPOOLCACHE_PATH=/mnt/nvme/spoolcache
# Confirm that the GPU is available before starting the fixture.
docker compose up -d vllm
docker compose ps
docker compose logs -f vllm
```

Compose mounts the model cache read-only and enables offline loading. The cache
bind mount defaults to the host user's `~/.cache/spoolcache`. Ctrl+C stops log
following; it does not stop the server. Stop it with `docker compose down`, which
preserves the host cache directory.

The fixture exposes vLLM development reset/RPC endpoints on the host network.
Use it only in an isolated development environment. Production deployments keep
`VLLM_SERVER_DEV_MODE=0`. The fixture is a daily functional baseline, not a
maximum-context or multi-node qualification.

## Runtime and CUDA contracts

Stop the model service so the test container has the GPU to itself. With the
build exports retained:

```bash
docker compose stop vllm
docker compose run --rm --no-deps -w /opt/spoolcache --entrypoint python3 vllm \
  -m unittest -v tests.test_gpu_mover tests.test_hma tests.test_vllm_contract \
  tests.test_vllm_cache_semantics_runtime \
  tests.test_vllm_model_namespace_runtime tests.test_vllm_pp_runtime
```

Run against the installed candidate wheel. Do not add `src/` to `PYTHONPATH`.
Generic feature/fault tests use the pinned Gemma fixture; DeepSeek/Qwen/GLM
belong to separate runtime-compatibility qualification.

## Verify a persistent hit

1. Establish two independent cold controls with distinct cache salts,
   both `spoolcache.skip_read=true` and `spoolcache.skip_write=true`, zero cached
   tokens and stable complete output. The flags leave vLLM's local caches active;
   each control salt must be fresh and distinct from the producer/restore salt.
2. Publish a sufficiently long exact prefix at a runtime-safe boundary.
3. Clear only process-local GPU/encoder/multimodal caches or restart the entire
   serving group while retaining persistent storage.
4. Replay the exact prompt/media/salt and require matching scheduler and every
   worker entry ID/span, the expected API cached-token count, valid payloads
   and the established output oracle.
5. Repeat after a complete group restart. Reject incomplete responses or missing
   usage evidence; do not synthesize token counts or infer correctness from HTTP 200.

In the isolated fixture, GPU-only resets are available through:

```bash
uv run --locked python benchmarks/reset_gpu_prefix_cache.py \
  --api http://127.0.0.1:8000 --multimodal
```

The helper keeps `reset_external=false`. Include encoder/media resets for
multimodal tests so those caches cannot mask a persistent restore. For shorter
producer/longer consumer tests, slice one deterministic token stream to prove
an exact shared prefix.

Authenticate entries with `benchmarks/verify_entry_content.py`. Supply the
expected deployment/rank/topology/layout identities, coordinates, span and
coverage from an independent trusted receipt. The tool's `--help` lists all
required fields; do not infer expectations from the untrusted manifest itself.

The text and multimodal clients accept `--skip-read` and `--skip-write`.
Use both with a fresh `--cache-salt` for cold controls. The interference benchmark
also uses both flags and generates a fresh salt for its GPU-only control. It
checks that priming had zero cached tokens before measuring GPU reuse.

## Fixed Gemma end-to-end regression

After starting the isolated fixture with an authenticated candidate image, run
`benchmarks/run_gemma_e2e.py`. The test matrix, prompt text and assertions are checked into
the repository; running it does not require an Agent to generate a test plan.
The client environment uses the ordinary development dependencies.

Prepare the three [pinned media files](LAB.md#media-fixtures) under `MEDIA_DIR`,
named `cats.jpg`, `archery.mp4` and `mary_had_lamb.ogg`. The runner checks their
sizes and hashes before sending requests. Keep the build exports from above:

```bash
: "${MEDIA_DIR:?Set the directory containing the pinned media fixtures}"
: "${QUALIFIED_IMAGE_ID:?Set the authenticated immutable Docker image ID}"
uv run --locked python benchmarks/run_gemma_e2e.py \
  --topology pp1 --head-container spoolcache-dev-vllm \
  --image-id "$QUALIFIED_IMAGE_ID" --media-dir "$MEDIA_DIR" \
  --output /tmp/spoolcache-gemma-pp1-receipt \
  --restart-command 'docker compose restart vllm'
```

For the running two-node fixture, use:

```bash
: "${MEDIA_DIR:?Set the directory containing the pinned media fixtures}"
: "${QUALIFIED_IMAGE_ID:?Set the authenticated immutable Docker image ID}"
: "${WORKER_HOST:?Set the worker SSH host}"
uv run --locked python benchmarks/run_gemma_e2e.py \
  --topology pp2 --head-container spoolcache-gemma-pp2-head \
  --worker-container spoolcache-gemma-pp2-worker --worker-host "$WORKER_HOST" \
  --image-id "$QUALIFIED_IMAGE_ID" --media-dir "$MEDIA_DIR" \
  --output /tmp/spoolcache-gemma-pp2-receipt \
  --restart-command 'scripts/gemma-pp2-dev.sh restart'
```

Each output directory must be new. The restart command is parsed as an argument
list without a shell; use a wrapper script for environment setup or multiple
commands. It must restart every participant and retain the same cache roots.
The runner checks that every container restarted with unchanged cache identities
and the expected image/model revision. It leaves the service running afterwards;
restore the prior lab state when finished.

The fixed sequence covers text, image, audio, video and mixed inputs:

1. Two independently salted cold controls with both persistent read/write skipped;
   require zero cached tokens and identical complete output hashes.
2. Publish a shorter prefix, clear process-local caches, then restore it into
   the longer consumer. Require 2,048 cached tokens (2,560 for mixed input).
3. Match the scheduler and every rank's entry/span and authenticate every
   payload against independently logged runtime identity and page geometry.
4. Verify independent `skip_read` / `skip_write` behavior, including unchanged
   persistent manifest inventories when writes are disabled.
5. Restart the whole group and repeat every restore, output and payload check.

Each run uses fresh salts and records the request commands and results.
A failed assertion exits nonzero and retains the evidence. `summary.json` is
written only after the whole sequence passes. The output oracle establishes
cache equivalence: a stable `OTHER` classification is recorded as such and does
not prove media-recognition accuracy. Fault injection, maximum context, storage
soak and additional runtime/topology compatibility are separate qualifications.

## Diagnose a mixed-input output mismatch

`benchmarks/probe_gemma_mixed_cache.py` reproduces one fixed Gemma PP=2 case
and compares disk restore with **native GPU prefix caching at the same 2,560-token
boundary**. Run it against the isolated two-node fixture:

```bash
: "${MEDIA_DIR:?Set the directory containing the pinned media fixtures}"
: "${QUALIFIED_IMAGE_ID:?Set the authenticated immutable Docker image ID}"
: "${WORKER_HOST:?Set the worker SSH host}"
uv run --locked python benchmarks/probe_gemma_mixed_cache.py \
  --worker-host "$WORKER_HOST" --image-id "$QUALIFIED_IMAGE_ID" \
  --media-dir "$MEDIA_DIR" --output /tmp/spoolcache-mixed-diagnostic
```

The diagnostic keeps complete video/image media in a shorter native-cache
producer, omitting the following audio/text. It proves the exact common token
prefix, verifies actual native/disk hit counts, compares complete generated
outputs and per-token log probabilities, and authenticates disk payloads on
both ranks. Native-only controls skip both SpoolCache operations and must leave
persistent manifest inventories unchanged. It also compares preserved versus
reset encoder caches and repeats cold/native/disk requests.

Its exit status means the diagnostic collected and checked its evidence;
`summary.json` classifies the outcome. `divergence-reproduced-by-native-cache`
means the default runtime reproduces the cold/cache difference without disk
restore. It is **not** an e2e correctness pass. The fixed e2e runner retains its
strict cold-output assertion. See the [diagnosis receipt](receipts/2026-09-08-gemma-mixed-diagnosis/README.md)
for the connector-free baseline and the limits of `VLLM_BATCH_INVARIANT=1`.

## Maintained benchmark tools

| Tool | Purpose |
| --- | --- |
| `probe_gemma_mixed_cache.py` | Reproduce and attribute a mixed-output difference using same-span native/disk controls |
| `run_gemma_e2e.py` | Fixed five-input regression, request controls, all-rank payload checks and complete-group restart |
| `bench_prefix_e2e.py` / `bench_multimodal_prefix_e2e.py` | Exact text or multimodal producer/consumer requests and independent read/write controls |
| `bench_decode.py` / `bench_restore_interference.py` | Decode baselines and persistent-restore interference |
| `bench_manifest_io.py` | Existing-entry direct-I/O timing |
| `bench_cuda_staging.py` / `bench_restore_pipeline.py` | Current gather/scatter staging and bounded CUDA event ownership |
| `analyze_store_economics.py` | Interpret matched store and restore timings |
| `soak_storage_maintenance.py` / `qualify_namespace_scaling.py` | Storage maintenance and filesystem namespace scaling |
| `verify_entry_content.py` / `verify_release_install.py` | Authenticate cache contents and installed wheel contents |
| `reset_gpu_prefix_cache.py` | Reset development process-local caches |

These scripts live in `benchmarks/`. The old single-image visual probe has been
superseded by the multimodal client. Layerwise and shared-staging experiment
scripts have been removed with the inactive experiment paths; their historical
receipts describe the recorded experiments only.

## Performance and long-context work

Report matched cold, miss/store and persistent-hit measurements, raw samples,
medians, image/model revisions and resource limits. A microbenchmark does not
establish end-to-end TTFT or concurrent decode performance. See the historical
[benchmark](BENCHMARK_2026-09-04.md) and [engineering notes](PERFORMANCE_IMPLEMENTATION_NOTES.md).

Increase context length gradually while monitoring all hosts' liveness,
`MemAvailable` and `SwapFree`. Stop at the recorded safety boundary; a connector
bypass does not bound the model's prefill memory. Do not replay a length that
previously required reboot. Current remaining qualification is tracked in [Goals](TODO_GOALS.md).

## Two-node lab

`scripts/gemma-pp2-dev.sh` provides `preflight`, `image-sync`, `model-sync`,
`start`, `stop`, `restart`, `status` and `logs`. Its pinned model, runtime and
upstream partition are qualification settings. Host wiring, path defaults,
media fixtures and fault procedures are documented in [Lab qualification](LAB.md).

## Contribute and release

Use Conventional Commits. Run relevant checks before proposing a change.
python-semantic-release owns package versions, changelog and tags; GitHub Actions
publishes the authenticated wheel to PyPI. Follow [Release](RELEASE.md) when a
release is intended. A local edit or successful test run does not publish anything.
