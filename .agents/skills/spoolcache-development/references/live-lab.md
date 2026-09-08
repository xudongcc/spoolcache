# SpoolCache live lab

## Repositories

- Standalone implementation: `/root/projects/xudongcc/spoolcache`
- DeepSeek DSpark deployment:
  `/root/projects/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- Qwen DSpark deployment:
  `/root/projects/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks`
- GLM DSpark deployment:
  `/root/projects/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
- Read-only design references unless the user explicitly expands scope:
  `/root/projects/FujitsuPolycom/sparkcache` and
  `/root/projects/LMCache/LMCache`

All three MiaAI-Lab deployment repositories use `main` and keep their local
SpoolCache integration as one rebased commit above the latest upstream. Never
switch DeepSeek back to the former `xudongcc`/`spoolcache` branch. SpoolCache
is a separate repository. All serving paths install the same immutable release wheel into their target
images; see `docs/RELEASE.md`. Production launchers do not synchronize source.

## Safety and secrets

- Source a deployment's `.env`/`.env.dspark` only inside a command that needs
  its values.
- Never print, grep broadly, diff, or include a private environment file in a
  response.
- Do not place an API key directly in process arguments. Let benchmark scripts
  read `VLLM_API_KEY`/`DSPARK_API_KEYS` from the environment.
- The two devices are dedicated to development, so coordinated service
  restarts are authorized when required. Still confirm both ranks recover and
  the API becomes healthy.
- The deployments share the same two GPUs and API port. Run only one model
  cluster at a time, and restore the service that was active before a
  qualification run.

## Default model roles

Use `google/gemma-4-E2B-it` for every live development or test that needs a
real model, except runtime-compatibility work. The initial reproducible revision
is `3e22461f65e89153144f8adb70e3b8c2cc9845a7`; record the exact revision in each
receipt and requalify before changing it. This single functional baseline covers
feature, correctness, fault-injection, PP, asynchronous-path, routine
performance, and release/install work.

Use the existing DeepSeek, Qwen, and GLM deployments only as the compatibility
matrix across distinct target runtimes and discovered layouts, or for an
explicit deployment-specific qualification already tracked in
`docs/TODO_GOALS.md`. CPU-only storage/protocol tests remain model-independent.
These roles are test policy only: they must never become a connector model
allowlist, profile, configuration option, or source-code branch.

## Local functional container

The standalone repository's `Dockerfile` and `compose.yaml` are the default
single-node functional environment. They extend the pinned official
`vllm/vllm-openai:v0.28.0` image, load the installed release wheel through the
public external connector module path, and serve the pinned Gemma revision at
TP=1/PP=1 with a 16K context. The development-only service always enables vLLM
DEBUG logging and `VLLM_SERVER_DEV_MODE=1` for process-local cache reset
qualification. The latter exposes `POST /reset_prefix_cache`,
`POST /reset_mm_cache`, and `POST /reset_encoder_cache`; use
`benchmarks/reset_gpu_prefix_cache.py --api http://127.0.0.1:8000` instead of
hand-writing reset requests. Because the official serving image omits optional
audio decoders, the development Dockerfile adds only the version-pinned
av/scipy/soundfile/soxr packages with `--no-deps`; verify that its Torch/CUDA/
NCCL versions still equal the official base. Keep vLLM/Torch/CUDA out of the
host `.venv`.

Before `docker compose up`, inspect running containers and stop the currently
active model deployment through its own launcher; never let this baseline
compete with a two-node qualification service for the same GPU. `docker compose
down` retains its named cache volume. Do not use `down -v` or otherwise delete
that volume merely to reset vLLM's process-local caches. This local Compose
baseline does not qualify PP=2, maximum context, cross-node quorum, or
external group recovery.

## Qualified targets

- Two DGX Spark nodes, TP=2, PP=1
- DeepSeek-V4-Flash-Vision-Exp, HMA five groups / 170 cache tensors
- vLLM `0.25.2.dev0+g752a3a504.d20260714`
- API `http://127.0.0.1:8888`
- Rank-local NVMe, required `O_DIRECT`
- Runtime layout and compatibility discovery are unconditional. Startup checks
  the complete public hook contract and records the installed vLLM package
  content digest; there is no version/profile/model compatibility selector.

Qwen3.8-Flash-Next and GLM-5.3-Flash use the same two-node TP=2 lab with their
own launchers, runtime-derived model locator/revision namespaces, target images,
and runtime-discovered layouts. No launcher-supplied checkpoint digest or model
profile participates in compatibility. Consult
`docs/PERFORMANCE_IMPLEMENTATION_NOTES.md` for their exact
receipts. Never infer support from a model name: query vLLM's runtime contracts
and treat the table as test coverage only.

G4 also qualified the pinned `google/gemma-4-E2B-it` revision on the same two
nodes at TP=1/PP=2 over the CX-7 addresses. The exact receipt is
`docs/receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json`. Its
`VLLM_PP_LAYER_PARTITION=12,23` value is evidence for that fixed upstream
model/runtime pair only: establish any such partition with SpoolCache bypassed,
and never translate it into connector code, a model profile, or a support
allowlist. A PP receipt must show the same entry/span restored by every stage,
different stage-local ownership identities where applicable, complete payload
authentication on each global rank, and an output oracle after a full group
restart.

Use the standalone repository's `scripts/gemma-pp2-dev.sh` for routine
two-node Gemma PP debugging. Its checked-in defaults match the qualified CX-7
lab and its optional host-only overrides live in ignored `.env.gemma-pp2`;
model, revision, image, TP/PP and the qualified upstream partition remain fixed
in the launcher. Run `preflight` before `start`; use the explicit `image-sync`
and `model-sync` commands only when either immutable input is absent or differs
on the worker. `start` uses identical installed-wheel images on both hosts, launches worker
before head, and waits for API readiness. `restart`
replaces the complete PP group without deleting either cache root. This is a qualification harness, not a model compatibility profile or a
SpoolCache supervisor.

Treat every one of these as a qualification fact, not a universal limit or a
promise for another model/runtime.

## Host and runtime checks

From the standalone repository:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
python3 -m compileall -q src tests benchmarks
```

The target containers may lack pytest. Run the standard-library suite with only
`tests/` and `benchmarks/` copied into `/opt/spoolcache-qualification`, and that
directory as the working directory. Never put `src/` on the import path: the
suite must exercise the wheel installed in site-packages. The Gemma development
image already contains these qualification tools under `/opt/spoolcache`.

Use the deployment repository's validation and start/stop scripts. Transfer the
complete release image ahead of startup and verify identical image IDs and wheel
SHA-256 on both hosts. Runtime config rendering uses the installed package inside
the image. Private environment transfer remains deployment-owned; never print
its contents. There is no SpoolCache source transfer or cleanup during startup.

## Maximum-context host safety

Treat a runtime's declared `max_model_len` as a tokenizer/scheduler limit, not
proof that the two Spark hosts have enough unified/system memory for a cold
prefill at that length. Increase qualification lengths through bounded steps.
For every long request, poll both hosts at a fixed short interval and require
SSH reachability, API/group readiness, `MemAvailable` above the receipt's OS
reserve, and `SwapFree` above its reserve. The current G3d safety receipts use
a two-second interval, 3 GiB minimum available memory, and 4 GiB minimum free
swap. Close the client request immediately when any bound is crossed, then
recheck the complete group before continuing.

Do not directly replay the GLM 992,769-token cold bypass: it stopped around
144,832 computed tokens, both prior-boot journals reported NVIDIA
`NV_ERR_NO_MEMORY` and hung tasks, Spark-1 reported global OOM, and both hosts
required a hard reboot. Its empty SpoolCache load/store plan proves the cache
data path was not involved; it does not make the model prefill safe. After any
long request, clear only the in-process GPU/encoder/multimodal caches when the
development endpoint is already enabled, retain external cache, and wait for
the OS reserve to recover before the next case. If an exact-boundary HMA store
reports `unsafe_boundary`, keep the fail-closed result and use a shorter,
authenticated prefix for the maximum consumer rather than retrying the unsafe
length or changing core semantics.

## Deployment health and external recovery

SpoolCache does not provide a process/container supervisor. Check the active
deployment with:

```bash
# GLM
./start.sh health

# Qwen or DeepSeek
./health.sh
```

The launcher health check proves API and container liveness only. For production
readiness, the external orchestrator must additionally consume bounded
SpoolCache metrics and require every rank identity, inventory quorum and
fatal-clear condition. Do not treat `/health` or one running container as proof
that a distributed group is ready. Container restart and all-rank replacement
policy is deployment-owned and is not implemented by SpoolCache.

### Pre-admission deep-scrub corruption qualification

For a production-mode targeted scrub (`VLLM_SERVER_DEV_MODE=0`), first create
and authenticate a new exact-boundary entry on every rank, establish a stable
output oracle with disjoint cold bypass controls, and prove one cross-restart
persistent hit. Then:

1. Select one uniquely referenced descriptor from the exact rank-local
   manifest. Verify regular-file type, stored/logical length, logical SHA-256,
   and zero padding. Back up only that object and manifest, fsync both files and
   their directory, and independently re-authenticate the backup.
2. Under the rank maintenance lock, flip exactly one live payload byte and
   fsync it. Do not alter the manifest, another rank, or the persistent root.
3. Run `python3 -m spoolcache.maintenance request` with the exact
   `--root <absolute-rank-root> --entry <entry-id>` arguments and poll `status`
   until the same entry has a durable
   `last_request_status=quarantined`. Require the live object and manifest to
   be absent, bounded quarantine artifacts to be present, the checksum and
   scrub-quarantine counters to advance, and scrub failures to remain zero.
4. The local receipt does not prove scheduler withdrawal. Wait for
   `spoolcache_rank_quorum_entries` to lose the entry. vLLM's worker stats
   channel may be idle when there are no scheduler iterations; in that case
   send one unrelated, small `spoolcache_bypass=true` request as an explicit
   report barrier. Never send the affected prompt before quorum has fallen.
5. Perform a complete deployment-managed TP restart to remove all process-local APC.
   Replay the exact affected consumer and require zero cached tokens, one
   external `rank_quorum` miss, the established complete-output oracle, no
   restore on either rank, and no post-admission fatal event.
6. Recompute/publish the exact safe-boundary prefix or restore the authenticated
   backup only through a separately verified recovery step. Authenticate all
   live payloads again before deleting the temporary backup. Preserve
   quarantine evidence; never make it visible by moving corrupt bytes back.

For a production-mode post-admission corruption qualification
(`VLLM_SERVER_DEV_MODE=0`):

1. Produce one exact-boundary prefix and establish its client output oracle.
   A single free-form completion hash is insufficient: repeat cold bypass
   controls with disjoint salts and zero cached tokens. If their outputs differ,
   record the nondeterminism and use a constrained suffix or semantic assertion
   that is stable before testing restored content.
2. From independent startup/log receipts, supply deployment, rank, topology,
   profile, layout, span and coverage expectations to
   `benchmarks/verify_entry_content.py` on every rank.
3. If a prior same-process request could remain in vLLM's GPU prefix cache,
   perform one complete deployment-managed TP restart and wait for readiness; do not
   enable a development reset endpoint merely to shorten the procedure.
4. Select one descriptor from the exact manifest. Before mutation, check file
   type/length and logical SHA-256 against the descriptor. Copy it to a narrow
   temporary backup, fsync the backup and directory, then verify the backup
   digest independently.
5. Flip one byte in place and fsync it. Do not rename, delete, truncate, alter
   the manifest, or touch any other cache entry. Replay the exact prompt/salt.
6. Bound the client wait. It must fail without a complete receipt; observe
   worker exit 70 and require the external orchestrator to stop every remaining
   rank before restart. As soon as the old group is confirmed down, restore
   the authenticated backup atomically and fsync its directory so startup
   inventory cannot advertise corrupt bytes.
7. Wait for the external orchestrator to restore full group readiness. Replay the exact request
   and require the expected external cached-token count, scheduler hit, the
   same entry/span on every rank, full offline payload authentication and the
   original output oracle. Remove only the temporary backup after all checks.

Never clear a persistent root during this workflow. If object identity,
backup authentication, rank state, or restoration timing is uncertain, stop
the experiment and preserve the evidence rather than guessing.

## Reset only the GPU prefix cache

When an isolated qualification run explicitly sets `VLLM_SERVER_DEV_MODE=1`,
after the one restart needed to load that setting, invoke the repository helper:

```bash
set -a
source .env.dspark
set +a
PYTHONPATH=/root/projects/xudongcc/spoolcache/src \
  /root/projects/xudongcc/spoolcache/.venv/bin/python \
  /root/projects/xudongcc/spoolcache/benchmarks/reset_gpu_prefix_cache.py \
  --multimodal
```

The helper explicitly sends `reset_external=false` and
`reset_running_requests=false`; `--multimodal` additionally clears the media
processor and encoder caches. A 200 response means vLLM accepted the resets; it
does not mean allocated KV memory is returned to the OS. Run while idle;
otherwise vLLM may be unable to reset referenced blocks.

If development endpoints must remain disabled, use
`benchmarks/bench_restore_interference.py`: it builds a disjoint working set
larger than reported GPU KV capacity and naturally evicts older prefixes.

## Persistent restore evidence

After reset, replay exactly the same tokenized prompt, cache salt, model,
sampling settings, and multimodal bytes/geometry. Require:

1. API `cached_tokens` equals the expected aligned SpoolCache span.
2. Scheduler logs `spoolcache: hit` for one entry.
3. Both TP workers log `spoolcache: restore` for that entry and span.
4. Output satisfies the fixture oracle, whose stability was established with
   independent cold bypass controls or a deterministic semantic assertion.

The client must reject missing usage/details, coercible token counts, empty
request IDs, cached counts above prompt counts, and incomplete streams before
emitting a receipt. Do not fill a missing field with zero merely to make the
result machine-readable.

For image/video qualification, use immutable or locally pinned media bytes.
Record the source URL, usage/license constraint, byte size, and SHA-256. A
successful request with empty `content` does not satisfy item 4; reasoning
models may consume a short output budget without producing visible content.
Run a `spoolcache_bypass=true` control, reset the in-process prefix/encoder/
multimodal caches with external reset disabled, then replay the exact request
through SpoolCache and compare the complete answer or its SHA-256.

Reusable qualification fixtures (download to a temporary directory; do not
commit the media bytes):

- COCO `val2017/000000039769.jpg`, 173,131 bytes, SHA-256
  `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`;
  source and terms: <https://cocodataset.org/#termsofuse>.
- PyTorchVideo `archery.mp4`, 549,197 bytes, SHA-256
  `8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562`;
  source: <https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4>.
  The repository is Apache-2.0, but verify the asset's own rights before any
  redistribution.
- vLLM public `mary_had_lamb.ogg`, 65,449 bytes, SHA-256
  `c8f0a87f8d7e44f2d6e0f88ec63f6401b4f153f53fd14a9d730a5d1ba9927c4e`;
  source:
  <https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg>.
  It is a qualification input from the vLLM public asset bucket; verify the
  asset's own rights before redistribution.

`bench_multimodal_prefix_e2e.py` accepts matching repeated `--media-kind` and
`--media-file` pairs for mixed prompts. Keep the exact ordered set unchanged
across producer, bypass, local-cache reset, restore, and post-restart phases.

Probabilistic/speculative execution can vary short generated suffixes even at
temperature zero. Prefer a seeded request plus a semantic oracle. If a free
answer differs only in formatting, record that failure and repeat with the same
semantically meaningful constrained candidate set; do not silently select a
favorable sample. Use low-level page checksums for byte-level mover correctness.
