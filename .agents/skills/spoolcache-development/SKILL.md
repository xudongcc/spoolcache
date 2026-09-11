---
name: spoolcache-development
description: Develop, review, test, benchmark, document, and deploy the SpoolCache patch-free vLLM KV connector. Use for changes in the spoolcache package, its vLLM V1/HMA integration, bounded-memory NVMe storage, text or multimodal prefix caching, runtime-discovered cache semantics, capacity/GC behavior, performance qualification, or the linked two-node DSpark lab deployments.
---

# SpoolCache Development

Treat correctness and bounded memory as product requirements. Preserve the
out-of-tree connector boundary: do not patch, overwrite, or hot-patch vLLM for
SpoolCache functionality.

## Establish context

1. Resolve the repository root from this skill directory; do not assume the
   caller's working directory.
2. Read `docs/SPOOLCACHE_DESIGN.md` before changing cache semantics, identity,
   storage, HMA handling, failure behavior, or deployment support.
3. Read `docs/TODO_GOALS.md` before starting planned development. Keep its goal
   status, acceptance criteria, dependencies, and completion receipts current;
   do not leave completed or newly discovered work only in a conversation.
4. Read `docs/PERFORMANCE_IMPLEMENTATION_NOTES.md` before optimizing or running
   performance tests. Read `docs/BENCHMARK_2026-09-04.md` before comparing with
   the qualified DeepSeek baseline.
5. Read [references/live-lab.md](references/live-lab.md) before touching the
   two-node service or its private environment file.
6. Inspect `git status` in every repository involved. Preserve unrelated and
   pre-existing changes; do not assume an untracked file belongs to the agent.

## Choose the test model

- Use DeepSeek, Qwen, and GLM only for runtime-compatibility development and
  qualification. That matrix exists to exercise different vLLM builds,
  architectures, modalities, and discovered cache layouts.
- For every other test that needs a real vLLM model, use
  `google/gemma-4-E2B-it` at the project-recorded pinned revision. This includes
  feature development, correctness, fault injection, PP, asynchronous paths,
  routine performance work, and release/install qualification.
- Keep CPU-only identity, quorum, storage, scrub, and fault-contract tests
  model-independent. Do not download a model merely to run a duck-typed test.
- Record the exact Hugging Face revision in every live receipt. Changing the
  pinned revision requires a new qualification receipt; it never changes core
  admission logic.
- The development-model policy belongs only to tests, documentation, and
  receipts. Never add a Gemma, DeepSeek, Qwen, or GLM branch, profile, allowlist,
  default, or environment variable to `src/spoolcache`.

## Preserve invariants

- Use only public vLLM V1 connector hooks and external module loading.
- Fail closed for unknown cache specs, layouts, callback contracts, media
  identity, topology, or post-admission restore errors.
- Treat every HMA group and every TP rank as one logical transaction. Never
  advertise partial state as a hit.
- Parse topology and physical-rank values as literal non-boolean integers; do
  not repair a changed runtime contract with `int()` or another coercion before
  deriving rank ownership, paths, or identities.
- Keep payload memory bounded by fixed staging slots. The internal pinned
  budget is two 64 MiB slots per rank plus one 64 MiB CPU authentication
  slot: 192 MiB explicit payload staging. Account for metadata, OS page cache
  and the model runtime separately. The sole backend uses buffered I/O.
- Apply scheduler admission bounds to the complete in-flight lifecycle, not
  just one transient map. In particular, count restore lookups awaiting
  allocation together with allocated plans awaiting connector metadata.
- Separate cross-role logical layout identity from worker physical geometry.
  Scheduler owns logical prefix IDs and sends them through connector metadata;
  every worker must match the scheduler's coordination receipt, while its
  rank-local manifests additionally bind its role-local model view and exact
  page byte geometry.
- Treat `KVCacheConfig.kv_cache_groups` as the block-table-owning layer set.
  vLLM may add cross-layer KV-sharing aliases only when it registers the final
  tensors. Omit an extra registered name only after proving it is the exact
  same Torch storage view as a group owner, and bind the proven alias mapping
  into rank ownership. Missing owners or independent extra tensors fail closed;
  never add a model-name exception for shared KV layers.
- Use LMCache's basic vLLM path as a scope reference, not as a feature list:
  trust the final registered tensor mapping and runtime group metadata, while
  retaining the extra manifest/quorum checks required by persistent HMA. The
  existing in-process connector, store, and mover are the cache data plane; do
  not create a standalone SpoolCache engine/server, controller, CPU L1, or
  supervisor merely to mirror an optional LMCache architecture.
- For PP, follow the same runtime-fact principle: derive PP/TP/DCP coordinates
  from vLLM's public process groups and cross-check them with public parallel
  config. Require every PP x TP participant through the PP-aware handshake;
  stage-local layer sets may differ, but ordered page-selection semantics must
  agree. Do not copy LMCache deployment helpers that assume TP is intra-node
  or PP is inter-node, and do not add a model-specific partition or stage map.
- Include text tokens, request cache_salt, model-locator/runtime identity, and
  qualified multimodal identifiers plus placeholder geometry in cache keys.
- Do not add model, architecture, or modality allowlists. For multimodal
  models, discover every enabled input from vLLM's public multimodal registry
  and deployment limits at startup, then apply one generic identity contract.
  A documented verified-model matrix is evidence only and must never become a
  runtime gate.
- Infer reusable cache semantics from vLLM's public semantic-kind resolver and
  public capabilities; never branch on a concrete or inherited cache-spec class
  name. A newly named implementation of a supported semantic kind must take the
  same path, while unimplemented semantic kinds remain fail closed. For a
  public aggregate group declaration, prefer its semantic kind and fall back
  to member declarations only when the group is unknown; reject group/member
  prefix-sharing contradictions. One-page scratch state must prove
  non-participation in prefix caching, exactly one request block-table page,
  and exactly one physical page under the current deployment config. A packed
  scratch group must repeat the page-ownership proof on the actual shared
  allocator/group. If an admission-bound capability exists, it must also prove
  exactly one block at the deployment's real public bounds. Never infer
  universality from sampled lengths, and reject booleans as integer results.
- Publish self-contained token files durably: write header and payload to a
  temporary, fsync it, publish the immutable key and fsync its shard before
  reporter admission. Each full-KV file covers the smallest multiple of the
  runtime's common reusable-group alignment that is at least 256 tokens;
  scratch capacity is excluded. Larger files stream through fixed
  credits, authenticating each range before GPU placement. Mixed HMA also needs
  an exact-boundary state file. Consecutive data keys and required state need
  every-rank quorum. Never serialize CUDA pointers, block IDs or request IDs.
- Pin the complete restore key set before releasing the namespace lock, and
  retain pins through the final CUDA drain. Release a failed read's staging
  credit before taking the quarantine lock so a concurrent writer cannot
  deadlock while waiting for that credit.
- Deep scrub one complete rank identity. Persist a lexical key cursor after
  complete file authentication, with fixed-size selection batches. Validate
  header identity, regular-file type, exact length and payload SHA-256 before
  success. Recheck concurrent target/cycle state under the rank lock; skip
  pinned keys. Files added behind the cursor are visited next cycle. There is
  no SQLite work queue or shared-reference graph.
- Reserve every worker inventory generation from a strict rank-local
  `state/generation.json` under the maintenance lock. Migrate legacy raw-clock
  epochs into a disjoint high domain and persist a separate initialization
  sentry; once that sentry exists, missing state fails startup. Atomically
  persist and directory-fsync every increment. Scheduler ordering may ignore a
  lower epoch only when its exact UUID/epoch is in bounded observed history;
  an unknown lower identity, equal epoch with a different UUID, or malformed
  identity withdraws the rank. A strictly newer epoch first withdraws the old
  image and recovers only from a complete checkpoint.
- Give each rank root exactly one lifetime inventory-owner lease, acquired
  before startup scanning and held until worker store close. Short-lived
  standalone maintenance must not compete for that lease; it leaves durable
  withdrawal markers which the sole reporter consumes under the maintenance
  lock before constructing its next stats report.
- Persist a key-specific withdrawal marker before quarantining known bad
  bytes. Metadata-only inventory must honor it across reopen. Only durable
  absence, complete replacement or authentication of that exact key can clear
  it. Keep publication/reporter admission and catalog scan/replacement inside
  their respective rank critical sections, so a stale add cannot undo removal.
  Withdrawing an intermediate key makes dependent chains miss through
  consecutive-key quorum; do not reintroduce object fences or reference GC.
- A marker operation has a durability receipt only after its parent directory
  fsync succeeds. Idempotent marker creation must re-fsync the parent even when
  the marker already exists, so a retry can complete a prior failed receipt.
  Apply the same rule to managed top-level namespaces, lock/marker namespaces,
  and token-file shard directories: validating an existing directory does
  not prove that its link survived an earlier failed ancestor fsync. Keep shard
  initialization inside the temporary payload's cleanup scope so a persistent
  parent-fsync failure cannot accumulate one full `.part` file per retry.
  Cleanup must still attempt both unlink and temporary-directory fsync, but a
  secondary cleanup failure must never replace the primary publication or
  durable-state exception; preserve the primary and attach bounded diagnostics.
  Before acknowledging an absent-entry marker, fsync its manifest shard and
  recheck absence under the maintenance lock. Never make marker deletion more
  durable than the namespace mutation it certifies.
- Stream actual durable entry withdrawal markers before every startup/stats
  report, removing all marked held keys before constructing the wire image.
  Never truncate withdrawals to the acknowledgement page or probe every healthy
  inventory key in an empty marker namespace. Acknowledge durably absent keys
  in fixed-size, cursor-driven pages, including markers created while no owner
  was online. Never mutate the marker directory during its active scan.
- Build bounded startup/rescan inventory by streaming validation before heap
  selection. Do not select the newest raw `limit` paths and then filter them:
  newer tombstones or corrupt manifests would starve older healthy offers even
  though memory remains O(limit). Do not quarantine while an active `scandir`
  stream is reading the same directory; defer only a fixed batch until every
  directory stream closes, leaving the remainder for later scans/scrub.
  Inventory inspection must not refresh LRU.
- Normalize every data-driven manifest decode failure into `ManifestError`,
  including JSON integer limits, deep recursion, non-finite canonicalization,
  and payload type/value errors. Lookup/scan may quarantine these as clean
  misses; do not catch `MemoryError`, filesystem errors outside the existing
  I/O boundary, or arbitrary implementation exceptions as corrupt data.
- Recover only recognized abandoned token temporaries under the rank lock;
  preserve unrecognized paths and quarantine evidence. Capacity GC starts at
  80% and attempts 20% of keys per round, at most four candidates per batch,
  with no fixed stop watermark. Use persisted file-mtime LRU, reverse batch
  touches to favor earlier prefix keys, and skip pinned keys. Keep namespace
  scans bounded in memory and qualify their cost under sustained pressure.
- Treat model locator/revision as a deployment namespace, not byte-level
  checkpoint attestation. SpoolCache authenticates its own KV payload and does
  not scan, copy, rewrite, or inventory an entire model repository. Operators
  own model-artifact immutability through images, immutable revisions, or an
  explicit model locator/revision or cache-path change. Derive model identity
  from public vLLM `ModelConfig`: prefer non-empty `model_weights`, otherwise `model`, and bind
  `revision`. Ignore served aliases and configuration class names. Do not add a
  launcher-supplied digest, model profile, or repository file receipt.
- Build releases from a clean committed tree using `scripts/build-release.py`.
  Version, changelog and tag ownership belongs to python-semantic-release in
  `.github/workflows/release.yml`; PyPI uploads use Trusted Publishing and the
  original authenticated workflow artifact. Follow `docs/RELEASE.md` for first
  publisher setup and retries. Never rebuild an already published version.
  Install the exact same authenticated wheel into all target images without
  dependency resolution. Serving must import the installed wheel; do not bring
  back source synchronization, source bind mounts, or source PYTHONPATH. Record
  commit, wheel SHA-256, immutable image ID and model revision in receipts.
  Run `benchmarks/verify_release_install.py` against the retained wheel to prove
  installed bytes, import origin and absence of stale modules. Keep production
  development endpoints disabled; use the separate Gemma qualification harness.
- Do not ship future-only planners in `src/spoolcache`. Layerwise restore,
  shared staging, async store, native movers and
  compression are not active Goals. Create a new Goal only when an apples-to-
  apples benchmark proves a falsifiable need and the current public ownership
  contract supports one replacement path. When a replacement ships, delete
  the superseded path instead of adding a model/profile selector between two
  implementations.
- Runtime compatibility is one unconditional automatic contract gate. It must
  inspect all public hooks and prove each SpoolCache override accepts every
  base call shape, then attest the bytes of the actually installed vLLM
  package. Never add strict/auto, version, model, architecture, modality, or
  profile selectors as a substitute for that proof. Any additional public
  runtime helper used by discovery must be included in the same startup
  signature gate, and execution must use the exact callable that gate checked.
- Keep orchestration out of the package. SpoolCache owns connector, storage,
  integrity and bounded telemetry; it does not own Docker/SSH lifecycle,
  readiness endpoints, restart policy, or TP/PP group reconstruction. A
  deployment may combine API/rank liveness with connector identity, inventory
  quorum and fatal metrics in its external orchestrator. Any missing, extra,
  mistyped, wrong-rank, or wrong-coordination startup inventory must still fail
  API startup.
- Treat `VLLM_PP_LAYER_PARTITION` as an upstream qualification-harness setting,
  never a SpoolCache compatibility knob. If a fixed model/runtime needs an
  explicit partition, first prove it with SpoolCache bypassed and record the
  result; it must not enter production identity logic, a model table, or the
  verified-model admission path.
- Bound inventory at every trust boundary: local held state, each checkpoint
  page and combined delta, pending checkpoint accumulation, rank/report count,
  identity lengths and numeric counters. A startup subset is a safe false
  negative and should roll forward without rank withdrawal; any actual gap or
  malformed report withdraws the rank until a complete checkpoint arrives.
  Large additions also roll forward in bounded deltas. Prioritize removals;
  oversized removals must still withdraw the rank immediately. Checkpoints
  describe the emitted sequence, never a mixture with unreported additions.
  Inventory capacity is derived from the conservative retained-memory allowance,
  not a fixed key count. Long-chain descriptors share the runtime layout and
  coverage validation uses per-layer cursors. Do not restore the former
  4,096-key, 4 MiB combined-header or million-token caps. The user also removed
  the per-chain 64 MiB descriptor quota: do not restore descriptor accounting
  or repeated whole-prefix admission on file publication. Retain individual-file
  framing/checksum checks. Report active-chain memory growth separately from
  the fixed payload staging and inventory allowances.
  Startup sends the complete bounded held catalog. Do not take a further key
  slice that cuts token chains and requires unrelated requests before reuse.
- Never print API keys or the contents of deployment `.env`/`.env.dspark`
  files.

## Implement changes

1. State a falsifiable behavior or performance goal.
2. Add a CPU-only contract test first when possible. Use duck-typed vLLM
   fixtures so host tests do not require the target image.
3. Make the smallest connector/storage change that satisfies the contract.
4. Add comments at ownership, ordering, identity, and failure boundaries; add
   durable notes for non-obvious experiments and rejected designs.
5. Update the design/support matrix whenever behavior or qualification scope
   changes. Do not describe a probe as a production implementation, and do not
   equate absence from the verified matrix with lack of runtime support.

## Validate in layers

Run the cheapest relevant layer first and stop on unexplained failure:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
python3 -m compileall -q src tests benchmarks
```

Then run the real-vLLM contract tests inside the target image. For GPU mover,
two-rank service, multimodal, persistence, or interference changes, run the
corresponding container/live test from `references/live-lab.md`.

Unless the work is explicitly compatibility qualification, real-model tests in
this layer must use the pinned `google/gemma-4-E2B-it` development checkpoint.
After the feature receipt passes, run DeepSeek/Qwen/GLM only as the separate
compatibility matrix required by the change. Do not maintain per-feature model
choices.

For local TP=1/PP=1 functional development, use the repository `Dockerfile`
and `compose.yaml`. They pin the official multi-architecture
`vllm/vllm-openai` image and the project Gemma revision; do not install a second
vLLM/Torch/CUDA stack into the host `.venv`. Confirm no other model service is
using the Spark GPU before starting Compose. Treat the 16K Compose service as a
daily functional baseline only: Q1 maximum-context, G4 PP=2, multi-node,
post-admission failure, and release qualification still require their dedicated
harness and receipts. A new official image digest must pass the real runtime
contract tests before it replaces the recorded development base.

For routine two-node TP=1/PP=2 Gemma debugging, use
`scripts/gemma-pp2-dev.sh` rather than reconstructing the G4 container commands
or changing a MiaAI-Lab launcher. Keep its model/revision/image/partition fixed;
put only host wiring and path overrides in ignored `.env.gemma-pp2`. Use its
explicit CX-7 `image-sync`/`model-sync` operations when needed, then `preflight`
and `start`. The launcher must continue to use identical installed-wheel images,
start worker before head, replace the complete PP group, retain cache
roots, and remain external to `src/spoolcache`. It remains a qualification harness, not a supervisor or model compatibility
mechanism.

Accept a persistent hit only when all of these agree:

- API `cached_tokens` is positive and equals the expected aligned span;
- when a shorter producer is reused by a longer consumer, construct both from
  one deterministic longer token stream (the generic prefix benchmark exposes
  `--prompt-source-tokens`) and slice the producer from it; similar-looking
  independently tokenized prompts are not proof of an exact prefix;
- usage, prompt details, request ID, and token counts are actually present and
  have exact non-boolean types; never synthesize absent evidence with defaults
  or `int()` coercion, and reject cached tokens greater than prompt tokens;
- scheduler logs one entry ID and every TP rank restores that same entry/span;
- output passes a deterministic or semantic correctness oracle; HTTP 200,
  valid token counts, or an empty reasoning-model `content` field is not an
  output-correctness result;
- do not treat one free-form completion hash as a deterministic oracle. First
  repeat cold bypass controls with disjoint cache salts and require zero cached
  tokens plus identical output, or use a constrained continuation/semantic
  assertion whose expected answer is stable. Record any discovered
  nondeterminism and redesign the oracle before judging restored KV content;
- logs contain no CUDA, NCCL, checksum, layout, or partial-rank error.

For multimodal qualification, test each modality independently and at least one
mixed prompt containing every jointly enabled modality. Record the
media source, license/usage constraint, byte size, and content SHA-256; send
pinned bytes or a data URL so the producer and consumer cannot observe changed
remote content. Compare a persistent-cache bypass control with a SpoolCache
restore using the same rendered prompt, media, salt, seed, and sampling. Clear
vLLM's GPU prefix, encoder, and multimodal caches between them while retaining
external state, and require the scheduler plus every TP rank to agree on the
entry and span. If an HMA layout publishes state only at exact boundaries,
align the producer prompt or document any priming request needed to reach that
boundary.

For maximum-context live qualification, never jump directly from a short
smoke test to the declared model limit. Increase lengths through bounded
steps, and guard every request by polling every participating host's SSH/API
health, rank readiness, Linux `MemAvailable`, and `SwapFree` at a fixed short
interval. Record the thresholds and observed minima. Abort before the fixed OS
reserve is crossed; after each long request, clear only process-local GPU and
multimodal caches if needed and wait for memory to recover before proceeding.
An empty SpoolCache load/store plan proves only that the connector data path
was bypassed; it does not make a unified-memory prefill safe. Never directly
replay a length that previously hung both hosts or required a reboot. If a
stateful producer overshoots its exact runtime-proven boundary, preserve the
`unsafe_boundary` skip and qualify the maximum consumer with a shorter,
authenticated prefix instead of weakening the rule or adding a model case.

Do not infer a hit from latency alone. Report raw samples, medians, environment,
and limitations; do not turn normal variance into a speedup claim.

Run storage soak qualification in a fresh subprocess. Use a separate sampler
process to poll the workload's current Linux VmRSS from `/proc/<pid>/status`
with a post-start baseline; inherited process-lifetime `ru_maxrss` is
diagnostic only and must not decide pass/fail. Include a native-allocation
negative test whose burst remains below the old process high-water but is still
captured by the sampler. Sample token metadata, staging and scrub/control-state
peaks throughout incremental work; idle measurements alone are not peak-memory
evidence. Keep historical SQLite queue/vacuum receipts scoped to the retired
backend.

## Handle GPU-only cache reset

In the isolated lab, vLLM development mode may expose
`POST /reset_prefix_cache`, `POST /reset_mm_cache`, and
`POST /reset_encoder_cache`. Call the prefix route with
`reset_external=false` and `reset_running_requests=false` to invalidate the
in-process GPU prefix cache while retaining SpoolCache NVMe entries. Use the
two multimodal routes when testing image, video, or audio requests so their
processor/encoder caches cannot mask an external KV restore. Never use
`reset_external=true` unless the user explicitly asks to delete external cache
state.

Development mode also exposes low-level RPC/debug endpoints. Keep it off by
default in examples and production guidance. If the endpoint is unavailable,
use an oversized disjoint working set to evict GPU prefixes without restarting;
do not claim that a same-process warm request proves an NVMe restore.

## Finish the work

- Re-run focused tests after each fix, then the complete host suite.
- For offline full-payload receipts, require expected deployment identity, rank
  identity/number, topology, and layout; validate the fixed internal layout
  protocol without a user option. Validate every caller-supplied digest and
  numeric bound before deriving a manifest path, then compare identity before
  any payload I/O or success status.
- Record live receipts in the performance notes, including failures and
  nondeterminism discovered during qualification.
- For a live corruption test, select one authenticated token file from a
  known prefix, record its expected digest/length and dependent chains, and make an
  authenticated narrow backup. Change one byte only after the scheduler has a
  quorum offer. Require the client to reject an incomplete stream, observe
  readiness false and fatal exit 70, prove that all ranks stop, and restore the
  exact file before the replacement group can offer it. Verify the restored
  digest, full payload set, cross-restart cached-token count, all-rank
  entry/span agreement, and output oracle. Never delete or reset the
  persistent cache root to prepare or clean up this test.
- For a pre-admission scrub qualification, request a targeted key through an
  identity-bound `TokenFileScrubber`; the retired snapshot request/status CLI
  is unavailable. Wait for that key to be `quarantined`, proving removal and
  the bounded quarantine metrics. The rank-local receipt is not yet a
  scheduler receipt: vLLM transports worker stats after an iteration. Wait for
  the quorum gauge to withdraw the entry; if the service is idle, drive one
  unrelated small request with `spoolcache.skip_read=true` and
  `spoolcache.skip_write=true` as a report barrier. Never
  use the affected prompt for that barrier. Then clear process-local APC by a
  complete deployment-managed TP restart and require the affected request to
  record an external rank-quorum miss, zero cached tokens, and the established
  output oracle without any post-admission fatal event.
- Update `docs/TODO_GOALS.md` when a goal changes state. Mark it complete only
  after every acceptance condition and P1/P2 review issue is closed, and record
  the relevant commits and qualification receipts.
- Recheck repository diffs/status and service health.
- State exactly which model, vLLM build, topology, and paths were tested.
  Distinguish implemented support from simulated coverage and untested scope.
