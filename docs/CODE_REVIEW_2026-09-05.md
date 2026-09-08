# Code review backlog — 2026-09-05

The following findings were recorded during review and resolved in the
follow-up implementation described under each item.

## CR-001 — P1 — Capability mode does not validate most hook signatures

Status: resolved. The compatibility gate now carries an explicit contract for
all 18 public connector callbacks SpoolCache currently overrides or relies on. It checks
parameter names, order, kind, and whether extensions are optional, then proves
the SpoolCache override accepts every base call shape. CPU tests cover removed,
reordered, newly mandatory, optional, keyword-only, `*args`, `**kwargs`, and
classmethod changes. The same gate also covers the constructor and HMA hook.

`verify_vllm_runtime()` validates compatible signatures for the connector
constructor and `SupportsHMA.request_finished_all_groups`, but the remaining
required connector hooks are checked only with `callable()`. A future vLLM
build can therefore add or reorder mandatory hook parameters, pass the startup
capability gate, and fail later when the engine invokes SpoolCache. This does
not meet the advertised startup fail-closed boundary.

Evidence: `src/spoolcache/vllm/compat.py`, around the `required_hooks` check.

Acceptance criteria:

- define the parameter contract used by every connector callback SpoolCache
  implements;
- reject removed, reordered, newly mandatory, or override-incompatible optional
  parameters at startup;
- add negative tests using fake vLLM base classes with signature drift.

## CR-002 — P2 — Qwen worker source synchronization can retain stale files

Status: resolved. The launcher now deletes only the fixed
`/tmp/qwen-spoolcache-source` staging tree before extracting the next snapshot.
A worker-side stale-file sentinel was removed by the next synchronization, the
new source was present, and the existing head/worker image preflight accepted
all 18 currently checked callbacks.

The Qwen launcher extracts a tar stream into the persistent
`/tmp/qwen-spoolcache-source` directory without first removing the old tree.
Tar overwrites files present in the new stream but does not delete a module
that was removed or renamed locally. A worker restart can consequently import
stale connector code that no longer exists on the head source tree.

Evidence:
`/root/projects/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/start.sh`, in the
SpoolCache preparation block.

Acceptance criteria:

- synchronize an exact source snapshot, either by replacing a staged tree
  atomically or by deleting the narrow remote staging directory before
  extraction;
- demonstrate that a file removed from the local source is absent on the
  worker after the next synchronization;
- retain the existing pre-load contract check on both nodes.

## CR-003 — P2 — Offline payload verification is not identity-bound

Status: resolved. The verifier now requires and reports deployment identity,
rank identity, physical rank, topology, profile, and layout expectations. It
compares them before the first payload lookup; negative tests cover wrong
deployment, rank, topology, profile, and layout values.

`benchmarks/verify_entry_content.py` validates the manifest checksum, object
hashes, padding, and caller-supplied group/page coverage, but it does not
compare the manifest's deployment, rank, topology, profile, or layout identity
against an expected value. A self-consistent manifest from the wrong
deployment or physical rank can therefore produce
`status=all-payloads-authenticated`.

Evidence: `benchmarks/verify_entry_content.py`, from manifest decoding through
the final status output.

Acceptance criteria:

- require expected deployment and physical-rank identity, plus topology,
  profile, and layout identity where applicable;
- fail when any expected identity differs, before reporting payload success;
- add negative tests for wrong deployment, wrong rank, and wrong layout.

## CR-004 — P1 — A post-admission rank failure can leave the TP request hung

Status: resolved. SpoolCache now crosses vLLM's catch-and-continue RPC boundary
by terminating the failed worker with fixed exit code 70. A CPU test patches
the process primitive and proves this branch is mandatory. Repeating the same
GLM fault made the request end in about 3 seconds; the worker monitor observed
the exit, shut down the executor, raised `EngineDeadError`, and stopped the API
container. After restoring the authenticated object, a complete two-node
restart again restored 14,336 tokens on both ranks with the original output
hash, and rank 0 passed full 77-object payload verification.

A live GLM TP=2 fault injection changed one byte in a rank-0
payload after the scheduler had admitted the entry. SpoolCache authenticated
the streamed bytes and raised `FatalRestoreError`, but vLLM's multiprocessing
worker loop caught the exception. Rank 1 had already entered the model
collective, so the HTTP stream remained open without producing a token and the
worker monitor did not initiate group shutdown.

Evidence: GLM request `cmpl-948b2eaf978a3900-0-affcdbad`, entry
`01502cde8f19`, rank-0 expected object digest `d6e057647054`; logs contain
`ObjectCorruptionError: object SHA-256 differs` followed by
`SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED`. The client was still blocked after
more than 50 seconds and had to be interrupted. The original object was
restored from an authenticated backup immediately after the experiment.

Acceptance criteria:

- a post-admission restore failure must terminate the failing worker instead of
  returning through vLLM's catch-and-continue RPC boundary;
- the vLLM worker monitor must observe the exit, close the complete engine
  group, and make the admitted request fail promptly rather than hang;
- a supervisor restart with the authenticated payload restored must recover
  the deployment, and the entry must pass full offline verification again;
- add a CPU test for the explicit worker-termination boundary and repeat the
  live TP=2 corruption injection.

## CR-005 — P1 — Multimodal qualification can accept an empty answer envelope

Status: resolved. The generic multimodal benchmark now constrains the answer
to an explicit set of labels, disables template reasoning through both public
template-key spellings without selecting on a model name, includes both
`reasoning` and `reasoning_content` in its diagnostic digest, and requires a
consumer response to contain exactly one allowed visible label. CPU tests cover
the structured-output request and rejection of an envelope with only hidden
reasoning. The Qwen image and video matrix was rerun: bypass and restore emitted
the same visible labels and full observable response digests (`CATS` and
`ARCHERY` respectively), while different fixtures produced cold `STREET` and
`KITCHEN` results.

The first generic client revision hashed a normalized response even when
`content`, `reasoning_content`, and `tool_calls` were all null. Qwen generated
eight internal reasoning tokens under the runtime's `reasoning` field, so the
client could assign identical hashes to two empty envelopes and call that a
content oracle. The client also passed only `thinking=false`; the target chat
template consumes the public `enable_thinking` spelling instead. HTTP success,
completion-token usage, and a hash of absent fields do not prove restored
content correctness.

Evidence: `benchmarks/bench_multimodal_prefix_e2e.py` and
`tests/test_multimodal_benchmark.py`; the live Qwen response reported
`completion_tokens=8`, `reasoning_tokens=8`, `content=null` before the fix.

Acceptance criteria:

- fail a consumer qualification run unless it has an observable semantic
  answer from a bounded, explicitly declared label set;
- ensure the output digest covers every reasoning-field spelling returned by
  the qualified vLLM runtimes;
- verify bypass and restore against the same semantic oracle, and retain full
  observable-message hashing when the runtime is deterministic;
- add a negative CPU test for a transport-successful empty answer envelope.

## CR-006 — P2 — Scheduler block IDs were coercive instead of fail-closed

Status: resolved. `_normalize_block_ids()` now accepts only literal
non-negative integers and rejects booleans, strings, negative IDs, and
non-sequence group values. The contract test covers both valid normalization
and each invalid shape.

The connector previously applied `int()` to every value returned by vLLM's
block-table hook. A future contract drift such as `true`, `"1"`, or another
coercible object could therefore select a physical page instead of failing at
the scheduler boundary. That weakens the same fail-closed guarantee enforced
for token IDs, HMA dimensions, and multimodal geometry.

Evidence: `src/spoolcache/vllm/connector.py::_normalize_block_ids` and
`tests/test_vllm_contract.py`.

Acceptance criteria:

- require every physical block ID to be a non-negative, non-boolean integer;
- reject malformed outer/group shapes before page selection or connector
  metadata publication;
- retain a positive test for normal list/tuple input from vLLM.

## CR-007 — P1 — Allocated external span was not checked against admission

Status: resolved. Admission now records the exact external-token delta returned
to vLLM. `update_state_after_alloc()` requires a non-boolean, non-negative
integer and, when nonzero, requires it to equal that recorded delta before it
publishes worker restore metadata. A legitimate zero from a non-selected
multi-connector path still cancels the pending admission. Store-progress
counts are no longer coerced with `int()`, so the existing strict boundary
validator sees malformed runtime values.

The previous allocation hook checked only whether `num_external_tokens` was
positive, then restored the full admitted `span_tokens`. A future scheduler or
multi-connector drift could allocate a different number of tokens while the
worker metadata still selected the larger stored span. The same code also
coerced computed/scheduled counts, allowing booleans or numeric strings around
the strict `_pre_forward_store_span()` contract.

Evidence: `src/spoolcache/vllm/connector.py` admission/allocation paths and
`tests/test_vllm_contract.py` runtime-integer cases.

Acceptance criteria:

- bind each pending admission to the exact external-token delta returned by
  `get_num_new_matched_tokens()`;
- reject nonzero allocation deltas that differ before worker metadata is sent;
- reject boolean, negative, or non-integer scheduler counts without coercion;
- retain the public zero-token cancellation behavior required by
  `MultiConnector`.

## CR-008 — P2 — Cache-salt identity accepted coercible non-strings

Status: resolved. Request salts now have one generic canonical boundary:
absent means the empty salt, a valid UTF-8 string is used byte-for-byte, and
every other type or unencodable string makes the request bypass persistent
restore/store. CPU tests cover booleans, integers, bytes, invalid Unicode,
empty/absent values, and a valid salt.

The scheduler previously converted truthy `cache_salt` values with `str()` in
both match and request-tracking paths. Internal API drift could therefore make
the integer `1` alias the string `"1"`, or give an opaque object's unstable
rendering a persistent identity. Prefix identity must never be repaired by
coercion.

Evidence: `src/spoolcache/vllm/connector.py::_request_cache_salt` and the
connector contract tests.

Acceptance criteria:

- accept only absent/empty or valid UTF-8 string salts;
- use accepted strings byte-for-byte in prefix identity;
- turn an invalid salt into a request-local persistent-cache bypass rather
  than an unsalted store;
- add positive and negative CPU cases without introducing an extra option.

## CR-009 — P2 — Unused `require_cache_salt` policy remained configurable

Status: resolved. The dataclass field and hand-written connector-JSON alias
were removed, and the removed-setting test now rejects
`spoolcache_require_cache_salt`. Valid salts remain part of the exact prefix
identity; absent salts use the empty value, and malformed salts bypass
persistence as defined by CR-008.

The environment renderer no longer emitted this setting, none of the four
deployments used it, and it did not affect layout or storage mechanics. Keeping
the hidden JSON-only switch contradicted the reduced essential configuration
surface and created a second policy path that deployment documentation could
not audit.

Evidence: `src/spoolcache/config.py`, `src/spoolcache/vllm/connector.py`, and
`tests/test_config_identity_prefix.py`.

Acceptance criteria:

- remove the unused field and alias instead of replacing it with another
  environment variable;
- reject the legacy lower-case setting as unknown;
- preserve normal salted/unsalted exact identity and malformed-salt bypass.

## CR-010 — P2 — Target images could shadow repository benchmark helpers

Status: resolved. `benchmarks/` is now an explicit repository package and the
target-image test recipe sets `/opt/spoolcache` as its working directory, so
regression imports the mounted SpoolCache qualification helpers instead of an
unrelated image package with the same top-level name.

The host virtualenv had no conflicting package and therefore passed, while the
DeepSeek target image kept `/vllm-workspace` as `sys.path[0]` and resolved five
test imports through that workspace's `benchmarks` package. This made the final image suite fail before exercising
the response validator, reset helper, interference helper, and offline payload
verifier tests. It could also make an operator invoke a different helper body
than the source tree under qualification.

Evidence: target-image `unittest discover` import failures,
`benchmarks/__init__.py`, and the corrected live-lab command.

Acceptance criteria:

- make repository helper imports deterministic and run target tests with the
  mounted repository as `sys.path[0]`;
- rerun the complete host and all three target-image suites;
- do not rename or special-case helpers per deployment.

## CR-011 — P1 — An unpinned DeepSeek revision produced a stable-looking identity

Status: resolved. The DeepSeek launcher now permits automatic checkpoint
identity derivation only from a lowercase 40-hex `DSPARK_REVISION`. An
operator may still use moving-tip weights only by supplying an explicit
64-hex `SPOOLCACHE_CHECKPOINT_SHA256`; the literal `unpinned` fallback was
removed. The deployment CI has a static regression gate for both conditions.

When the revision was empty, the launcher hashed the model ID together with
the literal string `unpinned`. Different checkpoint bytes fetched from a
moving branch would consequently reuse the same deployment identity and could
admit incompatible KV payloads after restart.

Evidence: the SpoolCache setup block in the DeepSeek launcher and its
`scripts/ci-validate.sh` recipe guard.

## CR-012 — P1 — Multimodal capability discovery coerced non-boolean results

Status: resolved. Startup now requires both `is_multimodal_model` and the
registry's `supports_multimodal_inputs()` result to be literal booleans. CPU
tests reject a string model flag and integer registry result; all three target
images retain the positive public-contract path.

Using `bool()` made values such as `"false"` and `1` look authoritative. A
future vLLM contract drift could therefore enable a persistent media identity
path even though the public capability result no longer had the shape that
SpoolCache had qualified.

Evidence: `_discover_multimodal_modalities()` and the connector contract test.

## CR-013 — P1 — Qualification clients accepted missing or coercible usage evidence

Status: resolved. Text and multimodal qualification now require the usage
object, prompt details, exact non-boolean integer counts, a non-empty request
ID, and a complete output. Text requires the forced completion count;
multimodal requires at least one completion token and an exact prompt count.
Both reject the impossible case where cached tokens exceed prompt tokens.
Negative tests cover absent fields, strings, booleans, wrong prompt counts,
empty IDs, truncated streams, and impossible cache usage.

Previously the multimodal client replaced absent usage/details with empty
mappings and converted values through `int()`. The text stream similarly
started from an empty usage mapping. A transport-successful but truncated or
schema-drifted response could thus emit a plausible receipt without proving
the claimed cache hit.

Evidence: `bench_prefix_e2e.py`, `bench_multimodal_prefix_e2e.py`, and their
unit tests.

## CR-014 — P2 — Offline verifier derived paths from unvalidated CLI identity

Status: resolved. All five caller-provided SHA-256 identities must now be
lowercase 64-hex values before the manifest path is constructed. Span, rank,
group, layer, and page expectations are also checked with strict positive or
non-negative bounds. A CLI test proves a traversal-shaped entry ID is rejected
before any manifest read; helper tests cover malformed digest and numeric
inputs.

The identity comparisons themselves were strict, but the verifier sliced and
joined `--entry-id` before validating its digest form. A malformed value could
escape the expected manifest shard path, weakening the claim that the final
receipt referred to exactly the operator-selected entry.

Evidence: `validate_expected_inputs()` in `verify_entry_content.py` and
`test_verify_entry_content.py`.

## CR-015 — P2 — DeepSeek source snapshots included the local virtualenv

Status: resolved. Both DeepSeek worker tar streams now exclude `.venv`, in
addition to replacing the narrow remote source directory before extraction.
The recipe CI requires exactly two exclusions so the TP=2 and optional TP=3
paths cannot diverge.

Copying a developer virtualenv is unnecessary, can be very large, and can
carry stale host-specific packages into a worker snapshot. It is not part of
the SpoolCache source artifact being qualified.

Evidence: both SpoolCache tar blocks in the DeepSeek launcher and the recipe
guard.

## CR-016 — P1 — Physical tensor-parallel rank was coercive

Status: resolved. The connector now accepts only a literal non-negative
integer from `get_tensor_model_parallel_rank()` and checks it against the
strictly parsed TP degree before constructing `RankIdentity` or a rank-local
path. The same strict parsing is used immediately for TP, PP, and DCP startup
values. Tests reject booleans, strings, negatives, and an out-of-topology rank.

The previous `int()` conversion could map `True` to rank 1 or `"0"` to rank 0
after an API drift. Since physical rank chooses the persistent shard, such a
coercion could swap otherwise shape-compatible TP payloads and corrupt model
output.

Evidence: connector initialization, `register_kv_caches()`, and
`_physical_tensor_parallel_rank()` contract tests.

## CR-017 — P2 — Restore admission bound did not count allocated plans

Status: resolved. Admission now counts the union of lookups awaiting
allocation and already allocated plans awaiting connector metadata. A
contract test reproduces vLLM's query-then-allocate ordering: the third request
is cold while two plans are pending, then becomes eligible when one slot is
released. All three CUDA/vLLM target suites pass this behavior.

vLLM invokes `get_num_new_matched_tokens()` and
`update_state_after_alloc()` consecutively for each request. Counting only
`_need_load` therefore returned to zero between requests and allowed one
scheduler output to accumulate an unbounded number of synchronous restores,
despite the fixed `MAX_PENDING_RESTORES` contract.

Evidence: restore admission in `get_num_new_matched_tokens()` and the connector
contract test.

## CR-018 — P1 — Remaining benchmark clients tolerated truncated response evidence

Status: resolved. The no-restart interference client now validates tokenize
counts, response IDs, text chunks, usage/details, exact generated-token counts,
and the invariant `cached_tokens <= prompt_tokens` before a request can
contribute to a PASS verdict. The decode throughput client likewise requires a
complete non-coercive usage receipt and observable output. CLI workload bounds
are validated before either benchmark starts. CPU tests cover missing,
boolean, string, impossible, and truncated count cases.

The initial CR-013 fix covered the direct text and multimodal qualification
clients, but the production-mode fallback still used empty defaults and
`int()` coercion. A string-valued cached count could therefore contribute to a
reported external-hit phase, while a cut-off foreground stream could silently
skew the interference comparison.

Evidence: `bench_restore_interference.py`, `bench_decode.py`,
`test_restore_interference_benchmark.py`, and `test_decode_benchmark.py`.
