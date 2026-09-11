# Current-code Gemma regression, 2026-09-08

Historical record for the pre-0.2.0 candidate: references to "current code" below
mean the artifact tested on this date, not the 0.3.0 token-file backend.

The overall no-regression qualification has an **unresolved PP=2 mixed-input
output difference**. Do not treat the fixed-fixture pass as resolving it.

| Run | Result |
| --- | --- |
| Installed vLLM 0.28.0/CUDA suite | 284 passed, one skip (no non-prefix scratch spec in this build) |
| Host suite, including fixed-runner failure assertions | 288 passed, 12 skips, 373 subtests passed |
| PP=1, TP=1, compiled, 16K | All five inputs restored before/after restart with matching cold output and authenticated payloads; request controls passed |
| PP=2, TP=1, eager, 8K, partition 12,23 | Maintained fixed runner passed all five inputs, both controls, both-stage payload checks and whole-group restart |
| Earlier PP=2 generated prompt | Mixed cold controls returned `DOGS_SPEECH_STREET` twice; restored output was `OTHER` despite 2,560 cached tokens and authenticated payloads on both stages |

The initial PP=1 mixed semantic-label assertion also failed: both cold controls
returned `OTHER`. That observation is retained in
[negative-observations.json](negative-observations.json); subsequent PP=1 mixed
checks used the explicitly stated cold-output-equivalence contract, not a claim
of correct media classification.

The fixed PP=2 prompt changed the input, not the output assertion. The earlier
failure remains open: no baseline comparison establishes whether it is a new
SpoolCache regression or pre-existing runtime behavior. Its initial generated
nonce was not retained, so the exact prompt cannot be replayed from API rows
alone. The maintained runner now fixes prompt text and retains each command,
run salt, media hash and request/result record. It exits nonzero on a mismatch;
no Agent is required to choose a plan or judge an output.

## Artifact and scope

Model: `google/gemma-4-E2B-it`, immutable revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, official vLLM 0.28.0 on two DGX Spark
hosts. PP=1 uses only the head. Exact image, source and wheel identities are in
[artifact.json](artifact.json) and [release.json](release.json).

The isolated candidate is `0.2.0rc1`, source commit
`eabe045bd4b55188846dc6fe6fa09bbebe06798c`. It was built twice with identical wheel
bytes from a clean committed snapshot of the current working package, with
python-semantic-release stamping the candidate version. The working package
files still match that snapshot apart from the candidate version stamp.
Both hosts used the same immutable installed-wheel image. No serving source
mounts or source PYTHONPATH were used. This was not a PyPI release or remote push.

The current package checks include `O_DIRECT`, path/capacity configuration and
independent `spoolcache.skip_read` / `spoolcache.skip_write`. Text/image/audio/video
restore 2,048 tokens; mixed restores 2,560. Each persistent hit requires matching
API span, scheduler/rank entry agreement, full payload authentication and cold
output equivalence. Wrong media labels in cold controls remain wrong labels;
see [cold-labels.json](cold-labels.json). These results do not establish media
recognition accuracy, new fault recovery, maximum context or throughput gains.

## Evidence

- [Summary](summary.json), [host test log](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/host-tests-final.log),
  [head installed wheel](installed-wheel.json), [worker installed wheel](worker-installed-wheel.json).
- [PP=1 payloads after restart](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp1/post-restart-payload-verification.json)
  and [request controls](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp1/request-flags.json).
- [PP=2 fixed runner summary](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-fixed/summary.json),
  [payloads after restart](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-fixed/post-restart-payload-verification.json),
  [request controls](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-fixed/request-flags.json).
- [Earlier PP=2 failure](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-exploratory-failure/failure.json),
  [interpretation](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-exploratory-failure/interpretation.json),
  [authenticated payloads](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/pp2-exploratory-failure/restore-payload-verification.json).
- [Raw logs, runner snapshots and authenticated candidate wheel](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/raw-evidence.tar.gz),
  [archive SHA-256](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/receipts/2026-09-08-current-gemma/raw-evidence.sha256).
- [Restored lab state](restoration.json).

PP=1 used the retained temporary orchestration; the maintained fixed runner was
executed on PP=2. Its CPU tests reject unstable controls, changed restored output,
invalid counters, missing rank restores, conflicting scheduler entries and
partial group restarts. A preliminary runner was cancelled before requests to
correct a test-only assumption: PP stage deployment identities differ, while
the common topology agrees. No package code changed during the live tests.

The PP=1 container's combined log includes an upstream `EngineDeadError` during
the requested Compose shutdown/restart; the replacement startup and requests
succeeded. The maintained verifier reads logs since each current container start
to distinguish previous shutdown diagnostics from errors in the tested process.

Both hosts initially had no active model services. Test groups were stopped and
the original development image tags restored afterwards. Persistent test roots
under `/root/.cache/spoolcache-qualification/spoolcache-gemma-current-iitfrkyo/`
were retained on their owning hosts; no existing persistent root was deleted.
