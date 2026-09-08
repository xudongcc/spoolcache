# G6: release and installation evidence

Recorded on 2026-09-08. G6 qualified an installed-wheel candidate on the pinned
Gemma fixture, then authenticated the first public 0.1.0 release and its installed
images. This directory is historical evidence; use the current
[release guide](../../RELEASE.md) for instructions.

## Artifacts

| Artifact | Source commit | Wheel SHA-256 |
| --- | --- | --- |
| Live-qualified candidate | `93e7ad49089e2b191eed1962b991fdad3653ce7d` | `9ab91e0eb4782b723891a379a7fe634fa1fae08f0329fa2e622ee29bb2e9925b` |
| Public 0.1.0 | `40528e685301bd5f45e1332255d69f492b3d8819` | `abaf71af05afd41c0d5a2873a003bc94af3a65b54dbe9db97c1015b2dd01bd28` |

The candidate commit predates the requested history squash. Its ID is local
provenance, not a promised remote revision link. Both wheels report 0.1.0 but
have different distribution bytes. The public installation receipt proves that
all 22 package files and metadata headers match the qualified candidate;
README description, its RECORD entry and ZIP timestamps differ. Qualification
was carried forward through that explicit equality check, not the version string.

## File map

| File | Purpose |
| --- | --- |
| [summary.json](summary.json) | Overall scope, artifact identities, results and limitations. |
| [release.json](release.json) | Clean-commit candidate build receipt. |
| [semantic-release-rehearsal.json](semantic-release-rehearsal.json) | Isolated version/build/install rehearsal; not publication proof. |
| [g6-clean-install-results.json](g6-clean-install-results.json) | Fresh candidate installs and tamper rejection. |
| [g6-runtime-stacks.json](g6-runtime-stacks.json) | Base versus candidate vLLM/Torch/CUDA/NCCL comparison. |
| [g6-image-quorum.json](g6-image-quorum.json) | Candidate image equality across participants. |
| [g6-official-install.json](g6-official-install.json), [g6-worker-install-results.json](g6-worker-install-results.json) | Candidate package authentication in runtime images. |
| [g6-runtime-results.json](g6-runtime-results.json) | Installed-wheel runtime test results. |
| [live-pp1.json](live-pp1.json), [live-pp2.json](live-pp2.json) | Text/media controls, restore, restart, payload and fault observations. |
| [negative-observations.json](negative-observations.json) | Failed controls, upstream limits and fault-test repetitions. |
| [raw-qualification.json](raw-qualification.json), [raw archive](raw-qualification.tar.gz) | Raw evidence and archive identity. |
| [publication.json](publication.json) | GitHub/PyPI bytes from successful workflow run 34180558915. |
| [pypi-install.json](pypi-install.json) | Fresh installation of the public wheel. |
| [published-installations.json](published-installations.json) | Public wheel authentication in four images on both nodes and candidate payload equality. |

## Recorded checks

Host tests: 258 passed, 12 skipped, 279 subtests. Each of four real CUDA/vLLM
runtime suites ran 264 tests. Official and DeepSeek skipped one unavailable
non-prefix scratch contract; Qwen and GLM skipped none. Each of the three
external deployment launcher suites passed five checks.

The live fixture was `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, with PP=1 and PP=2. Text, image,
audio and video restored 2,048 tokens; mixed prompts restored 2,560. Evidence
includes independently salted cold bypasses, process-local cache resets,
full-group restart, complete output hashes and every rank's authenticated
entry/span, payload length, digest, padding and page coverage.

Fault tests changed one previously authenticated object byte after quorum offer,
retained a narrow backup and restored exact bytes before recovery verification.
Persistent roots and unrelated entries were preserved. The final public wheel
was authenticated in eight installations across the four images and two hosts.

## Negative observations and limits

Initial PP=1 audio choices overlapped and cold controls chose SPEECH. Revised,
mutually exclusive song-title choices established a stable oracle; the failed
controls remain recorded. The first PP=1 fault lacked an observable child exit
code, so a repeat used external tracing of `exit_group` and observed exit 70.

PP=2 long padded image and mixed prompts stably returned OTHER in cold controls,
while a short image control returned CATS. Those long cases establish output
equivalence and payload integrity, not correct media classification. Alternate
prompt experiments remain in the evidence.

A remote PP worker exited 70 while the head health endpoint still succeeded and
the client stream hung. The first run reached a 90-second external deadline;
a repeated run observed exit 70 and performed a deployment-owned whole-group
stop. The incomplete stream was rejected, readiness became false after stop,
and both old ranks were absent before object restoration. This does not prove
automatic upstream propagation of remote-stage failure.

G6 did not rerun real-model DeepSeek/Qwen/GLM feature or maximum-context workloads,
measure a new performance gain, or require a 24-hour soak. Its review reported
no unresolved P1/P2 issues at that artifact boundary. Subsequent local changes
need their own qualification; see [Migration](../../MIGRATION.md).

Media bytes are not redistributed. [Lab qualification](../../LAB.md#media-fixtures)
records the fixture origins and hashes. The [documentation archive](../../archive/README.md)
preserves the original G6 narrative; machine receipts and raw evidence remain unchanged.
