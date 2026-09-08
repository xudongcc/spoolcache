# G6 release qualification

`summary.json` identifies the exact local qualification candidate, all four
runtime images, fixed model revision and the PP=1/PP=2 live receipts.
`semantic-release-rehearsal.json` records isolated PSR version/build/install
rehearsals; it is not a claim of a remote publication. The final publication
receipt records GitHub/PyPI bytes separately after the workflow runs.

The candidate wheel was built from pre-squash commit `93e7ad4`, SHA-256
`9ab91e0eb4782b723891a379a7fe634fa1fae08f0329fa2e622ee29bb2e9925b`.
All four candidate images on both hosts authenticate that same wheel and every
installed package file. Source bind mounts and source sync are removed from
all serving launchers. Historical commit IDs remain local evidence after the
user-requested squash; they are not remote GitHub revision links.

## Evidence and interpretation

- Host: 258 passed, 12 skipped, 279 subtests. Four real CUDA/vLLM runtime suites:
  264 tests each; official/DeepSeek skip one unavailable non-prefix scratch
  contract, Qwen/GLM skip none. Three deployment launcher suites: 5 passed each.
- Both topologies: two independently salted cold bypasses with zero cached
  tokens, cold producer, GPU/MM-only reset and persistent restore, full group
  restart, then repeat after restoring the fault-injected object. All five
  paths compare complete output hashes, including finish and reasoning fields.
- Text/image/audio/video restore 2,048 tokens; mixed restores 2,560. Every rank
  must name the same entry and span; full object digests, lengths, zero padding,
  independently discovered group/layer/page coverage and identities are checked.
- Faults modify one previously authenticated object byte after quorum offer.
  A narrow backup is retained until exact-byte restoration and full-group
  restart verification succeed. All cache roots and unrelated entries remain.

## Negative observations and scope

See `negative-observations.json` and the raw archive. PP=1 initially used
ambiguous song labels; two cold controls selected SPEECH. Mutually exclusive
song-title labels then established a stable semantic oracle. The first PP=1
fault lacked an observable exit code and was repeated with external `strace`
restricted to `exit_group`; the actual exit was 70.

PP=2 long padded image/mixed cold controls stably return OTHER, although a
short image control identifies CATS. These two paths establish cache output
equivalence and payload integrity, not correct media classification. Alternate
prompt/choice experiments are retained; no favorable sample is substituted.

On remote PP stage failure, the head health endpoint can remain successful and
an inference stream can hang. The first fault run hit a 90-second external
deadline, stopped both ranks and authenticated the restored object. The repeated
fault observes the remote worker exit 70 and then performs deployment-owned
whole-group stop: the incomplete client stream is rejected, API readiness is
false after stop, and both old ranks are absent before object restoration.
No automatic upstream cross-host readiness propagation is claimed.

These are correctness/install receipts, not new performance measurements or
new real-model DeepSeek/Qwen/GLM qualifications. No 24-hour soak was required.
Media bytes are not redistributed; fixture origins and rights are documented
in the live-lab skill reference, while the receipts bind their exact hashes.

## Review

Reviewed the installed-package boundary, reproducible clean-commit builder,
wheel origin and file authentication, image equality/frozen IDs, production
configuration removal, schema/identity rollback rules, and release workflow.
PSR owns versions/changelog/tags; lock updates fail closed, release commits and
tags push atomically after tests, stale workflow SHAs are rejected, publication
uses the retained authenticated artifact, and OIDC permission is isolated to
the PyPI job. No connector backend or vLLM patch was added for release work.
The two upstream/model limitations above remain explicit supported-scope
constraints. Final live recovery and remote publication status are recorded in
the machine receipts rather than inferred from this review.
