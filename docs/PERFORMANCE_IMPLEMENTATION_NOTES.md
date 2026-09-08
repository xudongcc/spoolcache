# Performance and engineering evidence

This is an index of measured results and engineering decisions. Current behavior
is specified in [Design](SPOOLCACHE_DESIGN.md); instructions are in
[Development](DEVELOPMENT.md). The complete pre-rewrite work log is preserved in
[the documentation archive](archive/README.md), including rejected approaches,
raw command transcripts and failed experiments.

## How to interpret results

A CPU contract, isolated CUDA probe, installed-runtime test and live API workload
answer different questions. Report exact source/wheel/image/model revisions,
topology, raw samples, medians and resource limits. Do not combine a new local
implementation with an old live measurement and call it a new qualification.

An accepted persistent hit needs exact prefix identity, API usage, all-participant
entry/span agreement, full payload authentication and a stable output oracle.
A latency difference, HTTP 200 or an empty answer envelope is insufficient.

## Current-code Gemma regression (2026-09-08)

The [current candidate receipt](receipts/2026-09-08-current-gemma/README.md)
records installed-wheel runtime/CUDA checks and short PP=1/PP=2 workloads after
the public configuration and request-control cleanup. PP=1 restored all five
input kinds after restart with matching cold outputs and authenticated payloads.

An exploratory PP=2 mixed request returned `DOGS_SPEECH_STREET` in two independent
cold controls and `OTHER` after restoring 2,560 tokens. Both stages restored the
same entry and every payload authenticated. This is an unresolved output
consistency finding; byte integrity does not establish correct model output.
The first runner did not retain its generated prompt nonce, so that exact prompt
cannot be replayed from its API rows alone. The maintained runner now uses fixed
prompt text and records each request command/salt. Results from the fixed prompt
remain separate from the initial failed prompt; no assertion was relaxed.

The [follow-up diagnosis](receipts/2026-09-08-gemma-mixed-diagnosis/README.md)
reproduces the class of failure with fully retained inputs. The 2,560-token
boundary is inside the image's `[2308, 2574)` token interval. At that same span,
three native GPU restores produce exactly the same generated-token probabilities
as disk restore, while cold controls differ. The original and reproduced
persistent prefixes have identical object descriptors/digests on both ranks.
Preserving the encoder cache does not remove the difference. Batch invariance
makes this particular output agree but leaves probability drift, so it is not a
complete fix. No package cache semantics or e2e assertions were changed.

Some long padded cold controls also return `OTHER` or an incorrect media label.
Record output equivalence separately from media-recognition accuracy. This is a
functional regression exercise, with no new throughput or TTFT improvement claim.

## Recorded performance

The [2026-09-04 matched benchmark](BENCHMARK_2026-09-04.md) used one fixed
DeepSeek Vision TP=2 deployment. Cross-restart hit TTFT improved from 4.910 s to
0.918 s for the 8K workload and from 19.216 s to 1.036 s for 32K. Misses that
synchronously stored new data cost 31.9% and 8.4% more TTFT in that small sample.
No decode regression was observed there; this is not a universal speed guarantee.

| Engineering result | Recorded observation | Interpretation |
| --- | --- | --- |
| Single-pass restore authentication | 133.727 MiB read and hashed once in 0.142086 s median; 941.17 MiB/s | Isolated rank-local I/O, not end-to-end API throughput |
| Two-slot CUDA pipeline | 170-object staging: 4.647030 → 3.696252 ms; synchronizations 170 → 1 | Bounded event ownership improved the probe |
| Post-change short C1 decode | 80.54 tok/s median vs original enabled 80.65 | About −0.14%, within recorded variance |
| Shared direct-I/O/pinned staging probe | Feasible for a fixed 128 MiB pool | Feasibility only; production still has two separate pools |

The historical baseline performed two authenticated payload reads. The current
restore path performs one physical read/hash pass. Each object authenticates in
host staging before its GPU copy; earlier objects can already be on the GPU
before the final object authenticates. Completion requires the entire restore
and its CUDA work to finish. Any post-admission failure remains fatal.

## Storage and integrity progression

Reviews in [September 5](CODE_REVIEW_2026-09-05.md) and
[September 6](CODE_REVIEW_2026-09-06.md) drove strict runtime-call checks,
identity-bound verification, complete admission accounting, monotonic generation,
exclusive inventory ownership and durable shared-reference withdrawal.

Later fixes covered tombstone provenance, ancestor directory fsync, crash-atomic
collision repair, bounded healthy startup selection, hostile manifest decoding,
incremental namespace snapshots and bounded scrub shutdown. The current design
retains those constraints; deleting historical supervisors and source overlays
did not remove persistent generation safety state.

| Evidence | Scope |
| --- | --- |
| [Million-item namespace receipt](receipts/2026-09-06-g3d-namespace-million.json) | Incremental work and bounded namespace handling |
| [Storage soak receipt](receipts/2026-09-06-g3d-storage-soak-1000.json) | Repeated maintenance, reclamation and resource observations |
| [G4 review](CODE_REVIEW_2026-09-07_G4.md) | PP-aware identity and complete-group ownership |
| [G6 release receipts](receipts/2026-09-08-g6/README.md) | Reproducible artifact, installed bytes and live fault recovery |

For new soak runs, sample current Linux VmRSS from a separate process with a
post-start baseline. Inherited `ru_maxrss` alone can conceal allocation bursts.
Observe SQLite and state-tree peaks during incremental work, before vacuum;
a deliberately enlarged queue must demonstrate peak size above final idle size.
Fault injection cannot establish every physical power-loss behavior, and finite
sampling can miss shorter transients.

## Long-context qualification

- [Qwen](receipts/2026-09-07-g3d-qwen-max-multimodal-cross-restart-summary.json):
  text/image/video consumers reached 260,800 tokens using 160,000-token prefixes.
  Maximum-length stateful producers that crossed the proven boundary remained
  `unsafe_boundary`; no guessed snapshot was published.
- [DeepSeek](receipts/2026-09-07-g3d-deepseek-196608-cross-restart-summary.json):
  a 196,608-token consumer restored 130,048 tokens after full-group restart and
  matched the independent cold output hash.
- [GLM](receipts/2026-09-07-g3d-glm-107k-cross-restart-summary.json):
  the recorded 107K cross-restart result is separate from incomplete maximum
  qualification. The earlier 992,769-token attempt required host recovery and
  must not be directly replayed.

A bypassed connector does not bound model prefill memory. Use progressive lengths
and all-host memory/liveness guards from [Lab qualification](LAB.md#long-context-safety).
Remaining work stays in [Q1](TODO_GOALS.md#q1-remaining-qualification).

## Multimodal, PP and release evidence

[Gemma PP=1](receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json)
and [G4 PP=2](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)
record text, image, audio, video and mixed controls, payload checks and complete
restart reuse. Final tensor sharing is accepted only by exact storage-view proof.

G6 adds installed-wheel and corruption/recovery evidence. Its PP=2 long padded
image/mixed controls returned stable `OTHER`; the receipt proves output equality
and payload integrity for those cases, not correct media classification. Remote
PP-stage failure could leave head health responsive until deployment-owned stop.
See [the negative observations](receipts/2026-09-08-g6/negative-observations.json).

Current CLI, direct-I/O and path changes have CPU/storage and launcher checks;
they are not a new live performance measurement. New optimization work requires
a measured need and a qualified ownership path, as recorded in [Goals](TODO_GOALS.md).
