# Evidence index

Receipts record particular tests, artifacts and observations. They are not
current setup instructions or a compatibility guarantee for later code. Start
with [Compatibility](../COMPATIBILITY.md) for supported scope and
[Development](../DEVELOPMENT.md) for how to collect new evidence.

## Token-file evidence

The [final token-file evidence](2026-09-11-token-files/README.md) records source
review, installed-wheel checks, Gemma and Qwen correctness, the historical
DeepSeek comparison and remaining limits. It replaces the intermediate
experiment narratives. Complete originals and raw tools/results remain in the
[verified local archive](HISTORICAL_RAW.md).

## Qualification groups

These groups predate the token-file backend. Their summaries and artifact
identities remain here; detailed rows and logs are available through
[historical evidence](HISTORICAL_RAW.md). They do not qualify 0.3.0.

| Evidence | What it establishes |
| --- | --- |
| [Million-record namespace](2026-09-06-g3d-namespace-million.json) | Bounded namespace stress results for the recorded implementation. |
| [1,000-iteration storage soak](2026-09-06-g3d-storage-soak-1000.json) | Recorded storage/maintenance convergence and resource observations. |
| [DeepSeek long context](2026-09-07-g3d-deepseek-196608-cross-restart-summary.json) | 196,608-token consumer with a shorter authenticated persistent prefix. |
| [Qwen long text and media](2026-09-07-g3d-qwen-max-multimodal-cross-restart-summary.json) | 260,800-token consumers and authenticated shorter prefixes. |
| [GLM bounded context](2026-09-07-g3d-glm-107k-cross-restart-summary.json) | The recorded bounded run, not maximum-context completion. |
| [Gemma PP=1](2026-09-07-gemma4-text-multimodal-cross-restart-summary.json) | Text and multimodal persistent reuse on the pinned fixture. |
| [G4 PP=2](2026-09-07-g4-gemma4-pp2-cross-restart-summary.json) | Stage-local ownership, full-rank restore and cross-restart evidence. |
| [Gemma mixed-output diagnosis](2026-09-08-gemma-mixed-diagnosis/README.md) | Reproducible same-span native/disk comparison and runtime reproducibility limits. |
| [Pre-0.2.0 Gemma candidate](2026-09-08-current-gemma/README.md) | Installed candidate and fixed regression; includes an unresolved PP=2 mixed-output difference. |
| [G6 release](2026-09-08-g6/README.md) | Candidate qualification, first public release and installed-artifact authentication. |

The historical archive retains individual performance, scrub, fault and recovery
results.
[Performance notes](../PERFORMANCE_IMPLEMENTATION_NOTES.md) explain their context;
[Goals](../TODO_GOALS.md) distinguishes completed work from remaining qualification.

## Interpretation rules

Read the source commit, runtime/image identity, model revision, topology,
parameters and limitations together with the result. A passing storage stress
test is not an end-to-end model or host-memory qualification. A cached-token
count without rank agreement, output checks and payload authentication is not
proof of correct persistent reuse.

Pre-squash commit IDs are local provenance references. G6 explicitly separates
candidate and public artifacts even when they share a version string. The
CLI/path/direct-I/O changes have since been released; their
[Gemma regression evidence](2026-09-08-current-gemma/README.md) identifies a
separately built candidate. A release or matching version string alone does not
carry live qualification from one artifact to another. Published artifacts are
listed on [GitHub Releases](https://github.com/xudongcc/spoolcache/releases).

Retained machine summaries are unchanged. Bulk raw evidence and the previous
prose, including failed experiments and superseded procedures, are preserved
in [release history and verified backups](HISTORICAL_RAW.md). New tests should
produce separately identified receipts instead of rewriting historical outcomes.
