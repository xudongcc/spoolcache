# Goals and status

Current contracts are in [Design](SPOOLCACHE_DESIGN.md); final results and exact
artifact scope are in the [release evidence](receipts/2026-09-11-token-files/README.md).
`[x]` means implemented and checked; `[-]` identifies remaining qualification.

## Token-file backend

- [x] Make runtime-aligned chained token files the sole backend and retain the
  standard connector entry point. Remove snapshot/slot storage, SQLite, native
  I/O and their exclusive tools/tests. Require Python 3.11+.
- [x] Derive chunk width from common reusable-group alignment with a 256-token
  floor. Stream larger files through fixed buffers; preserve complete HMA state
  and all-rank quorum without model presets.
- [x] Authenticate existing keys before skipping capture, protect active reads
  with per-key pins, and publish/withdraw files durably. Review all 27 modules
  and fix the reproduced lock-ordering and durability failures.
- [x] Use persisted LRU, favor earlier prefix keys and skip pins. Trigger GC at
  80%; attempt 20% of keys per round without a fixed stop watermark.
- [x] Remove fixed chain-key, aggregate-header and active-chain metadata quotas.
  Share geometry and validate incrementally. Keep payload staging and inventory
  bounded; send the complete admitted startup catalog through bounded reports.
- [x] Remove idle inventory scans and repeated quota/preflight work. Preserve
  removal ordering, generation, checkpoint and full-rank admission semantics.
- [x] Pass installed Python 3.11/CUDA tests and Gemma reset/restart regression on
  `9061441`; preserve the separate Qwen `6843d49` compatibility result and the
  earlier matched DeepSeek performance comparison.
- [x] Consolidate intermediate experiment records in a verified local archive;
  retain final results and current generic tests/auditors in PR #1.

## Remaining work

- [-] Measure shared-prefix traffic under sustained capacity pressure: actual
  hit rate, reused tokens, total latency, GC cost and memory. Save/GC directory
  scans and whole-rank index updates remain measured optimization candidates.
- [-] Qualify token-file PP>1 and maximum-context model workloads separately.
  Older snapshot results and synthetic long-chain probes do not qualify these.

The user cancelled the Qwen performance comparison; it is not an active task
or a release performance claim. The initial fixed-window, raw-block/io_uring,
message-boundary and per-chain quota experiments are retired.

## Permanent constraints

- Use public vLLM V1/HMA contracts; no model or runtime-version selectors.
- Require every cache group and participant before admitting a persistent hit.
- Keep fixed payload staging, bounded inventory and durable state ordering.
- Leave process supervision in deployment code and serve an authenticated wheel.
- Use pinned Gemma for ordinary live development, with other models reserved for
  compatibility or an explicit user request. Never delete caches as a reset.

## Previous releases

G1–G4 core/topology and G6 publication are recorded in the
[historical receipt index](receipts/README.md#qualification-groups). The former
G5 mandatory optimization program was withdrawn; GLM safe maximum-context
qualification remains incomplete. Those results apply to the original backend.
