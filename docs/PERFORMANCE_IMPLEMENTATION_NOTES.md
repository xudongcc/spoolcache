# Performance and engineering evidence

Current behavior is specified in [Design](SPOOLCACHE_DESIGN.md) and
[Token files](TOKEN_FILES.md). The [final evidence](receipts/2026-09-11-token-files/README.md)
retains exact artifacts, validation, measurements and limits. Intermediate
reports and retired tools are in the [local archive](receipts/HISTORICAL_RAW.md).

## Source review and operation probes

Checkpoint `9061441` reviews all 27 package modules and removes redundant
quota/preflight work plus idle inventory scans. At 100,000 held keys, the CPU
probe reduces empty-marker reconciliation from 546 ms to 0.024 ms and idle
reporting from 1.65 ms to 0.002 ms. Seven paired storage trials against the
preceding experiment show median save/restore changes within -2.4% to +1.2%,
with byte-oracle agreement. These are operation measurements, not model speedups.

Durable publication fsyncs, save/GC directory scans and rebuilding indexes after
small inventory changes remain scaling costs. Any replacement must preserve
full-rank admission, pinning and durable withdrawal semantics.

## Token-file qualification

The reviewed package passes installed Python 3.11/CUDA tests and a 32-request
Gemma text/multimodal regression with ten complete audits across reset/restart.
A preceding aligned-file package separately passes Qwen image/video/mixed
compatibility. These are correctness results; the requested later Qwen
performance comparison was cancelled.

The historical DeepSeek three-way experiment uses `f8ba70e` and warm OS page
cache. Against published 0.2.0, first-save TTFT falls 14.5–42.8%; ordinary file-hit
differences are -8.8%/-1.0%/+1.6% across 4K/8K/12K prompts. First save remains
9.9–12.0% slower than no reuse. The [results table and scope](receipts/2026-09-11-token-files/README.md#performance)
include 180 requests, 96 rank/prefix audits and same-span checks. These are not
measurements of the later reviewed package or controlled cold-NVMe throughput.

Separate-salt tests do not establish growing-dialogue space savings, actual
workload hit rate or eviction behavior under pressure. Token-file PP>1 and
maximum-context model workloads remain unqualified. CPU probes representing
millions of synthetic tokens do not establish million-token model support.

## Retired experiments and decisions

Snapshot objects, packed/slot backends, SQLite reference indexes, O_DIRECT and
io_uring were removed when token files became the sole backend. Their tuning
sweeps and discarded designs are historical evidence, not current setup options.
Earlier fixed 256-token files were replaced by runtime-aligned files with a
256-token minimum. The active-chain descriptor quota and fixed chain/header/key
caps are removed; individual-file framing checks and bounded staging/inventory
remain. See [resource bounds](SPOOLCACHE_DESIGN.md#resource-bounds).

The public package remains model-independent. Follow [Goals](TODO_GOALS.md) for
remaining qualification rather than restoring superseded optimization plans.
