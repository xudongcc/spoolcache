# Benchmark record — 2026-09-04

This historical qualification measured one fixed two-node DeepSeek Vision
configuration. It does not claim current-checkout performance on other models
or machines. The complete original report and reproduction commands are in
[the archive](archive/README.md); current procedures are in [Development](DEVELOPMENT.md).

## Environment

| Field | Recorded value |
| --- | --- |
| Hosts / topology | Two DGX Spark nodes, TP=2 |
| Model | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| Revision | `86f746b36186f0e567729a5c06a8c918caba82a9` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| Image digest | `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` |
| KV layout | `nvfp4_ds_mla`, five HMA groups / 170 layers |
| Storage | Rank-local NVMe, direct I/O, read-write mode |
| Staging per rank | 128 MiB pinned plus 128 MiB aligned I/O |

The matched sides used the same generated prompts, salts, settings and seeds.
Streaming-client TTFT for long prompts used medians of three runs; C1 decode
used eight waves and C6 used five. Complete group restarts cleared GPU prefix
state before persistent restores.

## Decode comparison

| Workload | Disabled | Enabled | Difference |
| --- | ---: | ---: | ---: |
| C1, 269 prompt + 128 output, decode | 80.41 tok/s | 80.65 tok/s | +0.3% |
| C1, aggregate/wall | 66.03 tok/s | 66.12 tok/s | +0.1% |
| C1, TTFT | 0.346 s | 0.348 s | +0.4% |
| C6, per-stream decode | 46.55 tok/s | 49.03 tok/s | +5.3% |
| C6, aggregate/wall | 193.77 tok/s | 212.99 tok/s | +9.9% |
| C6, TTFT | 0.944 s | 0.875 s | -7.3% |

C6 had substantial scheduler/fairness and wave-to-wave variance; its positive
differences are not evidence that SpoolCache accelerates decoding. C1 requests
were below the 1,024-token persistence threshold. The supported conclusion is
that these samples found no measurable decode regression.

GPU KV capacity was reported as 2,422,841 tokens disabled and 2,429,181 enabled
(+0.26%, interpreted as startup measurement noise).

## Cold, miss and persistent hit

| Prompt | Disabled cold TTFT | Enabled miss + store TTFT | Cross-restart hit TTFT | Restored tokens | Hit speedup vs disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K (8,207 actual) | 4.910 s | 6.474 s (+31.9%) | 0.918 s | 7,168 | 5.35x |
| 32K (32,777 actual) | 19.216 s | 20.822 s (+8.4%) | 1.036 s | 31,744 | 18.55x |

All six restore requests reported the expected 7,168 or 31,744 cached tokens
and the same entry/span on both ranks. The 31.9% / 8.4% miss penalty reflects
synchronous capture/publication. `restore-only` was not part of this comparison.

## Data-path baseline and later probe

The original path authenticated objects at lookup and again during restore,
reading twice the logical payload. These baseline measurements retain that
historical meaning:

| Node | Entry | Logical restore | Physical read | Median I/O time |
| --- | --- | ---: | ---: | ---: |
| head | visual, 31,744 tokens | 435.15 MiB/s | 870.30 MiB/s | 0.307 s |
| head | text, 38,912 tokens | 521.66 MiB/s | 1,043.31 MiB/s | 0.307 s |
| worker | visual, 31,744 tokens | 486.71 MiB/s | 973.42 MiB/s | 0.275 s |
| worker | text, 38,912 tokens | 598.73 MiB/s | 1,197.45 MiB/s | 0.268 s |

A later single-pass probe read/hashed 133.727 MiB once in 0.142086 s median
(941.17 MiB/s). The 170-object two-slot CUDA probe improved from 4.647030 ms to
3.696252 ms and reduced synchronizations from 170 to one. A shared-pool probe
passed feasibility checks; production retains separate pinned and aligned pools.
These are isolated probes, not a new matched API benchmark.

Five post-change C1 trials recorded 80.54 tok/s median against the original
80.65 enabled baseline (about −0.14%, within variance).

## Visual prefix and limitations

One visual request had 32,637 prompt tokens and restored 31,744 after restart.
Observed cold/warm TTFT was 21.26/1.17 s, with all four image colors preserved.
A different image with the same text missed. This is a functional receipt,
not a paired enabled/disabled A/B comparison.

Use immutable model/runtime inputs, identical prompts/salts and independent
cold controls for a new measurement. Preserve misses, failures and all raw
samples. Later context/PP/release observations are indexed in
[Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md).
