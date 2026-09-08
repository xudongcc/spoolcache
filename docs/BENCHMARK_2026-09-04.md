# DSpark performance qualification — 2026-09-04

## Result

On the qualified two-node DeepSeek-V4 Flash Vision deployment, enabling
SpoolCache did not produce a measurable decode-throughput or GPU KV-capacity
regression. A cache miss in `read-write` mode does have a synchronous store
cost: median TTFT increased by 31.9% at 8K and 8.4% at 32K in this small
sample. Cross-restart hits reduced median TTFT by 5.35x at 8K and 18.55x at
32K relative to the disabled baseline.

This is a qualification receipt for one fixed model/image/profile, not a
general performance guarantee.

## Post-optimization addendum

The tables below are the original matched baseline and remain unchanged for
auditability. After implementing the no-vLLM-patch P1–P6 work, the qualified
service was restarted and checked again:

- a text request restored 7,168 of 8,205 prompt tokens on both TP ranks;
- a visual request restored 4,096 of 5,136 prompt tokens on both TP ranks and
  returned the same correct description of the embedded white PNG;
- five short C1 trials measured median decode throughput of 80.54 tok/s versus
  the original enabled baseline of 80.65 tok/s (about -0.14%, within run
  variance);
- the single-pass rank-0 object probe read and hashed 133.727 MiB once in a
  median 0.142086 s (941.17 MiB/s, physical/logical ratio 1.0);
- the isolated two-slot CUDA pipeline reduced 170-object staging time from
  4.647030 ms to 3.696252 ms and stream synchronizations from 170 to one.

The shared 128 MiB `O_DIRECT + cudaHostRegister` probe also passed, but the
production path still uses separate 128 MiB pinned and 128 MiB aligned-I/O
pools per rank. See
[`PERFORMANCE_IMPLEMENTATION_NOTES.md`](PERFORMANCE_IMPLEMENTATION_NOTES.md)
for commands, raw measurements, failures, safety boundaries, and the visual
cross-restart procedure.

## Environment

- topology: 2 x DGX Spark, tensor parallel size 2;
- model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`;
- model revision: `86f746b36186f0e567729a5c06a8c918caba82a9`;
- served model: `deepseek-v4-flash-vision-exp`;
- vLLM: `0.25.2.dev0+g752a3a504.d20260714`;
- runtime image digest:
  `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`;
- KV format: `nvfp4_ds_mla`, HMA 5 groups / 170 layers;
- SpoolCache mode: `read-write`, required `O_DIRECT`;
- staging per rank: 2 x 64 MiB pinned slots plus 2 x 64 MiB aligned I/O slots;
- persistent store: rank-local NVMe;
- vLLM prefix cache was cleared by a full two-node process restart before the
  restore measurements.

The engine reported 2,422,841 GPU KV tokens with SpoolCache disabled and
2,429,181 with it enabled (+0.26%, normal startup measurement noise). No KV
capacity loss was observed.

## Matched A/B results

The same generated prompts, cache salts, model settings, and seeds were used
on both sides. TTFT is measured at the streaming client. Long-prompt values
are medians of three runs; decode C1 uses eight waves and C6 uses five waves.

| Workload | Disabled | Enabled | Difference |
| --- | ---: | ---: | ---: |
| C1, 269 prompt + 128 output, decode | 80.41 tok/s | 80.65 tok/s | +0.3% |
| C1, aggregate/wall | 66.03 tok/s | 66.12 tok/s | +0.1% |
| C1, TTFT | 0.346 s | 0.348 s | +0.4% |
| C6, per-stream decode | 46.55 tok/s | 49.03 tok/s | +5.3% |
| C6, aggregate/wall | 193.77 tok/s | 212.99 tok/s | +9.9% |
| C6, TTFT | 0.944 s | 0.875 s | -7.3% |

C6 has substantial wave-to-wave variance (including scheduler/fairness
effects), so the positive difference must not be interpreted as a SpoolCache
speedup. The supported conclusion is only that this run found no decode
regression. C1 is the cleaner connector-overhead comparison because these
short requests are below the 1,024-token persistence threshold.

| Prompt | Disabled cold TTFT | Enabled miss + store TTFT | Cross-restart hit TTFT | Restored tokens | Hit speedup vs disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K (8,207 actual) | 4.910 s | 6.474 s (+31.9%) | 0.918 s | 7,168 | 5.35x |
| 32K (32,777 actual) | 19.216 s | 20.822 s (+8.4%) | 1.036 s | 31,744 | 18.55x |

All six restore requests hit on the engine and restored the same entry on TP
rank 0 and rank 1. `cached_tokens` in the API response matched 7,168 or 31,744
for every request. There were no partial-rank hits.

The synchronous miss penalty is expected in the current alpha: capture and
publication occur at the model-runner boundary. It should be considered when
choosing `read-write` for workloads with little prefix reuse. A future
asynchronous store path is the main optimization for making misses approach
the disabled baseline. `restore-only` avoids creating new entries, but needs a
pre-populated store and was not part of this A/B.

## Data-path probes

At the time of the original qualification, authenticated manifest/object reads
validated the full payload and then streamed it, so physical read bytes were
twice the logical restored bytes. The measured baseline rates were:

| Node | Entry | Logical restore | Physical read | Median I/O time |
| --- | --- | ---: | ---: | ---: |
| head | visual, 31,744 tokens | 435.15 MiB/s | 870.30 MiB/s | 0.307 s |
| head | text, 38,912 tokens | 521.66 MiB/s | 1,043.31 MiB/s | 0.307 s |
| worker | visual, 31,744 tokens | 486.71 MiB/s | 973.42 MiB/s | 0.275 s |
| worker | text, 38,912 tokens | 598.73 MiB/s | 1,197.45 MiB/s | 0.268 s |

The 63.32 MiB CUDA staging probe measured approximately 3.23–3.29 ms for
capture and 3.35–3.50 ms for restore (18–20 GiB/s). Staging copies are not the
current bottleneck. The later P1 implementation eliminated the authenticated
double-read; its post-optimization result is recorded in the addendum above.

## Visual-prefix receipt

The visual qualification used a 32,637-token request and restored 31,744
tokens from both rank-local shards after a complete restart. Its observed cold
TTFT was 21.26 s and warm TTFT was 1.17 s. The answer preserved all four image
colors. A different image with the same text missed and created a distinct
entry. This is a functional cross-restart receipt, not a paired disabled A/B.

## Reproduction

Export the deployment's API environment, then run from the SpoolCache root:

```bash
PYTHONPATH=src python3 benchmarks/bench_decode.py \
  --model "$SERVED_MODEL_NAME" --prompt-tokens 256 \
  --concurrency 1 --repetitions 8 --nonce <unique-run-id>

PYTHONPATH=src python3 benchmarks/bench_decode.py \
  --model "$SERVED_MODEL_NAME" --prompt-tokens 256 \
  --concurrency 6 --repetitions 5 --nonce <unique-run-id>

PYTHONPATH=src python3 benchmarks/bench_prefix_e2e.py \
  --model "$SERVED_MODEL_NAME" --target-tokens 8192 \
  --nonce <unique-run-id> --cache-salt <unique-run-id>
```

For a valid enabled/disabled comparison, restart both ranks between modes and
keep the model revision, runtime image, prompts, salts, sampling settings, and
warmup procedure fixed. For a persistent-hit test, first store while enabled,
restart both ranks to clear vLLM's in-process prefix cache, and replay the exact
prompt and salt.
