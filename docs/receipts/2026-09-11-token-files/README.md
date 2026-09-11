# Token-file release evidence

This summary retains final results for the 0.3.0 feature. Intermediate designs,
failed attempts, raw measurements and their original artifact identities are in
the [verified local archive](../HISTORICAL_RAW.md). Source checkpoints below are
pre-squash provenance; a version string alone does not identify a tested wheel.

## Reviewed package

The full-source review covers all 27 package modules at
`9061441bb52ab422ad1a600cbb9563397aef3dc0`. It fixes restore/writer lock ordering,
durable withdrawal acknowledgement and cleanup ordering, removes redundant
active-chain quota/preflight work, and avoids repeated idle inventory scans.
Runtime code remains identical during the final documentation cleanup.

| Check | Result |
| --- | --- |
| Host pytest | 246 passed, 17 runtime skips, 306 subtests |
| Installed Python 3.11 unittest | 257 tests, 17 runtime/CUDA skips, passed |
| Installed Gemma CUDA/vLLM unittest | 257 tests, one skip, passed |
| Installed Qwen CUDA/vLLM unittest | 257 tests, passed |
| Installed wheel authentication | All 32 wheel-owned files verified |
| Earlier synthetic long-chain probe | 8,193 files, over two million synthetic tokens; 478,316 KiB peak RSS in its recorded Python 3.11 probe |

The reviewed development wheel retains version metadata 0.2.0 but is distinct
from published 0.2.0. Its SHA-256 is
`9a97c959b68e9c50a4a12372c21c72b667236f783f811682c27ae28231bbd4ea`.
Build dependencies are setuptools 80.9.0 and wheel 0.45.1; the package is pure
Python (`py3-none-any`) with a Python 3.11 minimum.

## Gemma correctness

On the exact reviewed wheel, `google/gemma-4-E2B-it` revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, official vLLM 0.28.0, TP=1/PP=1:
32 text/image/audio/video/mixed requests and ten complete prefix audits pass
across local cache resets and complete-container recreation. Text/image/audio/
video restore 2,048 tokens; mixed restores 2,560. All restored outputs match
repeated cold controls. Mixed `OTHER` proves equivalence, not recognition quality.
Restart advertises all 168 retained keys. No cache, CUDA or rank failure occurs.

Image: `sha256:874022e445b6c2f9dab52452a9402f35575fd98e67f2d8e15cbd7c40f97bd822`.
Raw source/build/test/model receipts are retained under
`/root/projects/xudongcc/spoolcache-backups/2026-09-11-unlimited-chain`.

## Qwen compatibility

The preceding `6843d49401c0cb7b8bc2792aa4c5986e8b681d7d` package qualifies
`nvidia/Qwen3.8-Flash-Next-NVFP4` revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, vLLM `0.1.dev20073+g8e685d198`,
Torch 2.13.0+cu130, TP=2/PP=1 with EP and three MTP tokens. Fifteen image/video/
mixed requests and twelve whole-rank prefix audits pass across group recreation.
Runtime-derived chunks are 1,664 tokens; all restores reuse 4,992 tokens.
Outputs match repeated cold controls: `CATS`, `ARCHERY`, `CATS_ARCHERY`.

Wheel SHA-256: `578230dce7bed657131a6ae6e187b84853824bdb303cc30e91e51994054e1fa0`.
Image: `sha256:e37999b69306cad0596988d1648877279378573e38b5d84e7cde994bf8f1e676`.
Raw receipts: `/root/projects/xudongcc/spoolcache-backups/2026-09-11-token-memory-budgets/qwen-retry`.
The final reviewed wheel's Qwen installed suite passes, but its requested
three-way model timing run was cancelled. No Qwen performance comparison is claimed.

## Performance

At 100,000 held keys, the final CPU operation probe changes empty-marker
reconciliation from 546 ms / 100,001 stat calls to 0.024 ms / one stat call.
Idle inventory reporting changes from 1.65 ms to 0.002 ms. Seven paired storage
trials against the preceding experiment show median save/restore changes within
-2.4% to +1.2%, with byte-oracle agreement. These are operation measurements,
not model latency improvements. Publication fsyncs and mutation/GC scans remain
scaling costs.

The historical DeepSeek comparison uses experimental checkpoint
`f8ba70efc00e9d0d4dceccecc9c876dbc07874d9`, not the later review build. Model:
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` revision
`86f746b36186f0e567729a5c06a8c918caba82a9`, vLLM
`0.25.2.dev0+g752a3a504.d20260714`, TP=2/PP=1. It passes 180 requests, 96 full-rank
payload audits and 48 equal-span/logical-byte comparisons. Concurrency is one,
with three measured contexts per length and warm OS page cache.

| Prompt tokens | No reuse TTFT | 0.2.0 first save | Experiment first save | 0.2.0 file hit | Experiment file hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,128 | 2.6179 s | 5.0285 s | 2.8760 s | 0.9569 s | 0.8730 s |
| 8,224 | 5.0312 s | 7.2307 s | 5.5841 s | 0.9310 s | 0.9217 s |
| 12,320 | 7.4352 s | 9.7328 s | 8.3262 s | 0.9641 s | 0.9793 s |

First-save TTFT is 14.5–42.8% below 0.2.0 and 9.9–12.0% above no reuse.
File-hit differences versus 0.2.0 are -8.8%/-1.0%/+1.6%; small differences do
not establish a general advantage. Buffered warm hits record zero physical reads;
0.2.0 uses O_DIRECT. This is not a cold-NVMe comparison. Separate-salt storage
allocation declines only 0.43%; the fixture does not measure growing-dialogue
space savings, eviction hit rate or sustained GC pressure.

## Qualification limits

Token-file PP>1, maximum-context model runs and sustained capacity-pressure
performance remain separate work. Synthetic token counts do not qualify a
million-token model workload. Stateful HMA needs an actually captured boundary;
full-KV files cannot recreate missing earlier recurrent/windowed state.

All model services are stopped and cache roots retained. Release CI must build
and authenticate the actual tagged wheel; the development wheel above is not a
published 0.3.0 artifact. No model-specific branch or configuration preset enters
SpoolCache runtime code.
