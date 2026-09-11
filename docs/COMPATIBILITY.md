# Compatibility and qualification

SpoolCache discovers public vLLM contracts, cache semantics and final tensor
ownership at startup. It has no model, architecture, modality or version
allowlist. Passing startup admission and qualifying a serving workload are
separate requirements.

## Implemented scope

| Area | Current behavior |
| --- | --- |
| Integration | External vLLM V1 connector with HMA support; no SpoolCache patch to vLLM |
| Storage | Rank-local runtime-aligned token files; buffered I/O and fixed CUDA/CPU staging |
| Prefixes | Exact tokens, cache salt and qualified multimodal identity in chained keys |
| Cache groups | Full, sliding and stateful semantics proved from runtime capabilities |
| Parallelism | Runtime-derived PP/TP/DCP participants; every required rank must agree |
| Integrity | Bound header identity, exact length, SHA-256 and complete HMA coverage |
| Failure | Safe miss before admission; fail-stop and deployment-owned recovery after admission |

The standard `spoolcache.vllm.connector.SpoolCacheConnector` is the sole entry
point. Chunk width is the smallest multiple of the reusable-group alignment
covering at least 256 tokens. Scratch capacity is excluded. Stateful HMA needs
an exact captured boundary as well as consecutive full-KV files. See
[Token files](TOKEN_FILES.md) and [resource bounds](SPOOLCACHE_DESIGN.md#resource-bounds).

Unknown contracts fail closed. LoRA, unsupported prompt embeddings and incomplete
multimodal identity do not receive ambiguous persistent hits. There is no remote
payload backend, encoder-cache persistence or GDS path.

## Token-file qualification

The [release evidence](receipts/2026-09-11-token-files/README.md) records exact
source, wheel, image and model identities. Later source or version changes do
not inherit live qualification from a matching version string alone.

| Fixture | Runtime / topology | Recorded result |
| --- | --- | --- |
| Gemma 4 E2B, checkpoint `9061441` | vLLM 0.28.0; TP=1/PP=1 | 32 text/image/audio/video/mixed requests, ten full-prefix audits across reset/restart |
| Qwen Flash Next, checkpoint `6843d49` | vLLM `0.1.dev20073+g8e685d198`; TP=2/PP=1, EP/MTP | 15 image/video/mixed requests, twelve rank/prefix audits across group recreation; 1,664-token chunks |
| DeepSeek Vision, earlier checkpoint `f8ba70e` | vLLM `0.25.2.dev0+g752a3a504.d20260714`; TP=2/PP=1 | Historical text three-way comparison: 180 requests and 96 rank/prefix audits |

Gemma mixed consumers return `OTHER` in both cold and cached paths: this proves
cache equivalence, not recognition quality. The cancelled Qwen timing run does
not establish a performance comparison. Earlier model timing results are not
measurements of the final reviewed package.

The final review covers all 27 modules and passes installed Python 3.11 and
CUDA/vLLM tests. Release CI separately checks the tagged wheel. Serving imports
installed wheel bytes, without source mounts. See [Release](RELEASE.md).

## Historical qualification

The [receipt index](receipts/README.md#qualification-groups) retains the previous
snapshot backend's PP=2, DeepSeek/Qwen/GLM long-context, fault and publication
results. Those records qualify their own artifacts; they do not qualify the
new token backend for PP>1 or the same context limits. Full intermediate
experiments are recoverable from the [local archive](receipts/HISTORICAL_RAW.md).

## Known limits

- Token-file PP>1, model maximum context and sustained GC/capacity pressure
  require separate qualification. CPU long-chain probes are not model prefill.
- Recurrent/windowed state must exist at the exact eligible boundary. Runtime
  scheduling can legitimately produce an `unsafe_boundary` save skip.
- In the recorded PP runtime, a remote worker exit 70 could leave the head health
  endpoint responsive. Deployment must observe all participants and replace
  the complete group after a fatal restore.
- Cold and prefix-cached model computation need not be numerically identical.
  The [Gemma diagnosis](receipts/2026-09-08-gemma-mixed-diagnosis/README.md)
  preserves an observed native/disk versus cold difference. Qualify outputs
  independently rather than inferring correctness from hit counts.
- The historical GLM 992,769-token attempt required host recovery and must not
  be replayed as a routine test. Fault injection does not prove every physical
  power-failure scenario.

Use [persistent-hit verification](DEVELOPMENT.md#verify-a-persistent-hit) to add
new evidence, including repeated cold controls, rank agreement and full payload
checks. Record failures as carefully as successful restores.
