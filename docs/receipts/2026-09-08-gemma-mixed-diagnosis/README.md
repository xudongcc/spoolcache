# Gemma mixed-input cache diagnosis, 2026-09-08

The reproducible cold/cache output difference occurs in **vLLM's native GPU
prefix cache even with no SpoolCache connector configured**. At the same
2,560-token boundary, native GPU caching and SpoolCache disk restore return
identical generated-token probability records. This finding does not implicate
the current SpoolCache disk read/write path or the recent configuration cleanup.
It also does not establish universal cold/cache output equivalence.

## Controlled result

All rows use the same immutable Gemma checkpoint, serving image, TP=1/PP=2,
partition `12,23`, eager execution, 8K limit, temperature 0, seed 0, media bytes
and consumer token sequence. Salts isolate cache state. Each cold request clears
process-local prefix, processor and encoder caches and skips both SpoolCache
operations. Native-only requests also skip both operations.

| Configuration/path | Cached tokens | Output | Probability comparison |
| --- | ---: | --- | --- |
| Default cold, repeated | 0 | `OTHER` | Cold repeats agree exactly |
| Default SpoolCache restore, repeated | 2560 | `DOGS_SPEECH_STREET` | Disk repeats agree exactly |
| Default native GPU cache, three repeats | 2560 | `DOGS_SPEECH_STREET` | Every returned probability record equals disk restore |
| Connector omitted entirely, native GPU cache, three repeats | 2560 | `DOGS_SPEECH_STREET` | Every returned probability record equals default disk restore |
| Connector omitted entirely, cold, two repeats | 0 | `OTHER` | Same cold/cache difference |
| Default native GPU cache, full producer | 4096 | `OTHER` | A different cache boundary is not the same control |
| `VLLM_BATCH_INVARIANT=1`, cold/native/disk | 0 / 2560 / 2560 | All `OTHER` | Native equals disk; cold still differs in probabilities |

Keeping the encoder cache while clearing only GPU prefix caching did not change
the default restored output or probabilities. The reset operation is therefore
not sufficient to explain the difference. These are output-equivalence tests,
not a claim that either label describes the media correctly.

## Why this boundary matters

The original mixed consumer contains video, image, audio and text. Its image
placeholder tokens occupy the zero-based interval `[2308, 2574)`. Restoring at
2,560 therefore resumes inside the image, with 14 image placeholder tokens left,
followed by audio/text. The installed Gemma implementation uses bidirectional
vision attention on sliding layers, so the resumed forward differs from a
complete cold prefill in both query shape and the use of cached states.

The native-cache control contains the same complete video and image but omits
the following audio/text. It has 2,577 tokens and a 2,575-token exact common prefix
with the 4,149-token consumer. The observed native APC hit rounds down to 2,560.
No block table, cache object, model implementation or runtime function was patched
to obtain this boundary. The disk producer has 4,096 tokens and shares an exact
prefix with the same consumer.

The repeated same-span results localize the divergence to the runtime's
cold-versus-cached continuation path. Changing official numerical execution mode
changes the result, consistent with sensitivity to that computation path. **The
specific arithmetic operator or kernel has not been isolated**, and the mode
change does not prove that rounding alone is the cause. There is no claim that
all multimodal partial-prefix paths are semantically or numerically equivalent.

vLLM documents [batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)
as a beta numerical-reproducibility feature. The experiment here used the exact
same image with only `VLLM_BATCH_INVARIANT=1` added to both worker environments,
and separate retained cache roots. It made this fixture's text agree, but did
not remove probability drift, so it is not presented as a complete fix or a new
SpoolCache setting.

## Relation to the first failure

The [earlier receipt](../2026-09-08-current-gemma/README.md) recorded a generated
prompt whose cold controls returned `DOGS_SPEECH_STREET` and whose disk restore
returned `OTHER`. Its full nonce was not retained, so that exact consumer cannot
be reconstructed. This diagnosis records a deterministic nonce and reproduces
the reverse label change with the same media prefix.

The original and reproduced prefixes have identical object descriptors and
SHA-256 digests: all 12 objects on PP rank 0 and all 3 on PP rank 1. Both entries
were independently authenticated in their respective tests. See the
[comparison](original-prefix-payload-comparison.json). The missing original
suffix nonce does not require guessing the persisted prefix bytes.

## Reproduce and review

The maintained [diagnostic](../../../benchmarks/probe_gemma_mixed_cache.py) performs
cold repeats, encoder retention/reset, authenticated disk restore and three
same-span native GPU controls. It checks exact token-prefix construction, API
counts and returned probabilities, and proves that native-only requests do not
change persistent manifests. See [the command](../../DEVELOPMENT.md#diagnose-a-mixed-input-output-mismatch).
A successful diagnostic exit means evidence was collected and classified;
`divergence-reproduced-by-native-cache` is not an e2e correctness pass.

- [Machine summary](summary.json), [current default diagnostic](maintained-default/summary.json),
  [default inputs and exact token sequences](maintained-default/inputs.json).
- [Both-rank payload verification](maintained-default/restore-payload-verification.json).
- [Connector-free result](no-connector-native/summary.json) and
  [both containers' image/command/log checks](no-connector-runtime.json).
- [Batch-invariant same-span native/disk control](invariant-variant-12/native-media-boundary/summary.json).
- [Host suite](host-tests.log): 292 passed, 12 runtime-dependent skips, 373 subtests.
  The four new CPU tests reject unequal spans/probabilities and distinguish label
  equality from numerical equality. Package code did not change, so the prior
  installed-wheel runtime qualification remains separately scoped evidence.
- [Artifact/environment state](state.json), [inspected runtime-source hashes](inspected-runtime-source.json),
  [raw logs/tool snapshots](raw-evidence.tar.gz), [archive digest](raw-evidence.sha256).
- [Restoration](restoration.json): original image tags restored, both GPUs idle,
  no active model services, all persistent roots retained.

The initial 13 prompt variants were bounded exploration, stopped after a stable
mismatch was found; their results remain in the raw archive. A text-truncation
attempt could not construct a 2,560-token native control because audio was still
present; it failed its prefix assertion before inference. Keeping complete video
and image while omitting later inputs solved that test construction. One client
started during a requested group transition received a closed connection; its
log is retained and the connector-free test was rerun after the replacement
container and API were verified. Neither preliminary attempt is a passing result.

No model/media bytes are redistributed in this receipt. The probe rebuilds the
requests from the [pinned external fixtures](../../LAB.md#media-fixtures), retained
parameters and token sequences. No package semantics or strict e2e output
assertion was changed, and no commit, push or release was performed in this task.
