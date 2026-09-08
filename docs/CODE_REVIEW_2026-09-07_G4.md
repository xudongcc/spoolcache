# G4 review: pipeline-parallel topology

Date: 2026-09-07. Reviewed scope: the G4 working tree based on parent `5dc7f61a`.
The original review found no outstanding P1/P2 issues at that boundary. This
summary preserves that conclusion; it does not approve subsequent local edits.
The [archive](archive/README.md) retains the complete review.

## Contracts reviewed

| Area | Required behavior |
| --- | --- |
| Rank ownership | Validate public PP/TP/DCP coordinates before opening a store; use global rank for the rank-local path. |
| Stage-local HMA | Preserve ordered runtime groups while deriving actual layer ownership and aliases from final tensors. A stage can have an empty group, but must own some layers. |
| Quorum | Require the complete PP × TP set. Reject missing, duplicate, mismatched or malformed rank identities and inventories. |
| Hook compatibility | Require PP-aware public handshake support for PP > 1; validate exposed hooks before model allocation. |
| Offline verification | Check expected identities and coordinates before payload paths, then authenticate exact group/layer/page coverage. |
| PP=1 regression | Retain the existing PP=1 handshake and coordination layout. |

Coordination identity describes semantics shared across ranks; storage identity
also binds each rank's physical ownership and tensor geometry. Forcing different
stages to have identical local layouts would reject valid partitions.

## Recorded validation

- Host: 249 passed, 12 skipped, 256 subtests passed; compilation and diff checks passed.
- Official vLLM 0.28 image with CUDA: 256 tests, 1 skipped.
- DeepSeek, Qwen and GLM target images: 256 tests each, respectively 6, 5 and 5 skipped.
- Three deployment launcher suites: 3 checks each.
- Two-node TP=1/PP=2: text, image, audio, video and mixed prompts, with cold and
  bypass controls, process-local resets, persistent restore, full-group restart,
  payload authentication on both ranks and output oracles.

The [G4 machine receipt](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)
records the concrete fixture and results. The receipt's 20 shared aliases leave
15 actual owners, split 12/3 between the stages.

## Scope boundaries

The pinned Gemma/vLLM fixture used `VLLM_PP_LAYER_PARTITION=12,23`. That upstream
setting belongs to the qualification harness. It is not a connector option,
model allowlist or generic partition recommendation. Other attempted partitions
failed upstream initialization or bypass output checks; their outcomes remain
in the archived review and receipts.

A healthy head API does not prove that a remote PP stage remains usable. Later
[G6 fault evidence](receipts/2026-09-08-g6/README.md) records this limitation and
the deployment-owned whole-group stop. See [Compatibility](COMPATIBILITY.md)
and [Lab qualification](LAB.md) before interpreting the historical result.
