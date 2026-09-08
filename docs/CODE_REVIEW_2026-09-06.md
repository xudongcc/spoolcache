# Code review: 2026-09-06

This is a summary of the review recorded on 2026-09-06. All findings below were
recorded as resolved at their respective reviewed boundaries. This is historical
evidence, not a fresh approval of the current checkout.

The [archived original](archive/README.md) preserves the complete findings,
remedies, commit references, follow-up reviews and validation output. Historical
commit IDs refer to local pre-squash provenance and may not resolve on GitHub.

## Findings

| Finding | Severity | Problem reviewed | Recorded status |
| --- | --- | --- | --- |
| CR-019 | P1 | Semantic resolver signatures were not fully checked. | Resolved |
| CR-020 | P2 | Public aggregate cache semantics were ignored. | Resolved |
| CR-021 | P1 | Packed scratch allocations lacked sufficient ownership proof. | Resolved |
| CR-022 | P1 | Inventory and statistics bounds did not cover every interface. | Resolved |
| CR-023 | P1 | The former supervisor lacked proof of stop ownership. | Resolved |
| CR-024 | P2 | Launcher failures could leave a partial TP group running. | Resolved |
| CR-025 | P2 | Readiness metric parsing mishandled a trailing label comma. | Resolved |
| CR-026 | P2 | The former supervisor had probe, configuration and state bounds gaps. | Resolved |
| CR-027 | P2 | Source overlays risked stale files or deleting a live source tree. | Resolved |
| CR-028 | P1 | Generation rollback detection depended on wall-clock behavior. | Resolved |
| CR-029 | P1 | Namespace, reporter and quarantine receipts had race conditions. | Resolved |
| CR-030 | P1 | Corrupt content-address collisions could leave broken references and stale offers. | Resolved |
| CR-031 | P1 | Multiple rank owners and standalone maintenance could race. | Resolved |
| CR-032 | P2 | Soak measurements could miss peaks or inherit earlier process high-water marks. | Resolved |
| CR-033 | P2 | Scrub restart boundaries and special namespace paths needed fail-closed handling. | Resolved |
| CR-034 | P1/P2 | Shared-reference withdrawal was not crash-atomic and could build unbounded lists. | Resolved |
| CR-035 | P1 | Object repair could incorrectly clear an entry; durable marker handling was incomplete. | Resolved |
| CR-036 | P2 | Shard receipt failures could accumulate full-sized temporary files. | Resolved |
| CR-037 | P2 | Selecting newest raw entries could starve healthy offers. | Resolved |
| CR-038 | P1 | JSON resource-limit failures could escape manifest error handling and block startup. | Resolved |
| CR-039 | P1 | Persistent fsync work could make the main shutdown path unbounded. | Resolved |

## How to read the findings today

This file originally accumulated several review passes. A resolved finding does
not imply that the entire later goal or every model qualification was complete
at the time of that pass. In particular, the GLM maximum-context qualification
remains bounded by the host-safety work in [Goals](TODO_GOALS.md).

The supervisor and serving source-overlay mechanisms discussed in CR-023,
CR-026 and CR-027 were subsequently retired. Current serving images contain the
installed wheel; the deployment owns all-rank recovery. The durable store still
requires exclusive rank ownership, monotonic generation and withdrawal fences,
bounded inventory and maintenance work, and explicit handling of filesystem
failures.

See [Design](SPOOLCACHE_DESIGN.md) and [Operations](OPERATIONS.md) for current
contracts. [Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md) separate
storage stress evidence from end-to-end model qualification.
