# Code review: 2026-09-05

This is a summary of the review recorded on 2026-09-05. All findings below were
recorded as resolved at their respective reviewed boundaries. This is historical
evidence, not a fresh approval of the current checkout.

The [archived original](archive/README.md) preserves the complete findings,
remedies, commit references, follow-up reviews and validation output. Historical
commit IDs refer to local pre-squash provenance and may not resolve on GitHub.

## Findings

| Finding | Severity | Problem reviewed | Recorded status |
| --- | --- | --- | --- |
| CR-001 | P1 | Public hook signatures did not prove call compatibility. | Resolved |
| CR-002 | P2 | Worker source synchronization could retain stale files. | Resolved |
| CR-003 | P2 | Offline payload verification lacked expected identities. | Resolved |
| CR-004 | P1 | An admitted restore failure could leave the distributed request hanging. | Resolved |
| CR-005 | P1 | Multimodal qualification accepted empty answers. | Resolved |
| CR-006 | P2 | Scheduler block IDs were accepted through coercion. | Resolved |
| CR-007 | P1 | Allocated external spans lacked admission validation. | Resolved |
| CR-008 | P2 | Non-string cache salts were coerced. | Resolved |
| CR-009 | P2 | An unused require_cache_salt option obscured the contract. | Resolved |
| CR-010 | P2 | Runtime image contents could shadow benchmark helpers. | Resolved |
| CR-011 | P1 | An unpinned model revision could produce a stable-looking identity. | Resolved |
| CR-012 | P1 | Multimodal capability flags were accepted through coercion. | Resolved |
| CR-013 | P1 | Benchmark clients accepted missing or coercible usage fields. | Resolved |
| CR-014 | P2 | Offline verification used paths before validating expected identities. | Resolved |
| CR-015 | P2 | Source snapshots included the host virtual environment. | Resolved |
| CR-016 | P1 | Physical rank values were accepted through coercion. | Resolved |
| CR-017 | P2 | Resource accounting omitted admitted, allocated restore plans. | Resolved |
| CR-018 | P1 | Benchmark clients accepted truncated responses. | Resolved |

## How to read the findings today

The lasting requirements are strict public-contract checks, rank and identity
validation, bounded admission accounting, and benchmark clients that reject
incomplete evidence. Post-admission failure stops the worker; recovery of the
whole distributed group belongs to the deployment.

Source synchronization remedies in CR-002 and CR-015 describe a retired serving
path. Current deployments install an authenticated wheel in the image. Runtime
identity does not independently authenticate a model repository's contents;
operators must pin and preserve the model artifacts used for qualification.

See [Design](SPOOLCACHE_DESIGN.md), [Development](DEVELOPMENT.md) and
[Release](RELEASE.md) for current behavior and verification procedures.
