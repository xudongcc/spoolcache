# Goals and status

This is the source of truth for planned work. Design is in
[Design](SPOOLCACHE_DESIGN.md), verified scope in [Compatibility](COMPATIBILITY.md),
and detailed historical observations in [Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md).

`[x]` means the recorded acceptance conditions were completed; `[-]` means work
remains. A documentation rewrite does not complete a goal or qualify new code.

## Status

| Goal | State | Outcome / remaining work |
| --- | --- | --- |
| G1–G3 core | [x] | Public runtime admission, persistent HMA correctness, identity, bounded inventory and maintenance |
| G4 | [x] | Generic PP>1 topology, stage-local ownership and complete PP×TP quorum |
| G5 | Withdrawn | No mandatory optimization program without a measured need |
| G6 | [x] | Reproducible installed-wheel serving and first GitHub/PyPI release |
| Q1 | [-], non-blocking | GLM safe maximum-context qualification and its final receipts remain |

CLI renaming, mandatory `O_DIRECT`, unified `SPOOLCACHE_PATH` and GB-based
`SPOOLCACHE_MAX_SIZE`, plus removal of the access-mode setting, are released
features. Independent request-level `spoolcache.skip_read` and
`spoolcache.skip_write` flags replace the old bypass; enabling both skips
persistent reads and writes. Their CPU/launcher checks are recorded in
[Compatibility](COMPATIBILITY.md#release-and-qualification-evidence). Published
versions are listed on [PyPI](https://pypi.org/project/spoolcache/) and
[GitHub Releases](https://github.com/xudongcc/spoolcache/releases). Qualification
remains scoped to the exact artifacts recorded in each receipt.

## Permanent constraints

- Use public vLLM V1/HMA interfaces; do not patch vLLM for SpoolCache.
- Derive model/runtime/layout/topology identity from public facts, without
  model, architecture, modality or version selectors.
- Authenticate every required cache group and participant before completing a hit.
- Keep staging, admission and inventory bounded. Use mandatory `O_DIRECT` for
  payloads; small metadata uses ordinary I/O.
- Preserve durable generation, withdrawal and publication ordering. Keep
  orchestration in deployment code and serving on an installed immutable wheel.
- Use pinned Gemma for general live feature/fault work; use DeepSeek/Qwen/GLM
  for separate runtime compatibility. CPU fixtures remain model-independent.

## G1–G3 core completion

Completed behavior includes automatic public-call checks and runtime package
attestation, runtime-discovered HMA semantics/final tensor ownership, exact text
and multimodal identity, immutable object/manifest publication, single-pass
restore authentication, all-rank inventory, capacity/GC, scrub and quarantine.
Model locator/revision identifies a namespace without scanning checkpoint files.

Key historical references: `1d74877`, `40cdccd`, `71e391b`, `07ae286`, `38fa442`,
`4a06e99`, `35abc08`, `a633f75` through `bb81636`, `51ec9a6` and `581c55b`.
`eb1c1a1` records exact storage-view handling of shared KV aliases.
These pre-squash IDs are local evidence; see the
[September 5](CODE_REVIEW_2026-09-05.md) and
[September 6](CODE_REVIEW_2026-09-06.md) review summaries.

## G4 completion

Acceptance covered public PP/TP/DCP coordinate checks, global-rank storage,
stage-local layouts, PP-aware startup handshake, complete participant quorum,
missing/duplicate/wrong-identity cases, offline identity-bound verification and
PP=1 regression. No model-specific partition was added to the connector.

The [G4 receipt](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json)
records two-node TP=1/PP=2 Gemma text/image/audio/video/mixed controls, full group
restart, same entry/span across stages, payload authentication and output checks.
The [G4 review](CODE_REVIEW_2026-09-07_G4.md) records no outstanding P1/P2 at that
reviewed boundary. The current lab launcher uses installed images and has no
source synchronization path.

## G6 completion

Acceptance covered clean-commit reproducible builds, one authenticated wheel in
all serving images, removal of source overlays, release identity/rollback rules,
Conventional Commits, GitHub Actions + python-semantic-release, PyPI Trusted
Publishing and Gemma PP=1/PP=2 functional/fault qualification.

The [G6 receipt index](receipts/2026-09-08-g6/README.md) records successful workflow
`34180558915`. The initial squashed feature commit was `80a19db`; the release
commit/tag was `40528e6` / `v0.1.0`. Public wheel SHA-256:

```text
abaf71af05afd41c0d5a2873a003bc94af3a65b54dbe9db97c1015b2dd01bd28
```

Candidate and public artifact identities remain distinct. Package-payload
equality and eight final image-install authentications carry the recorded live
qualification to the public wheel. PP=2 media-oracle and remote-health limits
remain explicit in those receipts.

## Namespace simplification

- [x] Remove `SPOOLCACHE_NAMESPACE` and JSON/Python namespace fields; retain only
  path and capacity settings plus the independent request read/write flags.
- [x] Remove operator namespace from prefix and coordination identity, using v2
  domains; preserve automatic model/runtime/layout identity and request salts.
- [x] Update launchers and documentation; retain old cache trees without migration.

These changes are released. CPU checks and qualification limits are recorded in
[Compatibility](COMPATIBILITY.md#release-and-qualification-evidence).

## Protocol residue audit

- [x] Trace configuration, per-step metadata, startup/inventory, layout and
  persistent schema consumers; remove the unused per-step metadata schema field.
- [x] Correct stale bypass/profile comments and the development report-barrier
  instructions to use independent request read/write flags.
- [x] Preserve active layout/manifest identity, quorum and crash-recovery checks.
- [x] Scan the full repository, remove the superseded single-image probe and
  inactive layerwise/shared-staging probes, and remove unused runtime imports.
- [x] Document the maintained benchmark tools and exact per-object authentication
  boundary; preserve historical receipts and removed-option rejection tests.

The [protocol boundaries](SPOOLCACHE_DESIGN.md#internal-protocol-boundaries) document
which identifiers remain necessary. This cleanup is released.

## CLI configuration output

- [x] Add `spoolcache config` using the existing validated environment renderer;
  print compact JSON without starting vLLM or creating cache directories.
- [x] Reject invalid settings with an error on stderr and no partial JSON.
- [x] Update current usage examples to pass `$(spoolcache config)` to vLLM;
  retain the existing maintenance request/status contracts.

CLI subprocess tests cover defaults, home expansion, capacity overrides and
invalid inputs. This command is available in published packages.

## Current Gemma regression and fixed e2e runner

- [x] Add a maintained five-input runner with repeated cold controls, exact
  persistent hit counts, independent request flags, all-rank payload verification
  and whole-group restart checks. See [Development](DEVELOPMENT.md#fixed-gemma-end-to-end-regression).
- [x] Reject unstable controls, wrong outputs, missing rank evidence and partial
  restarts through CPU tests; retain machine-readable live failures.
- [x] Build and authenticate an isolated current-code candidate and run the
  installed vLLM/CUDA tests and real Gemma PP=1/PP=2 checks.
- [x] Reproduce and diagnose the PP=2 mixed-input cold/restore difference using
  same-span native GPU controls; retain the strict cold-output gate and numerical
  reproducibility limit. See the [follow-up diagnosis](receipts/2026-09-08-gemma-mixed-diagnosis/README.md).

The [current receipt](receipts/2026-09-08-current-gemma/README.md) distinguishes
successful cases from the initial finding. The follow-up attributes the
reproduced difference to the runtime caching path; it does not claim that the
default runtime now guarantees cold/cache output equality. Historical G4/G6 qualification remains
scoped to its original artifacts. No current package behavior was changed to
make the live output assertion pass. The interface changes have since been
released; the regression receipts still identify the isolated candidate that was
tested. Publishing a release does not itself authenticate that wheel in the
recorded live deployments.

## Q1 remaining qualification

Q1 adds deployment evidence without changing core admission logic:

- [x] Million-item incremental namespace work and bounded shutdown/storage soak.
- [x] Qwen 260,800-token consumers restoring 160,000-token prefixes; unsafe
  maximum-length producers remain skipped.
- [x] DeepSeek 196,608-token consumer restoring a 130,048-token prefix.
- [ ] Progressively establish GLM's highest safe qualified context with all-host
  memory/liveness monitoring. Do not replay the failed 992,769-token request.
- [ ] Archive that GLM evidence, review its limits and restore the prior deployment
  state at the end of the qualification task.

A 24-hour dual-host soak was explicitly removed from required acceptance. It
is not a remaining G4/G6 task.

## Scope exclusions and optimization entry criteria

No current goal requires a separate cache server, CPU L1, remote replication,
Redis/S3, GDS/RDMA, encoder cache, model/checkpoint-directory inventory, automatic
cache migration, shared GPU prefixes or P/D disaggregation.

Async store, layerwise restore, incremental publication, shared staging, native
movers and compression are candidates only. Start work on one when a matched
benchmark demonstrates a falsifiable need and the public ownership contract
supports a qualified replacement path. A feasibility probe is not a shipped
implementation or an automatic future commitment.
