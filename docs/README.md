# Documentation

These guides describe the current SpoolCache source tree. Unreleased CLI,
configuration and deployment changes are listed in [Migration](MIGRATION.md).
The recorded published release is 0.1.0; its receipts do not qualify later edits.

## Start here

| Task | Guide |
| --- | --- |
| Install a published package or this checkout | [Installation](INSTALLATION.md) |
| Choose cache paths, capacity and request controls | [Configuration](CONFIGURATION.md) |
| Build a serving image and mount persistent storage | [Deployment](DEPLOYMENT.md) |
| Inspect cache state, diagnose failures and recover | [Operations](OPERATIONS.md) |
| Change code and verify persistent reuse | [Development](DEVELOPMENT.md) |
| Understand identity, ownership and storage guarantees | [Design](SPOOLCACHE_DESIGN.md) |
| Review tested runtime scope | [Compatibility](COMPATIBILITY.md) |
| Publish with GitHub Actions and python-semantic-release | [Release guide](RELEASE.md) |
| Upgrade from the previous interface | [Migration](MIGRATION.md) |
| Check completed and remaining work | [Goals](TODO_GOALS.md) |

## Evidence

[Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md) index the engineering
results and their limits. The [2026-09-04 benchmark](BENCHMARK_2026-09-04.md)
is one historical matched comparison. Review summaries cover
[2026-09-05](CODE_REVIEW_2026-09-05.md),
[2026-09-06](CODE_REVIEW_2026-09-06.md) and
[G4](CODE_REVIEW_2026-09-07_G4.md).

The [receipt index](receipts/README.md) separates machine evidence from current
instructions. [G6](receipts/2026-09-08-g6/README.md) records the first release.
[Archived documentation](archive/README.md) preserves the exact pre-rewrite
narratives, including superseded experiments and failures.

Specific models, machines and network interfaces belong to
[lab qualification](LAB.md). They are test fixtures, not package requirements.
The generated [changelog](../CHANGELOG.md) records published versions and remains
owned by python-semantic-release.
