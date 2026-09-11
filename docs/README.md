# Documentation

These guides describe the current SpoolCache source tree. Install published
packages from [PyPI](https://pypi.org/project/spoolcache/) and consult
[GitHub Releases](https://github.com/xudongcc/spoolcache/releases) for release notes.
For an older installation, use its release-tag documentation; see
[Migration](MIGRATION.md) when upgrading. Qualification receipts apply to their
recorded artifacts, not automatically to later releases or source edits.

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

The [token-file guide](TOKEN_FILES.md) documents the standard connector's
storage format and limits. Superseded design discussions
and experiment tooling are indexed in the performance notes and local backup.

## Evidence

[Performance notes](PERFORMANCE_IMPLEMENTATION_NOTES.md) index the engineering
results and their limits. The [final token-file evidence](receipts/2026-09-11-token-files/README.md)
records the source review and qualification relevant to the current backend.

The [receipt index](receipts/README.md) separates machine evidence from current
instructions. [G6](receipts/2026-09-08-g6/README.md) records the first release.
[Historical evidence](receipts/HISTORICAL_RAW.md) locates the older benchmarks,
reviews and raw outputs in release history and verified local backups.

Specific models, machines and network interfaces belong to
[lab qualification](LAB.md). They are test fixtures, not package requirements.
The generated [changelog](../CHANGELOG.md) records published versions and remains
owned by python-semantic-release.
