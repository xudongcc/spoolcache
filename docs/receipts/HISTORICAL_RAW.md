# Archived experiment evidence

Intermediate experiment tools, bulk logs and superseded narratives are retained
in release history or local backups as described below. The working tree contains
current runtime/tests and the
[final evidence summary](2026-09-11-token-files/README.md). Historical identities
remain tied to their original artifacts.

## After the 0.3.0 release

The post-release cleanup removes 282 historical files (3,060,254 bytes): raw
request/payload-verification rows, logs, old archives, three review documents,
the 2026-09-04 benchmark and three generated interference results. Current
runtime code, tests, benchmark tools, final summaries and artifact identities
remain in the working tree. The cleanup does not change the published release.

Every removed file is available in the
[v0.3.0 source tree](https://github.com/xudongcc/spoolcache/tree/v0.3.0), at commit
`3c2c2fbadd3fc47e759b88b2b49a0fbf7d53474b`. Links from retained narratives to
removed details now point to that release. Historical paths inside unchanged
machine receipts should also be resolved against that tree.

Historical entry points:

- [2026-09-04 DeepSeek benchmark](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/BENCHMARK_2026-09-04.md).
- Reviews from [2026-09-05](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/CODE_REVIEW_2026-09-05.md),
  [2026-09-06](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/CODE_REVIEW_2026-09-06.md)
  and [G4](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/CODE_REVIEW_2026-09-07_G4.md).
- [Pre-rewrite documentation archive](https://github.com/xudongcc/spoolcache/blob/v0.3.0/docs/archive/README.md).

An additional local backup is at
`/root/projects/xudongcc/spoolcache-backups/2026-09-11-post-release-cleanup`:

- `before-cleanup.bundle` passes `git bundle verify` and preserves the release
  tree on `codex/post-030-file-cleanup`. SHA-256:
  `6916bfda685748a92d35ae184b40b161cc08f6d79443f072951d0cba1193b389`.
- `removed-files.tar.gz` was compared byte-for-byte with all 282 originals
  before deletion. SHA-256:
  `537448037f3f8fbf4f8c2596fc4b6c6c011de22fe4ffd601c96d1834a899691d`.
- `removal-candidates.json` records each original path, size and SHA-256.

To inspect the complete historical tree without changing the current checkout:

```bash
git worktree add --detach /tmp/spoolcache-v0.3.0-evidence v0.3.0
```

## Before the 0.3.0 cleanup

Directory: `/root/projects/xudongcc/spoolcache-backups/2026-09-11-release-0.3.0`.

- `before-cleanup.bundle` preserves branch
  `codex/backup-before-030-cleanup-20260911` at `120d08a` and passes bundle verification.
  SHA-256: `578638d15361824ed7e25f08e87c2831a3c2663ae7bcb12086d7fbc3854fd639`.
- `intermediate-records.tar.gz` preserves all six removed narrative files,
  verified byte-for-byte before removal. `intermediate-records.json` records
  individual sizes and hashes. Archive SHA-256:
  `0150b447ee4e28ce01dc8fbd4f57daff65b0b0c8709bd3e308d45f968c7e30f0`.
- The cancelled Qwen comparison remains separately under
  `/root/projects/xudongcc/spoolcache-backups/2026-09-11-qwen-performance`.
  Its excluded warmup and failed helper attempts are not performance results.

## Verified local backups

The earlier bundles contain complete history and passed `git bundle verify`. Every
removed file was compared byte-for-byte with its original committed blob before
removal. Per-file paths, sizes and SHA-256 digests are in `removed-files.json`
beside each bundle.

| Backup | Original head | Local bundle directory |
| --- | --- | --- |
| Full 133-commit experiment | `15283dc4f1b4736c110b613c787ae57914870709` | `/root/projects/xudongcc/spoolcache-backups/2026-09-11-pr-cleanup/` |
| First squashed PR | `ffefa37bcd831c7f032c88900f165ff8284b50c3` | `/root/projects/xudongcc/spoolcache-backups/2026-09-11-pr-cleanup-round2/` |

Each directory contains `pre-cleanup.bundle`. SHA-256 values, in table order:

```text
6e4077974d5c113d95aacd5147e3cb6801c3b4f32152b5a8b090e3fd76155c50
873d9f623ea66f62b490184a7ecf045923c601216a356c6f24c368c61323d791
```

These are **local backups, not artifacts distributed by this PR**. To replay
historical measurements, obtain the full-experiment bundle and restore a new
checkout on the development host:

```bash
git clone --branch codex/backup-token-key-files-20260911 \
  /root/projects/xudongcc/spoolcache-backups/2026-09-11-pr-cleanup/pre-cleanup.bundle \
  /tmp/spoolcache-pr-before-cleanup
```

From the restored repository root, unpack the 3,804 older detail files that
were already archived at that checkpoint:

```bash
tar -xzf docs/receipts/historical-raw-details.tar.gz
```

Original reports, helper scripts and checksum manifests then have their original
layout. Restore the other bundle with branch
`codex/backup-token-key-files-cleanup1-20260911` to inspect the first cleanup.
Pre-squash commit IDs continue to identify historical evidence; cleanup does not
relabel old model measurements or installed-wheel results as new qualification.

## Before the token-only refactor

The complete pre-refactor tree `45c1fc21917f9ab8688ee0952e9c9fc8904476cf` is
preserved as `codex/backup-before-token-only-20260911` in the verified bundle
`/root/projects/xudongcc/spoolcache-backups/2026-09-11-token-only/before.bundle`.
Bundle SHA-256: `076e3f18727fed86c2c5d55da91c04591f9306c356c3a13c67e94be630271d62`.
This backup includes the retired snapshot/slot/native modules and their tests.
The same local directory holds the refactor's raw host, artifact, GPU and
process-fault validation logs. It is not distributed as part of the PR.

Final runtime source `2a671bc430a68d926fa3ac54399954da9dd6568e` is preserved
in `final-qualified.bundle` on `codex/qualified-token-only-final-20260911` in
that directory. `final-wheel/` and `final-qualification.json` identify the exact
installed package; the review receipt records its validation and limitations.
