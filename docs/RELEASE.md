# SpoolCache 0.1 release contract

`0.1.0` is published on [PyPI](https://pypi.org/project/spoolcache/0.1.0/)
and [GitHub](https://github.com/xudongcc/spoolcache/releases/tag/v0.1.0).
The [G6 receipts](receipts/2026-09-08-g6/README.md) distinguish the live-qualified
candidate from the final public wheel and its installed images. Goal status is
tracked in [`TODO_GOALS.md`](TODO_GOALS.md).

## Build once, install the same wheel

### Automatic GitHub Actions → PyPI releases

[`release.yml`](../.github/workflows/release.yml) uses
[python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
10.6.2. Pushes to `main` (or a manual workflow run on `main`) first run the CPU
matrix. PSR derives the next version from Conventional Commits, updates
`pyproject.toml`, `__version__`, `uv.lock` and `CHANGELOG.md`, then creates the
release commit/tag locally. The workflow builds from that clean commit, checks
an independent rebuild, validates package metadata and tests a fresh wheel
installation before atomically pushing the commit/tag and publishing GitHub
release notes and assets.

The separate `pypi` job downloads and authenticates that **same wheel** and
uploads it through PyPI Trusted Publishing (OIDC), with no PyPI API token.
It does not check out code or rebuild the package. Only this job receives
`id-token: write`; only the GitHub release job receives `contents: write`.
Actions are pinned to commit SHAs, release tools to exact versions, concurrent
releases are serialized, and a stale workflow cannot release a newer untested
commit. Branch protection must allow the release bot to push version commits;
do not disable protection or add a broad PAT to work around a rejected push.

The first publisher was configured and used successfully on 2026-09-08. For a
new project, create a
[pending publisher on PyPI](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `spoolcache` |
| GitHub owner | `xudongcc` |
| GitHub repository | `spoolcache` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

This account-side PyPI binding cannot be supplied by repository YAML. Configure
the matching GitHub environment before the first release. The PyPI name was
absent during setup; availability is not a reservation.

There are no historical release tags. Before the first push, development history
is squashed into one Conventional Commit (`feat:`); the original history is
retained only as a local backup reference. Pre-squash commit IDs in historical
qualification receipts identify those original local observations, not commits
published on `main`. The first release resolves to `0.1.0` from the initial
feature commit. Subsequently `fix:` produces a patch and
`feat:` a minor release; breaking changes increment minor while below 1.0.
`docs:`, `test:` and `chore:` alone do not release. Use Conventional Commit
messages for squash merges. PSR owns version/tag creation; do not hand-edit
versions or manually tag release candidates.

If only PyPI upload fails, use **Re-run failed jobs** on that workflow run so it
uses the retained original artifact (90-day retention). If a GitHub release
asset upload fails after the atomic push, repair that release from the original
workflow artifact with `semantic-release changelog --post-to-release-tag TAG`
and `semantic-release publish --tag TAG`; never rebuild and overwrite assets
of an already released version. A complete rerun after the tag exists is a
no-op version calculation, not an upload retry.

The local G6 qualification candidate and the CI-produced release have separate
commit/SHA receipts. The final public wheel was checked against the candidate:
every SpoolCache package file and all package metadata headers are identical;
only README description/RECORD bytes and ZIP timestamps differ. This exact
payload comparison carries the runtime qualification forward; installation
receipts separately authenticate the public wheel in each final image. Never
treat a shared version string alone as proof of identical artifacts.

### Local qualification / reproducibility

For a release rebuild, use the clean tagged release checkout (not a later
documentation commit). For a new candidate, use its clean committed checkout:

```bash
python3 scripts/build-release.py --output dist/release
```

The builder archives that exact commit, uses pinned setuptools/wheel versions
and the commit timestamp as `SOURCE_DATE_EPOCH`, and refuses a dirty checkout or
nonempty output directory. `dist/release/release.json` records the commit and
wheel SHA-256. An independent rebuild into another empty directory must have
the same wheel SHA-256. Keep the first wheel as the artifact installed everywhere.
The wheel includes the MIT license; vLLM, Torch, CUDA and model weights are not
redistributed in it and retain their respective licenses.

Export `SPOOLCACHE_WHEEL` (path relative to the Docker build context),
`SPOOLCACHE_WHEEL_SHA256`, and `SPOOLCACHE_COMMIT` from that receipt. These are
build arguments, not connector configuration. Build the official Gemma
development image with `docker compose build vllm`. To extend each existing
production runtime, use:

```bash
docker build -f Dockerfile.release \
  --build-arg BASE_IMAGE="$PINNED_RUNTIME_IMAGE" \
  --build-arg SPOOLCACHE_WHEEL="$SPOOLCACHE_WHEEL" \
  --build-arg SPOOLCACHE_WHEEL_SHA256="$SPOOLCACHE_WHEEL_SHA256" \
  --build-arg SPOOLCACHE_COMMIT="$SPOOLCACHE_COMMIT" \
  -t "$RELEASE_IMAGE" .
```

Pin the base by registry digest or an already-recorded local image ID. The
Dockerfiles authenticate the wheel before installing with `--no-deps`; no vLLM
or CUDA dependency resolution occurs. The wheel is retained under
`/opt/spoolcache-release` for independent install verification. Record the final
image ID/digest and configure each deployment's existing image variable
(`DSPARK_VLLM_IMAGE` for DeepSeek, `IMAGE` for Qwen/GLM). Transfer the complete
image to every participant before launch. Production launchers require the same
image ID on all nodes and use the installed package; source checkout mounts and
source transfer are removed. Development Compose and Gemma PP also use the
installed wheel. Tests and benchmarks are qualification tools only.

Authenticate installed files with `benchmarks/verify_release_install.py`, passing
the retained wheel path and its expected SHA-256. This checks the original wheel,
every packaged file, absence of stale extra modules, pip's wheel-origin receipt,
and the actual import path. A wheel version alone is insufficient evidence.

Production launchers force vLLM development endpoints off. Process-local reset
endpoints belong only to the independent Gemma Compose/PP qualification harness.
The connector configuration has five fields: absolute root, deployment namespace,
access mode, direct-I/O mode, and maximum managed bytes. Memory slots, admission,
inventory, scrub and low watermark bounds remain internal constants.

## Persistent compatibility and rollback

0.1 freezes `spoolcache-manifest/v1`, `spoolcache-manifest-envelope/v1`,
`spoolcache-deployment/v2`, and `spoolcache-coordination/v1`. The existing rank
identity encoding and PP=1 encoding remain unchanged. PP>1 binds discovered
stage/global-rank ownership; it does not reinterpret old PP=1 manifests.
Frozen schema names do not promise hits across changed model revisions, runtime
package bytes, **SpoolCache version**, tensor layouts, topology or operator namespaces: those inputs
intentionally change the identity and cause a safe miss.

Upgrade and rollback replace the **complete** TP/PP group with one qualified
image while retaining every rank's cache root. Authenticate representative
entries and check API output plus all-rank entry/span agreement after restart.
Never run two inventory owners on the same rank root, mix release artifacts
within a group, or roll back only `state/` while keeping manifests. Persisted
generation counters, sentries, tombstones and object fences are durable safety
state and must move with their rank root.

Rolling back to the previous qualified image is supported operationally by
retaining that image and model revision. A changed identity may cold-fill a
separate namespace. Compatibility with older alpha readers is not promised;
never coerce a schema or rename an old directory into a new identity.

No automatic migration daemon or implicit cache deletion is provided. The
existing monotonic generation initialization remains: it prevents replay of old
worker inventory and distinguishes a new root from lost durable state. It is a
correctness boundary, not a pre-release cleanup candidate. PyTorch's two
allocator environment names likewise remain necessary runtime compatibility.

Before manually archiving old namespaces, stop the complete group, enumerate
deployment/rank directories beneath the configured root, record their manifest
identities and sizes, and preserve the entire selected rank tree (including
state, markers and quarantine). Keep an archive receipt with path, identity,
release and date. Do not follow unknown links or select by model name alone.
For an explicit fresh cache, configure a new empty root or namespace and retain
the old root for rollback. Deleting an identified archived root is a separate
operator decision; launchers never clear it during restart or upgrade.

## Scope and limitations

The verified support matrix and existing performance/correctness receipts are
linked from [`README.md`](../README.md) and
[`PERFORMANCE_IMPLEMENTATION_NOTES.md`](PERFORMANCE_IMPLEMENTATION_NOTES.md).
Gemma release functionality is qualified on pinned vLLM 0.28.0, TP=1/PP=1 and
TP=1/PP=2 only after the new release receipt passes. DeepSeek/Qwen/GLM runtime
tests are compatibility evidence, not new per-model feature qualification.
PP=2 Gemma uses the upstream qualification-only partition `12,23` established in
G4; this is not a SpoolCache model rule. Other PP×TP combinations are covered
only to the extent explicitly recorded in CPU/runtime/live receipts.

Stores/restores remain synchronous, exact prefix only, rank-local NVMe, with
fixed separate pinned and aligned-I/O pools. Unknown contracts fail startup;
missing participants miss; post-admission payload failure terminates the worker
with exit 70 and requires deployment-owned complete-group recovery. No cache
supervisor, remote replication, CPU L1, asynchronous store, or encoder cache is
part of 0.1. In the qualified vLLM PP=2 runtime, a remote stage's exit 70 can
leave the head's `/health` responding while an inference stream hangs. That
endpoint is process liveness, not an all-stage readiness check. The deployment
must observe participant failure and stop the complete group; the G6 remote
fault receipt explicitly distinguishes health before that stop from readiness
afterwards. SpoolCache does not add an in-package process supervisor to mask
this upstream behavior.

The G6 PP=2 long padded image and mixed prompts produce a stable `OTHER` even
on independent cold bypasses; a short image control identifies cats. Their
release oracle proves complete output equivalence across cold, persistent
restore and restart plus per-rank payload integrity. It does not claim correct
long-context media classification. All negative controls are retained with the
qualification receipts.

The previously waived 24-hour soak is not a release prerequisite.
