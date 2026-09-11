# Releases

GitHub Actions and python-semantic-release own version calculation, changelog,
release commits, tags and GitHub assets. PyPI receives the same authenticated
wheel through Trusted Publishing. A local edit, test or candidate build does not
publish a package.

Published packages are available on [PyPI](https://pypi.org/project/spoolcache/),
with tags, notes and assets on
[GitHub Releases](https://github.com/xudongcc/spoolcache/releases).
The first release's [G6 evidence](receipts/2026-09-08-g6/README.md) distinguishes
the qualified candidate, public wheel and installed images. Legacy interface
changes are listed in [Migration](MIGRATION.md).

## Automated workflow

The checked-in [release workflow](../.github/workflows/release.yml) runs on a
push to `main` or a manual dispatch on `main`:

1. Run the CPU validation matrix and reject a stale workflow source commit.
2. Run python-semantic-release 10.6.2 to calculate the next version and create
   the local release commit/tag. Its build hook updates the lock file.
3. Build twice from that clean commit, compare wheel bytes and check metadata.
4. Install the actual wheel into a fresh environment; authenticate installed
   files, run the standard-library test suite and check the CLI.
5. Retain the original artifact, atomically push the release commit/tag and
   publish GitHub notes/assets.
6. In the separate `pypi` job, download and authenticate the retained wheel,
   then publish it with OIDC. That job does not rebuild the package.
7. Call the container workflow with that release's Git tag after PyPI succeeds.
   Build and test both architectures before publishing the combined GHCR tag.

Release concurrency is serialized. The GitHub release job has `contents: write`;
only the PyPI job has `id-token: write`. Release actions are pinned by commit and
tools by version in the workflow. Branch protection must allow the authorized
release bot path; a failed push is not a reason to silently bypass protection.

## Container publication

[Dockerfile](../Dockerfile) builds the serving image; the development fixture
uses [Dockerfile.development](../Dockerfile.development). The
[container workflow](../.github/workflows/container.yml) owns `VLLM_VERSION` and
passes `vllm/vllm-openai:v${VLLM_VERSION}` as `BASE_IMAGE`, without a digest.
SpoolCache's version comes from the published Git tag, with the leading `v`
removed. For example, vLLM `0.29.0` and SpoolCache tag `v0.3.0` publish:

```text
ghcr.io/xudongcc/vllm-openai-spoolcache:0.29.0-0.3.0
```

The workflow downloads the original GitHub release wheel and `release.json`,
checks the receipt against the tag's commit, and checks the wheel checksum and
embedded version. Both native runners (`ubuntu-24.04` for AMD64 and
`ubuntu-24.04-arm` for ARM64) install that same universal wheel with `--no-deps`.
Each image must pass installed-byte authentication, the real vLLM connector
contract and the release's test suite. These runners do not qualify CUDA or
model inference; GPU-only tests remain skipped.

Each successful architecture is pushed under a run-specific build tag. The
combined version tag is published only after both architectures pass, then its
platform inventory is checked. Failed jobs can be rerun in the same workflow;
the successful architecture's build tag remains available.

Publication uses `GITHUB_TOKEN` with `packages: write`; no Docker Hub credentials
or registry PAT are needed. Package visibility is managed separately in GHCR.
For a private package, pulling requires an account with package read access.

New SpoolCache releases call this reusable workflow directly because releases
created by `GITHUB_TOKEN` do not trigger another ordinary release-event workflow.
Changes to the production Dockerfile, container workflow or its input verifier
on `main` rebuild the latest published release. To rebuild a specific release,
run the `container` workflow on `main` with its Git tag:

```bash
gh workflow run container.yml --ref main -f tag=v0.3.0
```

Leaving `tag` empty selects the latest published release. Updating
`VLLM_VERSION` in the workflow selects a new upstream version; the Dockerfile
contains no default vLLM version. Retried builds resolve the upstream tag again.

## Commit and version policy

| Commit type | Normal release effect |
| --- | --- |
| `fix:` | Patch |
| `feat:` | Minor |
| Breaking change | Minor while below 1.0, under the configured policy |
| `docs:`, `test:`, `chore:` alone | No release |

Use Conventional Commits for changes and squash merges. Do not manually edit
version fields or create release tags. The initial development history was
squashed before first publication; pre-squash IDs in historical receipts are
local provenance identifiers, not promised public GitHub revisions.

## Trusted Publishing

The established PyPI publisher binding is:

| Field | Value |
| --- | --- |
| Project | `spoolcache` |
| Owner/repository | `xudongcc/spoolcache` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

The account-side publisher and matching GitHub environment must exist. Repository
YAML alone does not create a PyPI publisher. The first binding and publication
were recorded on 2026-09-08; see the [publication receipt](receipts/2026-09-08-g6/publication.json).

## Build a local candidate

The builder requires a clean committed checkout, including untracked files,
and a new empty output directory. It archives the exact commit, builds with
pinned setuptools/wheel versions and sets `SOURCE_DATE_EPOCH` from that commit.
If changes must remain uncommitted, finish local tests first and defer the
release candidate build.

The token-only package has no compiled extension and builds a `py3-none-any`
wheel. The same authenticated wheel is installable on Python 3.11 and 3.12;
serving still requires supported Linux locking, CUDA and vLLM contracts. The
builder accepts `--python /path/to/python` for reproducible interpreter choice.
No architecture-specific SpoolCache wheel matrix is required.

From a clean candidate checkout:

```bash
CANDIDATE_DIR="dist/candidate-$(git rev-parse --short HEAD)"
uv run --locked python scripts/build-release.py --output "$CANDIDATE_DIR"
```

`release.json` records the source commit, wheel filename and SHA-256. Export the
Docker build arguments directly from that receipt:

```bash
export SPOOLCACHE_WHEEL="$CANDIDATE_DIR/$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' \
  "$CANDIDATE_DIR/release.json")"
export SPOOLCACHE_WHEEL_SHA256=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["wheel_sha256"])' \
  "$CANDIDATE_DIR/release.json")
export SPOOLCACHE_COMMIT=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' \
  "$CANDIDATE_DIR/release.json")
```

These are build arguments, not connector settings. The wheel path is relative
to the Docker build context. Use [Deployment](DEPLOYMENT.md#build-a-serving-image)
for production images or [Development](DEVELOPMENT.md#build-the-development-image)
for the repository fixture.

An independent rebuild from the same clean commit into another empty directory
must yield identical bytes. Keep the first wheel as the artifact installed
everywhere. For a published version, use its exact release tag for reproducibility
investigation; never replace the published artifact with a rebuild.

## Authenticate an installation

Run the verifier in the environment where the wheel is installed:

```bash
: "${SPOOLCACHE_WHEEL:?Set the retained wheel path}"
: "${SPOOLCACHE_WHEEL_SHA256:?Set the trusted wheel SHA-256}"
python3 benchmarks/verify_release_install.py \
  --wheel "$SPOOLCACHE_WHEEL" --sha256 "$SPOOLCACHE_WHEEL_SHA256"
```

It verifies wheel bytes, packaged-file contents, import location, stale extra
modules and pip's wheel-origin receipt. Record the final immutable image IDs,
wheel SHA-256 and model revision. One version string is not an artifact identity.
Serving must import the installed package, without source overlays or binds.

## Retry a failed publication

If only PyPI upload failed, re-run the failed job in the original workflow run.
It uses the retained original artifact; current retention is 90 days. A complete
rerun after a tag exists may calculate no new version and is not an upload retry.

If GitHub asset publication failed after the atomic push, repair that release
using the original workflow artifact and the workflow's
`semantic-release changelog --post-to-release-tag TAG` /
`semantic-release publish --tag TAG` steps. Authenticate the artifact before any
repair. Do not rebuild or overwrite the content of an existing version.

## Qualification and rollout

CPU, installed-runtime, CUDA, persistent-hit and fault evidence apply to exact
artifacts. Do not carry a live result to a changed package solely because its
version string matches. G6's candidate/public distinction was bridged by an
explicit equality check of package payloads plus separate public-install receipts.

Upgrade or rollback the complete PP/TP group while preserving cache state.
[Migration](MIGRATION.md) and [Compatibility](COMPATIBILITY.md) describe current
interface changes, identity boundaries and the known PP failure limitation.
