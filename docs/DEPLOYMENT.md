# Deployment

Run SpoolCache inside the vLLM worker processes. The deployment provides the
runtime image, models, storage mounts, networking and process recovery.
SpoolCache provides the connector, persistent storage and bounded telemetry.

The package has no hardware-brand or model-name deployment selector. Specific
DGX Spark deployments are documented separately in [Lab qualification](LAB.md).

## Before enabling the connector

1. Verify the intended vLLM model, revision, topology and allocator without
   persistent caching. Keep that working serving configuration as the baseline.
2. Select local Linux storage supporting buffered I/O, file/directory fsync and OFD locks.
   Budget capacity per rank and leave room for staging writes and maintenance state.
3. Install one authenticated wheel in every participant's runtime image.
4. Generate connector configuration and pass it to vLLM. Keep production debug
   endpoints disabled with `VLLM_SERVER_DEV_MODE=0`.

Startup checks public vLLM contracts and runtime cache semantics. An enabled
`expandable_segments` allocator setting is rejected; adjust the deployment's
`PYTORCH_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF` before starting the runtime.
A successful package import does not establish compatibility.

## Container paths

Use `SPOOLCACHE_PATH` for the host source directory. The default is the host
user's `~/.cache/spoolcache`. The provided launchers mount that directory at
`/var/lib/spoolcache` and configure that container-visible path explicitly.

```text
host: SPOOLCACHE_PATH or ~/.cache/spoolcache
                       |
                       +-- bind mount --> container: /var/lib/spoolcache
                                           |
                                           +-- deployment digest / rank directory
```

The same variable name is used in both environments; its value describes the
path visible there. A host path is not automatically visible inside a container.
When generating JSON inside a container, set `SPOOLCACHE_PATH` to its mount target.

For example, this command illustrates the mount and renderer without loading a model:

```bash
: "${RUNTIME_IMAGE:?Set the image containing the current SpoolCache wheel}"
CACHE_DIR="${SPOOLCACHE_PATH:-$HOME/.cache/spoolcache}"
mkdir -p "$CACHE_DIR"
docker run --rm \
  --mount "type=bind,src=$CACHE_DIR,dst=/var/lib/spoolcache" \
  -e SPOOLCACHE_PATH=/var/lib/spoolcache \
  --entrypoint spoolcache "$RUNTIME_IMAGE" config
```

Use an expanded absolute path for `CACHE_DIR` in this Docker example. The
provided Compose files and shell launchers also handle a literal `~/...` value.
Shell launchers require paths without spaces or shell metacharacters.

On multiple hosts, an omitted path or literal `~/...` resolves under each host's
own user home. An absolute override uses the same pathname on each machine.
The disks remain rank-local; SpoolCache does not replicate payloads between hosts.

## Build a serving image

Obtain a wheel and trusted `release.json` from a release or follow the clean-tree
candidate build in [Release](RELEASE.md#build-a-local-candidate). Export the wheel
path, SHA-256 and source commit as shown there. Then extend the qualified runtime:

```bash
: "${PINNED_RUNTIME_IMAGE:?Set the qualified runtime image digest}"
: "${SPOOLCACHE_WHEEL:?Set the wheel path relative to this build context}"
: "${SPOOLCACHE_WHEEL_SHA256:?Set its trusted SHA-256}"
: "${SPOOLCACHE_COMMIT:?Set the source commit from release.json}"
docker build -f Dockerfile.release \
  --build-arg BASE_IMAGE="$PINNED_RUNTIME_IMAGE" \
  --build-arg SPOOLCACHE_WHEEL="$SPOOLCACHE_WHEEL" \
  --build-arg SPOOLCACHE_WHEEL_SHA256="$SPOOLCACHE_WHEEL_SHA256" \
  --build-arg SPOOLCACHE_COMMIT="$SPOOLCACHE_COMMIT" \
  -t spoolcache-runtime:local .
```

The Dockerfile authenticates the wheel, installs with `--no-deps` and retains it
under `/opt/spoolcache-release`. Serving imports installed package files.
Deploy the same immutable image content to every participant; mutable tag names
alone do not prove image equality. Do not bind-mount a development source tree
or add it to serving `PYTHONPATH`.

## Distributed serving

Retain the topology that works for the underlying vLLM deployment. SpoolCache
derives PP/TP/DCP coordinates and stage-local layer ownership from public runtime
facts. Every required PP×TP participant must agree before a persistent hit;
partial inventories cannot supply a hit.

Do not infer support for a topology solely from a successful startup or a CPU
fixture. Qualify the exact runtime/image/model revision and full participant set
using the [development checks](DEVELOPMENT.md#verify-a-persistent-hit).

## Start, observe and recover

Use the deployment's existing service manager or launcher. Observe participant
liveness together with SpoolCache readiness, inventory quorum and fatal state.
vLLM's `/health` alone is insufficient: a qualified remote PP-stage failure left
the head endpoint responsive while inference hung.

After a post-admission restore failure, SpoolCache terminates the affected
worker with exit code 70. The deployment must replace the complete serving
group and reject incomplete responses. Keep cache directories intact during
ordinary restart and recovery. See [Operations](OPERATIONS.md#failure-and-recovery).

Upgrade and rollback replace the complete group with one qualified artifact.
Changed package/runtime/model identities can select a new cache namespace.
The [migration guide](MIGRATION.md) explains interface and path changes without
assuming old cache state can be renamed into a new identity.
