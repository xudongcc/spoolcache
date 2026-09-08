# SpoolCache

SpoolCache persists vLLM KV cache shards on local NVMe so requests with a shared
prefix can reuse cached state across engine restarts. It runs as an external
vLLM connector, with fixed staging buffers and no changes to vLLM's source.

- Exact text and multimodal prefix identity, derived from the serving runtime.
- Complete cache-group and PP/TP participant agreement before a persistent hit.
- Mandatory `O_DIRECT` for KV payloads; authenticated, immutable disk objects.
- Bounded inventory, capacity reclamation, background integrity checks and metrics.

SpoolCache is experimental. Runtime support is determined by public-interface
checks and discovered cache semantics. See [compatibility and evidence](docs/COMPATIBILITY.md)
for tested environments and limitations.

> These docs describe the current checkout, including unreleased interface changes.
> The recorded PyPI release is 0.1.0. Use the [installation guide](docs/INSTALLATION.md)
> for that release and the [migration table](docs/MIGRATION.md) when upgrading.

## Install

Install SpoolCache in the same Python environment as vLLM:

```bash
python -m pip install spoolcache==0.1.0
```

To use the interfaces documented below, install this checkout from its repository
root in that environment:

```bash
python -m pip install -e .
spoolcache --help
```

Python 3.10+ is required. Serving additionally requires Linux, a working
CUDA/vLLM runtime and storage supporting SpoolCache's aligned `O_DIRECT` I/O.
The package does not install vLLM, PyTorch, CUDA or model weights.
Production images use a verified wheel; see [deployment](docs/DEPLOYMENT.md).

## Use

Start with a model and immutable revision that already work in your vLLM runtime.
Set `MODEL_ID` and `MODEL_REVISION` accordingly, then run:

```bash
: "${MODEL_ID:?Set your model identifier}"
: "${MODEL_REVISION:?Set its immutable revision}"
# Optional: SPOOLCACHE_PATH defaults to ~/.cache/spoolcache.
# export SPOOLCACHE_PATH=/mnt/nvme/spoolcache

KV_CONFIG=$(python -m spoolcache.vllm.config_json)
vllm serve "$MODEL_ID" --revision "$MODEL_REVISION" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-transfer-config "$KV_CONFIG"
```

Keep your model's normal memory, topology and serving options. The configuration
renderer does not start vLLM; pass its output with `--kv-transfer-config`.
The startup gate rejects incompatible runtime and allocator contracts, including
an enabled `expandable_segments` allocator setting.

Ordinary requests need no extra SpoolCache field. Reuse requires an exact shared
prefix, compatible identities and a runtime-safe persistence boundary. A shorter
latency or nonzero API cached-token count alone does not prove an NVMe hit;
follow the [verification procedure](docs/DEVELOPMENT.md#verify-a-persistent-hit).

Per-request flags `spoolcache.skip_read` and `spoolcache.skip_write` independently
control persistent reads and writes through `kv_transfer_params`. Both default
to false and leave vLLM's own caches enabled. See [request controls](docs/CONFIGURATION.md#request-level-read-and-write-controls).

## Configure

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPOOLCACHE_PATH` | `~/.cache/spoolcache` | Cache directory visible to the process; `~` is expanded. |
| `SPOOLCACHE_MAX_SIZE` | `200` | Capacity target per rank in GB (1 GB = 1024³ bytes); decimals allowed. |

Connector JSON uses `spoolcache_path` and `spoolcache_max_size`. Configuring the
connector enables both persistent reads and writes. To disable SpoolCache, omit its connector configuration. See the
[configuration reference](docs/CONFIGURATION.md).

Docker launchers also use `SPOOLCACHE_PATH` for the host directory, defaulting to
each host user's `~/.cache/spoolcache`. They map it to the configured container
path. See [container paths](docs/DEPLOYMENT.md#container-paths).

## Operate

Inspect or request an integrity check for one owned rank directory:

```bash
spoolcache status --root /absolute/cache/deployment-digest/rank-0000
spoolcache request --root /absolute/cache/deployment-digest/rank-0000 \
  --entry 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`request` queues work for the running worker's scrubber. Metrics appear on
vLLM's `/metrics` endpoint. Capacity reclamation and integrity checks run in the
worker; the deployment owns process restarts and complete-group recovery.
Read the [operations guide](docs/OPERATIONS.md) before manipulating cache state.

## Develop

From this checkout:

```bash
uv sync --python 3.12 --locked --group dev
uv run --locked pytest -q
```

CPU storage tests use real `O_DIRECT`; a GPU is unnecessary. Set `TMPDIR` to a
supported filesystem when needed. GPU/runtime tests require their target image.
The [development guide](docs/DEVELOPMENT.md) covers wheel builds, containers,
multimodal checks and the isolated two-node harness.

## Documentation

[Documentation index](docs/README.md) · [Installation](docs/INSTALLATION.md) ·
[Configuration](docs/CONFIGURATION.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Operations](docs/OPERATIONS.md) · [Development](docs/DEVELOPMENT.md) ·
[Design](docs/SPOOLCACHE_DESIGN.md) · [Releases](docs/RELEASE.md) ·
[Goals](docs/TODO_GOALS.md)

SpoolCache is distributed under the [MIT license](LICENSE).
