# Configuration

SpoolCache has two public connector settings. The environment renderer converts
them to vLLM's `--kv-transfer-config` JSON and validates values before returning it:

```bash
spoolcache config
```

The command prints one compact JSON object to stdout, reading `SPOOLCACHE_PATH`
and `SPOOLCACHE_MAX_SIZE`. It does not start vLLM or create cache directories.
Invalid settings produce an error on stderr and a nonzero exit code.

Environment variables alone do not attach the connector to vLLM. Pass the result
with `--kv-transfer-config` as shown in the [README](../README.md#use).

## Settings

| Environment variable | Connector JSON key | Default |
| --- | --- | --- |
| `SPOOLCACHE_PATH` | `spoolcache_path` | Current user's `~/.cache/spoolcache` |
| `SPOOLCACHE_MAX_SIZE` | `spoolcache_max_size` | `200` GB per rank |

The Python configuration fields and accepted short JSON aliases are `path`
and `max_size`. Unknown keys and duplicate
aliases are errors. JSON capacity values must be finite numbers, not booleans or numeric strings.
The environment renderer parses its capacity string first.

## Paths

Omitting the path uses the current process user's home, resolved when the
configuration is created. Explicit `~/...` values are expanded too. A relative
path, empty value or filesystem root is rejected by the Python configuration.
The worker creates its owned directories during initialization; rendering JSON
creates none.

Docker launchers use the same `SPOOLCACHE_PATH` name on the host. They bind-mount
that directory into the container and set the container-visible path explicitly.
The provided launchers use `/var/lib/spoolcache` inside the container. Defaults
in a multi-host launcher resolve under each host's user home; an absolute override
selects that same absolute pathname on both machines. Shell launchers accept
paths without spaces or shell metacharacters. See [Deployment](DEPLOYMENT.md).

## Enabling persistent caching

Configuring `SpoolCacheConnector` enables both persistent reads and writes,
whether configuration comes from the renderer or direct JSON. To disable it,
omit the connector from vLLM's transfer configuration. vLLM's process-local
prefix cache remains independently controlled by vLLM.

## Cache isolation

Model locator/revision, runtime, topology and cache layout are bound into cache
identity automatically. Use separate `SPOOLCACHE_PATH` directories to isolate
services or start with an empty persistent cache; each rank directory permits
only one inventory owner. Requests can use vLLM's `cache_salt` for separate
prefix keys. Keep model artifacts immutable under the same locator/revision.

## Capacity

Capacity uses GB in the LMCache convention: **1 GB = 1024³ bytes (1 GiB)**.
The default is `200`, preserving the previous 214,748,364,800-byte limit.
Fractional values such as `0.5` are supported; conversion to bytes rounds down.
The resulting limit must be at least two bytes. Internal storage accounting and
metrics continue to use integer bytes. For example:

```bash
export SPOOLCACHE_MAX_SIZE=200
```

Capacity is managed per rank, with background reclamation toward 90% of the
configured maximum. It is a maintenance target, not a reservation or filesystem
quota: in-flight writes, control state and quarantine require additional space.
For example, 200 GiB per rank is not a 200 GiB total budget for a multi-rank group.

## Direct JSON

This is the current interface; see [Migration](MIGRATION.md) for legacy settings:

```json
{
  "kv_connector": "SpoolCacheConnector",
  "kv_connector_module_path": "spoolcache.vllm.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "spoolcache_path": "~/.cache/spoolcache",
    "spoolcache_max_size": 200
  }
}
```

## Fixed implementation choices

KV payload I/O always requires `O_DIRECT`. Unsupported roots fail initialization;
there is no buffered payload fallback. Manifests and SQLite state use ordinary I/O.
Each rank has separate 128 MiB pinned and 128 MiB aligned-I/O staging pools.
Chunking, admission, inventory, scrub and GC low-watermark bounds are internal
constants, documented in [Design](SPOOLCACHE_DESIGN.md#resource-bounds).

No model profile, compatibility selector, checkpoint-directory digest or I/O-mode
option is accepted. Model identity and cache layout come from public runtime facts.

## Request-level read and write controls

Pass independent boolean flags through vLLM's `kv_transfer_params`:

```json
{
  "kv_transfer_params": {
    "spoolcache.skip_read": true,
    "spoolcache.skip_write": true
  }
}
```

| `skip_read` | `skip_write` | Persistent restore | New persistent writes |
| --- | --- | --- | --- |
| false / omitted | false / omitted | Allowed | Allowed |
| true | false / omitted | Skipped | Allowed |
| false / omitted | true | Allowed | Skipped |
| true | true | Skipped | Skipped |

Only the literal JSON boolean `true` enables a flag; strings and integers are
ignored. These flags control request KV reuse and publication, not filesystem
permissions or the worker's background integrity and capacity maintenance.
`skip_read` prevents a persistent hit from being admitted, while `skip_write`
prevents new store tracking. Neither flag disables vLLM's GPU prefix,
multimodal or encoder caches. Both flags together therefore do not guarantee
a completely cold request. Use a fresh `cache_salt` or reset process-local caches
as appropriate, and verify zero cached tokens for a cold control. See the
[persistent-hit checks](DEVELOPMENT.md#verify-a-persistent-hit).
