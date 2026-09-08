# Migration

The changes below are present in the current checkout and remain unpublished. The recorded release is 0.1.0. Install a
wheel containing the new interfaces before switching production configuration.

## Interface changes after 0.1.0

| Surface | Published 0.1.0 / previous launchers | Current checkout |
| --- | --- | --- |
| Maintenance command | `spoolcache-maintenance` | `spoolcache` |
| Connector configuration command | `python -m spoolcache.vllm.config_json` | `spoolcache config`; the same renderer supplies validated JSON |
| Python maintenance entry point | `spoolcache.maintenance:main` | Unchanged |
| Environment path | `SPOOLCACHE_CONTAINER_ROOT` | `SPOOLCACHE_PATH` |
| Connector JSON path | `spoolcache_root` / `root` | `spoolcache_path` / `path` |
| Python configuration field | `SpoolCacheConfig.root` | `SpoolCacheConfig.path` |
| Renderer default path | `/var/lib/spoolcache` | Current user's `~/.cache/spoolcache` |
| Direct JSON path omission | Root required | Uses the same home-directory default |
| Capacity environment variable | `SPOOLCACHE_MAX_BYTES` (bytes) | `SPOOLCACHE_MAX_SIZE` (GB) |
| Capacity JSON / Python field | `spoolcache_max_bytes` / `max_bytes` | `spoolcache_max_size` / `max_size` |
| Default capacity per rank | `214748364800` bytes | `200` GB; same byte limit |
| Internal per-step metadata | Unchecked `SpoolCacheMetadata.schema` field | Removed; plans still travel through vLLM, with no change to disk formats from this cleanup |
| Operator namespace | `SPOOLCACHE_NAMESPACE`, `spoolcache_deployment_namespace`, `deployment_namespace` | Removed; use separate cache paths or request `cache_salt` for isolation |
| Access mode | `SPOOLCACHE_ACCESS_MODE`, `spoolcache_access_mode`, `access_mode`, `AccessMode` | Removed; configuring the connector enables reads and writes |
| Request control | `spoolcache_bypass=true` skipped reads and writes | Independent `spoolcache.skip_read` / `spoolcache.skip_write`; set both true to skip persistent reads and writes |
| Benchmark request option | `--bypass-spoolcache` | `--skip-read` / `--skip-write`; combine both with a fresh salt for cold controls |
| Payload I/O modes | `required`, `best-effort`, `disabled` | Mandatory `O_DIRECT` |
| I/O configuration | `SPOOLCACHE_DIRECT_IO`, `spoolcache_direct_io`, `DirectIOMode`, store `direct_io` argument | Removed |
| Host path setting | `SPOOLCACHE_HOST_ROOT` and per-worker variants | `SPOOLCACHE_PATH` |
| Gemma PP host roots | Separate head/worker cache-root overrides | One `SPOOLCACHE_PATH` |
| Development Compose cache | Named volume | Host bind mount from `SPOOLCACHE_PATH` |

Old JSON keys are rejected, rather than treated as aliases. The environment
renderer no longer reads the old variables. Update shell automation and Python
callers together. The `request` and `status` maintenance subcommands, arguments
and JSON response semantics are unchanged. The new `config` subcommand renders connector JSON.

GB follows the LMCache convention: 1 GB = 1024³ bytes (GiB). Divide an old
byte limit by `1073741824` when converting it; do not copy the old byte count
into the new size field. Fractional sizes are accepted and rounded down to
whole bytes internally. `SpoolCacheConfig.max_bytes` remains a derived read-only
property for internal accounting, not a constructor or JSON setting.

To disable SpoolCache, omit its connector from vLLM configuration. An old
`SPOOLCACHE_ACCESS_MODE=disabled` environment variable no longer disables it;
old access-mode JSON keys are rejected. There are no separate deployment-level
store-only or restore-only modes.

The old `spoolcache_bypass` request key is no longer interpreted. Its behavior
is available by setting both `spoolcache.skip_read` and `spoolcache.skip_write`
to JSON `true`. The briefly introduced, unreleased `spoolcache.skip_save` name
was renamed to `spoolcache.skip_write` and has no alias. Each flag defaults to
false; vLLM's process-local caches remain active. Cold controls still require
an independent salt or suitable local reset and verified zero cached tokens.
Historical bypass receipts retain their original meaning for their artifact.

## Namespace removal and cache identity

Model locator/revision, runtime, topology and layout identity checks remain
mandatory. Services requiring separate storage use different `SPOOLCACHE_PATH`
directories; requests requiring separate prefix keys use `cache_salt`.
`SPOOLCACHE_NAMESPACE` is no longer read and its JSON/Python fields are rejected.

The prefix hash domain is now `spoolcache-exact-prefix/v2`, and scheduler/worker
coordination uses `spoolcache-coordination/v2`. The namespace component has been
removed from both encodings. Old prefix keys are not reused, even when the old
namespace was `default`; all participants must run the same new artifact.
Existing files remain on disk without automatic migration or deletion.

## Choose the intended path explicitly

Omitting `SPOOLCACHE_PATH` uses `~/.cache/spoolcache` for the process user.
In host launchers, a default or literal `~/...` resolves under each host's own
home; an absolute override selects the same pathname on every host. Containers
use their explicit mount target, `/var/lib/spoolcache` in the provided launchers.

Existing cache directories and named volumes are not moved, deleted or merged.
If you need an old cache location, configure its mount explicitly. Do not mistake
a new default directory for a loss of the previous data.

## Upgrade and rollback

1. Record the old wheel/image identity, model revision, namespace and cache paths.
2. Qualify the new immutable image and new configuration together.
3. Stop the complete serving group and start every participant with the same
   qualified artifact. Keep rank directories and their durable state intact.
4. Verify outputs, inventory quorum and payload integrity before accepting the
   deployment as qualified.

Package version is part of cache identity. A new version can select a fresh
namespace even when persistent schema names are unchanged. Do not promise hits
across versions or rename old directories into the new identity.

Rollback uses the retained qualified image and matching model revision. Preserve
complete rank trees, including generation counters, sentries, fences, tombstones
and quarantine. Do not restore only SQLite state or only object files.

No automatic migration service or implicit cache deletion is provided.
[Operations](OPERATIONS.md#preserve-and-retire-data) explains safe archival.
Version calculation remains owned by [python-semantic-release](RELEASE.md);
this update is classified as a regular `feat` for version calculation.
