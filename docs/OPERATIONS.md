# Operations

SpoolCache stores each physical rank's state under the configured cache directory:

```text
SPOOLCACHE_PATH/
  deployment-identity-digest/
    rank-0000/
      .spoolcache-root
      manifests/
      objects/
      state/
      tmp/
      quarantine/
```

The ownership marker and durable state belong to the rank directory. Preserve
the complete directory when archiving or restoring it. Do not copy only the
payloads or remove generation state to force a startup to succeed.

## Inspect progress

The installed `spoolcache` command exposes `status` and `request`. It is a
maintenance client; the worker performs the background work.

```bash
spoolcache status --root /absolute/cache/deployment-digest/rank-0000
```

`--root` names one existing owned **rank directory**, not the top-level
`SPOOLCACHE_PATH`. It must be absolute, canonical and non-symlinked. The command
reads SQLite state without modifying it. `initialized: false` means the scrub
state database has not been initialized; it is not a full cache-health verdict.

## Request an integrity check

```bash
spoolcache request --root /absolute/cache/deployment-digest/rank-0000 \
  --entry 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

The entry ID must be a 64-character lowercase hexadecimal digest. The command
persists a targeted request and returns a nonce. Repeating the same pending
entry is idempotent; a different pending request must complete first.

A queued request does not prove authentication has finished. Poll `status` for
that entry's outcome: `authenticated`, `absent` or `quarantined`. The running
worker's scheduled scrubber must be present to process the request.

A rank-local quarantine receipt is not yet a scheduler receipt. Worker inventory
changes travel through vLLM's stats channel after an iteration. Wait for quorum
withdrawal before replaying a known-bad entry. In an isolated test, an unrelated
small request with `spoolcache.skip_write=true` can provide the reporting iteration;
never use the affected prompt as that barrier.

## Background maintenance

Workers run a rank-local scrubber with fixed internal bounds:
64 MiB/s, at most 64 work items per step, a 60-second initial delay for a new
store and a six-hour cycle interval. Durable SQLite work tables track namespace
snapshots, live references and progress at complete-object boundaries.

Scrub validates identity, file type, stored and logical lengths, SHA-256 and
zero padding. Known-bad objects and referring manifests are withdrawn and
quarantined. Startup inventory alone is metadata validation; payloads are fully
authenticated during restore or scrub.

Capacity maintenance reclaims proven crash orphans before evicting healthy LRU
entries. It works toward 90% of `SPOOLCACHE_MAX_SIZE`. Monitor both live-cache
usage and total managed/control/quarantine storage: the setting is not a hard
filesystem quota. Do not delete unknown files or quarantine evidence automatically.

Shutdown requests scrub cancellation and waits up to five seconds. A timeout
produces a structured `spoolcache-scrub-shutdown/v1` receipt; a daemon finalizer
retains resources until the reader exits. A timeout receipt does not mean the
background I/O operation completed successfully.

## Metrics

Read vLLM's existing `/metrics` endpoint. SpoolCache uses a `spoolcache_` prefix
and bounded label sets. The surface covers:

| Area | Useful signals |
| --- | --- |
| Reuse | Lookup outcomes, restored tokens, restore/store bytes and duration |
| Capacity | Live cache, total managed usage, quarantine and cleanup |
| Integrity | Quarantine, scrub progress, failures and shutdown timeouts |
| Distributed state | Required/ready ranks, generation, quorum and fatal readiness |

Combine these with worker liveness and inference results. A lower latency or
an API cached-token count can also reflect vLLM's GPU cache. For the current
metric definitions, inspect [telemetry.py](../src/spoolcache/telemetry.py) and
[the exporter](../src/spoolcache/vllm/prometheus.py).

## Failure and recovery

| Observation | Meaning and response |
| --- | --- |
| `O_DIRECT` unavailable at initialization | Choose a supported filesystem/mount. Payload fallback is intentionally absent. |
| Runtime compatibility or layout startup error | Inspect the rejected public contract and qualify the runtime. Do not bypass the gate. |
| Persistent miss | Check exact prefix, identity, span/boundary eligibility and all-rank inventory. A miss is safe. |
| Store skipped or failed | Publication was not completed; inference may continue without that new entry. Inspect storage/admission metrics. |
| Corruption detected before a hit commitment | Withdraw/quarantine it and serve a normal miss. |
| Restore failure after admission | Reject the incomplete response; replace the full serving group after the affected worker exits 70. |
| Lost generation sentry/state | Preserve the directory for investigation. Do not reset durable identity bookkeeping. |

The package does not supervise Docker, SSH or remote processes. In the qualified
PP=2 runtime, a remote worker exit could leave the head's `/health` responsive.
Observe every participant and stop the full group before restart.

## Preserve and retire data

To start cold, choose a new cache path and retain the previous directory.
To archive old data, stop all owners, identify the exact deployment/rank trees,
record their identities and sizes, and preserve manifests, payloads, state,
markers and quarantine together. Cache deletion is an explicit operator action;
ordinary launch, restart and upgrade do not clear persistent state.

See [Migration](MIGRATION.md) for renamed settings and
[Lab qualification](LAB.md) for controlled corruption procedures.
