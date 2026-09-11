# Operations

SpoolCache stores each physical rank's state under the configured cache directory:

```text
SPOOLCACHE_PATH/
  deployment-identity-digest/
    rank-0000/
      .spoolcache-root
      manifests/
      objects/       # empty reserved directory; no separate payload backend
      state/
      tmp/
      quarantine/
```

The ownership marker and durable state belong to the rank directory. Preserve
the complete directory when archiving or restoring it. Do not copy only the
payloads or remove generation state to force a startup to succeed.

## Inspect progress

Use vLLM's `/metrics` endpoint and the worker logs. The worker persists a bounded
`state/token-scrub.json` cursor, but reading that file alone is not a cache-health
verdict. The CLI exposes `spoolcache config`; old snapshot `status` and `request`
commands are removed. Targeted checks in qualification use the token scrubber
against an already identity-bound store.

A rank-local quarantine receipt is not yet a scheduler receipt. Worker inventory
changes travel through vLLM's stats channel after an iteration. In an isolated
test, an unrelated request with both `spoolcache.skip_read=true` and
`spoolcache.skip_write=true` can provide that
reporting iteration before replaying an affected entry.

## Background maintenance

Each file contains an authenticated header and exactly one payload. Full-KV
files cover the smallest multiple of the common reusable-group alignment that
is at least 256 tokens; mixed HMA additionally needs a complete boundary-state
file. Larger files stream through fixed-size buffers. There are no separate
payload objects or SQLite reference/index tables.

Workers scrub at 64 MiB/s, at most 64 keys per step, with a 60-second initial
delay and six-hour cycle interval. Durable progress advances only after a whole
file is checked. Identity, file type, exact length and SHA-256 are authenticated.
Scrub skips active keys, quarantines corruption and reconciles worker inventory.
Metadata-only startup scanning does not authenticate payload bytes.

Capacity collection starts at 80% of `SPOOLCACHE_MAX_SIZE`. A round attempts
20% of keys, at most four per batch, using file-mtime LRU and skipping pins.
There is no fixed stop watermark. Reverse touches favor earlier prefix files;
deleting an intermediate key makes longer chains miss. Repeated directory scans
remain a scaling cost. The limit is not a hard filesystem quota: control state,
temporary writes and quarantine need additional space.

Shutdown cancels scrub and waits up to five seconds. A timeout returns a
`spoolcache-scrub-shutdown/v1` receipt; a daemon finalizer retains resources
until the reader exits. The timeout does not certify that I/O completed.

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
| File-lock or filesystem initialization error | Use a supported local Linux filesystem and preserve the rejected root for inspection. |
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
