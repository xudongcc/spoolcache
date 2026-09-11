# Token-file connector

`SpoolCacheConnector` in `spoolcache.vllm.connector` is the sole public connector.
It stores each complete runtime-aligned full-KV interval in one self-contained file
per rank. Chained keys include preceding tokens, salt, media and deployment
identity. Consecutive key availability across every rank determines the reusable
prefix. There is no explicit prefix tree or logical/physical packing layer.

Use the JSON emitted by `spoolcache config` without changing its connector fields.
The earlier experimental `SpoolCacheTokenConnector` entry point is removed.
Install an authenticated wheel with Python 3.11+. Version 0.2.0 uses the earlier
snapshot backend; follow [Migration](MIGRATION.md) when upgrading.

The file schema participates in the deployment namespace. Old snapshot and slot
roots are preserved without migration or reinterpretation. The package contains
no arena/index backend, prefix bundles, O_DIRECT path or io_uring extension.
CUDA host registration remains part of the fixed transfer buffers.

## Contracts and current limits

- A small-file transfer batch contains at most 16 files in one 64 MiB credit.
  Larger files stream through consecutive ranges of at most 64 MiB, with each
  range authenticated before GPU placement and one final whole-file check.
  Two CUDA credits rotate; a separate 64 MiB CPU authentication slot gives
  192 MiB explicit payload staging per rank. OS page cache and runtime/model
  allocations are separate from this budget.
- Saves occur synchronously at the pre-forward ownership boundary. Existing
  keys are fully authenticated before skipping GPU capture. New files require
  payload and directory durability before advertisement.
- Mixed HMA needs an additional complete state file for the exact captured
  boundary. Full-KV files from a longer prefix cannot manufacture missing
  earlier sliding/recurrent state. Every rank must have both full data and the
  required state before a hit is admitted.
- Chunk width is `ceil(256 / A) * A`, where A is the runtime LCM of reusable
  groups' logical page spans, with scratch capacity excluded. This is an aligned
  minimum of 256 tokens, not `lcm(256, A)`. One data interval or complete boundary state can
  contain at most 512 transfer ranges (32 GiB), subject to the 64 KiB header
  bound. Chain descriptors share layout views and use incremental coverage
  checks; their memory grows with keys and transfer checksums. There is no
  separate chain key-count, encoded-header-total or descriptor-memory bound;
  see [Design](SPOOLCACHE_DESIGN.md#resource-bounds).
- Active restores pin every key through the final CUDA drain. GC skips pinned
  keys. Reverse batch touches favor earlier prefix files; durable mtime drives
  LRU selection, including after restart, without a duplicate in-memory recency
  table. Metadata lookup and inventory scans do not refresh recency. The
  inventory uses a conservative 512 MiB per-rank memory allowance rather than
  a fixed key count. Startup sends its complete memory-bounded image.
- Capacity collection starts at 80% and a round attempts about 20% of keys,
  at most four per batch. There is no fixed stop watermark. Repeated directory
  scans remain a scaling cost; large-directory/GC-pressure qualification for
  token files remains outstanding.
- The worker runs its token-file scrubber. The removed snapshot `request` and
  `status` CLI commands do not apply to this backend. Scrub metrics use the
  existing vLLM exporter; `spoolcache config` remains the CLI entry point.

The [final evidence](receipts/2026-09-11-token-files/README.md) records source
review, model checks and historical performance with their exact artifact scope.
Earlier performance numbers are not measurements of a later review build;
warm file-hit measurements are not cold-NVMe throughput results.

The Gemma regression on checkpoint `9061441`, TP=1/PP=1, passes 32 requests and
ten complete prefix audits across reset/restart. Mixed consumers consistently answer
`OTHER`; cache output agreement does not establish mixed recognition quality.
