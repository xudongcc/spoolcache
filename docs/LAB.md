# Lab qualification

This guide describes the project's reproducible test fixtures and linked
laboratory deployments. The generic product instructions are in
[Deployment](DEPLOYMENT.md). Machine addresses, models and partitions here are
qualification inputs, not runtime support rules.

## Repositories and fixtures

| Purpose | Repository / entry point |
| --- | --- |
| Generic connector and single-GPU fixture | This repository's Dockerfile and Compose |
| Two-node functional fixture | [gemma-pp2-dev.sh](../scripts/gemma-pp2-dev.sh) |
| DeepSeek runtime compatibility | [MiaAI-Lab DeepSeek deployment](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) |
| Qwen runtime compatibility | [MiaAI-Lab Qwen deployment](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) |
| GLM runtime compatibility | [MiaAI-Lab GLM deployment](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) |

The existing lab uses two DGX Spark hosts connected by CX-7. Only one model group
may use the shared GPUs/API port at a time. Check current services before a live
test and restore the previously active deployment afterwards. Local deployment
repositories retain one integration commit above their upstream `main`; routine
working changes must not be mistaken for pushed updates.

General live feature, correctness and fault work uses `google/gemma-4-E2B-it` at
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Runtime compatibility work separately
uses the linked DeepSeek/Qwen/GLM stacks. CPU storage and protocol tests need no
model download. Exact qualified image identities are in [the receipts](receipts/README.md).

## Single-node fixture

Follow [Development](DEVELOPMENT.md#run-the-single-gpu-development-fixture).
The pinned official vLLM 0.28.0 container runs TP=1/PP=1, a 16K context and the
installed wheel. It mounts host `SPOOLCACHE_PATH` (default `~/.cache/spoolcache`)
at `/var/lib/spoolcache`. Model artifacts are read-only and offline at serving time.

The fixture enables vLLM development endpoints. They expose reset and debug/RPC
operations and are appropriate only in the isolated lab. Production launchers
keep them disabled. Stopping Compose preserves the host cache directory.

## Two-node fixture

Use the [environment example](../scripts/gemma-pp2-dev.env.example) for host-only
overrides. Copy it to ignored `.env.gemma-pp2` if needed; do not print private
environment files. The checked-in model, revision, image, TP/PP and partition
remain fixed for reproducibility.

```bash
scripts/gemma-pp2-dev.sh --help
scripts/gemma-pp2-dev.sh preflight
# Run these explicit operations only when the corresponding artifact is missing:
# scripts/gemma-pp2-dev.sh image-sync
# scripts/gemma-pp2-dev.sh model-sync
scripts/gemma-pp2-dev.sh start
scripts/gemma-pp2-dev.sh status
```

`start` verifies the hosts and image equality, launches worker before head and
waits for readiness. `stop` and `restart` operate on the complete PP group.
`logs head` / `logs worker` follow the selected container. Cache roots are retained.

One `SPOOLCACHE_PATH` selects host cache storage. Defaults or literal `~/...`
resolve against each host's home; absolute overrides are used on both hosts.
The container path is `/var/lib/spoolcache`. The previous separate Gemma head/
worker cache-root variables are removed; old directories remain untouched.

The upstream `VLLM_PP_LAYER_PARTITION=12,23` was established for this exact
Gemma/vLLM fixture. It is not a SpoolCache configuration option or model rule.
A different runtime partition must first work with the connector omitted and pass
its own output oracle. The launcher is not a general process supervisor.

## Installed-runtime checks

Run tests inside each exact candidate image, using only `tests/` and `benchmarks/`
as qualification tools. Do not inject `src/` into the import path. The Gemma
image contains those tools under `/opt/spoolcache`; the installed package remains
the data path. Authenticate it against the retained wheel with
`benchmarks/verify_release_install.py`.

Record source commit, wheel SHA-256, immutable image ID, model revision, runtime
build, topology and stage ownership. For production images, keep the upstream
model/runtime setup unchanged and install the same wheel without dependency
resolution. Do not reintroduce source synchronization.

## Media fixtures

Use immutable bytes or a data URL for controls and restores. Record origin,
rights, size and SHA-256; media files are not redistributed in this repository.

| Fixture | Recorded origin | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| COCO cats image, `val2017/000000039769.jpg` | [COCO](https://cocodataset.org/#termsofuse) | 173131 | `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e` |
| Archery video | [PyTorchVideo fixture](https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4) | 549197 | `8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562` |
| Mary Had a Little Lamb audio | [vLLM fixture](https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg) | 65449 | `c8f0a87f8d7e44f2d6e0f88ec63f6401b4f153f53fd14a9d730a5d1ba9927c4e` |

Verify the downloaded bytes against the recorded sizes and digests before a new test.
Repository licensing does not by itself establish redistribution rights to
all included upstream media. Test each enabled modality separately and a mixed
request containing every jointly enabled modality. Keep media order and geometry
identical between controls and restores.

## Persistent-hit procedure

Follow [the generic procedure](DEVELOPMENT.md#verify-a-persistent-hit). Include
API usage fields, exact expected aligned cached tokens, scheduler and every
worker entry/span agreement, payload authentication and complete output checks.
Use a stable semantic oracle or independently repeated identical cold controls.
An empty content field, HTTP 200 or one favorable completion hash is not proof.

Clear GPU prefix, encoder and multimodal caches while keeping
`reset_external=false`. A full deployment-managed group restart also clears
process-local state. Keep salt, media, sampling and the exact token prefix fixed.

## Controlled corruption

A corruption test is deliberate mutation of one authenticated cache object,
with a narrow backup and exact recovery receipt. Never clear a whole cache root
as setup or cleanup. Confirm the entry's deployment/rank/layout/topology and
full payload before touching a byte.

Select one uniquely referenced object from the exact rank manifest. Check regular
file type, lengths, hash and padding; back up that object and manifest, fsync the
backups and their directory, then independently authenticate the backup. Perform
the mutation under the rank maintenance lock, without changing another entry
or its manifest. Keep a fixed external client deadline for every fault request.
If backup integrity, old-rank shutdown or recovery timing is uncertain, stop and
preserve the evidence. A stopped experiment must not leave a known corrupt object
advertised by a running group.

For pre-admission scrub:

1. Flip one byte in the selected authenticated object and fsync it.
2. Request that exact entry through `spoolcache request` and wait for its durable
   `quarantined` result and bounded quarantine metrics.
3. Wait for scheduler quorum withdrawal. If idle, use an unrelated skip-write request
   as a stats-transport barrier, not the affected prompt.
4. Restart the whole group to remove GPU cache state and require an external miss,
   zero cached tokens and the established output oracle.
5. Restore/republish only through a verified path; preserve quarantine evidence.

For post-admission failure:

1. Wait until every participant offers the authenticated entry, then back up and
   change one object byte after the offer.
2. Replay the matching request. Require an incomplete response to be rejected and
   observe the affected worker's exit 70.
3. Stop the complete group through the deployment manager. A live head `/health`
   does not establish remote-stage readiness.
4. Restore the exact backed-up bytes atomically, fsync and authenticate all
   relevant payloads before replacing the group.
5. Verify the cross-restart hit, participant agreement and output oracle again.
   Remove a backup only after that recovery proof succeeds.

## Long-context safety

Increase context lengths through bounded steps while monitoring every host's
SSH/API liveness, `MemAvailable` and `SwapFree`. Record thresholds and minima and
abort before the fixed reserve is crossed. The historical G3d guard sampled every
two seconds with 3 GiB available-memory and 4 GiB swap-free reserves.

A connector bypass does not limit unified-memory prefill. The GLM 992,769-token
attempt previously required host recovery and must not be directly replayed.
A stateful producer beyond its runtime-proven boundary remains `unsafe_boundary`;
qualify a longer consumer using a shorter authenticated prefix instead.

Never print API keys or private `.env` files. Let clients read authentication
from their environment. Use only the deployment's existing lifecycle and health
commands, and preserve the exact failed observations in [receipts](receipts/README.md).
