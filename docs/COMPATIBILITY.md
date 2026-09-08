# Compatibility and qualification

SpoolCache discovers public vLLM contracts, cache semantics and final tensor
ownership at startup. It has no model, architecture, modality or version
allowlist. Acceptance by that gate and qualification of a complete serving
workload are different kinds of evidence.

## Implemented scope

| Area | Current behavior |
| --- | --- |
| Integration | External vLLM V1 connector with HMA support; no vLLM source patch |
| Storage | Local, rank-owned persistent objects; mandatory `O_DIRECT` payload I/O |
| Prefixes | Exact text and qualified multimodal identity |
| Cache groups | Full/windowed/stateful semantics when runtime capabilities prove safe ownership and reuse |
| Parallelism | Runtime-derived PP/TP participants and DCP coordinates; complete-group quorum |
| Integrity | Manifest identity, logical payload hash, stored length and zero padding |
| Failure | Safe pre-admission miss; post-admission fail-stop and deployment-owned recovery |

Unknown contracts fail closed. LoRA, unsupported prompt embeddings and incomplete
multimodal identity do not receive ambiguous persistent hits. There is no remote
payload backend, encoder-cache persistence or GDS path.

## Recorded environments

The table summarizes historical qualification, not a restriction on other
models or a promise about arbitrary future versions. Exact artifact identities
are in the linked receipts.

| Fixture/deployment | Recorded runtime | Topology | Evidence |
| --- | --- | --- | --- |
| Gemma 4 E2B functional fixture | Official vLLM 0.28.0 | TP=1/PP=1 | [Text, image, video, audio and mixed cross-restart receipt](receipts/2026-09-07-gemma4-text-multimodal-cross-restart-summary.json) |
| Gemma 4 E2B G4 fixture | Official vLLM 0.28.0 | TP=1/PP=2 | [All five paths and stage-local ownership](receipts/2026-09-07-g4-gemma4-pp2-cross-restart-summary.json) |
| DeepSeek Vision compatibility deployment | `0.25.2.dev0+g752a3a504.d20260714` in G6 runtime checks | TP=2/PP=1 | [196,608-token consumer / 130,048-token restored prefix](receipts/2026-09-07-g3d-deepseek-196608-cross-restart-summary.json) |
| Qwen Flash Next compatibility deployment | `0.1.dev20073+g8e685d198` in G6 runtime checks | TP=2/PP=1 | [260,800-token text/image/video consumers / 160,000-token prefixes](receipts/2026-09-07-g3d-qwen-max-multimodal-cross-restart-summary.json) |
| GLM Flash EXL3 compatibility deployment | `0.1.dev20051+g487ecf187` in G6 runtime checks | TP=2/PP=1 | [107K cross-restart receipt](receipts/2026-09-07-g3d-glm-107k-cross-restart-summary.json) |

Runtime versions in this table identify the G6 runtime checks. Consult each
workload receipt for its own source/image/model revision; do not combine nearby
observations into an unrecorded qualification.

The compatibility deployments declared image input for DeepSeek and image/video
for Qwen/GLM; none declared audio in those runs. The Gemma fixture covers audio
as well. Runtime-enabled modalities are discovered automatically rather than
looked up in this table.

## Published artifact and local changes

[G6](receipts/2026-09-08-g6/README.md) qualified an exact candidate and separately
authenticated the published 0.1.0 wheel in four runtime images on both hosts.
The qualification carry-forward used equality of package payloads, not merely a
matching version string.

The current unreleased package was built as an isolated `0.2.0rc1` candidate
from a committed snapshot of the working tree. Its installed bytes were
verified on both hosts; the package source still matches the working tree
except for the candidate's python-semantic-release version stamp. Real vLLM
0.28.0/CUDA tests completed with 284 passes and one runtime-dependent skip.
The host suite passed 292 tests and 373 subtests, with 12 runtime-dependent skips.

The new [Gemma regression receipt](receipts/2026-09-08-current-gemma/README.md)
records PP=1 text/image/audio/video/mixed persistence and restart checks, and
PP=2 testing. An initial PP=2 mixed prompt failed the strict cold-output gate:
matching cold controls differed from restored output despite matching all-rank
entries and authenticated payloads.
The [follow-up diagnosis](receipts/2026-09-08-gemma-mixed-diagnosis/README.md)
attributes a reproducible instance to the runtime's prefix-cache computation:
same-span native GPU caching and disk restore produce identical token
probabilities, while cold computation differs. The original cold-output gate
remains strict; a different passing prompt or batch-invariant output does not
establish universal numerical equivalence. This candidate has
not been published.

Checks cover the two-setting configuration, unchanged 200 GiB capacity,
fractional sizes, rejected legacy namespace keys, model/salt separation,
legacy prefix-key separation, default restore/store behavior, allocator gating, startup
metrics, and all combinations of `spoolcache.skip_read` / `spoolcache.skip_write`.
Skip-read avoids persistent lookup while retaining eligible store plans;
skip-write retains restore plans while suppressing stores. Benchmark clients
forward both flags independently. Cold controls also need fresh salts or local
cache resets and verified zero cached tokens. See [Migration](MIGRATION.md).

The repository residue audit also checked current source/configuration/docs and
all three linked launcher integrations. Fourteen Python CLI help commands,
Python compilation, shell syntax and documentation links/examples passed.
Removed scripts were the superseded single-image client and inactive layerwise
and shared-staging probes. Historical receipts/results remain unchanged; that
audit alone did not qualify a CUDA or live-model artifact. The separately linked
current Gemma receipt records the later live checks and their unresolved limit.

CLI subprocess checks also cover `spoolcache config`: default and overridden
settings, compact JSON output, no cache-directory creation, and invalid settings
reported on stderr without partial JSON. Existing request/status checks still pass.

## Known limits

- The G6 PP=2 long padded image and mixed cold controls consistently returned
  `OTHER`, while a short image control returned `CATS`. Those long cases prove
  restored-output equality and payload integrity, not correct media classification.
- A remote PP-stage exit 70 could leave the head health endpoint responsive and
  an inference stream hung. Recovery requires deployment-owned whole-group stop.
- Stateful producers may exceed a proven snapshot boundary. Qwen's attempted
  260,800-token producers were skipped as `unsafe_boundary`; successful long
  consumers reused a shorter authenticated prefix.
- Maximum-context GLM qualification remains incomplete. The earlier 992,769-token
  attempt required host recovery; it is not safe to replay as a routine test.
- Fault injection is not a proof of behavior under every physical power failure.
  Runtime acceptance and CPU fixtures do not qualify every PP×TP placement.

Use [Development](DEVELOPMENT.md#verify-a-persistent-hit) to add a reproducible
qualification. Record failures and unchanged cold controls as carefully as hits.
