# Live-lab working reference

The maintained human guide is [Lab qualification](../../../../docs/LAB.md).
Read it before running a live test. Installation, development and recovery
procedures are linked there; historical receipts remain evidence for their
recorded artifacts, not automatic qualification of later edits.

## Local repositories

| Role | Path |
| --- | --- |
| Implementation | `/root/projects/xudongcc/spoolcache` |
| DeepSeek deployment | `/root/projects/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` |
| Qwen deployment | `/root/projects/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` |
| GLM deployment | `/root/projects/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` |
| Read-only design reference | `/root/projects/FujitsuPolycom/sparkcache` |
| Read-only design reference | `/root/projects/LMCache/LMCache` |

Keep each deployment integration on `main`, as one local rebased commit above
its upstream unless the user changes that policy. Do not restore the former
DeepSeek `xudongcc`/`spoolcache` branch. Inspect current Git state before acting;
this reference does not assert that a checkout is clean or synchronized.

## Operating rules

- Use the pinned Gemma fixture for ordinary real-model development. Use the
  three deployment runtimes for compatibility work or explicitly tracked
  deployment-specific qualification. These roles must never enter core model
  selectors or configuration.
- Install and authenticate the same immutable wheel and image on both hosts.
  Serving launchers do not synchronize source. Runtime tests must import the
  installed package, with only tests and benchmark helpers copied in.
- Treat private environment files as secrets. Source them only within commands
  that need them; never print or broadly search their contents, or place API
  keys in process arguments.
- The lab shares GPUs and ports. Run one model group at a time, replace the
  complete group when recovery requires it, and restore the previously active
  service after a qualification run. Session instructions govern authorization.
- Never delete a persistent cache root to reset process-local caches or recover
  an experiment. Fault tests require narrow authenticated backups, bounded
  client deadlines, confirmed old-rank shutdown and verified restoration.
- Follow the lab's host-memory guardrails. Do not replay the unsafe GLM
  992,769-token cold prefill. A configured context limit is not host-memory
  qualification.
- Require complete output evidence, full rank quorum and offline payload
  authentication. One HTTP health response or cached-token counter is
  insufficient proof of persistent reuse or distributed recovery.

The [original reference](../../../../docs/archive/README.md) is archived for
provenance. Use the maintained guide for current commands and path settings.
