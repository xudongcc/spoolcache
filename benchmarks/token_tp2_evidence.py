"""Two-host token-file qualification using actual process argv and rank logs."""

from __future__ import annotations

import ast
import json
import re

from benchmarks.token_file_evidence import TokenFileEvidence


def validate_process(command, artifact, mode):
    """Bind a serving process to the pinned fixture and exact connector mode."""
    if mode not in ("on", "off"):
        raise ValueError("unknown connector mode")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("missing process argv")
    if artifact["model"] not in command or "--revision" not in command:
        raise ValueError("missing pinned model/revision")
    if command[command.index("--revision") + 1] != artifact["revision"]:
        raise ValueError("model revision differs")
    if command[command.index("--tensor-parallel-size") + 1] != "2":
        raise ValueError("requires TP=2")
    if ("--kv-transfer-config" in command) != (mode == "on"):
        raise ValueError("connector mode differs")
    prefix_caching = artifact.get("prefix_caching")
    if prefix_caching is not None:
        if type(prefix_caching) is not bool:
            raise ValueError("prefix caching expectation must be boolean")
        enabled = "--enable-prefix-caching" in command
        disabled = "--no-enable-prefix-caching" in command
        if (enabled, disabled) != (prefix_caching, not prefix_caching):
            raise ValueError("cross-request prefix caching mode differs")
    if mode == "on":
        config = json.loads(command[command.index("--kv-transfer-config") + 1])
        if (config.get("kv_connector"), config.get("kv_connector_module_path")) != (
            artifact.get("kv_connector", "SpoolCacheConnector"),
            artifact.get("kv_connector_module_path", "spoolcache.vllm.connector"),
        ):
            raise ValueError("wrong connector")
    return command


class TokenTP2Evidence(TokenFileEvidence):
    store_event = "token save"

    def __init__(self, args, artifact):
        self.args, self.artifact = args, artifact
        self.containers = [args.head_container, args.worker_container]
        if not all(self.containers) or not args.worker_host:
            raise ValueError("both hosts/containers are required")
        self.identities = []

    def process(self, rank, mode):
        container = self.containers[rank]
        state = json.loads(
            self.call(
                rank,
                [
                    "docker",
                    "inspect",
                    "--format",
                    (
                        '{"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}},'
                        '"running":{{json .State.Running}},"labels":{{json .Config.Labels}}}'
                    ),
                    container,
                ],
            )
        )
        artifact = self.artifact
        if state["image"] != artifact["image_id"] or state["running"] is not True:
            raise ValueError("wrong image or stopped rank")
        labels = state["labels"]
        if (
            labels.get("io.spoolcache.commit") != artifact["commit"]
            or labels.get("io.spoolcache.wheel.sha256") != artifact["wheel_sha256"]
        ):
            raise ValueError("image artifact labels differ")
        # The deployment entrypoint execs vLLM. Inspect actual PID 1 argv,
        # rather than treating a Compose shell program as vLLM arguments.
        command = json.loads(
            self.call(
                rank,
                [
                    "docker",
                    "exec",
                    container,
                    "python3",
                    "-c",
                    "from pathlib import Path;import json;print(json.dumps(Path('/proc/1/cmdline').read_bytes().decode().rstrip('\\0').split('\\0')))",
                ],
            )
        )
        state["argv"] = validate_process(command, artifact, mode)
        return state

    def refresh(self, require_restarted=False):
        previous, identities = self.identities, []
        logs = self.logs()
        if len(logs) != 2:
            raise ValueError("both TP rank logs are required")
        for rank, log in enumerate(logs):
            state = self.process(rank, "on")
            lines = re.findall(
                r"spoolcache: rank identity rank=" + str(rank) + r" .*", log
            )
            ready = re.findall(
                r"spoolcache: worker ready rank=" + str(rank) + r".*?root=(\S+)", log
            )
            if not lines or not ready:
                raise ValueError("missing rank identity/readiness")
            identity = dict(re.findall(r"(\w+)=([^ ]+)", lines[-1]))
            if any(
                identity.get(k) != v
                for k, v in {
                    "rank": str(rank),
                    "pp_rank": "0",
                    "tp_rank": str(rank),
                    "dcp_rank": "0",
                }.items()
            ):
                raise ValueError("incorrect TP participant coordinates")
            identity.update(state, root=ready[-1])
            credits = re.findall(
                r"spoolcache: worker ready rank=" + str(rank)
                + r"[^\n]* transfer_bytes=(\d+)", log,
            )
            if not credits:
                raise ValueError("missing runtime transfer credit")
            identity["transfer_bytes"] = int(credits[-1])
            layouts = re.findall(
                r"spoolcache: HMA runtime layout profile=\S+ groups=(\[.*?\]) "
                r"layers=\d+ alignment=\d+ logical_digest=\w+ physical_digest=(\w+)",
                log,
            )
            matches = [
                ast.literal_eval(groups)
                for groups, digest in layouts
                if identity["hma_layout"].startswith(digest)
            ]
            if not matches:
                raise ValueError("missing independently logged HMA geometry")
            identity["groups"] = matches[-1]
            if require_restarted:
                if (
                    len(previous) != 2
                    or identity["started"] == previous[rank]["started"]
                ):
                    raise ValueError("both ranks must restart")
                for key in (
                    "deployment",
                    "rank_identity",
                    "topology",
                    "hma_layout",
                    "root",
                ):
                    if identity[key] != previous[rank][key]:
                        raise ValueError("persistent identity changed across restart")
            identities.append(identity)
        # Deployment namespaces bind each worker's runtime/model view and may
        # differ across hosts. Authenticate them per rank; shared topology and
        # this fixture's physical geometry must agree, as must restored keys.
        for key in ("topology", "hma_layout"):
            if len({i[key] for i in identities}) != 1:
                raise ValueError("TP ranks disagree on runtime identity")
        self.identities = identities
