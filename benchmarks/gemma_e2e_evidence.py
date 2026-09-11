"""Independent runtime/log and full-payload evidence for the fixed Gemma fixture."""
from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess


class Evidence:
    def __init__(self, args, model, revision):
        self.args, self.model, self.revision = args, model, revision
        self.containers = [args.head_container]
        if args.topology == "pp2":
            self.containers.append(args.worker_container)
        self.identities = []

    def call(self, rank, command):
        if rank:
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "--", self.args.worker_host, shlex.join(command)]
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=180)

    def logs(self):
        result = []
        for rank, container in enumerate(self.containers):
            started = self.call(rank, ["docker", "inspect", "--format", "{{.State.StartedAt}}", container]).strip()
            result.append(re.sub(r"\x1b\[[0-9;]*m", "", self.call(
                rank, ["docker", "logs", "--since", started, container])))
        return result

    def refresh(self, require_restarted=False):
        previous = self.identities
        identities = []
        for rank, (container, log) in enumerate(zip(self.containers, self.logs())):
            # Inspect only the fields needed here; environment variables can contain secrets.
            state = json.loads(self.call(rank, ["docker", "inspect", "--format",
                '{"id":{{json .Id}},"image":{{json .Image}},"started":{{json .State.StartedAt}},'
                '"running":{{json .State.Running}},"cmd":{{json .Config.Cmd}}}', container]))
            if state["image"] != self.args.image_id or state["running"] is not True:
                raise ValueError(f"rank {rank}: wrong image or stopped container")
            command = state["cmd"]
            if self.model not in command or "--revision" not in command:
                raise ValueError(f"rank {rank}: missing pinned model/revision")
            if command[command.index("--revision") + 1] != self.revision:
                raise ValueError(f"rank {rank}: model revision mismatch")
            lines = re.findall(r"spoolcache: rank identity rank=" + str(rank) + r" .*", log)
            if not lines:
                raise ValueError(f"rank {rank}: missing runtime identity")
            identity = dict(re.findall(r"(\w+)=([^ ]+)", lines[-1]))
            identity.update(state)
            identity["root"] = re.findall(
                r"spoolcache: worker ready rank=" + str(rank) + r".*?root=(\S+)", log)[-1]
            identity["transfer_bytes"] = int(re.findall(
                r"spoolcache: worker ready rank=" + str(rank)
                + r"[^\n]* transfer_bytes=(\d+)", log)[-1])
            layouts = re.findall(
                r"spoolcache: HMA runtime layout profile=\S+ groups=(\[.*?\]) "
                r"layers=\d+ alignment=\d+ logical_digest=\w+ physical_digest=(\w+)", log)
            matches = [ast.literal_eval(groups) for groups, digest in layouts
                       if identity["hma_layout"].startswith(digest)]
            if not matches:
                raise ValueError(f"rank {rank}: missing independent HMA layout")
            identity["groups"] = matches[-1]
            if require_restarted:
                if not previous or identity["started"] == previous[rank]["started"]:
                    raise ValueError(f"rank {rank}: whole-group restart not proved")
                for key in ("deployment", "rank_identity", "topology", "hma_layout", "root"):
                    if identity[key] != previous[rank][key]:
                        raise ValueError(f"rank {rank}: identity changed across restart: {key}")
            identities.append(identity)
        # PP stages have distinct model views and deployment digests. Only the
        # common topology must agree; authenticate each stage against its own facts.
        if len({i["topology"] for i in identities}) != 1:
            raise ValueError("ranks disagree on topology")
        self.identities = identities
