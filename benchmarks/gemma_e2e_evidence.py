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

    def manifests(self):
        result = []
        for rank, container in enumerate(self.containers):
            result.append(json.loads(self.call(rank, ["docker", "exec", container, "python3", "-c",
                "from pathlib import Path; import json,sys; "
                "print(json.dumps(sorted(p.name for p in Path(sys.argv[1]).glob('manifests/*/*.json'))))",
                self.identities[rank]["root"]])))
        return result

    def verify(self, phase, samples):
        logs = self.logs()
        receipts = []
        for rank, log in enumerate(logs):
            (self.args.output / f"{phase}-rank{rank}.log").write_text(log)
            if re.search(r"Traceback \(most recent call last\)|NCCL error|CUDA error|"
                         r"spoolcache:.*(?:fatal|checksum mismatch)", log, re.IGNORECASE):
                raise ValueError(f"rank {rank}: runtime error in logs")
        for kind, sample in samples.items():
            request, span = sample["request_id"], sample["cached_tokens"]
            hits = re.findall(r"spoolcache: hit request=" + re.escape(request) +
                              r"\S* tokens=(\d+) entry=([0-9a-f]+)", logs[0])
            if not hits or {int(s) for s, _ in hits} != {span} or len({e for _, e in hits}) != 1:
                raise ValueError(f"{kind}: missing or conflicting scheduler hit")
            entry_prefix = hits[-1][1]
            for rank, container in enumerate(self.containers):
                restores = re.findall(r"spoolcache: restore rank=" + str(rank) + r" request=" +
                    re.escape(request) + r"\S* tokens=(\d+) entry=([0-9a-f]+)", logs[rank])
                if not restores or set(restores) != {(str(span), entry_prefix)}:
                    raise ValueError(f"{kind}: rank {rank} restore disagrees with scheduler")
                identity = self.identities[rank]
                root, groups = identity["root"], identity["groups"]
                entry = self.call(rank, ["docker", "exec", container, "python3", "-c",
                    "from pathlib import Path; import sys; "
                    "p=list(Path(sys.argv[1]).glob('manifests/*/'+sys.argv[2]+'*.json')); "
                    "assert len(p)==1,p; print(p[0].stem)", root, entry_prefix]).strip()
                pages = []
                for group in groups:
                    if (group["policy"] not in ("full", "sliding") or group["running_state_tail_pages"] != 0
                            or group["eagle"] or group["dcp_shards"] != 1):
                        raise ValueError("unqualified fixture cache layout")
                    tokens = span if group["policy"] == "full" else min(span, group["window"])
                    if tokens % group["block"]:
                        raise ValueError("fixture span is not page aligned")
                    pages.append(tokens // group["block"] if group["layers"] else 0)
                expected = {
                    "rank-root": root, "entry-id": entry, "expected-span": span,
                    "expected-deployment-identity-digest": identity["deployment"],
                    "expected-rank-identity-digest": identity["rank_identity"],
                    "expected-physical-rank": rank, "expected-tp-degree": 1,
                    "expected-pp-degree": len(self.containers), "expected-dcp-degree": 1,
                    "expected-pp-rank": rank, "expected-tp-rank": 0, "expected-dcp-rank": 0,
                    "expected-topology-digest": identity["topology"],
                    "expected-layout-digest": identity["hma_layout"],
                    "expected-groups": len(groups), "expected-layers": sum(g["layers"] for g in groups),
                    "expected-group-layers": ",".join(str(g["layers"]) for g in groups),
                    "expected-pages": ",".join(map(str, pages)),
                }
                command = ["docker", "exec", "-w", "/opt/spoolcache", container,
                           "python3", "benchmarks/verify_entry_content.py"]
                for key, value in expected.items():
                    command += ["--" + key, str(value)]
                verified = json.loads(self.call(rank, command))
                if verified["status"] != "all-payloads-authenticated":
                    raise ValueError(f"{kind}: rank {rank} payload authentication failed")
                receipts.append({"kind": kind, "rank": rank, "expected": expected, "verification": verified})
                print(kind, phase, "rank", rank, verified["status"], flush=True)
        for suffix, value in (("payload-verification", receipts), ("runtime-identities", self.identities)):
            (self.args.output / f"{phase}-{suffix}.json").write_text(json.dumps(value, indent=2) + "\n")
