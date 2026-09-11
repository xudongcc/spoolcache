"""Independent bounded file audit plus runtime evidence for token-file fixtures."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import struct
import sys
from pathlib import Path

from benchmarks.gemma_e2e_evidence import Evidence

SCHEMA = "spoolcache-token-key-file/v3"
FRAME = struct.Struct("<8sI32s")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def valid_digest(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid expected/file digest")
    return value


def state_key(key):
    return hashlib.sha256(
        b"spoolcache-hma-boundary/v1\0" + bytes.fromhex(valid_digest(key))
    ).hexdigest()


def audit(root, entry, span, identity):
    """Require independent runtime identity/geometry before reading any payload."""
    valid_digest(entry)
    if type(span) is not int or not 0 < span < (1 << 63):
        raise ValueError("invalid expected span")
    expected_binding = {
        "deployment_digest": valid_digest(identity["deployment"]),
        "rank_digest": valid_digest(identity["rank_identity"]),
        "rank": identity["physical_rank"],
        "topology_digest": valid_digest(identity["topology"]),
        "profile": "vllm-runtime-kv-v1",
        "layout_digest": valid_digest(identity["hma_layout"]),
        "storage_schema": SCHEMA,
    }
    if type(expected_binding["rank"]) is not int or expected_binding["rank"] < 0:
        raise ValueError("invalid expected rank")
    groups = identity["groups"]
    if not groups or [g["index"] for g in groups] != list(range(len(groups))):
        raise ValueError("invalid expected groups")
    for group in groups:
        for name in ("layers", "block", "page_bytes", "dcp_shards"):
            if type(group[name]) is not int or group[name] <= 0:
                raise ValueError("invalid expected group geometry")
        if (
            type(group["eagle"]) is not bool
            or group["policy"] not in ("full", "sliding", "recurrent_align", "circular_one")
        ):
            raise ValueError("unsupported expected reuse policy")
        if group["policy"] == "sliding" and (
            type(group["window"]) is not int or group["window"] <= 1
        ):
            raise ValueError("invalid expected window")
    aligned_groups = [g for g in groups if g["policy"] != "circular_one"]
    quantum = math.lcm(*(g["block"] * g["dcp_shards"] for g in aligned_groups))
    quantum = ((256 + quantum - 1) // quantum) * quantum
    if not any(g["policy"] == "full" for g in groups) or span % quantum:
        raise ValueError("invalid expected aligned span")
    transfer_bytes = identity["transfer_bytes"]
    if type(transfer_bytes) is not int or not 0 < transfer_bytes <= 64 * 1024**2:
        raise ValueError("invalid expected transfer credit")
    expected_binding.update(chunk_tokens=quantum, transfer_bytes=transfer_bytes)
    root = Path(root)
    lock = os.open(root / "state/maintenance.lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise ValueError("invalid rank lock")
        # Production writers take SH on this gate before their namespace lock.
        # EX here keeps every file stable throughout the independent audit.
        fcntl.flock(lock, fcntl.LOCK_EX)
        page_sizes = {}
        group_geometries = {}

        def header(key, kind, expected_span, expected_parent=None):
            valid_digest(key)
            path = root / "manifests" / key[:2] / (key + ".kv")
            if path.parent.is_symlink():
                raise ValueError("symlink shard")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("nonregular payload")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    frame = stream.read(FRAME.size)
                    if len(frame) != FRAME.size:
                        raise ValueError("short frame")
                    magic, size, checksum = FRAME.unpack(frame)
                    if magic != b"SPCTOK02" or not 0 < size <= 64 * 1024:
                        raise ValueError("invalid frame")
                    raw = stream.read(size)
                if len(raw) != size or hashlib.sha256(raw).digest() != checksum:
                    raise ValueError("header checksum differs")
                obj = json.loads(raw)
                if set(obj) != {
                    "schema",
                    "kind",
                    "binding",
                    "key",
                    "parent",
                    "span_tokens",
                    "segments",
                    "byte_length",
                    "sha256",
                    "block_sha256",
                }:
                    raise ValueError("unexpected header fields")
                if (obj["schema"], obj["kind"], obj["key"]) != (SCHEMA, kind, key):
                    raise ValueError("file type/key differs")
                if digest(obj["binding"]) != digest(expected_binding):
                    raise ValueError("runtime binding differs")
                if (
                    type(obj["span_tokens"]) is not int
                    or obj["span_tokens"] != expected_span
                ):
                    raise ValueError("boundary differs")
                valid_digest(obj["sha256"])
                if kind == "state":
                    if obj["parent"] != expected_parent or key != state_key(
                        expected_parent
                    ):
                        raise ValueError("state parent differs")
                elif expected_span == quantum:
                    if obj["parent"] is not None:
                        raise ValueError("invalid root parent")
                else:
                    valid_digest(obj["parent"])
                segments = obj["segments"]
                if not isinstance(segments, list):
                    raise TypeError("invalid segments")
                expected = []
                for g in groups:
                    if (g["policy"] == "full") != (kind == "data"):
                        continue
                    names = [
                        x["layer_name"]
                        for x in segments
                        if x["group_index"] == g["index"]
                    ]
                    if len(names) != g["layers"] or names != sorted(set(names)):
                        raise ValueError("layer owners differ")
                    if digest({"layers": names})[:12] != g["layer_names_digest"]:
                        raise ValueError("runtime layer names differ")
                    page_tokens = g["block"] * g["dcp_shards"]
                    if kind == "data":
                        if quantum % page_tokens:
                            raise ValueError("data page width differs")
                        count = quantum // page_tokens
                        start = (expected_span - quantum) // page_tokens
                    else:
                        if g["policy"] != "circular_one" and expected_span % page_tokens:
                            raise ValueError("state alignment differs")
                        count = (
                            min(expected_span // page_tokens, math.ceil((g["window"] - 1) / page_tokens))
                            if g["policy"] == "sliding" else 1
                        )
                        start = 0
                    group_sizes = []
                    for name in names:
                        segment = next(
                            x
                            for x in segments
                            if x["group_index"] == g["index"]
                            and x["layer_name"] == name
                        )
                        segment_length = segment["byte_length"]
                        if (
                            type(segment_length) is not int
                            or segment_length <= 0
                            or segment_length % count
                        ):
                            raise ValueError("invalid layer size")
                        page_bytes = segment_length // count
                        owner = (g["index"], name)
                        if page_sizes.setdefault(owner, page_bytes) != page_bytes:
                            raise ValueError("layer size changes across chain")
                        group_sizes.append(page_bytes)
                        expected.append(
                            {
                                "group_index": g["index"],
                                "layer_name": name,
                                "page_start": start,
                                "page_count": count,
                                "byte_length": count * page_bytes,
                            }
                        )
                    # Infer the two public manager relationships from runtime
                    # bytes and validated owner coverage, never a model or TP
                    # preset: one page per layer, or a packed sum of members.
                    if all(size == g["page_bytes"] for size in group_sizes):
                        geometry = "uniform-layer-pages"
                    elif sum(group_sizes) == g["page_bytes"]:
                        geometry = "packed-group-pages"
                    else:
                        raise ValueError("runtime group byte total/page size differs")
                    if group_geometries.setdefault(g["index"], geometry) != geometry:
                        raise ValueError("manager geometry changes across chain")
                if digest(segments) != digest(expected):
                    raise ValueError("runtime page coverage differs")
                length = sum(x["byte_length"] for x in expected)
                if (
                    type(obj["byte_length"]) is not int
                    or obj["byte_length"] != length
                    or not 0 < length <= 512 * transfer_bytes
                ):
                    raise ValueError("payload length differs")
                hashes = obj["block_sha256"]
                if not isinstance(hashes, list) or len(hashes) != math.ceil(length / transfer_bytes):
                    raise ValueError("invalid transfer checksums")
                for value in hashes:
                    valid_digest(value)
                if info.st_size != FRAME.size + size + length:
                    raise ValueError("stored length differs")
                # Buffered header inspection may read ahead; explicitly seek
                # before the later bounded payload pass.
                os.lseek(fd, FRAME.size + size, os.SEEK_SET)
                return fd, obj, FRAME.size + size
            except BaseException:
                os.close(fd)
                raise

        # The rank gate keeps files stable across two bounded traversals.
        # Prove complete metadata coverage before payload I/O without retaining
        # every decoded file header.
        byte_count = 0

        def authenticate(key, kind, boundary, parent=None):
            nonlocal byte_count
            fd, obj, offset = header(key, kind, boundary, parent)
            try:
                remaining = obj["byte_length"]
                checksum = hashlib.sha256()
                for expected_hash in obj["block_sha256"]:
                    block_remaining = min(transfer_bytes, remaining)
                    block_checksum = hashlib.sha256()
                    while block_remaining:
                        data = os.read(fd, min(1024**2, block_remaining))
                        if not data:
                            raise ValueError("short payload")
                        checksum.update(data)
                        block_checksum.update(data)
                        block_remaining -= len(data)
                        remaining -= len(data)
                        byte_count += len(data)
                    if block_checksum.hexdigest() != expected_hash:
                        raise ValueError("transfer block checksum differs")
                if checksum.hexdigest() != obj["sha256"]:
                    raise ValueError("payload checksum differs")
            finally:
                os.close(fd)
            return obj["parent"], offset

        key, charged, file_count = entry, 0, 0
        for boundary in range(span, 0, -quantum):
            fd, obj, offset = header(key, "data", boundary)
            os.close(fd)
            key = obj["parent"]
            charged += offset
            file_count += 1
        if key is not None:
            raise ValueError("chain has no root")
        has_state = any(g["policy"] != "full" for g in groups)
        if has_state:
            fd, _, offset = header(state_key(entry), "state", span, entry)
            os.close(fd)
            charged += offset
            file_count += 1
        key = entry
        for boundary in range(span, 0, -quantum):
            key, _ = authenticate(key, "data", boundary)
        if has_state:
            authenticate(state_key(entry), "state", span, entry)
        return {
            "status": "all-payloads-authenticated",
            "entry_id": entry,
            "span_tokens": span,
            "files": file_count,
            "payload_bytes": byte_count,
            "metadata_bytes": charged,
            "group_page_geometry": group_geometries,
        }
    finally:
        os.close(lock)


class TokenFileEvidence(Evidence):
    def manifests(self):
        return [
            json.loads(
                self.call(
                    rank,
                    [
                        "docker",
                        "exec",
                        container,
                        "python3",
                        "-c",
                        "from pathlib import Path; import sys,json; print(json.dumps(sorted(p.name for p in Path(sys.argv[1]).glob('manifests/*/*.kv'))))",
                        self.identities[rank]["root"],
                    ],
                )
            )
            for rank, container in enumerate(self.containers)
        ]

    def verify(self, phase, samples):
        logs = self.logs()
        receipts = []
        for rank, log in enumerate(logs):
            (self.args.output / f"{phase}-rank{rank}.log").write_text(log)
            if re.search(
                r"Traceback \(most recent call last\)|NCCL error|CUDA error|spoolcache:.*(?:fatal|checksum mismatch)",
                log,
                re.IGNORECASE,
            ):
                raise ValueError("runtime error in log")
        for kind, sample in samples.items():
            request, span = sample["request_id"], sample["cached_tokens"]
            hits = re.findall(
                r"spoolcache: hit request="
                + re.escape(request)
                + r"\S* tokens=(\d+) entry=([0-9a-f]+)",
                logs[0],
            )
            if (
                not hits
                or {int(s) for s, _ in hits} != {span}
                or len({k for _, k in hits}) != 1
            ):
                raise ValueError("missing/conflicting scheduler hit")
            prefix = hits[-1][1]
            for rank, container in enumerate(self.containers):
                restored = re.findall(
                    r"spoolcache: restore rank="
                    + str(rank)
                    + r" request="
                    + re.escape(request)
                    + r"\S* tokens=(\d+) entry=([0-9a-f]+)",
                    logs[rank],
                )
                if not restored or set(restored) != {(str(span), prefix)}:
                    raise ValueError(f"rank {rank} restore disagrees with scheduler")
                identity = dict(self.identities[rank], physical_rank=rank)
                entry = self.call(
                    rank,
                    [
                        "docker",
                        "exec",
                        container,
                        "python3",
                        "-c",
                        "from pathlib import Path; import sys; p=list(Path(sys.argv[1]).glob('manifests/*/'+sys.argv[2]+'*.kv')); assert len(p)==1,p; print(p[0].stem)",
                        identity["root"],
                        prefix,
                    ],
                ).strip()
                # The audited helper is streamed into this container; only
                # benchmark code is injected, never serving package source.
                import subprocess

                command = [
                    "docker",
                    "exec",
                    "-i",
                    container,
                    "python3",
                    "-",
                    json.dumps(
                        {
                            "root": identity["root"],
                            "entry": entry,
                            "span": span,
                            "identity": identity,
                        }
                    ),
                ]
                if rank:
                    command = [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        "--",
                        self.args.worker_host,
                        shlex.join(command),
                    ]
                result = subprocess.check_output(
                    command, input=Path(__file__).read_text(), text=True, timeout=180
                )
                receipt = json.loads(result)
                if receipt["status"] != "all-payloads-authenticated":
                    raise ValueError("audit failed")
                receipts.append(
                    {"kind": kind, "expected": identity, "verification": receipt}
                )
        (self.args.output / f"{phase}-runtime-identities.json").write_text(
            json.dumps(self.identities, indent=2) + "\n"
        )
        (self.args.output / f"{phase}-payload-verification.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        print(
            f"{phase}: {len(receipts)} complete prefix payload audits passed",
            flush=True,
        )


if __name__ == "__main__":
    print(json.dumps(audit(**json.loads(sys.argv[1]))))
