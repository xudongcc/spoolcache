"""Deterministic token-file byte-oracle smoke benchmark; not model performance."""

import argparse
import hashlib
import json
import os
import platform
import struct
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from spoolcache.config import aligned_chunk_tokens
from spoolcache.prefix import prefix_digests
from spoolcache.token_files import TokenFileStore
from tests.token_fixtures import cpu_mover, layout_for


def run(root, tokens):
    layout = replace(layout_for(), num_manager_blocks=(tokens + 4096) // 64 + 10)
    binding = {
        "expected_deployment_digest": "b" * 64,
        "expected_rank_digest": "c" * 64,
        "expected_rank": 0,
        "expected_topology_digest": "d" * 64,
        "expected_profile": layout.profile,
        "expected_layout_digest": layout.digest,
    }
    results = []
    with TokenFileStore(root, layout=layout, slot_bytes=64 * 1024 * 1024,
                       slot_count=1, **binding) as store:
        mover = cpu_mover(layout, slot_bytes=64 * 1024 * 1024)
        producer = tuple(range(tokens))
        workloads = (
            ("producer", producer),
            ("repeat", producer),
            ("extension", producer + tuple(range(tokens, tokens + 4096))),
            ("branch", producer[: tokens // 2] + (999,) * (tokens // 2)),
        )
        for name, prompt in workloads:
            # Deterministic opaque bytes include all preceding tokens, so the
            # divergent tail really changes KV and shared prefixes stay equal.
            digest, pages = hashlib.sha256(), []
            for start in range(0, len(prompt), 64):
                digest.update(struct.pack("<64q", *prompt[start : start + 64]))
                pages.append(digest.digest() * 128)

            def capture(segments, selected, pages=pages):
                yield b"".join(
                    pages[page - 1]
                    for s in segments
                    for page in selected[s.group_index][
                        s.page_start : s.page_start + s.page_count
                    ]
                )

            mover._capture_packed = capture
            keys = prefix_digests(prompt, deployment_digest="b" * 64, chunk_tokens=aligned_chunk_tokens(layout.alignment_tokens))
            ids = (tuple(range(1, len(pages) + 1)),)
            counts = {"write": 0, "writev": 0, "readv": 0, "preadv": 0, "fsync": 0}

            def counted(operation, original, counts=counts):
                def invoke(*args, **kwargs):
                    counts[operation] += 1
                    return original(*args, **kwargs)

                return invoke

            from contextlib import ExitStack

            with ExitStack() as stack:
                for operation in counts:
                    stack.enter_context(
                        patch.object(
                            os,
                            operation,
                            new=counted(operation, getattr(os, operation)),
                        )
                    )
                started = time.perf_counter()
                mover.commit_keys(store, prefixes=keys, block_tables=ids)
                save_ms = (time.perf_counter() - started) * 1000
                save_counts = counts.copy()
                for operation in counts:
                    counts[operation] = 0
                restored = hashlib.sha256()

                @contextmanager
                def receiver(descriptor, restored=restored):
                    yield restored.update

                started = time.perf_counter()
                with store.restore_view(keys[-1].digest) as lease:
                    if not lease.result.is_hit:
                        raise AssertionError("saved prefix is not restorable")
                    store.stream_objects(lease.descriptors, receiver, lease=lease)
                restore_ms = (time.perf_counter() - started) * 1000
            expected = hashlib.sha256(b"".join(pages)).hexdigest()
            if restored.hexdigest() != expected:
                raise AssertionError("restored opaque bytes differ")
            results.append(
                {
                    "workload": name,
                    "tokens": len(prompt),
                    "save_ms": save_ms,
                    "restore_ms": restore_ms,
                    "save_calls": save_counts,
                    "restore_calls": counts.copy(),
                    "cache_bytes": store.disk_usage_bytes(),
                    "expected_sha256": expected,
                    "restore_ok": True,
                }
            )
    return {"root": str(root), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    output = {
        "kind": "CPU deterministic storage fixture",
        "python": platform.python_version(),
        "note": "warm OS caches; Python syscall counts; not a 0.2.0 comparison",
        "runs": [],
    }
    for repeat in range(args.repeats):
        for tokens in (4096, 16384):
            receipt = run(args.root / f"r{repeat}-{tokens}", tokens)
            receipt['repeat'] = repeat
            output['runs'].append(receipt)
            print(json.dumps({'completed': str(receipt['root'])}), flush=True)
    (args.root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
