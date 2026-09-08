#!/usr/bin/env python3
"""Benchmark the authenticated O_DIRECT leg of an existing cache entry."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from spoolcache.store import ManifestStore

MIB = 1024 * 1024


def resolve_entry(rank_root: Path, prefix: str) -> str:
    matches = sorted(rank_root.glob(f"manifests/{prefix[:2]}/{prefix}*.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one manifest for {prefix!r}, found {len(matches)}"
        )
    return matches[0].stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-root", type=Path, required=True)
    parser.add_argument("--entry", required=True, help="full entry ID or unique prefix")
    parser.add_argument("--slot-mib", type=int, default=64)
    parser.add_argument("--slot-count", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    entry_id = resolve_entry(args.rank_root, args.entry)
    slot_bytes = args.slot_mib * MIB
    with ManifestStore(
        args.rank_root,
        slot_bytes=slot_bytes,
        slot_count=args.slot_count,
    ) as store:
        initial = store.lookup(entry_id, verify_payloads=False)
        if not initial.is_hit or initial.manifest is None:
            raise SystemExit(f"entry is unavailable: {initial.reason}")
        manifest = initial.manifest
        largest = max(item.byte_length for item in manifest.objects)
        if largest > slot_bytes:
            raise SystemExit("entry contains an object larger than the staging slot")
        destination = bytearray(largest)
        rows: list[dict[str, float]] = []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            probed = store.lookup(entry_id, verify_payloads=False)
            probe_seconds = time.perf_counter() - started
            if not probed.is_hit:
                raise SystemExit(f"metadata probe failed: {probed.reason}")

            copied = 0

            def sink(view: memoryview) -> None:
                nonlocal copied
                destination[: len(view)] = view
                copied += len(view)

            started = time.perf_counter()
            for descriptor in manifest.objects:
                store.stream_object(descriptor, sink)
            transfer_seconds = time.perf_counter() - started
            if copied != manifest.logical_bytes:
                raise SystemExit("streamed byte count differs from manifest")
            rows.append(
                {
                    "probe_seconds": probe_seconds,
                    "transfer_seconds": transfer_seconds,
                    "total_seconds": probe_seconds + transfer_seconds,
                }
            )

    total_mib = manifest.logical_bytes / MIB
    median_probe = statistics.median(row["probe_seconds"] for row in rows)
    median_transfer = statistics.median(row["transfer_seconds"] for row in rows)
    median_total = statistics.median(row["total_seconds"] for row in rows)
    result = {
        "entry": entry_id,
        "span_tokens": manifest.span_tokens,
        "objects": len(manifest.objects),
        "logical_mib": total_mib,
        "largest_object_mib": largest / MIB,
        "slot_mib": args.slot_mib,
        "slot_count": args.slot_count,
        "direct_io": True,
        "repetitions": args.repetitions,
        "median_probe_seconds": median_probe,
        "median_transfer_seconds": median_transfer,
        "median_restore_io_seconds": median_total,
        "effective_restore_mib_s": total_mib / median_total,
        "physical_read_mib_s": total_mib / median_total,
        "physical_to_logical_read_ratio": 1.0,
        "samples": rows,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
