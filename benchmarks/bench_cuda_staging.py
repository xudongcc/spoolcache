#!/usr/bin/env python3
"""Measure the current GPU gather/pinned-copy/scatter staging sequence."""

from __future__ import annotations

import argparse
import json
import statistics
import time

MIB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-mib", type=int, default=64)
    parser.add_argument("--page-bytes", type=int, default=991_040)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    slot_bytes = args.slot_mib * MIB
    page_count = slot_bytes // args.page_bytes
    payload_bytes = page_count * args.page_bytes
    pages = torch.empty(
        (page_count, 1, args.page_bytes), dtype=torch.uint8, device="cuda"
    )
    pinned = torch.empty(slot_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    indexes = torch.arange(page_count, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()

    capture_samples: list[float] = []
    restore_samples: list[float] = []
    for repetition in range(args.repetitions + 1):
        started = time.perf_counter()
        gathered = torch.index_select(pages, 0, indexes).contiguous()
        pinned[:payload_bytes].copy_(gathered.view(-1), non_blocking=False)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - started

        started = time.perf_counter()
        staged = pinned[:payload_bytes].to(device="cuda", non_blocking=True)
        pages.index_copy_(0, indexes, staged.view_as(pages))
        torch.cuda.synchronize()
        restore_seconds = time.perf_counter() - started
        if repetition:
            capture_samples.append(capture_seconds)
            restore_samples.append(restore_seconds)

    median_capture = statistics.median(capture_samples)
    median_restore = statistics.median(restore_samples)
    payload_mib = payload_bytes / MIB
    result = {
        "slot_mib": args.slot_mib,
        "page_bytes": args.page_bytes,
        "page_count": page_count,
        "payload_mib": payload_mib,
        "repetitions": args.repetitions,
        "median_capture_seconds": median_capture,
        "median_capture_mib_s": payload_mib / median_capture,
        "median_restore_seconds": median_restore,
        "median_restore_mib_s": payload_mib / median_restore,
        "capture_samples": capture_samples,
        "restore_samples": restore_samples,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
