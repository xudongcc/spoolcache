#!/usr/bin/env python3
"""Compare per-object synchronization with bounded two-slot CUDA pipelining."""

from __future__ import annotations

import argparse
import json
import statistics
import time

MIB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-mib", type=int, default=1)
    parser.add_argument("--objects", type=int, default=170)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.object_mib <= 0 or args.objects <= 2 or args.repetitions <= 0:
        raise SystemExit("object size, count, and repetitions must be positive")

    object_bytes = args.object_mib * MIB
    pinned = [
        torch.full(
            (object_bytes,),
            index + 1,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        for index in range(2)
    ]
    destination = torch.empty(object_bytes, dtype=torch.uint8, device="cuda")
    events = [torch.cuda.Event(blocking=False) for _ in range(2)]
    torch.cuda.synchronize()

    sequential_samples: list[float] = []
    pipeline_samples: list[float] = []
    for repetition in range(args.repetitions + 1):
        started = time.perf_counter()
        for index in range(args.objects):
            destination.copy_(pinned[index % 2], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        sequential = time.perf_counter() - started

        pending = [False, False]
        started = time.perf_counter()
        for index in range(args.objects):
            slot = index % 2
            if pending[slot]:
                events[slot].synchronize()
            destination.copy_(pinned[slot], non_blocking=True)
            events[slot].record(torch.cuda.current_stream())
            pending[slot] = True
        torch.cuda.current_stream().synchronize()
        pipelined = time.perf_counter() - started
        if repetition:
            sequential_samples.append(sequential)
            pipeline_samples.append(pipelined)

    sequential_median = statistics.median(sequential_samples)
    pipeline_median = statistics.median(pipeline_samples)
    result = {
        "schema": "spoolcache-restore-pipeline/v1",
        "object_mib": args.object_mib,
        "objects": args.objects,
        "logical_mib": args.object_mib * args.objects,
        "slots": 2,
        "repetitions": args.repetitions,
        "sequential_syncs": args.objects,
        "pipeline_terminal_syncs": 1,
        "sequential_median_seconds": sequential_median,
        "pipeline_median_seconds": pipeline_median,
        "speedup": sequential_median / max(pipeline_median, 1e-12),
        "sequential_samples": sequential_samples,
        "pipeline_samples": pipeline_samples,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
