#!/usr/bin/env python3
"""Calculate Store break-even points from matched TTFT observations.

This tool does not benchmark the model by itself.  It turns cold, miss+Store,
and persistent-hit medians from the same A/B workload into an explicit answer:
how many expected future reuses are needed to repay the synchronous Store cost?
Keeping the input explicit prevents dated measurements from becoming hidden
production policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StoreEconomics:
    """Matched benchmark observations; never a live admission policy."""

    cold_seconds: float
    miss_and_store_seconds: float
    restore_hit_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.cold_seconds,
            self.miss_and_store_seconds,
            self.restore_hit_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("store economics timings must be finite and non-negative")

    @property
    def store_penalty_seconds(self) -> float:
        return max(0.0, self.miss_and_store_seconds - self.cold_seconds)

    @property
    def hit_savings_seconds(self) -> float:
        return self.cold_seconds - self.restore_hit_seconds

    @property
    def break_even_future_reuses(self) -> float | None:
        savings = self.hit_savings_seconds
        if savings <= 0:
            return None
        return self.store_penalty_seconds / savings

    def net_saved_seconds(self, expected_future_reuses: float) -> float:
        if not math.isfinite(expected_future_reuses) or expected_future_reuses < 0:
            raise ValueError("expected future reuses must be finite and non-negative")
        return (
            expected_future_reuses * self.hit_savings_seconds
            - self.store_penalty_seconds
        )


def _case(raw: str) -> tuple[str, StoreEconomics]:
    parts = raw.split(",")
    if len(parts) != 4 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "case must be LABEL,COLD_SECONDS,MISS_STORE_SECONDS,HIT_SECONDS"
        )
    label = parts[0]
    try:
        values = tuple(float(value) for value in parts[1:])
        economics = StoreEconomics(*values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return label, economics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=_case,
        required=True,
        help="LABEL,COLD_SECONDS,MISS_STORE_SECONDS,HIT_SECONDS; repeatable",
    )
    parser.add_argument(
        "--expected-reuses",
        nargs="+",
        type=float,
        default=(0.0, 1.0, 2.0),
    )
    args = parser.parse_args()

    rows = []
    for label, economics in args.case:
        rows.append(
            {
                "label": label,
                "cold_seconds": economics.cold_seconds,
                "miss_and_store_seconds": economics.miss_and_store_seconds,
                "restore_hit_seconds": economics.restore_hit_seconds,
                "store_penalty_seconds": economics.store_penalty_seconds,
                "hit_savings_seconds": economics.hit_savings_seconds,
                "break_even_expected_future_reuses": (
                    economics.break_even_future_reuses
                ),
                "net_saved_seconds": {
                    str(reuses): economics.net_saved_seconds(reuses)
                    for reuses in args.expected_reuses
                },
            }
        )
    print(
        json.dumps(
            {
                "schema": "spoolcache-store-economics/v1",
                "cases": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
