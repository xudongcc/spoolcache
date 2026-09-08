from __future__ import annotations

import unittest
from dataclasses import dataclass

import spoolcache.admission as admission
from benchmarks.analyze_store_economics import StoreEconomics
from spoolcache.admission import admit_store_plans


@dataclass(frozen=True)
class Plan:
    request_id: str
    entry_id: str
    span_tokens: int


class StoreAdmissionTests(unittest.TestCase):
    def test_benchmark_economics_is_not_a_runtime_api(self) -> None:
        self.assertFalse(hasattr(admission, "StoreEconomics"))

    def test_deduplicates_and_prefers_longest_prefix_with_stable_ties(self) -> None:
        plans = (
            Plan("short", "entry-short", 8_192),
            Plan("long-first", "entry-long-a", 32_768),
            Plan("duplicate", "entry-long-a", 32_768),
            Plan("long-second", "entry-long-b", 32_768),
        )

        result = admit_store_plans(plans, max_plans=2)

        self.assertEqual(
            tuple(plan.request_id for plan in result.admitted),
            ("long-first", "long-second"),
        )
        self.assertEqual(result.skipped_duplicate, 1)
        self.assertEqual(result.skipped_budget, 1)

    def test_zero_budget_skips_every_optional_store(self) -> None:
        result = admit_store_plans(
            (Plan("request", "entry", 8_192),),
            max_plans=0,
        )
        self.assertEqual(result.admitted, ())
        self.assertEqual(result.skipped_budget, 1)

    def test_conflicting_entry_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting"):
            admit_store_plans(
                (
                    Plan("one", "same-entry", 8_192),
                    Plan("two", "same-entry", 32_768),
                ),
                max_plans=1,
            )

    def test_measured_ttft_produces_reuse_break_even(self) -> None:
        # These are the matched 8K values in BENCHMARK_2026-09-04.md.
        economics = StoreEconomics(
            cold_seconds=4.910,
            miss_and_store_seconds=6.474,
            restore_hit_seconds=0.918,
        )
        self.assertAlmostEqual(economics.store_penalty_seconds, 1.564)
        self.assertAlmostEqual(economics.hit_savings_seconds, 3.992)
        self.assertAlmostEqual(
            economics.break_even_future_reuses or 0.0,
            1.564 / 3.992,
        )
        self.assertGreater(economics.net_saved_seconds(1), 0)
        self.assertLess(economics.net_saved_seconds(0), 0)

    def test_non_beneficial_hit_has_no_break_even(self) -> None:
        economics = StoreEconomics(1.0, 1.2, 1.1)
        self.assertIsNone(economics.break_even_future_reuses)
        self.assertLess(economics.net_saved_seconds(10), 0)


if __name__ == "__main__":
    unittest.main()
