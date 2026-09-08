from __future__ import annotations

import json
import math
import unittest

from spoolcache.telemetry import (
    LOOKUP_LABELS,
    MAX_HISTOGRAM_OBSERVATIONS,
    MAX_METRIC_RECORDS,
    MAX_METRIC_VALUE,
    METRIC_DEFINITIONS,
    MetricKind,
    TelemetryBuffer,
    aggregate_gauges,
    empty_metric_payload,
    iter_metric_records,
    merge_metric_payloads,
    metric_payload_is_empty,
    reduce_metric_payload,
    validate_metric_payload,
)


class TelemetryTests(unittest.TestCase):
    def test_required_metric_surface_and_labels_are_finite(self) -> None:
        required = {
            "spoolcache_lookup_total",
            "spoolcache_hit_tokens_total",
            "spoolcache_restore_bytes_total",
            "spoolcache_restore_seconds",
            "spoolcache_store_bytes_total",
            "spoolcache_store_seconds",
            "spoolcache_store_skipped_total",
            "spoolcache_post_admission_failure_total",
            "spoolcache_rank_quorum_entries",
            "spoolcache_rank_generation_changes_total",
            "spoolcache_pinned_pool_bytes",
            "spoolcache_delayed_store_requests",
            "spoolcache_disk_bytes",
            "spoolcache_managed_disk_bytes",
            "spoolcache_quarantine_bytes",
            "spoolcache_quarantined_entries_total",
            "spoolcache_scrub_namespace_items_total",
            "spoolcache_scrub_shutdown_failures_total",
            "spoolcache_required_ranks",
            "spoolcache_ready_ranks",
            "spoolcache_readiness",
        }
        self.assertLessEqual(required, set(METRIC_DEFINITIONS))
        for name, definition in METRIC_DEFINITIONS.items():
            self.assertTrue(name.startswith("spoolcache_"))
            self.assertEqual(
                len(definition.label_names),
                len(definition.allowed_labels[0])
                if definition.allowed_labels
                else 0,
            )
            self.assertLessEqual(len(definition.allowed_labels), 32)
            for labels in definition.allowed_labels:
                self.assertTrue(all(len(value) <= 32 for value in labels))
        self.assertIn(("hit", "ready_entry"), LOOKUP_LABELS)
        self.assertIn(("miss", "rank_quorum"), LOOKUP_LABELS)

    def test_buffer_drains_deltas_but_repeats_current_gauges(self) -> None:
        telemetry = TelemetryBuffer(source="rank:7")
        telemetry.increment(
            "spoolcache_lookup_total",
            labels=("hit", "ready_entry"),
        )
        telemetry.increment("spoolcache_hit_tokens_total", value=1024)
        telemetry.observe("spoolcache_restore_seconds", 0.125)
        telemetry.set_gauge("spoolcache_disk_bytes", 4096)

        first = telemetry.drain()
        validate_metric_payload(first)
        counters = {
            (record.name, record.labels): record.value
            for record in iter_metric_records(first, MetricKind.COUNTER)
        }
        self.assertEqual(
            counters[("spoolcache_lookup_total", ("hit", "ready_entry"))],
            1,
        )
        self.assertEqual(counters[("spoolcache_hit_tokens_total", ())], 1024)
        self.assertEqual(
            [record.values for record in iter_metric_records(
                first, MetricKind.HISTOGRAM
            )],
            [(0.125,)],
        )
        self.assertEqual(aggregate_gauges(first)["spoolcache_disk_bytes", ()], 4096)

        second = telemetry.drain()
        self.assertEqual(tuple(iter_metric_records(second, MetricKind.COUNTER)), ())
        self.assertEqual(tuple(iter_metric_records(second, MetricKind.HISTOGRAM)), ())
        self.assertEqual(
            aggregate_gauges(second)["spoolcache_disk_bytes", ()], 4096
        )

    def test_merge_sums_deltas_and_retains_latest_gauge_per_source(self) -> None:
        rank0 = TelemetryBuffer(source="rank:0")
        rank1 = TelemetryBuffer(source="rank:1")
        rank0.increment("spoolcache_restore_bytes_total", value=1024)
        rank1.increment("spoolcache_restore_bytes_total", value=2048)
        rank0.set_gauge("spoolcache_pinned_pool_bytes", 4096)
        rank1.set_gauge("spoolcache_pinned_pool_bytes", 8192)
        rank0.observe("spoolcache_restore_seconds", 0.1)
        rank1.observe("spoolcache_restore_seconds", 0.2)
        merged = merge_metric_payloads(rank0.drain(), rank1.drain())

        counters = tuple(iter_metric_records(merged, MetricKind.COUNTER))
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0].value, 3072)
        self.assertEqual(
            aggregate_gauges(merged)["spoolcache_pinned_pool_bytes", ()],
            12288,
        )
        histograms = tuple(iter_metric_records(merged, MetricKind.HISTOGRAM))
        self.assertEqual(histograms[0].values, (0.1, 0.2))

        rank0.set_gauge("spoolcache_pinned_pool_bytes", 6144)
        newer = merge_metric_payloads(merged, rank0.drain())
        self.assertEqual(
            aggregate_gauges(newer)["spoolcache_pinned_pool_bytes", ()],
            14336,
        )

    def test_unknown_or_unbounded_labels_fail_closed(self) -> None:
        telemetry = TelemetryBuffer(source="scheduler")
        invalid_calls = (
            lambda: telemetry.increment("not_spoolcache", value=1),
            lambda: telemetry.increment(
                "spoolcache_lookup_total",
                labels=("hit", "full-entry-id-" + "a" * 64),
            ),
            lambda: telemetry.increment(
                "spoolcache_store_skipped_total", labels=("model-name",)
            ),
            lambda: telemetry.set_gauge(
                "spoolcache_disk_bytes", float("nan")
            ),
            lambda: TelemetryBuffer(source="request:secret"),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

        malformed = empty_metric_payload()
        malformed["counters"] = [
            {
                "name": "spoolcache_lookup_total",
                "labels": ["hit", "tenant-a"],
                "value": 1,
            }
        ]
        with self.assertRaises(ValueError):
            validate_metric_payload(malformed)

    def test_histogram_memory_is_bounded_and_loss_is_reported(self) -> None:
        telemetry = TelemetryBuffer(source="rank:0")
        for index in range(MAX_HISTOGRAM_OBSERVATIONS + 3):
            telemetry.observe("spoolcache_store_seconds", index / 1000)
        payload = telemetry.drain()
        histogram = tuple(iter_metric_records(payload, MetricKind.HISTOGRAM))[0]
        self.assertEqual(len(histogram.values), MAX_HISTOGRAM_OBSERVATIONS)
        dropped = {
            (record.name, record.labels): record.value
            for record in iter_metric_records(payload, MetricKind.COUNTER)
        }
        self.assertEqual(
            dropped[("spoolcache_telemetry_dropped_total", ("histogram",))],
            3,
        )

    def test_payload_is_primitive_bounded_and_safe_to_log(self) -> None:
        telemetry = TelemetryBuffer(source="scheduler")
        telemetry.increment(
            "spoolcache_lookup_total", labels=("miss", "rank_quorum")
        )
        telemetry.set_gauge(
            "spoolcache_readiness", 0, labels=("inventory_quorum",)
        )
        payload = telemetry.drain()
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("prompt", encoded.lower())
        self.assertNotIn("token_ids", encoded)
        self.assertNotIn("entry_id", encoded)
        summary = reduce_metric_payload(payload)
        self.assertEqual(
            summary["spoolcache_lookup_total{miss,rank_quorum}"], 1
        )
        self.assertEqual(
            summary["spoolcache_readiness{inventory_quorum}"], 0
        )
        self.assertFalse(metric_payload_is_empty(payload))
        self.assertTrue(metric_payload_is_empty(empty_metric_payload()))

    def test_numeric_validation_rejects_bool_negative_and_nonfinite(self) -> None:
        telemetry = TelemetryBuffer(source="scheduler")
        for invalid in (
            True,
            -1,
            math.inf,
            -math.inf,
            math.nan,
            "1",
            MAX_METRIC_VALUE + 1,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    telemetry.increment(
                        "spoolcache_hit_tokens_total", value=invalid
                    )

        telemetry.increment(
            "spoolcache_hit_tokens_total", value=MAX_METRIC_VALUE
        )
        with self.assertRaises(ValueError):
            telemetry.increment("spoolcache_hit_tokens_total", value=1)

    def test_transport_record_count_is_bounded(self) -> None:
        payload = empty_metric_payload()
        payload["gauges"] = [
            {
                "name": "spoolcache_disk_bytes",
                "labels": [],
                "source": f"rank:{index}",
                "value": 0,
            }
            for index in range(MAX_METRIC_RECORDS + 1)
        ]
        with self.assertRaisesRegex(ValueError, "record count"):
            validate_metric_payload(payload)

    def test_merge_revalidates_counter_and_record_count_bounds(self) -> None:
        maximum = TelemetryBuffer(source="scheduler")
        maximum.increment(
            "spoolcache_hit_tokens_total", value=MAX_METRIC_VALUE
        )
        extra = TelemetryBuffer(source="scheduler")
        extra.increment("spoolcache_hit_tokens_total")
        with self.assertRaisesRegex(ValueError, "merged counter"):
            merge_metric_payloads(maximum.drain(), extra.drain())

        first = empty_metric_payload()
        first["gauges"] = [
            {
                "name": "spoolcache_disk_bytes",
                "labels": [],
                "source": f"rank:{index}",
                "value": 0,
            }
            for index in range(MAX_METRIC_RECORDS)
        ]
        second = empty_metric_payload()
        second["gauges"] = [
            {
                "name": "spoolcache_disk_bytes",
                "labels": [],
                "source": f"rank:{MAX_METRIC_RECORDS}",
                "value": 0,
            }
        ]
        with self.assertRaisesRegex(ValueError, "record count"):
            merge_metric_payloads(first, second)

    def test_histogram_summary_rejects_numeric_overflow(self) -> None:
        payload = empty_metric_payload()
        payload["histograms"] = [
            {
                "name": "spoolcache_restore_seconds",
                "labels": [],
                "values": [MAX_METRIC_VALUE, 1],
            }
        ]
        with self.assertRaisesRegex(ValueError, "histogram summary"):
            reduce_metric_payload(payload)


if __name__ == "__main__":
    unittest.main()
