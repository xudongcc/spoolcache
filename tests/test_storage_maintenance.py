"""Bounded shutdown, scheduling and reporter reconciliation for token scrub."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import spoolcache.maintenance as maintenance_module
from spoolcache.maintenance import ScheduledDeepScrubber
from spoolcache.token_scrub import TokenFileScrubber
from tests.token_fixtures import open_store

class TokenScrubSchedulingTests(unittest.TestCase):
    def test_scheduled_close_timeout_is_bounded_and_machine_readable(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingScrubber:
            def status(self):
                entered.set()
                release.wait(5)
                return SimpleNamespace(
                    last_completed_unix_ns=0,
                    phase="idle",
                    target_entry="",
                )

            def repair_invalid_state(self):
                return False

        scheduler = ScheduledDeepScrubber(
            BlockingScrubber(),  # type: ignore[arg-type]
            poll_seconds=0.01,
            startup_delay_seconds=3600,
            cycle_interval_seconds=3600,
        )
        scheduler.start()
        self.assertTrue(entered.wait(1))
        fallback_release = threading.Timer(0.2, release.set)
        fallback_release.start()
        try:
            with mock.patch.object(
                maintenance_module,
                "SCRUB_SHUTDOWN_TIMEOUT_SECONDS",
                0.01,
                create=True,
            ):
                started = time.monotonic()
                report = scheduler.close()
                elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.1)
            self.assertEqual(report.schema, "spoolcache-scrub-shutdown/v1")
            self.assertEqual(report.status, "timeout")
            self.assertTrue(report.thread_alive)
            self.assertEqual(
                scheduler.drain_metrics().get("shutdown_failures"),
                1,
            )
        finally:
            release.set()
            fallback_release.cancel()
        stopped = scheduler.close()
        self.assertEqual(stopped.status, "stopped")
        self.assertFalse(stopped.thread_alive)


    def test_scheduler_rejects_nonfinite_or_nonpositive_internal_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                scrubber = TokenFileScrubber(store)
                invalid = (
                    {"bytes_per_second": 0},
                    {"bytes_per_second": 1.5},
                    {"step_bytes": True},
                    {"item_budget": -1},
                    {"poll_seconds": 0},
                    {"poll_seconds": float("nan")},
                    {"startup_delay_seconds": -1},
                    {"startup_delay_seconds": float("inf")},
                    {"cycle_interval_seconds": float("-inf")},
                )
                for arguments in invalid:
                    with self.subTest(arguments=arguments):
                        with self.assertRaises(ValueError):
                            ScheduledDeepScrubber(scrubber, **arguments)


    def test_inventory_rescan_epoch_cannot_lose_a_concurrent_withdrawal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                scheduler = ScheduledDeepScrubber(TokenFileScrubber(store))
                self.assertIsNone(scheduler.pending_inventory_rescan_epoch())
                scheduler.require_inventory_rescan()
                first = scheduler.pending_inventory_rescan_epoch()
                self.assertIsInstance(first, int)
                scheduler.require_inventory_rescan()
                second = scheduler.pending_inventory_rescan_epoch()
                self.assertGreater(second, first)
                scheduler.acknowledge_inventory_rescan(first)
                self.assertEqual(
                    scheduler.pending_inventory_rescan_epoch(),
                    second,
                )
                scheduler.acknowledge_inventory_rescan(second)
                self.assertIsNone(scheduler.pending_inventory_rescan_epoch())

                scheduler.require_inventory_rescan(force_withdrawal=True)
                forced = scheduler.pending_inventory_rescan_epoch()
                self.assertTrue(
                    scheduler.inventory_rescan_requires_withdrawal(forced)
                )
                scheduler.acknowledge_inventory_rescan(forced)
                self.assertFalse(
                    scheduler.inventory_rescan_requires_withdrawal(forced)
                )
