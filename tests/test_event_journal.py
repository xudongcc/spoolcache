from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from spoolcache.event_journal import PersistentEventJournal
from spoolcache.telemetry import MAX_METRIC_VALUE


class PersistentEventJournalTests(unittest.TestCase):
    def test_startup_removes_only_exact_component_temporaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            exact = state / f".spoolcache-events.json.{'a' * 32}.tmp"
            unrelated = state / ".operator-state.tmp"
            malformed = state / ".spoolcache-events.json.short.tmp"
            exact.write_bytes(b"interrupted")
            unrelated.write_bytes(b"keep")
            malformed.write_bytes(b"keep")
            PersistentEventJournal(state)
            self.assertFalse(exact.exists())
            self.assertEqual(unrelated.read_bytes(), b"keep")
            self.assertEqual(malformed.read_bytes(), b"keep")

            unsafe = state / f".spoolcache-events.json.{'b' * 32}.tmp"
            unsafe.symlink_to(Path(directory) / "outside")
            with self.assertRaisesRegex(ValueError, "temporary.*regular"):
                PersistentEventJournal(state)

    def test_bounded_counter_totals_survive_process_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            journal = PersistentEventJournal(state)
            journal.increment(
                "spoolcache_post_admission_failure_total", ("payload",)
            )
            journal.increment(
                "spoolcache_quarantined_entries_total",
                ("manifest_validation",),
                amount=2,
            )
            journal.increment("spoolcache_scrub_shutdown_failures_total")
            reopened = PersistentEventJournal(state)
            self.assertEqual(
                reopened.totals()[
                    ("spoolcache_post_admission_failure_total", ("payload",))
                ],
                1,
            )
            self.assertEqual(
                reopened.totals()[
                    (
                        "spoolcache_quarantined_entries_total",
                        ("manifest_validation",),
                    )
                ],
                2,
            )
            self.assertEqual(
                reopened.totals()[
                    ("spoolcache_scrub_shutdown_failures_total", ())
                ],
                1,
            )
            encoded = (state / "spoolcache-events.json").read_text("utf-8")
            self.assertNotIn("entry_id", encoded)
            self.assertNotIn("prompt", encoded.lower())
            self.assertEqual(json.loads(encoded)["schema"], "spoolcache-events/v1")

    def test_only_crash_relevant_bounded_counters_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            journal = PersistentEventJournal(state)
            invalid = (
                ("spoolcache_hit_tokens_total", ()),
                ("spoolcache_post_admission_failure_total", ("rank-7",)),
                ("spoolcache_quarantined_entries_total", ("a" * 64,)),
            )
            for name, labels in invalid:
                with self.subTest(name=name, labels=labels):
                    with self.assertRaises(ValueError):
                        journal.increment(name, labels)

    def test_malformed_or_symlinked_journal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            path = state / "spoolcache-events.json"
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event journal"):
                PersistentEventJournal(state)
            path.unlink()
            target = Path(directory) / "outside"
            target.write_text("{}", encoding="utf-8")
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                PersistentEventJournal(state)
            path.unlink()
            os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, "regular file"):
                PersistentEventJournal(state)

    def test_counter_bound_survives_reopen_and_rejects_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            journal = PersistentEventJournal(state)
            journal.increment(
                "spoolcache_post_admission_failure_total",
                ("payload",),
                amount=MAX_METRIC_VALUE,
            )
            reopened = PersistentEventJournal(state)
            with self.assertRaisesRegex(ValueError, "fixed bound"):
                reopened.increment(
                    "spoolcache_post_admission_failure_total", ("payload",)
                )
            self.assertEqual(
                reopened.totals()[
                    ("spoolcache_post_admission_failure_total", ("payload",))
                ],
                MAX_METRIC_VALUE,
            )


if __name__ == "__main__":
    unittest.main()
