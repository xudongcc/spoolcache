from __future__ import annotations

import hashlib
import multiprocessing
import os
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import spoolcache.maintenance as maintenance_module
import spoolcache.store as store_module
from spoolcache.maintenance import (
    SCRUB_DATABASE_NAME,
    DeepScrubber,
    ScheduledDeepScrubber,
    read_scrub_status,
    request_deep_scrub,
)
from spoolcache.quorum import InventoryReporter, QuorumCatalog
from spoolcache.store import ManifestStore, ObjectSource


DEPLOYMENT = "a" * 64
RANK_IDENTITY = "b" * 64
TOPOLOGY = "c" * 64
LAYOUT = "d" * 64
PROFILE = "vllm-runtime-kv-v1"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def open_store(root: Path) -> ManifestStore:
    return ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=2,
        expected_deployment_digest=DEPLOYMENT,
        expected_rank_digest=RANK_IDENTITY,
        expected_rank=0,
        expected_topology_digest=TOPOLOGY,
        expected_profile=PROFILE,
        expected_layout_digest=LAYOUT,
    )


def commit_entry(
    store: ManifestStore,
    *,
    entry_id: str,
    payloads: tuple[bytes, ...],
    created_at_unix_ns: int = 1,
    deployment_identity_digest: str = DEPLOYMENT,
) -> object:
    return store.commit(
        entry_id=entry_id,
        deployment_identity_digest=deployment_identity_digest,
        rank_identity_digest=RANK_IDENTITY,
        span_tokens=1024,
        physical_rank=0,
        topology_digest=TOPOLOGY,
        profile=PROFILE,
        layout_digest=LAYOUT,
        sources=tuple(
            ObjectSource(
                group_index=0,
                layer_name=f"layer-{index}",
                page_start=index,
                page_count=1,
                chunks=(payload,),
            )
            for index, payload in enumerate(payloads)
        ),
        created_at_unix_ns=created_at_unix_ns,
    )


def finish_cycle(scrubber: DeepScrubber, *, limit: int = 100) -> list[object]:
    reports: list[object] = []
    for _ in range(limit):
        report = scrubber.step(payload_budget_bytes=4096, item_budget=2)
        reports.append(report)
        if report.cycle_completed:
            return reports
    raise AssertionError("deep scrub did not complete within its bounded test steps")


def advance_to_phase(
    scrubber: DeepScrubber,
    phase: str,
    *,
    limit: int = 100,
) -> None:
    for _ in range(limit):
        if scrubber.status().phase == phase:
            return
        scrubber.step(payload_budget_bytes=4096, item_budget=2)
    raise AssertionError(f"deep scrub did not reach {phase!r}")


def scrub_one_object_then_exit(root_text: str) -> None:
    store = open_store(Path(root_text))
    scrubber = DeepScrubber(store)
    scrubber.start_cycle()
    advance_to_phase(scrubber, "manifests")
    report = scrubber.step(payload_budget_bytes=4096, item_budget=2)
    if report.objects_authenticated != 1:
        os._exit(89)
    os._exit(88)


class DeepScrubberTests(unittest.TestCase):
    def test_cycle_start_is_constant_and_snapshot_steps_are_item_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(8):
                    entry = digest(f"incremental-snapshot-{index}")
                    path = root / "manifests" / entry[:2] / f"{entry}.json"
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"{}")

                scrubber = DeepScrubber(store)
                with mock.patch.object(
                    store,
                    "iter_managed_manifest_paths",
                    side_effect=AssertionError("start_cycle scanned manifests"),
                ):
                    scrubber.start_cycle()
                self.assertEqual(scrubber.status().phase, "snapshot_manifests")

                original = store.iter_managed_manifest_paths
                observed = 0

                def counted_paths():
                    nonlocal observed
                    for path in original():
                        observed += 1
                        yield path

                with mock.patch.object(
                    store,
                    "iter_managed_manifest_paths",
                    side_effect=counted_paths,
                ):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=3,
                    )
                self.assertEqual(report.namespace_items_scanned, 3)
                self.assertEqual(observed, 3)
                self.assertEqual(scrubber.status().phase, "snapshot_manifests")

    def test_incremental_snapshot_resumes_after_store_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(5):
                    commit_entry(
                        store,
                        entry_id=digest(f"snapshot-resume-{index}"),
                        payloads=(bytes((index + 1,)) * 1024,),
                    )
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                first = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertEqual(first.namespace_items_scanned, 2)
                self.assertEqual(scrubber.status().phase, "snapshot_manifests")

            with open_store(root) as reopened:
                resumed = DeepScrubber(reopened)
                reports = finish_cycle(resumed, limit=200)
                self.assertTrue(reports[-1].cycle_completed)
                status = resumed.status()
                self.assertEqual(status.last_cycle_manifests_authenticated, 5)
                self.assertEqual(status.last_cycle_objects_authenticated, 5)

    def test_local_snapshot_stream_resets_when_another_scrubber_starts_a_cycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(5):
                    commit_entry(
                        store,
                        entry_id=digest(f"snapshot-external-cycle-{index}"),
                        payloads=(bytes((index + 1,)) * 1024,),
                    )
                first = DeepScrubber(store)
                first.start_cycle()
                first.step(payload_budget_bytes=4096, item_budget=2)

                second = DeepScrubber(store)
                finish_cycle(second, limit=200)
                second.start_cycle()

                first.step(payload_budget_bytes=4096, item_budget=2)
                finish_cycle(first, limit=200)
                self.assertEqual(
                    first.status().last_cycle_manifests_authenticated,
                    5,
                )

    def test_incremental_snapshot_rechecks_live_post_cutoff_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                payload = b"post-cutoff-live-reference" * 200
                object_digest, _, _ = store.put_object((payload,))
                object_path = store.object_path(object_digest)
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                scrubber.step(payload_budget_bytes=4096, item_budget=1)

                entry = digest("post-cutoff-live-entry")
                commit_entry(store, entry_id=entry, payloads=(payload,))
                reports = finish_cycle(scrubber, limit=200)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertTrue(object_path.exists())
                self.assertTrue(store.lookup(entry, verify_payloads=True).is_hit)

    def test_incremental_snapshot_tolerates_delete_after_manifest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("snapshot-delete-interleave")
            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"delete-after-snapshot" * 200,),
                )
                object_path = root / manifest.objects[0].relative_path
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                advance_to_phase(scrubber, "manifests")

                self.assertTrue(store.invalidate(entry))
                reports = finish_cycle(scrubber, limit=200)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertFalse(object_path.exists())
                self.assertEqual(
                    scrubber.status().last_cycle_manifests_authenticated,
                    0,
                )

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

    def test_incremental_snapshot_cancellation_rolls_back_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(8):
                    commit_entry(
                        store,
                        entry_id=digest(f"snapshot-cancel-{index}"),
                        payloads=(bytes((index + 1,)) * 1024,),
                    )
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                probes = 0

                def cancel_after_two_items() -> bool:
                    nonlocal probes
                    probes += 1
                    return probes > 3

                with self.assertRaises(maintenance_module._ScrubCancelled):
                    scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=64,
                        cancel_requested=cancel_after_two_items,
                    )
                self.assertEqual(scrubber.status().phase, "snapshot_manifests")
                reports = finish_cycle(scrubber, limit=200)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertEqual(
                    scrubber.status().last_cycle_manifests_authenticated,
                    8,
                )

    def test_deep_scrub_requires_a_complete_rank_local_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with ManifestStore(
                root,
                slot_bytes=4096,
                slot_count=1,
            ) as store:
                with self.assertRaisesRegex(ValueError, "complete expected identity"):
                    DeepScrubber(store)

    def test_external_request_is_idempotent_bounded_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("external-request")
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=(b"payload",))
                scrubber = DeepScrubber(store)
                first_nonce = request_deep_scrub(root, entry)
                self.assertEqual(request_deep_scrub(root, entry), first_nonce)
                with self.assertRaisesRegex(RuntimeError, "different"):
                    request_deep_scrub(root, digest("other"))
                report = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertTrue(report.request_completed)
                status = read_scrub_status(root)
                self.assertTrue(status["initialized"])
                self.assertEqual(status["last_request_entry"], entry)
                self.assertEqual(status["last_request_status"], "authenticated")

    def test_scrubber_startup_removes_only_exact_request_temporaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                exact = (
                    root
                    / "state"
                    / f".deep-scrub-request.json.{'a' * 32}.tmp"
                )
                unrelated = root / "state" / ".operator-notes.tmp"
                malformed = root / "state" / ".deep-scrub-request.json.short.tmp"
                exact.write_bytes(b"interrupted atomic write")
                unrelated.write_bytes(b"keep")
                malformed.write_bytes(b"keep")
                DeepScrubber(store)
                self.assertFalse(exact.exists())
                self.assertEqual(unrelated.read_bytes(), b"keep")
                self.assertEqual(malformed.read_bytes(), b"keep")

    def test_malformed_external_request_is_quarantined_without_cache_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("malformed-request")
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=(b"payload",))
                scrubber = DeepScrubber(store)
                request_path = root / "state" / "deep-scrub-request.json"
                request_path.write_bytes(b'{"schema":"wrong"}\n')
                with self.assertRaisesRegex(ValueError, "malformed"):
                    scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertFalse(request_path.exists())
                self.assertTrue(
                    tuple((root / "quarantine").glob("control-*.bad"))
                )
                self.assertTrue(store.lookup(entry, verify_payloads=True).is_hit)

    def test_unreadable_request_controls_are_quarantined_without_following_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = base / "outside"
            outside.write_bytes(b"outside remains untouched")
            for mode in ("invalid-json", "oversized", "symlink"):
                with self.subTest(mode=mode):
                    root = base / mode / "rank"
                    entry = digest(f"request-control-{mode}")
                    with open_store(root) as store:
                        commit_entry(store, entry_id=entry, payloads=(b"payload",))
                        scrubber = DeepScrubber(store)
                        request_path = root / "state" / "deep-scrub-request.json"
                        if mode == "invalid-json":
                            request_path.write_bytes(b"{")
                        elif mode == "oversized":
                            request_path.write_bytes(b"x" * 4097)
                        else:
                            request_path.symlink_to(outside)
                        self.assertTrue(scrubber.has_pending_request())
                        with self.assertRaisesRegex(ValueError, "control file"):
                            scrubber.step(
                                payload_budget_bytes=4096,
                                item_budget=2,
                            )
                        self.assertFalse(os.path.lexists(request_path))
                        self.assertTrue(
                            tuple((root / "quarantine").glob("control-*.bad"))
                        )
                        self.assertTrue(
                            store.lookup(entry, verify_payloads=True).is_hit
                        )
            self.assertEqual(outside.read_bytes(), b"outside remains untouched")

    def test_persisted_cursor_cannot_escape_the_owned_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            outside = Path(directory) / "outside"
            outside.write_bytes(b"keep")
            with open_store(root) as store:
                DeepScrubber(store)
                database = root / "state" / SCRUB_DATABASE_NAME
                connection = sqlite3.connect(database)
                try:
                    encoded = connection.execute(
                        "SELECT value FROM meta WHERE key = 'state'"
                    ).fetchone()[0]
                    payload = json.loads(encoded)
                    payload["phase"] = "manifests"
                    payload["cycle"] = 1
                    payload["cycle_started_unix_ns"] = 1
                    payload["active_manifest"] = "../../outside"
                    payload["active_manifest_digest"] = "e" * 64
                    connection.execute(
                        "UPDATE meta SET value = ? WHERE key = 'state'",
                        (json.dumps(payload).encode(),),
                    )
                    connection.commit()
                finally:
                    connection.close()
                recovered = DeepScrubber(store)
                self.assertEqual(recovered.status().phase, "idle")
                self.assertTrue(
                    tuple((root / "quarantine").glob("control-deep-scrub-*.bad"))
                )
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_unknown_database_schema_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                DeepScrubber(store)
                database = root / "state" / SCRUB_DATABASE_NAME
                connection = sqlite3.connect(database)
                try:
                    connection.execute("PRAGMA user_version=2")
                    connection.commit()
                finally:
                    connection.close()
                with self.assertRaisesRegex(ValueError, "schema"):
                    read_scrub_status(root)
                with self.assertRaisesRegex(ValueError, "schema version"):
                    DeepScrubber(store)
                self.assertTrue(database.exists())

    def test_status_rejects_a_dangling_database_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root):
                pass
            database = root / "state" / SCRUB_DATABASE_NAME
            database.symlink_to(root / "state" / "missing-database")
            with self.assertRaisesRegex(ValueError, "regular owned file"):
                read_scrub_status(root)

    def test_scheduler_repairs_current_schema_state_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                scrubber = DeepScrubber(store)
                database = root / "state" / SCRUB_DATABASE_NAME
                connection = sqlite3.connect(database)
                try:
                    connection.execute(
                        "UPDATE meta SET value = ? WHERE key = 'state'",
                        (b"not-json",),
                    )
                    connection.commit()
                finally:
                    connection.close()
                scheduler = ScheduledDeepScrubber(
                    scrubber,
                    poll_seconds=0.01,
                    startup_delay_seconds=3600,
                    cycle_interval_seconds=3600,
                )
                scheduler.start()
                try:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        try:
                            if scrubber.status().phase == "idle":
                                break
                        except ValueError:
                            pass
                        time.sleep(0.01)
                    else:
                        self.fail("scheduled scrub did not repair its state")
                finally:
                    scheduler.close()
                self.assertTrue(
                    tuple((root / "quarantine").glob("control-deep-scrub-*.bad"))
                )
                self.assertGreaterEqual(
                    scheduler.drain_metrics().get("failures", 0),
                    1,
                )

    def test_scheduled_request_is_rate_bounded_and_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("scheduled")
            with open_store(root) as store:
                commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"a" * 4096, b"b" * 4096),
                )
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                scheduler = ScheduledDeepScrubber(
                    scrubber,
                    bytes_per_second=8192,
                    step_bytes=4096,
                    item_budget=4,
                    poll_seconds=0.001,
                    startup_delay_seconds=3600,
                    cycle_interval_seconds=3600,
                )
                started = time.monotonic()
                scheduler.start()
                try:
                    deadline = started + 5
                    while time.monotonic() < deadline:
                        status = scrubber.status()
                        if status.last_request_status == "authenticated":
                            break
                        time.sleep(0.01)
                    else:
                        self.fail("scheduled target scrub did not complete")
                    elapsed = time.monotonic() - started
                    # One fixed-size object is the allowed initial burst; the
                    # second cannot start until the first byte budget elapses.
                    self.assertGreaterEqual(elapsed, 0.4)
                finally:
                    scheduler.close()
                metrics = scheduler.drain_metrics()
                self.assertEqual(metrics["payload_bytes"], 8192)
                self.assertEqual(metrics["objects_authenticated"], 2)
                self.assertEqual(metrics["manifests_authenticated"], 1)
                self.assertIsNone(scheduler.pending_inventory_rescan_epoch())

    def test_scheduler_rejects_nonfinite_or_nonpositive_internal_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                scrubber = DeepScrubber(store)
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

    def test_one_step_cannot_overshoot_the_byte_budget_once_per_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(4):
                    commit_entry(
                        store,
                        entry_id=digest(f"burst-{index}"),
                        payloads=(bytes((index + 1,)) * 4096,),
                    )
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                advance_to_phase(scrubber, "manifests")
                report = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=64,
                )
                self.assertEqual(report.payload_bytes, 4096)
                self.assertEqual(report.objects_authenticated, 1)

    def test_single_object_larger_than_step_is_rate_limited_per_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("large-rate-limited-object")
            with open_store(root) as store:
                commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"x" * (3 * 4096),),
                )
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                scheduler = ScheduledDeepScrubber(
                    scrubber,
                    bytes_per_second=3 * 4096,
                    step_bytes=4096,
                    item_budget=4,
                    poll_seconds=0.001,
                    startup_delay_seconds=3600,
                    cycle_interval_seconds=3600,
                )
                started = time.monotonic()
                scheduler.start()
                try:
                    deadline = started + 5
                    while time.monotonic() < deadline:
                        if scrubber.status().last_request_status == "authenticated":
                            break
                        time.sleep(0.01)
                    else:
                        self.fail("large target scrub did not complete")
                finally:
                    scheduler.close()
                self.assertGreaterEqual(time.monotonic() - started, 0.9)
                self.assertEqual(
                    scheduler.drain_metrics()["payload_bytes"],
                    3 * 4096,
                )

    def test_rate_limited_object_can_stop_and_resume_at_object_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("cancel-and-resume")
            with open_store(root) as store:
                commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"x" * (4 * 4096),),
                )
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                scheduler = ScheduledDeepScrubber(
                    scrubber,
                    bytes_per_second=1,
                    step_bytes=4096,
                    item_budget=1,
                    poll_seconds=0.001,
                    startup_delay_seconds=3600,
                    cycle_interval_seconds=3600,
                )
                scheduler.start()
                time.sleep(0.05)
                started_close = time.monotonic()
                scheduler.close()
                self.assertLess(time.monotonic() - started_close, 1.0)

                # A cancelled object is not recorded as authenticated. The
                # durable target request resumes safely from index zero.
                status = scrubber.status()
                self.assertEqual(status.target_next_object_index, 0)
                report = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=1,
                )
                self.assertTrue(report.request_completed)
                self.assertEqual(report.request_status, "authenticated")

    def test_payload_cursor_resumes_after_process_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("resume")
            payloads = (b"a" * 4096, b"b" * 4096, b"c" * 4096)
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=payloads)
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                advance_to_phase(scrubber, "manifests")
                first = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertEqual(first.objects_authenticated, 1)
                self.assertEqual(first.payload_bytes, 4096)
                status = scrubber.status()
                self.assertEqual(status.phase, "manifests")
                self.assertEqual(status.next_object_index, 1)

            with open_store(root) as reopened:
                resumed = DeepScrubber(reopened)
                self.assertEqual(resumed.status().next_object_index, 1)
                reports = finish_cycle(resumed)
                self.assertTrue(reports[-1].cycle_completed)
                status = resumed.status()
                self.assertEqual(status.phase, "idle")
                self.assertEqual(status.last_cycle_objects_authenticated, 3)
                self.assertEqual(
                    status.last_cycle_payload_bytes,
                    sum(len(payload) for payload in payloads),
                )

    def test_payload_cursor_survives_abrupt_scrubber_process_exit(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("crash-resume")
            with open_store(root) as store:
                commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"a" * 4096, b"b" * 4096),
                )
            process = context.Process(
                target=scrub_one_object_then_exit,
                args=(str(root),),
            )
            process.start()
            process.join(10)
            self.assertEqual(process.exitcode, 88)
            with open_store(root) as store:
                scrubber = DeepScrubber(store)
                self.assertEqual(scrubber.status().next_object_index, 1)
                reports = finish_cycle(scrubber)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertEqual(
                    scrubber.status().last_cycle_objects_authenticated,
                    2,
                )

    def test_target_receipt_retries_a_manifest_replaced_after_payload_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("target-linearization")
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=(b"old" * 1024,))
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                original_stream = store.stream_object
                replaced = False

                def stream_then_replace(*args: object, **kwargs: object) -> None:
                    nonlocal replaced
                    original_stream(*args, **kwargs)  # type: ignore[arg-type]
                    if replaced:
                        return
                    replaced = True
                    self.assertTrue(store.invalidate(entry))
                    commit_entry(
                        store,
                        entry_id=entry,
                        payloads=(b"new" * 1024,),
                        created_at_unix_ns=2,
                    )

                with mock.patch.object(
                    store,
                    "stream_object",
                    side_effect=stream_then_replace,
                ):
                    first = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertFalse(first.request_completed)
                self.assertEqual(scrubber.status().last_request_status, "")

                second = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertTrue(second.request_completed)
                self.assertEqual(second.request_status, "authenticated")

    def test_target_receipt_reports_absent_if_entry_is_deleted_after_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("target-deleted-linearization")
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=(b"old" * 1024,))
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                original_stream = store.stream_object

                def stream_then_delete(*args: object, **kwargs: object) -> None:
                    original_stream(*args, **kwargs)  # type: ignore[arg-type]
                    self.assertTrue(store.invalidate(entry))

                with mock.patch.object(
                    store,
                    "stream_object",
                    side_effect=stream_then_delete,
                ):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertTrue(report.request_completed)
                self.assertEqual(report.request_status, "absent")
                self.assertEqual(report.manifests_authenticated, 0)

    def test_cycle_does_not_count_a_manifest_replaced_after_payload_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("cycle-linearization")
            with open_store(root) as store:
                commit_entry(store, entry_id=entry, payloads=(b"old" * 1024,))
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                advance_to_phase(scrubber, "manifests")
                original_stream = store.stream_object
                replaced = False

                def stream_then_replace(*args: object, **kwargs: object) -> None:
                    nonlocal replaced
                    original_stream(*args, **kwargs)  # type: ignore[arg-type]
                    if replaced:
                        return
                    replaced = True
                    self.assertTrue(store.invalidate(entry))
                    commit_entry(
                        store,
                        entry_id=entry,
                        payloads=(b"new" * 1024,),
                        created_at_unix_ns=2,
                    )

                with mock.patch.object(
                    store,
                    "stream_object",
                    side_effect=stream_then_replace,
                ):
                    first = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertEqual(first.manifests_authenticated, 0)
                self.assertTrue(scrubber.status().active_manifest)
                finish_cycle(scrubber)
                self.assertEqual(
                    scrubber.status().last_cycle_manifests_authenticated,
                    1,
                )

    def test_cycle_namespace_is_snapshotted_once_into_durable_work_queues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                for index in range(5):
                    commit_entry(
                        store,
                        entry_id=digest(f"snapshot-{index}"),
                        payloads=(bytes((index + 1,)) * 2048,),
                    )
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                advance_to_phase(scrubber, "objects")
                with (
                    mock.patch.object(
                        store,
                        "iter_managed_manifest_paths",
                        side_effect=AssertionError("manifest namespace rescanned"),
                    ),
                    mock.patch.object(
                        store,
                        "iter_managed_object_paths",
                        side_effect=AssertionError("object namespace rescanned"),
                    ),
                ):
                    reports = finish_cycle(scrubber, limit=100)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertEqual(
                    scrubber.status().last_cycle_manifests_authenticated,
                    5,
                )

    def test_corrupt_shared_object_quarantines_all_manifests_and_withdraws(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            first, second = digest("first"), digest("second")
            payload = b"shared" * 600
            withdrawn: list[str] = []
            reasons: list[str] = []
            with open_store(root) as store:
                first_manifest = commit_entry(
                    store,
                    entry_id=first,
                    payloads=(payload,),
                )
                commit_entry(
                    store,
                    entry_id=second,
                    payloads=(payload,),
                    created_at_unix_ns=2,
                )
                store.set_withdraw_hook(withdrawn.append)
                store.set_quarantine_hook(reasons.append)
                object_path = root / first_manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    original = handle.read(1)
                    handle.seek(0)
                    handle.write(bytes((original[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

                scrubber = DeepScrubber(store)
                scrubber.request(first)
                for _ in range(20):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                    if report.request_completed:
                        break
                else:
                    self.fail("targeted scrub did not finish")

                self.assertEqual(report.request_status, "quarantined")
                self.assertEqual(report.entries_quarantined, 2)
                self.assertEqual(report.objects_quarantined, 1)
                self.assertEqual(set(withdrawn), {first, second})
                self.assertEqual(reasons.count("payload_checksum"), 2)
                self.assertFalse(store.lookup(first).is_hit)
                self.assertFalse(store.lookup(second).is_hit)
                self.assertFalse(object_path.exists())
                self.assertTrue(
                    tuple((root / "quarantine").glob("object-*.bad"))
                )

    def test_shared_quarantine_continues_after_one_directory_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            first, second = digest("fsync-first"), digest("fsync-second")
            payload = b"shared-fsync" * 400
            withdrawn: list[str] = []
            with open_store(root) as store:
                first_manifest = commit_entry(
                    store,
                    entry_id=first,
                    payloads=(payload,),
                )
                commit_entry(
                    store,
                    entry_id=second,
                    payloads=(payload,),
                    created_at_unix_ns=2,
                )
                store.set_withdraw_hook(withdrawn.append)
                object_path = root / first_manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.write(b"X")
                    handle.flush()
                    os.fsync(handle.fileno())

                real_fsync = store_module._fsync_directory
                failed = False

                def fail_one_manifest_fsync(path: Path) -> None:
                    nonlocal failed
                    if not failed and path.parent == root / "manifests":
                        failed = True
                        raise OSError("injected manifest directory fsync failure")
                    real_fsync(path)

                scrubber = DeepScrubber(store)
                scrubber.request(first)
                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_one_manifest_fsync,
                    ),
                    self.assertRaisesRegex(RuntimeError, "complete receipt"),
                ):
                    scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertTrue(failed)
                self.assertEqual(set(withdrawn), {first, second})
                self.assertFalse(store._manifest_path(first).exists())
                self.assertFalse(store._manifest_path(second).exists())
                self.assertFalse(object_path.exists())

                resumed = scrubber.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertTrue(resumed.request_completed)
                self.assertEqual(resumed.request_status, "quarantined")

    def test_target_quarantine_outcome_survives_pre_acknowledgement_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("quarantine-receipt-recovery")
            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"payload" * 700,),
                )
                path = root / manifest.objects[0].relative_path
                with path.open("r+b") as handle:
                    handle.seek(1)
                    handle.write(b"X")
                    handle.flush()
                    os.fsync(handle.fileno())
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                quarantine = store.quarantine_object

                def fail_after_quarantine(*args: object, **kwargs: object) -> int:
                    quarantine(*args, **kwargs)
                    raise RuntimeError("simulated loss before request acknowledgement")

                with mock.patch.object(
                    store,
                    "quarantine_object",
                    side_effect=fail_after_quarantine,
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated loss"):
                        scrubber.step(
                            payload_budget_bytes=4096,
                            item_budget=2,
                        )

                resumed = DeepScrubber(store)
                report = resumed.step(
                    payload_budget_bytes=4096,
                    item_budget=2,
                )
                self.assertTrue(report.request_completed)
                self.assertEqual(report.request_status, "quarantined")
                self.assertEqual(
                    resumed.status().last_request_status,
                    "quarantined",
                )

    def test_pre_rename_failure_cannot_re_admit_known_bad_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("pre-rename-fail-closed")
            reporter = InventoryReporter(
                rank=0,
                generation="rank-0",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            catalog = QuorumCatalog(
                expected_ranks=(0,),
                max_entries=8,
                max_report_entries=8,
            )
            reporter.replace({entry: 1024})
            startup = reporter.startup(8)
            catalog.apply_startup(
                rank=0,
                generation=reporter.generation,
                generation_epoch=reporter.generation_epoch,
                entries=startup,
            )
            self.assertTrue(catalog.has_quorum(entry, 1024))

            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"payload" * 700,),
                )
                store.set_withdraw_hook(reporter.remove)
                object_path = root / manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(17)
                    original = handle.read(1)
                    handle.seek(17)
                    handle.write(bytes((original[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

                manifest_path = store._manifest_path(entry)
                real_replace = store_module.os.replace
                failed = False

                def fail_live_manifest_rename(source: object, target: object) -> None:
                    nonlocal failed
                    if not failed and Path(source) == manifest_path:
                        failed = True
                        raise OSError("injected manifest rename failure")
                    real_replace(source, target)

                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                with mock.patch.object(
                    store_module.os,
                    "replace",
                    side_effect=fail_live_manifest_rename,
                ):
                    for _ in range(8):
                        try:
                            scrubber.step(
                                payload_budget_bytes=4096,
                                item_budget=2,
                            )
                        except RuntimeError as error:
                            self.assertIn("complete receipt", str(error))
                            break
                    else:
                        self.fail("injected pre-rename failure was not reached")

                self.assertTrue(failed)
                self.assertTrue(manifest_path.exists())
                self.assertEqual(store.lookup(entry).reason, "withdrawn")
                catalog.apply_report(reporter.next_report(8))
                self.assertFalse(catalog.has_quorum(entry))

                # Model the forced empty report followed by repeated metadata
                # rescans. The still-live, self-consistent manifest must remain
                # excluded while its payload has not been reauthenticated.
                reporter.replace({})
                catalog.apply_report(reporter.next_report(8))
                for _ in range(2):
                    reporter.replace(
                        {
                            offer.entry_id: offer.span_tokens
                            for offer in store.scan_offers(8)
                        }
                    )
                    catalog.apply_report(reporter.next_report(8))
                    self.assertFalse(catalog.has_quorum(entry))

                # Removing the fault lets the resumable target scrub finish the
                # quarantine. A standalone scrubber leaves the durable signal
                # for the live inventory owner to consume.
                for _ in range(8):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                    if report.request_completed:
                        break
                else:
                    self.fail("target scrub did not retry quarantine")
                self.assertEqual(report.request_status, "quarantined")
                self.assertFalse(manifest_path.exists())
                self.assertTrue(store._inventory_withdrawal_path(entry).exists())
                pending = store.pending_inventory_withdrawals((entry,))
                reporter.remove(entry)
                store.acknowledge_absent_inventory_withdrawals(pending)
                self.assertFalse(store._inventory_withdrawal_path(entry).exists())
                self.assertFalse(catalog.has_quorum(entry))

    def test_full_authentication_releases_a_durably_repaired_object_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            old_entry = digest("repaired-live-manifest")
            new_entry = digest("repair-trigger")
            payload = b"authenticated-repair" * 400
            fired = False

            def fail_after_atomic_replace(stage: str) -> None:
                nonlocal fired
                if not fired and stage == "object-collision-after-replace":
                    fired = True
                    raise OSError("injected object-directory receipt failure")

            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=old_entry,
                    payloads=(payload,),
                )
                object_path = root / manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(13)
                    byte = handle.read(1)
                    handle.seek(13)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())
                store._fault_hook = fail_after_atomic_replace
                with self.assertRaisesRegex(
                    RuntimeError,
                    "collision repair did not get a complete receipt",
                ):
                    commit_entry(
                        store,
                        entry_id=new_entry,
                        payloads=(payload,),
                    )
                self.assertTrue(fired)
                self.assertEqual(store.lookup(old_entry).reason, "withdrawn")
                self.assertEqual(store.read_object_bytes(manifest.objects[0]), payload)
                self.assertTrue(
                    store._object_withdrawal_path(
                        manifest.objects[0].sha256
                    ).exists()
                )

                store._fault_hook = None
                scrubber = DeepScrubber(store)
                scrubber.request(old_entry)
                real_fsync_directory = store_module._fsync_directory
                failed_receipt = False

                def fail_repair_directory_once(path: Path) -> None:
                    nonlocal failed_receipt
                    if not failed_receipt and path == object_path.parent:
                        failed_receipt = True
                        raise OSError("injected repaired-object fsync failure")
                    real_fsync_directory(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        side_effect=fail_repair_directory_once,
                    ),
                    self.assertRaisesRegex(OSError, "repaired-object fsync"),
                ):
                    scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                self.assertTrue(failed_receipt)
                self.assertTrue(store._inventory_withdrawal_path(old_entry).exists())

                for _ in range(8):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                    if report.request_completed:
                        break
                else:
                    self.fail("target authentication did not complete")
                self.assertEqual(report.request_status, "authenticated")
                self.assertEqual(report.inventory_released, 1)
                self.assertFalse(store._inventory_withdrawal_path(old_entry).exists())
                self.assertFalse(
                    store._object_withdrawal_path(
                        manifest.objects[0].sha256
                    ).exists()
                )
                self.assertTrue(store.lookup(old_entry, verify_payloads=True).is_hit)
                scheduler = ScheduledDeepScrubber(scrubber)
                self.assertIsNone(scheduler.pending_inventory_rescan_epoch())
                scheduler._accumulate(report)
                self.assertIsNotNone(scheduler.pending_inventory_rescan_epoch())

    def test_final_authentication_cannot_release_a_new_corrupt_object_fence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("late-object-fence")
            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"late-fence-payload" * 400,),
                )
                descriptor = manifest.objects[0]
                store._mark_inventory_withdrawn(entry)
                store._mark_object_withdrawn(descriptor.sha256)
                object_path = root / descriptor.relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(23)
                    byte = handle.read(1)
                    handle.seek(23)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

                with self.assertRaisesRegex(
                    store_module.ObjectCorruptionError,
                    "final authentication",
                ):
                    store.release_authenticated_inventory(manifest)
                self.assertTrue(store._inventory_withdrawal_path(entry).exists())
                self.assertTrue(
                    store._object_withdrawal_path(descriptor.sha256).exists()
                )
                self.assertEqual(store.lookup(entry).reason, "withdrawn")

    def test_truncated_or_missing_payload_is_quarantined_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for mode in ("truncated", "missing"):
                with self.subTest(mode=mode):
                    root = base / mode
                    entry = digest(f"incomplete-{mode}")
                    withdrawn: list[str] = []
                    reasons: list[str] = []
                    with open_store(root) as store:
                        manifest = commit_entry(
                            store,
                            entry_id=entry,
                            payloads=(b"payload" * 700,),
                        )
                        store.set_withdraw_hook(withdrawn.append)
                        store.set_quarantine_hook(reasons.append)
                        path = root / manifest.objects[0].relative_path
                        if mode == "truncated":
                            with path.open("r+b") as handle:
                                handle.truncate(17)
                                handle.flush()
                                os.fsync(handle.fileno())
                        else:
                            path.unlink()
                        scrubber = DeepScrubber(store)
                        scrubber.request(entry)
                        report = scrubber.step(
                            payload_budget_bytes=4096,
                            item_budget=2,
                        )
                        self.assertTrue(report.request_completed)
                        self.assertEqual(report.request_status, "quarantined")
                        self.assertEqual(withdrawn, [entry])
                        self.assertEqual(reasons, ["payload_size"])
                        self.assertFalse(store.lookup(entry).is_hit)

    def test_scrub_all_identity_failures_precede_any_payload_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = {
                "expected_deployment_digest": DEPLOYMENT,
                "expected_rank_digest": RANK_IDENTITY,
                "expected_rank": 0,
                "expected_topology_digest": TOPOLOGY,
                "expected_profile": PROFILE,
                "expected_layout_digest": LAYOUT,
            }
            mismatches = {
                "expected_deployment_digest": "e" * 64,
                "expected_rank_digest": "e" * 64,
                "expected_rank": 1,
                "expected_topology_digest": "e" * 64,
                "expected_profile": "different-profile",
                "expected_layout_digest": "e" * 64,
            }
            for index, (field, wrong_value) in enumerate(mismatches.items()):
                with self.subTest(field=field):
                    root = Path(directory) / f"rank-{index}"
                    entry = digest(f"wrong-identity-{field}")
                    with ManifestStore(
                        root,
                        slot_bytes=4096,
                        slot_count=1,
                    ) as writer:
                        commit_entry(writer, entry_id=entry, payloads=(b"payload",))

                    actual = dict(expected)
                    actual[field] = wrong_value
                    with ManifestStore(
                        root,
                        slot_bytes=4096,
                        slot_count=1,
                        **actual,
                    ) as store:
                        scrubber = DeepScrubber(store)
                        scrubber.request(entry)
                        with mock.patch.object(
                            store,
                            "stream_object",
                            side_effect=AssertionError("payload was opened"),
                        ):
                            report = scrubber.step(
                                payload_budget_bytes=4096,
                                item_budget=2,
                            )
                        self.assertTrue(report.request_completed)
                        self.assertEqual(report.request_status, "quarantined")
                        self.assertFalse(store.lookup(entry).is_hit)

    def test_orphans_and_temps_require_a_complete_age_safe_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                old_digest, _, _ = store.put_object((b"old-orphan",))
                old_path = store.object_path(old_digest)
                (root / "tmp" / "object-abandoned.part").write_bytes(b"partial")
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                new_digest, _, _ = store.put_object((b"new-orphan",))
                new_path = store.object_path(new_digest)

                reports = finish_cycle(scrubber)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertFalse(old_path.exists())
                self.assertTrue(new_path.exists())
                self.assertFalse((root / "tmp" / "object-abandoned.part").exists())
                # A live manifest remains authoritative even if an old-cycle
                # reference index misses it; the final locked recheck protects
                # its immutable object.
                live_entry = digest("live-entry")
                live = commit_entry(
                    store,
                    entry_id=live_entry,
                    payloads=(b"live" * 500,),
                )
                live_path = root / live.objects[0].relative_path

                scrubber.start_cycle()
                finish_cycle(scrubber)
                self.assertFalse(new_path.exists())
                self.assertTrue(live_path.exists())
                self.assertTrue(store.lookup(live_entry, verify_payloads=True).is_hit)

    def test_unknown_managed_paths_are_quarantined_without_following_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            outside = Path(directory) / "outside"
            outside.write_bytes(b"keep")
            with open_store(root) as store:
                strange_manifest = root / "manifests" / "not-a-shard"
                strange_manifest.write_bytes(b"not a manifest")
                strange_object = root / "objects" / "ff" / "unexpected.bin"
                strange_object.parent.mkdir()
                strange_object.symlink_to(outside)
                strange_tmp = root / "tmp" / "unexpected"
                strange_tmp.mkdir()
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                reports = finish_cycle(scrubber, limit=500)
                self.assertTrue(reports[-1].cycle_completed)
                self.assertFalse(strange_manifest.exists())
                self.assertFalse(strange_object.exists())
                self.assertFalse(strange_tmp.exists())
                self.assertGreaterEqual(
                    len(tuple((root / "quarantine").iterdir())),
                    3,
                )
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_surrogateescape_names_are_quarantined_without_sqlite_text(self) -> None:
        for namespace in ("manifests", "objects", "tmp"):
            with self.subTest(namespace=namespace):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "rank"
                    with open_store(root) as store:
                        raw_path = os.fsencode(root / namespace) + b"/\xff"
                        descriptor = os.open(
                            raw_path,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                        )
                        try:
                            os.write(descriptor, b"unexpected")
                        finally:
                            os.close(descriptor)
                        scrubber = DeepScrubber(store)
                        scrubber.start_cycle()
                        reports = finish_cycle(scrubber)
                        self.assertTrue(reports[-1].cycle_completed)
                        self.assertFalse(os.path.lexists(raw_path))
                        self.assertTrue(tuple((root / "quarantine").iterdir()))

    def test_inventory_rescan_epoch_cannot_lose_a_concurrent_withdrawal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root) as store:
                scheduler = ScheduledDeepScrubber(DeepScrubber(store))
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

    def test_scrub_withdrawal_removes_quorum_before_restore_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            entry = digest("quorum-withdraw")
            reporter0 = InventoryReporter(
                rank=0,
                generation="rank-0",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            reporter1 = InventoryReporter(
                rank=1,
                generation="rank-1",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            catalog = QuorumCatalog(
                expected_ranks=(0, 1),
                max_entries=8,
                max_report_entries=8,
            )
            for reporter in (reporter0, reporter1):
                reporter.replace({entry: 1024})
                startup = reporter.startup(8)
                catalog.apply_startup(
                    rank=reporter.rank,
                    generation=reporter.generation,
                    generation_epoch=reporter.generation_epoch,
                    entries=startup,
                )
            self.assertTrue(catalog.has_quorum(entry, 1024))

            with open_store(root) as store:
                manifest = commit_entry(
                    store,
                    entry_id=entry,
                    payloads=(b"payload" * 600,),
                )
                store.set_withdraw_hook(reporter0.remove)
                path = root / manifest.objects[0].relative_path
                with path.open("r+b") as handle:
                    handle.seek(3)
                    handle.write(b"X")
                    handle.flush()
                    os.fsync(handle.fileno())
                scrubber = DeepScrubber(store)
                scrubber.request(entry)
                while True:
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=2,
                    )
                    if report.request_completed:
                        break

            catalog.apply_report(reporter0.next_report(8))
            self.assertFalse(catalog.has_quorum(entry))

    def test_repeated_capacity_gc_scrub_and_rank_reopen_remain_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            max_bytes = 32 * 1024
            low_bytes = 24 * 1024
            withdrawn: set[str] = set()
            store = open_store(root)
            try:
                store.set_withdraw_hook(withdrawn.add)
                for cycle in range(40):
                    entry = digest(f"soak-entry-{cycle}")
                    commit_entry(
                        store,
                        entry_id=entry,
                        payloads=(bytes((cycle,)) * 3072,),
                        created_at_unix_ns=cycle + 1,
                    )
                    manifest_path = (
                        root / "manifests" / entry[:2] / f"{entry}.json"
                    )
                    os.utime(manifest_path, ns=(cycle + 1, cycle + 1))
                    if cycle % 4 == 0:
                        store.put_object((f"orphan-{cycle}".encode() * 300,))
                    if store.disk_usage_bytes() > max_bytes:
                        store.maintain_capacity(
                            max_bytes=max_bytes,
                            low_watermark_bytes=low_bytes,
                        )
                    if cycle % 5 == 4:
                        scrubber = DeepScrubber(store)
                        scrubber.start_cycle()
                        finish_cycle(scrubber, limit=500)
                    if cycle % 7 == 6:
                        store.close()
                        store = open_store(root)
                        store.set_withdraw_hook(withdrawn.add)

                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                finish_cycle(scrubber, limit=500)
                if store.disk_usage_bytes() > max_bytes:
                    store.maintain_capacity(
                        max_bytes=max_bytes,
                        low_watermark_bytes=low_bytes,
                    )
                self.assertLessEqual(store.disk_usage_bytes(), max_bytes)
                self.assertFalse(tuple((root / "tmp").iterdir()))
                self.assertLessEqual(len(tuple(store.iter_manifest_paths())), 8)
                self.assertGreater(len(withdrawn), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
