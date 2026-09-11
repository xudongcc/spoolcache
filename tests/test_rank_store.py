"""Shared rank ownership and durability contracts on the token-file backend."""
import errno
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import spoolcache.rank_store as store_module
import spoolcache.token_files as token_module
from spoolcache.config import INVENTORY_ENTRY_BYTES, INVENTORY_MEMORY_BYTES, inventory_capacity
from spoolcache.quorum import InventoryReporter, QuorumCatalog
from tests.token_fixtures import open_store

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

class RankStoreTests(unittest.TestCase):
    def _store(self, root, **kwargs):
        return open_store(root, **kwargs)

    def test_inventory_generation_epoch_survives_clock_rollback_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                with mock.patch.object(store_module.time, "time_ns", return_value=500):
                    self.assertEqual(
                        store.reserve_inventory_generation_epoch(),
                        store_module._PERSISTENT_GENERATION_EPOCH_BASE + 500,
                    )

            state = root / "state" / store_module.GENERATION_STATE_NAME
            abandoned = state.parent / f".{state.name}.{'a' * 32}.tmp"
            unrelated = state.parent / f".{state.name}.not-a-nonce.tmp"
            abandoned.write_bytes(b"abandoned")
            unrelated.write_bytes(b"operator-owned")
            with self._store(root) as reopened:
                with mock.patch.object(store_module.time, "time_ns", return_value=100):
                    self.assertEqual(
                        reopened.reserve_inventory_generation_epoch(),
                        store_module._PERSISTENT_GENERATION_EPOCH_BASE + 501,
                    )
            self.assertFalse(abandoned.exists())
            self.assertEqual(unrelated.read_bytes(), b"operator-owned")
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8")),
                {
                    "schema": store_module.GENERATION_STATE_SCHEMA,
                    "epoch": store_module._PERSISTENT_GENERATION_EPOCH_BASE + 501,
                },
            )
            self.assertEqual(
                json.loads(
                    (root / "state" / store_module.GENERATION_REQUIRED_NAME).read_text(
                        encoding="utf-8"
                    )
                ),
                {"schema": store_module.GENERATION_REQUIRED_SCHEMA},
            )


    def test_inventory_generation_state_is_strict_and_crash_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                with mock.patch.object(store_module.time, "time_ns", return_value=100):
                    self.assertEqual(
                        store.reserve_inventory_generation_epoch(),
                        store_module._PERSISTENT_GENERATION_EPOCH_BASE + 100,
                    )
                state = root / "state" / store_module.GENERATION_STATE_NAME
                real_replace = store_module.os.replace

                def fail_generation_replace(source: object, target: object) -> None:
                    if Path(target) == state:
                        raise OSError(errno.EIO, "injected generation replace failure")
                    real_replace(source, target)

                with (
                    mock.patch.object(
                        store_module.os,
                        "replace",
                        side_effect=fail_generation_replace,
                    ),
                    mock.patch.object(store_module.time, "time_ns", return_value=1),
                    self.assertRaisesRegex(OSError, "generation replace"),
                ):
                    store.reserve_inventory_generation_epoch()
                self.assertEqual(
                    json.loads(state.read_text())["epoch"],
                    store_module._PERSISTENT_GENERATION_EPOCH_BASE + 100,
                )
                self.assertFalse(
                    tuple(state.parent.glob(f".{state.name}.????????????????????????????????.tmp"))
                )

                real_fsync = store_module._fsync_directory

                def fail_state_directory(path: Path) -> None:
                    if path == state.parent:
                        raise OSError(errno.EIO, "injected generation fsync failure")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        side_effect=fail_state_directory,
                    ),
                    mock.patch.object(store_module.time, "time_ns", return_value=1),
                    self.assertRaisesRegex(OSError, "generation fsync"),
                ):
                    store.reserve_inventory_generation_epoch()
                # Rename was the visibility point. A failed receipt may skip an
                # epoch, but reopening must never reuse it.
                self.assertEqual(
                    json.loads(state.read_text())["epoch"],
                    store_module._PERSISTENT_GENERATION_EPOCH_BASE + 101,
                )

            with self._store(root) as reopened:
                with mock.patch.object(store_module.time, "time_ns", return_value=1):
                    self.assertEqual(
                        reopened.reserve_inventory_generation_epoch(),
                        store_module._PERSISTENT_GENERATION_EPOCH_BASE + 102,
                    )


    def test_generation_migrates_legacy_state_and_missing_initialized_state_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                state = root / "state" / store_module.GENERATION_STATE_NAME
                state.write_text(
                    json.dumps(
                        {
                            "schema": store_module.GENERATION_STATE_SCHEMA,
                            "epoch": 1000,
                        }
                    ),
                    encoding="utf-8",
                )
                with mock.patch.object(store_module.time, "time_ns", return_value=500):
                    migrated = store.reserve_inventory_generation_epoch()
                self.assertEqual(
                    migrated,
                    store_module._PERSISTENT_GENERATION_EPOCH_BASE + 1000,
                )
                state.unlink()
                with self.assertRaisesRegex(ValueError, "missing after initialization"):
                    store.reserve_inventory_generation_epoch()


    def test_first_upgraded_generation_is_above_legacy_wall_clock_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                # A pre-upgrade root has no generation state or sentry. Even a
                # rolled-back wall clock must enter the disjoint high domain.
                with mock.patch.object(store_module.time, "time_ns", return_value=500):
                    epoch = store.reserve_inventory_generation_epoch()
            self.assertGreater(epoch, 1000)
            self.assertGreaterEqual(
                epoch,
                store_module._PERSISTENT_GENERATION_EPOCH_BASE,
            )


    def test_rolling_upgrade_epoch_replaces_legacy_scheduler_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            stale_entry = digest("legacy-scheduler-offer")
            scheduler = QuorumCatalog(
                expected_ranks=(0,),
                max_bytes=(8) * INVENTORY_ENTRY_BYTES,
                max_report_entries=8,
            )
            scheduler.apply_startup(
                rank=0,
                generation="legacy-worker",
                generation_epoch=1000,
                entries=((stale_entry, 256),),
            )
            self.assertTrue(scheduler.has_quorum(stale_entry))

            with self._store(root) as store:
                with mock.patch.object(store_module.time, "time_ns", return_value=500):
                    upgraded_epoch = store.reserve_inventory_generation_epoch()
            scheduler.apply_startup(
                rank=0,
                generation="upgraded-worker",
                generation_epoch=upgraded_epoch,
                entries=(),
            )
            self.assertTrue(scheduler.is_ready)
            self.assertFalse(scheduler.has_quorum(stale_entry))


    def test_inventory_owner_lease_rejects_overlapping_worker_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            first = self._store(root)
            second = self._store(root)
            try:
                first.acquire_inventory_owner()
                with self.assertRaisesRegex(RuntimeError, "already has a live owner"):
                    second.acquire_inventory_owner()
                first.close()
                second.acquire_inventory_owner()
            finally:
                first.close()
                second.close()


    def test_inventory_generation_state_corruption_fails_closed(self) -> None:
        invalid_states = (
            b"not-json",
            b'{"schema":"spoolcache-generation/v2","epoch":1}',
            b'{"schema":"spoolcache-generation/v1","epoch":true}',
            b'{"schema":"spoolcache-generation/v1","epoch":0}',
            b'{"schema":"spoolcache-generation/v1","epoch":1,"extra":0}',
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, encoded in enumerate(invalid_states):
                with self.subTest(index=index):
                    root = base / f"invalid-{index}"
                    with self._store(root) as store:
                        state = root / "state" / store_module.GENERATION_STATE_NAME
                        state.write_bytes(encoded)
                        with self.assertRaises(ValueError):
                            store.reserve_inventory_generation_epoch()

            root = base / "overflow"
            with self._store(root) as store:
                state = root / "state" / store_module.GENERATION_STATE_NAME
                state.write_text(
                    json.dumps(
                        {
                            "schema": store_module.GENERATION_STATE_SCHEMA,
                            "epoch": (1 << 63) - 1,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "overflowed"):
                    store.reserve_inventory_generation_epoch()

            root = base / "symlink"
            with self._store(root) as store:
                state = root / "state" / store_module.GENERATION_STATE_NAME
                state.symlink_to(root / ".spoolcache-root")
                with self.assertRaisesRegex(ValueError, "regular file"):
                    store.reserve_inventory_generation_epoch()


    def test_inventory_withdrawal_namespace_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cases = ("bad-name", "file", "symlink", "nonempty")
            for index, case in enumerate(cases):
                with self.subTest(case=case):
                    root = base / f"case-{index}"
                    with self._store(root):
                        pass
                    namespace = root / "state" / "inventory-withdrawn"
                    name = "not-a-digest" if case == "bad-name" else digest(case)
                    path = namespace / name
                    if case == "file" or case == "bad-name":
                        path.write_bytes(b"unexpected")
                    elif case == "symlink":
                        path.symlink_to(root / ".spoolcache-root")
                    else:
                        path.mkdir()
                        (path / "child").write_bytes(b"unexpected")
                    with self.assertRaisesRegex(
                        ValueError,
                        "inventory-withdrawal",
                    ):
                        self._store(root)


    def test_absent_withdrawal_markers_are_consumed_in_bounded_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entries = tuple(digest(f"offline-marker-{index}") for index in range(7))
            with self._store(root) as maintenance:
                for index, entry in enumerate(entries):
                    maintenance.commit_chunk(entry, None, 256,
                                             bytes((index + 1,)) * 16384)
                    self.assertTrue(maintenance.evict(entry))

            with self._store(root) as owner:
                owner.acquire_inventory_owner()
                self.assertEqual(owner.scan_offers(32), ())
                cursor: str | None = None
                pages = 0
                while tuple((root / "state" / "inventory-withdrawn").iterdir()):
                    batch, cursor = owner.inventory_withdrawal_marker_batch(
                        2,
                        after=cursor,
                    )
                    self.assertLessEqual(len(batch), 2)
                    self.assertTrue(batch)
                    owner.acknowledge_absent_inventory_withdrawals(batch)
                    pages += 1
                    self.assertLess(pages, 10)
                self.assertGreaterEqual(pages, 4)

    def test_marker_page_withdraws_all_held_keys_before_bounded_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            with self._store(Path(directory) / "cache") as store:
                marked = tuple(digest(f"withdraw-{i}") for i in range(7))
                healthy = tuple(digest(f"healthy-{i}") for i in range(5))
                worker = InventoryReporter(
                    rank=0, generation="owner", generation_epoch=1,
                    max_bytes=32 * INVENTORY_ENTRY_BYTES, max_report_entries=2,
                )
                worker.replace(dict.fromkeys(marked + healthy, 256))
                scheduler = QuorumCatalog(
                    expected_ranks=(0,), max_bytes=32 * INVENTORY_ENTRY_BYTES,
                    max_report_entries=2,
                )
                scheduler.apply_startup(rank=0, generation="owner",
                    generation_epoch=1, entries=worker.startup())
                self.assertEqual(scheduler.quorum_count, 12)
                for key in marked:
                    store._mark_inventory_withdrawn(key)
                # All markers are before this cursor. Wrapping must still
                # withdraw them all, not just the two selected for cleanup.
                with store._exclusive():
                    batch, _ = store.inventory_withdrawal_marker_batch(
                        2, after="f" * 64, withdraw=worker.remove,
                    )
                    self.assertEqual(set(worker.held_entry_ids()), set(healthy))
                    report = worker.next_report(2)
                    store.acknowledge_absent_inventory_withdrawals(batch)
                self.assertEqual(len(batch), 2)
                self.assertEqual(sum(store._inventory_is_withdrawn(k)
                                     for k in marked), 5)
                scheduler.apply_report(report)
                self.assertFalse(scheduler.is_ready)
                self.assertEqual(scheduler.quorum_count, 0)
                for _ in range(2):
                    scheduler.apply_report(worker.next_report(2))
                self.assertTrue(scheduler.is_ready)
                self.assertEqual(scheduler.quorum_count, len(healthy))
                self.assertTrue(all(not scheduler.has_quorum(k) for k in marked))


    def test_absent_withdrawal_ack_persists_manifest_absence_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("absent-marker-directory-receipt")
            with self._store(root) as store:
                store.commit_chunk(entry, None, 256, b"x" * 16384)
                manifest_path = store._manifest_path(entry)
                manifest_shard = manifest_path.parent
                real_fsync = store_module._fsync_directory

                def fail_manifest_shard(path: Path) -> None:
                    if path == manifest_shard:
                        raise OSError(errno.EIO, "injected manifest shard fsync")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_manifest_shard,
                    ),
                    mock.patch.object(token_module, '_fsync_directory', new=fail_manifest_shard),
                    self.assertRaisesRegex(OSError, "manifest shard fsync"),
                ):
                    store.evict(entry)
                marker = store._inventory_withdrawal_path(entry)
                self.assertFalse(manifest_path.exists())
                self.assertTrue(marker.exists())

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_manifest_shard,
                    ),
                    self.assertRaisesRegex(OSError, "manifest shard fsync"),
                ):
                    store.acknowledge_absent_inventory_withdrawals((entry,))
                self.assertTrue(marker.exists())

                fsynced: list[Path] = []

                def record_fsync(path: Path) -> None:
                    fsynced.append(path)
                    real_fsync(path)

                with mock.patch.object(
                    store_module,
                    "_fsync_directory",
                    new=record_fsync,
                ):
                    store.acknowledge_absent_inventory_withdrawals((entry,))
                self.assertFalse(marker.exists())
                self.assertLess(
                    fsynced.index(manifest_shard),
                    fsynced.index(marker.parent),
                )


    def test_existing_withdrawal_marker_retries_parent_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                object_digest = digest("retry-object-fence-receipt")
                marker = store._inventory_withdrawal_path(object_digest)
                real_fsync = store_module._fsync_directory

                def fail_marker_parent(path: Path) -> None:
                    if path == marker.parent:
                        raise OSError(errno.EIO, "injected marker fsync")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_marker_parent,
                    ),
                    self.assertRaisesRegex(OSError, "marker fsync"),
                ):
                    store._mark_inventory_withdrawn(object_digest)
                self.assertTrue(marker.exists())

                fsynced: list[Path] = []

                def record_fsync(path: Path) -> None:
                    fsynced.append(path)
                    real_fsync(path)

                with mock.patch.object(
                    store_module,
                    "_fsync_directory",
                    new=record_fsync,
                ):
                    store._mark_inventory_withdrawn(object_digest)
                self.assertIn(marker.parent, fsynced)


    def test_existing_withdrawal_namespace_retries_ancestor_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            state = root / "state"
            namespace = state / "inventory-withdrawn"
            real_fsync = store_module._fsync_directory
            failed = False

            def fail_namespace_parent_once(path: Path) -> None:
                nonlocal failed
                if not failed and path == state and namespace.exists():
                    failed = True
                    raise OSError(errno.EIO, "injected namespace parent fsync")
                real_fsync(path)

            with (
                mock.patch.object(
                    store_module,
                    "_fsync_directory",
                    new=fail_namespace_parent_once,
                ),
                self.assertRaisesRegex(OSError, "namespace parent fsync"),
            ):
                self._store(root)
            self.assertTrue(failed)
            self.assertTrue(namespace.is_dir())

            fsynced: list[Path] = []

            def record_fsync(path: Path) -> None:
                fsynced.append(path)
                real_fsync(path)

            with mock.patch.object(
                store_module,
                "_fsync_directory",
                new=record_fsync,
            ):
                with self._store(root) as reopened:
                    initialization_receipts = tuple(fsynced)
                    fsynced.clear()
                    reopened._mark_inventory_withdrawn(
                        digest("namespace-parent-retry-marker")
                    )
            self.assertIn(state, initialization_receipts)
            self.assertIn(namespace, fsynced)


    def test_nonempty_unowned_root_and_symlink_marker_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            unowned = base / "unowned"
            unowned.mkdir()
            (unowned / "user-data").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unowned"):
                self._store(unowned)

            owned = base / "owned"
            with self._store(owned):
                pass
            marker = owned / ".spoolcache-root"
            marker.unlink()
            target = base / "outside-marker"
            target.write_text('{"schema":"spoolcache-root/v1"}\n', encoding="utf-8")
            marker.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "marker.*symlink"):
                self._store(owned)
