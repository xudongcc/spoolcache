from __future__ import annotations

import hashlib
import errno
import json
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from spoolcache.manifest import ObjectDescriptor, RankManifest, decode_manifest, encode_manifest
from spoolcache.quorum import InventoryReporter, QuorumCatalog
import spoolcache.store as store_module
from spoolcache.store import ManifestStore, ObjectSource


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def commit_one(
    store: ManifestStore,
    *,
    entry: str,
    payload: bytes,
    created: int = 1,
) -> RankManifest:
    return store.commit(
        entry_id=entry,
        deployment_identity_digest="a" * 64,
        rank_identity_digest="b" * 64,
        span_tokens=256,
        physical_rank=0,
        topology_digest="c" * 64,
        profile="vllm-runtime-kv-v1",
        layout_digest="d" * 64,
        sources=(
            ObjectSource(
                group_index=0,
                layer_name="layer",
                page_start=0,
                page_count=1,
                chunks=(payload[:3000], payload[3000:]),
            ),
        ),
        created_at_unix_ns=created,
    )


def crash_commit_at(root_text: str, stage: str) -> None:
    root = Path(root_text)

    def crash(observed: str) -> None:
        if observed == stage:
            os._exit(86)

    with ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=1,
        expected_deployment_digest="a" * 64,
        expected_rank_digest="b" * 64,
        expected_rank=0,
        fault_hook=crash,
    ) as store:
        commit_one(
            store,
            entry=digest(f"crash-{stage}"),
            payload=b"z" * 5000,
        )


def crash_after_n_objects(root_text: str, target_count: int) -> None:
    root = Path(root_text)
    completed = 0

    def crash(observed: str) -> None:
        nonlocal completed
        if observed == "object-after-directory-fsync":
            completed += 1
            if completed == target_count:
                os._exit(87)

    with ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=1,
        expected_deployment_digest="a" * 64,
        expected_rank_digest="b" * 64,
        expected_rank=0,
        fault_hook=crash,
    ) as store:
        store.commit(
            entry_id=digest(f"multi-crash-{target_count}"),
            deployment_identity_digest="a" * 64,
            rank_identity_digest="b" * 64,
            span_tokens=256,
            physical_rank=0,
            topology_digest="c" * 64,
            profile="vllm-runtime-kv-v1",
            layout_digest="d" * 64,
            sources=tuple(
                ObjectSource(
                    group_index=0,
                    layer_name=f"layer-{index}",
                    page_start=index,
                    page_count=1,
                    chunks=(bytes((index + 1,)) * 3000,),
                )
                for index in range(3)
            ),
            created_at_unix_ns=1,
        )


def crash_collision_repair_at(
    root_text: str,
    stage: str,
    entry_id: str,
    payload: bytes,
) -> None:
    root = Path(root_text)

    def crash(observed: str) -> None:
        if observed == stage:
            os._exit(88)

    with ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=1,
        expected_deployment_digest="a" * 64,
        expected_rank_digest="b" * 64,
        expected_rank=0,
        fault_hook=crash,
    ) as store:
        commit_one(store, entry=entry_id, payload=payload)


def crash_object_quarantine_at(
    root_text: str,
    stage: str,
    descriptor: ObjectDescriptor,
) -> None:
    root = Path(root_text)

    def crash(observed: str) -> None:
        if observed == stage:
            os._exit(89)

    with ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=1,
        expected_deployment_digest="a" * 64,
        expected_rank_digest="b" * 64,
        expected_rank=0,
        fault_hook=crash,
    ) as store:
        store.quarantine_object(descriptor, reason="payload_checksum")


class ManifestCodecTests(unittest.TestCase):
    def test_envelope_detects_payload_tampering(self) -> None:
        payload = b"page"
        object_digest = hashlib.sha256(payload).hexdigest()
        manifest = RankManifest(
            entry_id="1" * 64,
            deployment_identity_digest="2" * 64,
            rank_identity_digest="3" * 64,
            span_tokens=256,
            physical_rank=0,
            topology_digest="4" * 64,
            profile="profile",
            layout_digest="5" * 64,
            objects=(
                ObjectDescriptor(
                    group_index=0,
                    layer_name="layer",
                    page_start=0,
                    page_count=1,
                    byte_length=4,
                    stored_length=4,
                    sha256=object_digest,
                    relative_path=(
                        f"objects/{object_digest[:2]}/{object_digest}.spool"
                    ),
                ),
            ),
            created_at_unix_ns=1,
        )
        encoded = encode_manifest(manifest)
        self.assertEqual(decode_manifest(encoded).manifest, manifest)
        tampered = encoded.replace(b'"span_tokens":256', b'"span_tokens":512')
        with self.assertRaisesRegex(ValueError, "digest differs"):
            decode_manifest(tampered)

    def test_noncanonical_json_number_is_a_manifest_error(self) -> None:
        encoded = json.dumps(
            {
                "schema": "spoolcache-manifest-envelope/v1",
                "payload": {"value": float("nan")},
                "payload_sha256": "0" * 64,
            },
            allow_nan=True,
        ).encode("utf-8")
        with self.assertRaisesRegex(
            store_module.ManifestError,
            "canonical JSON",
        ):
            decode_manifest(encoded)


class ManifestStoreTests(unittest.TestCase):
    def _store(self, root: Path, **extra: object) -> ManifestStore:
        return ManifestStore(
            root,
            slot_bytes=4096,
            slot_count=1,
            expected_deployment_digest="a" * 64,
            expected_rank_digest="b" * 64,
            expected_rank=0,
            **extra,
        )

    def test_commit_restart_stream_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            payload = bytes(range(256)) * 40
            entry = digest("entry")
            with self._store(root) as store:
                manifest = commit_one(store, entry=entry, payload=payload)
                self.assertEqual(store.buffer_budget_bytes, 4096)
                self.assertEqual(store.read_object_bytes(manifest.objects[0]), payload)
                self.assertTrue(store.lookup(entry, verify_payloads=True).is_hit)
            with self._store(root) as reopened:
                result = reopened.lookup(entry, verify_payloads=True)
                self.assertTrue(result.is_hit)
                self.assertEqual(reopened.scan_offers(10)[0].entry_id, entry)

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
                max_entries=8,
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

    def test_standalone_quarantine_signals_the_live_inventory_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("cross-instance-withdrawal")
            owner = self._store(root)
            maintenance = self._store(root)
            reporter = InventoryReporter(
                rank=0,
                generation="owner",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            catalog = QuorumCatalog(
                expected_ranks=(0,),
                max_entries=8,
                max_report_entries=8,
            )
            try:
                owner.acquire_inventory_owner()
                manifest = commit_one(owner, entry=entry, payload=b"shared" * 900)
                reporter.replace({entry: 256})
                catalog.apply_startup(
                    rank=0,
                    generation=reporter.generation,
                    generation_epoch=reporter.generation_epoch,
                    entries=reporter.startup(8),
                )
                self.assertTrue(catalog.has_quorum(entry))

                path = root / manifest.objects[0].relative_path
                with path.open("r+b") as handle:
                    handle.seek(3)
                    byte = handle.read(1)
                    handle.seek(3)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())
                maintenance.quarantine_object(
                    manifest.objects[0],
                    reason="payload_checksum",
                )
                self.assertTrue(
                    maintenance._inventory_withdrawal_path(entry).exists()
                )

                # This is the same reconciliation performed before every
                # connector inventory report.
                with owner._exclusive():
                    pending = owner.pending_inventory_withdrawals(
                        reporter.held_entry_ids()
                    )
                    for pending_entry in pending:
                        reporter.remove(pending_entry)
                    report = reporter.next_report(8)
                    owner.acknowledge_absent_inventory_withdrawals(pending)
                catalog.apply_report(report)
                self.assertFalse(catalog.has_quorum(entry))
                self.assertFalse(owner._inventory_withdrawal_path(entry).exists())
            finally:
                owner.close()
                maintenance.close()

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

    def test_pre_rename_quarantine_failure_stays_withdrawn_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            first, second = digest("withdrawn-first"), digest("withdrawn-second")
            withdrawn: list[str] = []
            failed_entry = ""
            with self._store(root) as store:
                manifest = commit_one(store, entry=first, payload=b"shared" * 800)
                commit_one(store, entry=second, payload=b"shared" * 800, created=2)
                store.set_withdraw_hook(withdrawn.append)
                object_path = root / manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(7)
                    handle.write(b"X")
                    handle.flush()
                    os.fsync(handle.fileno())

                real_replace = store_module.os.replace
                failed = False

                def fail_first_manifest_rename(source: object, target: object) -> None:
                    nonlocal failed, failed_entry
                    source_path = Path(source)
                    target_path = Path(target)
                    if (
                        not failed
                        and source_path.suffix == ".json"
                        and source_path.parent.parent == root / "manifests"
                        and target_path.parent == root / "quarantine"
                    ):
                        failed = True
                        failed_entry = source_path.stem
                        raise OSError(errno.EIO, "injected pre-rename failure")
                    real_replace(source, target)

                with (
                    mock.patch.object(
                        store_module.os,
                        "replace",
                        side_effect=fail_first_manifest_rename,
                    ),
                    self.assertRaisesRegex(RuntimeError, "complete receipt"),
                ):
                    store.quarantine_object(
                        manifest.objects[0],
                        reason="payload_checksum",
                    )
                self.assertTrue(failed)
                self.assertEqual(set(withdrawn), {first, second})
                self.assertTrue(store._manifest_path(failed_entry).exists())
                self.assertEqual(store.lookup(failed_entry).reason, "withdrawn")
                self.assertEqual(store.scan_offers(10), ())

            with self._store(root) as reopened:
                self.assertEqual(reopened.lookup(failed_entry).reason, "withdrawn")
                self.assertEqual(reopened.scan_offers(10), ())
                # A standalone maintenance instance leaves the successful
                # removal signal durable for the live inventory owner.
                self.assertTrue(
                    reopened._quarantine_manifest(
                        reopened._manifest_path(failed_entry),
                        reason="payload_checksum",
                    )
                )
                self.assertTrue(
                    reopened._inventory_withdrawal_path(failed_entry).exists()
                )
                pending = reopened.pending_inventory_withdrawals((failed_entry,))
                self.assertIn(failed_entry, pending)
                reopened.acknowledge_absent_inventory_withdrawals(pending)
                self.assertFalse(
                    reopened._inventory_withdrawal_path(failed_entry).exists()
                )

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

    def test_object_withdrawal_namespace_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cases = ("bad-name", "file", "symlink", "nonempty")
            for index, case in enumerate(cases):
                with self.subTest(case=case):
                    root = base / f"case-{index}"
                    with self._store(root):
                        pass
                    namespace = root / "state" / "object-withdrawn"
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
                        "object-withdrawal",
                    ):
                        self._store(root)

    def test_absent_withdrawal_markers_are_consumed_in_bounded_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entries = tuple(digest(f"offline-marker-{index}") for index in range(7))
            with self._store(root) as maintenance:
                for index, entry in enumerate(entries):
                    commit_one(
                        maintenance,
                        entry=entry,
                        payload=bytes((index + 1,)) * 4096,
                        created=index + 1,
                    )
                    self.assertTrue(maintenance.invalidate(entry))

            with self._store(root) as owner:
                owner.acquire_inventory_owner()
                self.assertEqual(owner.scan_offers(32), ())
                self.assertEqual(owner.pending_inventory_withdrawals(()), ())
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

    def test_absent_withdrawal_ack_persists_manifest_absence_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("absent-marker-directory-receipt")
            with self._store(root) as store:
                commit_one(store, entry=entry, payload=b"x" * 4096)
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
                    self.assertRaisesRegex(RuntimeError, "complete receipt"),
                ):
                    store.invalidate(entry)
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

    def test_unreferenced_object_ack_persists_manifest_namespace_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                for entry, payload in (
                    ("00" + "a" * 62, b"a" * 4096),
                    ("ff" + "b" * 62, b"b" * 4096),
                ):
                    commit_one(store, entry=entry, payload=payload)
                object_digest = digest("unreferenced-object-fence")
                store._mark_object_withdrawn(object_digest)
                marker = store._object_withdrawal_path(object_digest)
                failed_shard = root / "manifests" / "ff"
                real_fsync = store_module._fsync_directory

                def fail_one_shard(path: Path) -> None:
                    if path == failed_shard:
                        raise OSError(errno.EIO, "injected namespace fsync")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_one_shard,
                    ),
                    self.assertRaisesRegex(OSError, "namespace fsync"),
                ):
                    store.acknowledge_unreferenced_object_withdrawals(
                        (object_digest,)
                    )
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
                    store.acknowledge_unreferenced_object_withdrawals(
                        (object_digest,)
                    )
                self.assertFalse(marker.exists())
                marker_fsync = fsynced.index(marker.parent)
                self.assertLess(
                    fsynced.index(root / "manifests" / "00"),
                    marker_fsync,
                )
                self.assertLess(
                    fsynced.index(root / "manifests" / "ff"),
                    marker_fsync,
                )
                self.assertLess(fsynced.index(root / "manifests"), marker_fsync)

    def test_existing_withdrawal_marker_retries_parent_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                object_digest = digest("retry-object-fence-receipt")
                marker = store._object_withdrawal_path(object_digest)
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
                    store._mark_object_withdrawn(object_digest)
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
                    store._mark_object_withdrawn(object_digest)
                self.assertIn(marker.parent, fsynced)

    def test_existing_withdrawal_namespace_retries_ancestor_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            state = root / "state"
            namespace = state / "object-withdrawn"
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
                    reopened._mark_object_withdrawn(
                        digest("namespace-parent-retry-marker")
                    )
            self.assertIn(state, initialization_receipts)
            self.assertIn(namespace, fsynced)

    def test_repeated_shard_receipt_failure_never_accumulates_temp_objects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                real_fsync = store_module._fsync_directory

                def fail_object_namespace(path: Path) -> None:
                    if path == root / "objects":
                        raise OSError(errno.EIO, "injected shard parent fsync")
                    real_fsync(path)

                with mock.patch.object(
                    store_module,
                    "_fsync_directory",
                    new=fail_object_namespace,
                ):
                    for _ in range(3):
                        with self.assertRaisesRegex(
                            OSError,
                            "shard parent fsync",
                        ):
                            store.put_object((b"bounded-temp-payload" * 256,))
                        self.assertEqual(tuple((root / "tmp").iterdir()), ())

    def test_object_cleanup_failure_preserves_primary_publication_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                real_fsync = store_module._fsync_directory

                def fail_primary_and_cleanup(path: Path) -> None:
                    if path == root / "objects":
                        raise OSError(errno.EIO, "primary shard receipt")
                    if path == root / "tmp":
                        raise OSError(errno.EIO, "secondary tmp receipt")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_primary_and_cleanup,
                    ),
                    self.assertLogs(store_module.logger, level="ERROR") as logged,
                    self.assertRaisesRegex(OSError, "primary shard receipt") as caught,
                ):
                    store.put_object((b"dual-failure-object" * 256,))
                self.assertEqual(tuple((root / "tmp").iterdir()), ())
                self.assertIn("secondary tmp receipt", "\n".join(logged.output))
                if hasattr(caught.exception, "add_note"):
                    self.assertTrue(
                        any(
                            "secondary tmp receipt" in note
                            for note in caught.exception.__notes__
                        )
                    )

    def test_manifest_cleanup_failure_preserves_primary_publication_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                base = commit_one(
                    store,
                    entry=digest("manifest-cleanup-base"),
                    payload=b"manifest-cleanup-payload" * 256,
                )
                entry = digest("manifest-cleanup-target")
                candidate = RankManifest(
                    entry_id=entry,
                    deployment_identity_digest=base.deployment_identity_digest,
                    rank_identity_digest=base.rank_identity_digest,
                    span_tokens=base.span_tokens,
                    physical_rank=base.physical_rank,
                    topology_digest=base.topology_digest,
                    profile=base.profile,
                    layout_digest=base.layout_digest,
                    objects=base.objects,
                    created_at_unix_ns=2,
                )
                destination = store._manifest_path(entry)
                real_link = store_module.os.link
                real_fsync = store_module._fsync_directory
                primary_failed = False

                def fail_manifest_link(
                    source: object,
                    target: object,
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    nonlocal primary_failed
                    if Path(target) == destination:
                        primary_failed = True
                        raise OSError(errno.EIO, "primary manifest link")
                    real_link(source, target, *args, **kwargs)

                def fail_manifest_cleanup(path: Path) -> None:
                    if primary_failed and path == root / "tmp":
                        raise OSError(errno.EIO, "secondary manifest tmp receipt")
                    real_fsync(path)

                with (
                    mock.patch.object(
                        store_module.os,
                        "link",
                        new=fail_manifest_link,
                    ),
                    mock.patch.object(
                        store_module,
                        "_fsync_directory",
                        new=fail_manifest_cleanup,
                    ),
                    self.assertLogs(store_module.logger, level="ERROR") as logged,
                    self.assertRaisesRegex(OSError, "primary manifest link") as caught,
                ):
                    store.publish_manifest(candidate)
                self.assertEqual(tuple((root / "tmp").iterdir()), ())
                self.assertIn("secondary manifest tmp receipt", "\n".join(logged.output))
                if hasattr(caught.exception, "add_note"):
                    self.assertTrue(
                        any(
                            "secondary manifest tmp receipt" in note
                            for note in caught.exception.__notes__
                        )
                    )

    def test_marker_paging_advances_past_a_live_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entries = tuple(digest(f"paged-marker-{index}") for index in range(5))
            with self._store(root) as store:
                for index, entry in enumerate(entries):
                    commit_one(
                        store,
                        entry=entry,
                        payload=bytes((index + 1,)) * 4096,
                        created=index + 1,
                    )
                    store._mark_inventory_withdrawn(entry)
                live = min(entries)
                for entry in entries:
                    if entry != live:
                        store._manifest_path(entry).unlink()

                cursor: str | None = None
                for _ in range(10):
                    batch, cursor = store.inventory_withdrawal_marker_batch(
                        1,
                        after=cursor,
                    )
                    store.acknowledge_absent_inventory_withdrawals(batch)
                    remaining = tuple(
                        path.name
                        for path in (
                            root / "state" / "inventory-withdrawn"
                        ).iterdir()
                    )
                    if remaining == (live,):
                        break
                self.assertEqual(remaining, (live,))

    def test_inventory_scan_does_not_change_manifest_lru_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                entries = (digest("scan-old"), digest("scan-new"))
                for index, entry in enumerate(entries, start=1):
                    commit_one(
                        store,
                        entry=entry,
                        payload=bytes((index,)) * 4096,
                    )
                    os.utime(
                        store._manifest_path(entry),
                        ns=(index, index),
                    )
                before = tuple(
                    store._manifest_path(entry).stat().st_mtime_ns
                    for entry in entries
                )
                self.assertEqual(len(store.scan_offers(10)), 2)
                after = tuple(
                    store._manifest_path(entry).stat().st_mtime_ns
                    for entry in entries
                )
                self.assertEqual(after, before)

    def test_newer_withdrawn_manifest_cannot_starve_bounded_offer_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            healthy = digest("older-healthy-offer")
            withdrawn = digest("newer-withdrawn-offer")
            with self._store(root) as store:
                commit_one(store, entry=healthy, payload=b"h" * 4096)
                commit_one(store, entry=withdrawn, payload=b"w" * 4096)
                os.utime(store._manifest_path(healthy), ns=(1, 1))
                os.utime(store._manifest_path(withdrawn), ns=(2, 2))
                store._mark_inventory_withdrawn(withdrawn)

                offers = store.scan_offers(1)

                self.assertEqual(tuple(item.entry_id for item in offers), (healthy,))
                self.assertEqual(
                    store._manifest_path(healthy).stat().st_mtime_ns,
                    1,
                )

    def test_resource_limited_json_is_quarantined_by_lookup_and_scan(self) -> None:
        malformed = b'{"x":' + (b"1" * 5000) + b"}\n"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for operation in ("lookup", "scan"):
                with self.subTest(operation=operation):
                    root = base / operation
                    corrupt = digest(f"resource-json-{operation}")
                    healthy = digest(f"resource-json-healthy-{operation}")
                    with self._store(root) as store:
                        commit_one(store, entry=corrupt, payload=b"c" * 4096)
                        path = store._manifest_path(corrupt)
                        path.write_bytes(malformed)
                        with path.open("rb") as handle:
                            os.fsync(handle.fileno())
                        if operation == "lookup":
                            result = store.lookup(corrupt)
                            self.assertFalse(result.is_hit)
                            self.assertEqual(result.reason, "corrupt:ManifestError")
                        else:
                            commit_one(
                                store,
                                entry=healthy,
                                payload=b"h" * 4096,
                            )
                            iterator_active = False
                            real_iterator = store.iter_manifest_paths
                            real_quarantine = store._quarantine_manifest

                            def tracked_iterator():
                                nonlocal iterator_active
                                iterator_active = True
                                try:
                                    yield from real_iterator()
                                finally:
                                    iterator_active = False

                            def quarantine_after_iteration(
                                target: Path,
                                *,
                                reason: str,
                            ) -> bool:
                                self.assertFalse(iterator_active)
                                return real_quarantine(target, reason=reason)

                            with (
                                mock.patch.object(
                                    store,
                                    "iter_manifest_paths",
                                    new=tracked_iterator,
                                ),
                                mock.patch.object(
                                    store,
                                    "_quarantine_manifest",
                                    new=quarantine_after_iteration,
                                ),
                            ):
                                offers = store.scan_offers(1)
                            self.assertEqual(
                                tuple(item.entry_id for item in offers),
                                (healthy,),
                            )
                        self.assertFalse(path.exists())
                        self.assertTrue(tuple((root / "quarantine").iterdir()))

    def test_scan_deferred_quarantine_batch_is_fixed_and_converges(self) -> None:
        malformed = b'{"x":' + (b"1" * 5000) + b"}\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                paths: list[Path] = []
                for index in range(store_module._SCAN_QUARANTINE_BATCH + 1):
                    path = store._manifest_path(digest(f"deferred-{index}"))
                    store._ensure_shard_directory(path.parent)
                    path.write_bytes(malformed)
                    paths.append(path)

                self.assertEqual(store.scan_offers(1), ())
                self.assertEqual(sum(path.exists() for path in paths), 1)
                self.assertEqual(
                    len(tuple((root / "quarantine").iterdir())),
                    store_module._SCAN_QUARANTINE_BATCH,
                )

                self.assertEqual(store.scan_offers(1), ())
                self.assertFalse(any(path.exists() for path in paths))

    def test_restore_probe_and_stream_read_and_hash_payload_once(self) -> None:
        """Metadata probe plus authenticated transfer is one payload pass."""

        class CountingHash:
            def __init__(self, initial: bytes = b"") -> None:
                self._inner = real_sha256(initial)
                self.updated_bytes = len(initial)

            def update(self, value: object) -> None:
                self.updated_bytes += len(value)  # type: ignore[arg-type]
                self._inner.update(value)  # type: ignore[arg-type]

            def hexdigest(self) -> str:
                return self._inner.hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            payload = bytes(range(251)) * 41
            entry = digest("single-pass")
            with self._store(root) as store:
                manifest = commit_one(store, entry=entry, payload=payload)
                descriptor = manifest.objects[0]
                result = store.lookup(entry, verify_payloads=False)
                self.assertTrue(result.is_hit)

                real_read = store_module._read_into_exact
                physical_read_bytes = 0

                def counted_read(
                    file_descriptor: int,
                    target: memoryview,
                ) -> int:
                    nonlocal physical_read_bytes
                    count = real_read(file_descriptor, target)
                    physical_read_bytes += count
                    return count

                real_sha256 = hashlib.sha256
                hashers: list[CountingHash] = []

                def counted_sha256(initial: bytes = b"") -> CountingHash:
                    hasher = CountingHash(initial)
                    hashers.append(hasher)
                    return hasher

                restored = bytearray()
                with (
                    mock.patch.object(
                        store_module,
                        "_read_into_exact",
                        new=counted_read,
                    ),
                    mock.patch.object(
                        store_module.hashlib,
                        "sha256",
                        new=counted_sha256,
                    ),
                ):
                    store.stream_object(
                        descriptor,
                        lambda view: restored.extend(view),
                    )

                self.assertEqual(bytes(restored), payload)
                self.assertEqual(physical_read_bytes, descriptor.stored_length)
                self.assertEqual(len(hashers), 1)
                self.assertEqual(hashers[0].updated_bytes, descriptor.byte_length)

    def test_payload_corruption_becomes_a_clean_miss_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("corrupt")
            with self._store(root) as store:
                manifest = commit_one(store, entry=entry, payload=b"x" * 6000)
                object_path = root / manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(10)
                    handle.write(b"y")
                    handle.flush()
                    os.fsync(handle.fileno())
                result = store.lookup(entry, verify_payloads=True)
                self.assertFalse(result.is_hit)
                self.assertTrue(tuple((root / "quarantine").glob("*.bad")))

    def test_manifest_removal_always_attempts_offer_withdrawal(self) -> None:
        real_fsync = store_module._fsync_directory
        for failed_directory in ("source", "quarantine"):
            with self.subTest(failed_directory=failed_directory):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "owned"
                    entry = digest(f"quarantine-fsync-{failed_directory}")
                    withdrawn: list[str] = []
                    with self._store(root) as store:
                        commit_one(store, entry=entry, payload=b"x" * 4096)
                        store.set_withdraw_hook(withdrawn.append)
                        manifest_path = store._manifest_path(entry)
                        failure_path = (
                            manifest_path.parent
                            if failed_directory == "source"
                            else root / "quarantine"
                        )

                        def fail_one_directory(path: Path) -> None:
                            if path == failure_path:
                                raise OSError(errno.EIO, "injected directory fsync")
                            real_fsync(path)

                        with (
                            mock.patch.object(
                                store_module,
                                "_fsync_directory",
                                new=fail_one_directory,
                            ),
                            self.assertRaisesRegex(
                                RuntimeError,
                                "complete receipt",
                            ),
                        ):
                            store._quarantine_manifest(
                                manifest_path,
                                reason="manifest_validation",
                            )
                        self.assertFalse(manifest_path.exists())
                        self.assertEqual(withdrawn, [entry])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("quarantine-callback-failure")
            with self._store(root) as store:
                commit_one(store, entry=entry, payload=b"x" * 4096)
                manifest_path = store._manifest_path(entry)

                def fail_withdrawal(_entry_id: str) -> None:
                    raise RuntimeError("injected withdrawal failure")

                store.set_withdraw_hook(fail_withdrawal)
                with self.assertRaisesRegex(RuntimeError, "complete receipt"):
                    store._quarantine_manifest(
                        manifest_path,
                        reason="manifest_validation",
                    )
                self.assertFalse(manifest_path.exists())

    def test_invalidate_and_capacity_withdraw_after_directory_fsync_failure(self) -> None:
        real_fsync = store_module._fsync_directory
        for operation in ("invalidate", "capacity"):
            with self.subTest(operation=operation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "owned"
                    entry = digest(f"{operation}-fsync")
                    withdrawn: list[str] = []
                    with self._store(root) as store:
                        commit_one(store, entry=entry, payload=b"x" * 4096)
                        store.set_withdraw_hook(withdrawn.append)
                        manifest_path = store._manifest_path(entry)

                        def fail_manifest_directory(path: Path) -> None:
                            if path == manifest_path.parent:
                                raise OSError(errno.EIO, "injected directory fsync")
                            real_fsync(path)

                        with (
                            mock.patch.object(
                                store_module,
                                "_fsync_directory",
                                new=fail_manifest_directory,
                            ),
                            self.assertRaisesRegex(
                                RuntimeError,
                                "complete receipt",
                            ),
                        ):
                            if operation == "invalidate":
                                store.invalidate(entry)
                            else:
                                before = store.disk_usage_bytes()
                                store.maintain_capacity(
                                    max_bytes=before - 1,
                                    low_watermark_bytes=1,
                                )
                        self.assertFalse(manifest_path.exists())
                        self.assertEqual(withdrawn, [entry])

    def test_quarantine_hook_uses_only_bounded_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            reasons: list[str] = []
            with ManifestStore(
                root,
                slot_bytes=4096,
                slot_count=1,
            ) as store:
                store.set_quarantine_hook(reasons.append)
                manifest = store.commit(
                    entry_id="1" * 64,
                    deployment_identity_digest="2" * 64,
                    rank_identity_digest="3" * 64,
                    span_tokens=256,
                    physical_rank=0,
                    topology_digest="4" * 64,
                    profile="test",
                    layout_digest="5" * 64,
                    sources=(
                        ObjectSource(
                            group_index=0,
                            layer_name="layer",
                            page_start=0,
                            page_count=1,
                            chunks=(b"payload",),
                        ),
                    ),
                )
                path = root / manifest.objects[0].relative_path
                with path.open("r+b") as handle:
                    handle.write(b"damaged")
                result = store.lookup(manifest.entry_id, verify_payloads=True)
                self.assertFalse(result.is_hit)
            self.assertEqual(reasons, ["payload_checksum"])

    def test_manifest_fault_is_never_published_and_temp_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"

            def fault(stage: str) -> None:
                if stage == "manifest-before-fsync":
                    raise RuntimeError("injected crash")

            with self._store(root, fault_hook=fault) as store:
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    commit_one(store, entry=digest("fault"), payload=b"z" * 5000)
                self.assertFalse(store.lookup(digest("fault")).is_hit)
                (root / "tmp" / "abandoned.part").write_bytes(b"partial")
            with self._store(root) as recovered:
                self.assertFalse((root / "tmp" / "abandoned.part").exists())
                self.assertEqual(recovered.recover(), 0)

    def test_every_publication_crash_boundary_is_absent_or_fully_valid(self) -> None:
        before_visibility = (
            "object-before-write",
            "object-after-write",
            "object-before-fsync",
            "object-after-fsync",
            "object-before-link",
            "object-after-link",
            "object-before-directory-fsync",
            "object-after-directory-fsync",
            "manifest-before-write",
            "manifest-after-write",
            "manifest-before-fsync",
            "manifest-after-fsync",
            "manifest-before-link",
        )
        after_visibility = (
            "manifest-after-link",
            "manifest-before-directory-fsync",
            "manifest-after-directory-fsync",
        )
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for stage in (*before_visibility, *after_visibility):
                with self.subTest(stage=stage):
                    root = base / stage
                    with self._store(root):
                        pass
                    process = context.Process(
                        target=crash_commit_at,
                        args=(str(root), stage),
                    )
                    process.start()
                    process.join(10)
                    self.assertFalse(process.is_alive())
                    self.assertEqual(process.exitcode, 86)
                    entry = digest(f"crash-{stage}")
                    with self._store(root) as recovered:
                        result = recovered.lookup(entry, verify_payloads=True)
                        if stage in before_visibility:
                            self.assertFalse(result.is_hit)
                        else:
                            self.assertTrue(result.is_hit)
                        self.assertFalse(tuple((root / "tmp").iterdir()))

    def test_enospc_never_publishes_partial_entry_and_retry_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("enospc")
            with self._store(root) as store:
                def raise_enospc(*_args: object, **_kwargs: object) -> None:
                    raise OSError(errno.ENOSPC, "disk full")

                with mock.patch.object(
                    store_module,
                    "_write_exact",
                    new=raise_enospc,
                ):
                    with self.assertRaises(OSError) as caught:
                        commit_one(store, entry=entry, payload=b"z" * 5000)
                self.assertEqual(caught.exception.errno, errno.ENOSPC)
                self.assertFalse(store.lookup(entry).is_hit)
                self.assertFalse(tuple((root / "tmp").iterdir()))
                manifest = commit_one(store, entry=entry, payload=b"z" * 5000)
                self.assertTrue(store.lookup(entry, verify_payloads=True).is_hit)
                self.assertTrue((root / manifest.objects[0].relative_path).exists())

    def test_enospc_at_every_publication_boundary_is_retryable(self) -> None:
        before_visibility = (
            "object-before-write",
            "object-after-write",
            "object-before-fsync",
            "object-after-fsync",
            "object-before-link",
            "object-after-link",
            "object-before-directory-fsync",
            "object-after-directory-fsync",
            "manifest-before-write",
            "manifest-after-write",
            "manifest-before-fsync",
            "manifest-after-fsync",
            "manifest-before-link",
        )
        after_visibility = (
            "manifest-after-link",
            "manifest-before-directory-fsync",
            "manifest-after-directory-fsync",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for stage in (*before_visibility, *after_visibility):
                with self.subTest(stage=stage):
                    root = base / f"enospc-{stage}"
                    entry = digest(f"enospc-{stage}")
                    fired = False

                    def inject(observed: str) -> None:
                        nonlocal fired
                        if not fired and observed == stage:
                            fired = True
                            raise OSError(errno.ENOSPC, "injected disk full")

                    with self._store(root, fault_hook=inject) as store:
                        with self.assertRaises(OSError) as caught:
                            commit_one(store, entry=entry, payload=b"z" * 5000)
                        self.assertEqual(caught.exception.errno, errno.ENOSPC)
                        self.assertTrue(fired)
                        result = store.lookup(entry, verify_payloads=True)
                        self.assertEqual(
                            result.is_hit,
                            stage in after_visibility,
                        )
                        self.assertFalse(tuple((root / "tmp").iterdir()))

                    with self._store(root) as recovered:
                        commit_one(recovered, entry=entry, payload=b"z" * 5000)
                        self.assertTrue(
                            recovered.lookup(entry, verify_payloads=True).is_hit
                        )
                        self.assertFalse(tuple((root / "tmp").iterdir()))

    def test_corrupt_object_collision_never_leaves_a_reported_broken_reference(self) -> None:
        stages = (
            "object-withdrawal-after-fence",
            "object-withdrawal-after-reference",
            "object-collision-before-evidence-link",
            "object-collision-after-evidence-link",
            "object-collision-before-quarantine-directory-fsync",
            "object-collision-after-quarantine-directory-fsync",
            "object-collision-before-replace",
            "object-collision-after-replace",
            "object-collision-before-object-directory-fsync",
            "object-collision-after-object-directory-fsync",
            "object-collision-before-fence-release",
            "object-collision-after-fence-release",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for stage in stages:
                with self.subTest(stage=stage):
                    root = base / stage
                    fired = False

                    def fault(observed: str) -> None:
                        nonlocal fired
                        if not fired and observed == stage:
                            fired = True
                            raise OSError(errno.EIO, f"injected {stage}")

                    old_entry = digest(f"old-{stage}")
                    new_entry = digest(f"new-{stage}")
                    payload = b"collision-payload" * 400
                    reporter = InventoryReporter(
                        rank=0,
                        generation="worker",
                        generation_epoch=1,
                        max_entries=8,
                        max_report_entries=8,
                    )
                    catalog = QuorumCatalog(
                        expected_ranks=(0,),
                        max_entries=8,
                        max_report_entries=8,
                    )
                    with self._store(root, fault_hook=fault) as store:
                        # Do not arm a collision checkpoint during the initial
                        # publication; none is reached while the object is new.
                        manifest = commit_one(
                            store,
                            entry=old_entry,
                            payload=payload,
                        )
                        reporter.replace({old_entry: 256})
                        catalog.apply_startup(
                            rank=0,
                            generation=reporter.generation,
                            generation_epoch=reporter.generation_epoch,
                            entries=reporter.startup(8),
                        )
                        self.assertTrue(catalog.has_quorum(old_entry))
                        store.set_withdraw_hook(reporter.remove)
                        object_path = root / manifest.objects[0].relative_path
                        with object_path.open("r+b") as handle:
                            handle.seek(5)
                            byte = handle.read(1)
                            handle.seek(5)
                            handle.write(bytes((byte[0] ^ 1,)))
                            handle.flush()
                            os.fsync(handle.fileno())

                        with self.assertRaisesRegex(
                            RuntimeError,
                            "collision repair did not get a complete receipt",
                        ):
                            commit_one(store, entry=new_entry, payload=payload)
                        self.assertTrue(fired)
                        with store._exclusive():
                            pending = store.pending_inventory_withdrawals(
                                reporter.held_entry_ids()
                            )
                            for entry_id in pending:
                                reporter.remove(entry_id)
                            report = reporter.next_report(8)
                        catalog.apply_report(report)
                        self.assertFalse(catalog.has_quorum(old_entry))
                        self.assertEqual(store.lookup(old_entry).reason, "withdrawn")
                        self.assertTrue(
                            store._inventory_withdrawal_path(old_entry).exists()
                            or store._object_withdrawal_path(
                                manifest.objects[0].sha256
                            ).exists()
                        )

                        # Before the atomic switch, the old object remains at
                        # its live name. Afterwards the complete replacement is
                        # present. It is never missing beneath the manifest.
                        self.assertTrue(object_path.exists())
                        if stage in {
                            "object-collision-after-replace",
                            "object-collision-before-object-directory-fsync",
                            "object-collision-after-object-directory-fsync",
                            "object-collision-before-fence-release",
                            "object-collision-after-fence-release",
                        }:
                            self.assertEqual(
                                store.read_object_bytes(manifest.objects[0]),
                                payload,
                            )

    def test_successful_corrupt_object_collision_switch_is_atomic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            old_entry, new_entry = digest("collision-old"), digest("collision-new")
            payload = b"atomic-replacement" * 500
            withdrawn: list[str] = []
            with self._store(root) as store:
                manifest = commit_one(store, entry=old_entry, payload=payload)
                store.set_withdraw_hook(withdrawn.append)
                object_path = root / manifest.objects[0].relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(9)
                    byte = handle.read(1)
                    handle.seek(9)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

                commit_one(store, entry=new_entry, payload=payload)
                self.assertEqual(withdrawn, [old_entry])
                # Repairing one content address cannot prove the rest of an
                # older manifest. Its entry tombstone remains until a full
                # deep authentication (or complete replacement commit).
                self.assertEqual(store.lookup(old_entry).reason, "withdrawn")
                self.assertTrue(store.lookup(new_entry, verify_payloads=True).is_hit)
                self.assertFalse(
                    store._object_withdrawal_path(
                        manifest.objects[0].sha256
                    ).exists()
                )
                evidence = tuple((root / "quarantine").glob("*.bad"))
                self.assertEqual(len(evidence), 1)

    def test_one_object_repair_cannot_release_a_manifest_with_another_bad_object(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            old_entry = digest("two-bad-objects-old")
            new_entry = digest("repair-only-first-object")
            payload_a = b"first-object" * 500
            payload_b = b"second-object" * 500
            with self._store(root) as store:
                manifest = store.commit(
                    entry_id=old_entry,
                    deployment_identity_digest="a" * 64,
                    rank_identity_digest="b" * 64,
                    span_tokens=256,
                    physical_rank=0,
                    topology_digest="c" * 64,
                    profile="vllm-runtime-kv-v1",
                    layout_digest="d" * 64,
                    sources=(
                        ObjectSource(0, "layer-a", 0, 1, (payload_a,)),
                        ObjectSource(0, "layer-b", 1, 1, (payload_b,)),
                    ),
                    created_at_unix_ns=1,
                )
                for descriptor in manifest.objects:
                    path = root / descriptor.relative_path
                    with path.open("r+b") as handle:
                        handle.seek(11)
                        byte = handle.read(1)
                        handle.seek(11)
                        handle.write(bytes((byte[0] ^ 1,)))
                        handle.flush()
                        os.fsync(handle.fileno())

                commit_one(store, entry=new_entry, payload=payload_a)

                self.assertTrue(
                    store._inventory_withdrawal_path(old_entry).exists()
                )
                self.assertEqual(store.lookup(old_entry).reason, "withdrawn")
                self.assertNotIn(
                    old_entry,
                    {offer.entry_id for offer in store.scan_offers(8)},
                )
                descriptor_b = next(
                    item
                    for item in manifest.objects
                    if item.sha256 == hashlib.sha256(payload_b).hexdigest()
                )
                self.assertFalse(
                    store._verify_path(
                        root / descriptor_b.relative_path,
                        digest=descriptor_b.sha256,
                        logical_length=descriptor_b.byte_length,
                        stored_length=descriptor_b.stored_length,
                    )
                )

    def test_process_loss_during_collision_repair_cannot_readmit_old_manifest(self) -> None:
        stages = (
            "object-withdrawal-after-fence",
            "object-withdrawal-after-reference",
            "object-collision-before-evidence-link",
            "object-collision-after-evidence-link",
            "object-collision-before-quarantine-directory-fsync",
            "object-collision-after-quarantine-directory-fsync",
            "object-collision-before-replace",
            "object-collision-after-replace",
            "object-collision-before-object-directory-fsync",
            "object-collision-after-object-directory-fsync",
            "object-collision-before-fence-release",
            "object-collision-after-fence-release",
        )
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for stage in stages:
                with self.subTest(stage=stage):
                    root = base / stage
                    old_entry = digest(f"crash-collision-old-{stage}")
                    new_entry = digest(f"crash-collision-new-{stage}")
                    payload = b"crash-collision-payload" * 300
                    with self._store(root) as store:
                        manifest = commit_one(
                            store,
                            entry=old_entry,
                            payload=payload,
                        )
                        object_path = root / manifest.objects[0].relative_path
                        with object_path.open("r+b") as handle:
                            handle.seek(11)
                            byte = handle.read(1)
                            handle.seek(11)
                            handle.write(bytes((byte[0] ^ 1,)))
                            handle.flush()
                            os.fsync(handle.fileno())

                    process = context.Process(
                        target=crash_collision_repair_at,
                        args=(str(root), stage, new_entry, payload),
                    )
                    process.start()
                    process.join(10)
                    self.assertFalse(process.is_alive())
                    self.assertEqual(process.exitcode, 88)

                    with self._store(root) as reopened:
                        self.assertEqual(
                            reopened.lookup(old_entry).reason,
                            "withdrawn",
                        )
                        self.assertEqual(reopened.scan_offers(8), ())
                        self.assertTrue(
                            reopened._inventory_withdrawal_path(old_entry).exists()
                            or reopened._object_withdrawal_path(
                                manifest.objects[0].sha256
                            ).exists()
                        )

    def test_shared_collision_withdrawal_is_atomic_before_reference_walk(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entries = tuple(digest(f"shared-crash-{index}") for index in range(3))
            payload = b"shared-crash-payload" * 300
            with self._store(root) as store:
                manifests = tuple(
                    commit_one(
                        store,
                        entry=entry,
                        payload=payload,
                        created=index + 1,
                    )
                    for index, entry in enumerate(entries)
                )
                descriptor = manifests[0].objects[0]
                object_path = root / descriptor.relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(13)
                    byte = handle.read(1)
                    handle.seek(13)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

            process = context.Process(
                target=crash_collision_repair_at,
                args=(
                    str(root),
                    "object-withdrawal-after-reference",
                    digest("shared-crash-new"),
                    payload,
                ),
            )
            process.start()
            process.join(10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 88)

            with self._store(root) as reopened:
                self.assertTrue(
                    reopened._object_withdrawal_path(descriptor.sha256).exists()
                )
                self.assertEqual(reopened.scan_offers(16), ())
                for entry in entries:
                    self.assertEqual(reopened.lookup(entry).reason, "withdrawn")

    def test_shared_quarantine_withdrawal_is_atomic_before_reference_walk(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entries = tuple(digest(f"shared-quarantine-{index}") for index in range(3))
            payload = b"shared-quarantine-payload" * 300
            with self._store(root) as store:
                manifests = tuple(
                    commit_one(
                        store,
                        entry=entry,
                        payload=payload,
                        created=index + 1,
                    )
                    for index, entry in enumerate(entries)
                )
                descriptor = manifests[0].objects[0]
                object_path = root / descriptor.relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(17)
                    byte = handle.read(1)
                    handle.seek(17)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())

            process = context.Process(
                target=crash_object_quarantine_at,
                args=(str(root), "object-quarantine-after-reference", descriptor),
            )
            process.start()
            process.join(10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 89)

            with self._store(root) as reopened:
                self.assertTrue(
                    reopened._object_withdrawal_path(descriptor.sha256).exists()
                )
                self.assertEqual(reopened.scan_offers(16), ())
                for entry in entries:
                    self.assertFalse(reopened.lookup(entry).is_hit)

    def test_shared_quarantine_reference_count_is_streamed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            payload = b"bounded-shared-payload" * 220
            with self._store(root) as store:
                manifests = tuple(
                    commit_one(
                        store,
                        entry=digest(f"bounded-shared-{index}"),
                        payload=payload,
                        created=index + 1,
                    )
                    for index in range(129)
                )
                descriptor = manifests[0].objects[0]
                object_path = root / descriptor.relative_path
                with object_path.open("r+b") as handle:
                    handle.seek(19)
                    byte = handle.read(1)
                    handle.seek(19)
                    handle.write(bytes((byte[0] ^ 1,)))
                    handle.flush()
                    os.fsync(handle.fileno())
                self.assertEqual(
                    store.quarantine_object(
                        descriptor,
                        reason="payload_checksum",
                    ),
                    129,
                )
                self.assertEqual(store.scan_offers(256), ())
                self.assertFalse(
                    store._object_withdrawal_path(descriptor.sha256).exists()
                )

    def test_process_loss_after_each_object_count_leaves_only_collectable_orphans(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for target_count in (1, 2, 3):
                with self.subTest(target_count=target_count):
                    root = base / str(target_count)
                    with self._store(root):
                        pass
                    process = context.Process(
                        target=crash_after_n_objects,
                        args=(str(root), target_count),
                    )
                    process.start()
                    process.join(10)
                    self.assertEqual(process.exitcode, 87)
                    entry = digest(f"multi-crash-{target_count}")
                    with self._store(root) as recovered:
                        self.assertFalse(recovered.lookup(entry).is_hit)
                        self.assertEqual(
                            len(tuple(recovered.iter_object_paths())),
                            target_count,
                        )
                        before = recovered.disk_usage_bytes()
                        report = recovered.maintain_capacity(
                            max_bytes=before - 1,
                            low_watermark_bytes=1,
                        )
                        self.assertEqual(report.objects_removed, target_count)
                        self.assertFalse(tuple(recovered.iter_object_paths()))

    def test_same_entry_retry_is_idempotent_but_conflicting_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("immutable-entry")
            with self._store(root) as store:
                first = commit_one(
                    store,
                    entry=entry,
                    payload=b"same" * 2000,
                    created=1,
                )
                second = commit_one(
                    store,
                    entry=entry,
                    payload=b"same" * 2000,
                    created=2,
                )
                self.assertEqual(first.objects, second.objects)
                with self.assertRaisesRegex(ValueError, "incompatible immutable"):
                    commit_one(
                        store,
                        entry=entry,
                        payload=b"different" * 1000,
                        created=3,
                    )
                result = store.lookup(entry, verify_payloads=True)
                self.assertTrue(result.is_hit)
                self.assertEqual(result.manifest.objects, first.objects)

    def test_capacity_maintenance_cannot_collect_an_inflight_commit_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            object_durable = threading.Event()
            release_writer = threading.Event()
            maintenance_done = threading.Event()
            failures: list[BaseException] = []

            def fault(stage: str) -> None:
                if stage == "object-after-directory-fsync":
                    object_durable.set()
                    if not release_writer.wait(5):
                        raise RuntimeError("test writer release timed out")

            with self._store(root, fault_hook=fault) as store:
                entry = digest("inflight")

                def writer() -> None:
                    try:
                        commit_one(store, entry=entry, payload=b"x" * 5000)
                    except BaseException as error:
                        failures.append(error)

                def maintainer() -> None:
                    try:
                        store.maintain_capacity(
                            max_bytes=2,
                            low_watermark_bytes=1,
                        )
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        maintenance_done.set()

                writer_thread = threading.Thread(target=writer)
                writer_thread.start()
                self.assertTrue(object_durable.wait(5))
                maintenance_thread = threading.Thread(target=maintainer)
                maintenance_thread.start()
                self.assertFalse(maintenance_done.wait(0.1))
                release_writer.set()
                writer_thread.join(5)
                maintenance_thread.join(5)
                self.assertFalse(writer_thread.is_alive())
                self.assertFalse(maintenance_thread.is_alive())
                self.assertEqual(failures, [])
                # Maintenance may evict the now-complete entry to satisfy the
                # tiny watermark, but it never observes or deletes a partial
                # object transaction.
                self.assertFalse(tuple((root / "tmp").iterdir()))

    def test_recovery_cannot_remove_an_inflight_public_object_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            object_written = threading.Event()
            release_writer = threading.Event()
            recovery_done = threading.Event()
            failures: list[BaseException] = []

            def fault(stage: str) -> None:
                if stage == "object-after-write":
                    object_written.set()
                    if not release_writer.wait(5):
                        raise RuntimeError("test writer release timed out")

            with self._store(root, fault_hook=fault) as store:
                result: list[tuple[str, int, int]] = []

                def writer() -> None:
                    try:
                        result.append(store.put_object((b"x" * 5000,)))
                    except BaseException as error:
                        failures.append(error)

                def recoverer() -> None:
                    try:
                        store.recover()
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        recovery_done.set()

                writer_thread = threading.Thread(target=writer)
                writer_thread.start()
                self.assertTrue(object_written.wait(5))
                recovery_thread = threading.Thread(target=recoverer)
                recovery_thread.start()
                self.assertFalse(recovery_done.wait(0.1))
                release_writer.set()
                writer_thread.join(5)
                recovery_thread.join(5)

                self.assertEqual(failures, [])
                self.assertEqual(len(result), 1)
                self.assertTrue(store.object_path(result[0][0]).exists())
                self.assertFalse(tuple((root / "tmp").iterdir()))

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

    def test_capacity_evicts_oldest_manifest_and_unreferenced_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            withdrawn: list[str] = []
            with self._store(root) as store:
                store.set_withdraw_hook(withdrawn.append)
                older = digest("older")
                newer = digest("newer")
                commit_one(store, entry=older, payload=b"a" * 7000, created=1)
                commit_one(store, entry=newer, payload=b"b" * 7000, created=2)
                older_path = root / "manifests" / older[:2] / f"{older}.json"
                newer_path = root / "manifests" / newer[:2] / f"{newer}.json"
                os.utime(older_path, ns=(1, 1))
                os.utime(newer_path, ns=(2, 2))
                before = store.disk_usage_bytes()
                report = store.maintain_capacity(
                    max_bytes=before - 1,
                    low_watermark_bytes=before - 5000,
                )
                self.assertGreaterEqual(report.manifests_removed, 1)
                self.assertGreaterEqual(report.objects_removed, 1)
                self.assertLess(report.bytes_after, report.bytes_before)
                self.assertFalse(store.lookup(older).is_hit)
                self.assertEqual(withdrawn, [older])
                self.assertLessEqual(report.bytes_after, before - 5000)

    def test_capacity_uses_high_trigger_and_reaches_low_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            with self._store(root) as store:
                for index in range(4):
                    entry = digest(f"watermark-{index}")
                    commit_one(
                        store,
                        entry=entry,
                        payload=bytes((index + 1,)) * 6000,
                        created=index + 1,
                    )
                    manifest_path = (
                        root / "manifests" / entry[:2] / f"{entry}.json"
                    )
                    os.utime(manifest_path, ns=(index + 1, index + 1))
                before = store.disk_usage_bytes()
                unchanged = store.maintain_capacity(
                    max_bytes=before,
                    low_watermark_bytes=before - 1,
                )
                self.assertEqual(unchanged.bytes_after, before)
                self.assertEqual(unchanged.manifests_removed, 0)

                low = before // 2
                report = store.maintain_capacity(
                    max_bytes=before - 1,
                    low_watermark_bytes=low,
                )
                self.assertGreater(report.manifests_removed, 0)
                self.assertLessEqual(report.bytes_after, low)
                self.assertEqual(store.disk_usage_bytes(), report.bytes_after)

    def test_capacity_reclaims_crash_orphans_before_healthy_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("healthy-before-orphan")
            with self._store(root) as store:
                commit_one(store, entry=entry, payload=b"healthy" * 800)
                _orphan_digest, _logical, orphan_stored = store.put_object(
                    (b"orphan" * 900,)
                )
                before = store.disk_usage_bytes()
                report = store.maintain_capacity(
                    max_bytes=before - 1,
                    low_watermark_bytes=before - orphan_stored,
                )
                self.assertEqual(report.manifests_removed, 0)
                self.assertEqual(report.objects_removed, 1)
                self.assertTrue(store.lookup(entry, verify_payloads=True).is_hit)

    def test_reference_read_error_conservatively_retains_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            entry = digest("transient-reference-read-error")
            with self._store(root) as store:
                manifest = commit_one(store, entry=entry, payload=b"x" * 4096)
                object_path = root / manifest.objects[0].relative_path
                with mock.patch.object(
                    store,
                    "_read_manifest_file",
                    side_effect=OSError(errno.EIO, "transient manifest read"),
                ):
                    reclaimed = store.remove_orphan_object(
                        object_path,
                        not_newer_than_unix_ns=time.time_ns(),
                    )
                self.assertEqual(reclaimed, 0)
                self.assertTrue(object_path.exists())

    def test_managed_disk_usage_exposes_non_evictable_occupancy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            outside = Path(directory) / "outside"
            outside.write_bytes(b"outside must not be counted")
            with self._store(root) as store:
                commit_one(
                    store,
                    entry=digest("usage"),
                    payload=b"cache" * 1000,
                )
                (root / "quarantine" / "evidence.bad").write_bytes(b"1234567")
                (root / "tmp" / "abandoned.part").write_bytes(b"12345")
                unknown = root / "objects" / "not-a-shard"
                unknown.mkdir()
                (unknown / "unknown").write_bytes(b"123")
                (root / "quarantine" / "outside-link").symlink_to(outside)

                usage = store.managed_disk_usage()
                self.assertEqual(usage.cache_bytes, store.disk_usage_bytes())
                self.assertEqual(usage.quarantine_bytes, 7)
                self.assertEqual(usage.temporary_bytes, 5)
                self.assertEqual(usage.unrecognized_bytes, 3)
                self.assertGreater(usage.control_bytes, 0)
                self.assertEqual(
                    usage.total_bytes,
                    usage.cache_bytes
                    + usage.quarantine_bytes
                    + usage.temporary_bytes
                    + usage.state_bytes
                    + usage.unrecognized_bytes
                    + usage.control_bytes,
                )


if __name__ == "__main__":
    unittest.main()
