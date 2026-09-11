"""The key-file experiment must share prefixes without snapshot sidecars."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from spoolcache.buffers import AlignedBufferPool
from spoolcache.errors import LayoutError, ManifestError
from spoolcache.prefix import prefix_digests
from spoolcache.token_files import TokenFileStore
from spoolcache.token_mover import TokenPageMover
from spoolcache.token_scrub import TokenFileScrubber
from tests.token_fixtures import cpu_mover, layout_for


class TokenFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cache"
        self.layout = layout_for(tokens_per_page=256, page_bytes=16384)
        self.binding = {
            "expected_deployment_digest": "b" * 64,
            "expected_rank_digest": "c" * 64,
            "expected_rank": 0,
            "expected_topology_digest": "d" * 64,
            "expected_profile": self.layout.profile,
            "expected_layout_digest": self.layout.digest,
        }
        self.store = self.open_store()
        self.addCleanup(self.store.close)
        self.mover = cpu_mover(self.layout)
        self.mover.__class__ = TokenPageMover
        capture = self.mover._capture_packed

        def captured(segments, selected):
            yield from capture(segments, selected)

        self.mover._capture_packed = captured

    def open_store(self):
        return TokenFileStore(
            self.root,
            layout=self.layout,
            slot_bytes=16384,
            slot_count=1,
            **self.binding,
        )

    def keys(self, tokens):
        return prefix_digests(tokens, deployment_digest="b" * 64, chunk_tokens=self.layout.alignment_tokens)

    def save(self, tokens):
        keys = self.keys(tokens)
        return self.mover.commit_keys(
            self.store,
            prefixes=keys,
            block_tables=(tuple(range(1, len(tokens) // self.layout.groups[0].logical_tokens_per_page + 1)),),
        )

    def test_one_file_per_runtime_chunk_shared_branch_and_extension_skip_capture(self):
        tokens = tuple(range(1024))
        self.save(tokens)
        keys = self.keys(tokens)
        self.assertEqual(len(list(self.store.iter_manifest_paths())), 4)
        self.assertFalse(list((self.root / "objects").iterdir()))
        self.assertFalse(list((self.root / "state").glob("*.sqlite*")))
        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(tokens)
            self.assertEqual(capture.call_count, 0)
            self.save(tokens[:512] + (999,) * 512)
            self.assertEqual(capture.call_count, 2)
            self.save(tokens + tuple(range(1024, 1280)))
            self.assertEqual(capture.call_count, 3)
        self.assertEqual(len(list(self.store.iter_manifest_paths())), 7)
        with self.store.restore_view(keys[-1].digest) as view:
            output = []

            @contextmanager
            def receiver(desc):
                yield lambda data: output.append(bytes(data))

            self.store.stream_objects(view.descriptors, receiver, lease=view)
        self.assertEqual(
            b"".join(output), b"".join(bytes((i,)) * 16384 for i in range(1, 5))
        )

    def test_missing_middle_blocks_long_hit_but_leaves_shorter_prefix(self):
        tokens = tuple(range(1024))
        self.save(tokens)
        keys = self.keys(tokens)
        self.store.evict(keys[2].digest)
        self.assertFalse(self.store.lookup(keys[-1].digest).is_hit)
        self.assertTrue(self.store.lookup(keys[1].digest, verify_payloads=True).is_hit)

    def test_restart_no_direct_io_or_native_dependency(self):
        self.save(tuple(range(1024)))
        self.store.close()
        original = os.open

        def ordinary(path, flags, *args, **kwargs):
            self.assertFalse(flags & os.O_DIRECT)
            return original(path, flags, *args, **kwargs)

        with patch("os.open", side_effect=ordinary), self.open_store() as reopened:
            self.assertEqual(len(reopened.scan_offers(limit=20)), 4)
            self.assertTrue(
                reopened.lookup(
                    self.keys(range(1024))[-1].digest, verify_payloads=True
                ).is_hit
            )

    def test_corruption_never_reaches_receiver_and_recapture_repairs(self):
        tokens = tuple(range(1024))
        self.save(tokens)
        key = self.keys(tokens)[0].digest
        with self.store._manifest_path(key).open("r+b") as f:
            f.seek(-1, 2)
            f.write(b"!")
        self.assertFalse(self.store.lookup(key, verify_payloads=True).is_hit)
        self.assertNotIn(key, {x.entry_id for x in self.store.scan_offers(limit=20)})
        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(tokens)
            self.assertEqual(capture.call_count, 1)
        self.assertTrue(
            self.store.lookup(self.keys(tokens)[-1].digest, verify_payloads=True).is_hit
        )

    def test_pins_allow_unrelated_eviction_and_protect_entire_chain(self):
        self.save(tuple(range(1024)))
        self.save((999,) * 256)
        keys = self.keys(range(1024))
        with self.store.restore_view(keys[-1].digest):
            for key in keys:
                self.assertFalse(self.store.evict(key.digest))
            self.assertTrue(self.store.evict(self.keys((999,) * 256)[0].digest))
        self.assertTrue(self.store.evict(keys[-1].digest))

    def test_failed_publication_not_offered_and_no_temporary_leak(self):
        def fail(stage):
            if stage == "token_before_publish":
                raise OSError("injected")

        self.store._fault_hook = fail
        with self.assertRaisesRegex(OSError, "injected"):
            self.save(tuple(range(256)))
        self.assertEqual(self.store.scan_offers(limit=20), ())
        self.assertFalse(list((self.root / "tmp").iterdir()))

    def test_unsupported_geometry_fails_before_capture(self):
        for layout in (
            layout_for("sliding"),
            layout_for(page_bytes=513 * 64 * 1024 * 1024),
        ):
            with self.subTest(layout=layout.digest), self.assertRaises(LayoutError):
                TokenPageMover.validate_layout(layout, 64 * 1024 * 1024)

    def test_batch_caps_16_keys_and_one_buffer_and_reads_exact_payloads(self):
        self.mover._pool.slot_bytes = 256 * 1024
        tokens = tuple(range(5120))
        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(tokens)
            self.assertEqual(
                [len(call.args[0]) for call in capture.call_args_list], [16, 4]
            )
        received, batches = [], []

        @contextmanager
        def receiver(desc):
            yield lambda data: received.append(len(data))

        with (
            AlignedBufferPool(slot_bytes=256 * 1024, slot_count=2) as pool,
            self.store.restore_view(self.keys(tokens)[-1].digest) as lease,
        ):
            self.store.stream_objects(
                lease.descriptors,
                receiver,
                lease=lease,
                buffer_pool=pool,
                on_batch_complete=lambda slot: batches.append(slot.index),
            )
        self.assertEqual(received, [16384] * 20)
        self.assertEqual(len(batches), 2)

    def test_post_admission_corruption_rejects_payload_and_fences_key(self):
        tokens = tuple(range(512))
        self.save(tokens)
        keys = self.keys(tokens)
        received = []

        @contextmanager
        def receiver(desc):
            yield lambda data: received.append(desc.key)

        with self.store.restore_view(keys[-1].digest) as lease:
            with self.store._manifest_path(keys[0].digest).open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"!")
            with self.assertRaises(ManifestError):
                self.store.stream_objects(lease.descriptors, receiver, lease=lease)
            self.assertEqual(received, [])
            self.assertTrue(self.store._inventory_is_withdrawn(keys[0].digest))

    def test_link_failure_tombstone_survives_restart_then_repair(self):
        def fail(stage):
            if stage == "token_after_link":
                raise OSError("after link")

        self.store._fault_hook = fail
        with self.assertRaisesRegex(OSError, "after link"):
            self.save(tuple(range(256)))
        key = self.keys(range(256))[0].digest
        self.store.close()
        self.store = self.open_store()
        self.assertEqual(self.store.scan_offers(limit=20), ())
        self.assertFalse(self.store.lookup(key).is_hit)
        self.save(tuple(range(256)))
        self.assertTrue(self.store.lookup(key, verify_payloads=True).is_hit)
        self.store.close()

    def test_scrub_cursor_resume_and_targeted_corruption(self):
        self.save(tuple(range(1024)))
        scrub = TokenFileScrubber(self.store)
        scrub.start_cycle()
        report = scrub.step(payload_budget_bytes=16384, item_budget=1)
        self.assertEqual(report.objects_authenticated, 1)
        cursor = scrub.status().cursor
        reopened = TokenFileScrubber(self.store)
        self.assertEqual(reopened.status().cursor, cursor)
        for _ in range(5):
            report = reopened.step(payload_budget_bytes=16384, item_budget=1)
            if report.cycle_completed:
                break
        self.assertTrue(report.cycle_completed)
        key = self.keys(range(1024))[2].digest
        with self.store._manifest_path(key).open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        reopened.request(key)
        report = reopened.step(payload_budget_bytes=16384, item_budget=1)
        self.assertEqual(report.request_status, "quarantined")
        self.assertFalse(self.store.lookup(key).is_hit)

    def test_gc_20_percent_preserves_prefix_recency_after_inventory_and_restart(self):
        self.save(tuple(range(5120)))
        keys = self.keys(range(5120))
        paths = [self.store._manifest_path(key.digest) for key in keys]
        recency = [path.stat().st_mtime_ns for path in paths]
        self.assertTrue(self.store.lookup(keys[-1].digest).is_hit)
        self.assertEqual(len(self.store.scan_offers(limit=20)), 20)
        self.assertEqual([path.stat().st_mtime_ns for path in paths], recency)
        self.store.close()
        with self.open_store() as reopened:
            report = reopened.maintain_capacity_batch(trigger_bytes=1)
            self.assertEqual(report.objects_removed, 4)
            self.assertEqual(report.candidates_remaining, 0)
            self.assertTrue(reopened.lookup(keys[15].digest, verify_payloads=True).is_hit)
            self.assertFalse(reopened.lookup(keys[16].digest).is_hit)

    def test_short_vectored_writes_and_reads_preserve_all_bytes(self):
        writev, readv = os.writev, os.readv

        def short_write(fd, views):
            return writev(fd, [views[0][:127]])

        def short_read(fd, views):
            return readv(fd, [views[0][:113]])

        with (
            patch("os.writev", side_effect=short_write),
            patch("os.readv", side_effect=short_read),
        ):
            self.save(tuple(range(512)))
            self.assertTrue(
                self.store.lookup(
                    self.keys(range(512))[-1].digest, verify_payloads=True
                ).is_hit
            )

    def test_concurrent_scrub_target_is_not_overwritten_by_cycle_cursor(self):
        self.save(tuple(range(1024)))
        scrub = TokenFileScrubber(self.store)
        scrub.start_cycle()
        key = self.keys(range(1024))[-1].digest
        report = scrub.step(
            payload_budget_bytes=65536,
            item_budget=4,
            on_payload_read=lambda length: scrub.request(key),
        )
        self.assertEqual(report.objects_authenticated, 1)
        self.assertEqual(report.inventory_released, 0)
        self.assertEqual(scrub.status().target_entry, key)
        report = scrub.step(payload_budget_bytes=65536, item_budget=4)
        self.assertEqual(report.request_status, "authenticated")
