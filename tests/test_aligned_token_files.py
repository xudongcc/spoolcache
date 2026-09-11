"""Runtime-sized files must stay bounded independently of GPU page size."""

import hashlib
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from spoolcache.errors import ManifestError
from spoolcache.hma import HMALayout
from spoolcache.prefix import prefix_digests
from spoolcache.token_mover import TokenPageMover
from tests.token_fixtures import cpu_mover, layout_for, open_store


class AlignedTokenFileTests(unittest.TestCase):
    def test_oversized_data_or_state_header_fails_before_capture(self):
        full = layout_for(tokens_per_page=256, page_bytes=64).groups[0]
        state = replace(
            layout_for("sliding", tokens_per_page=256, page_bytes=64).groups[0],
            group_index=1,
            layers=(replace(full.layers[0], name="state"),),
        )
        prefixes = prefix_digests(range(256), deployment_digest="b" * 64, chunk_tokens=256)
        for oversized in (0, 1):
            with self.subTest(oversized_group=oversized):
                groups = [full, state]
                group = groups[oversized]
                groups[oversized] = replace(group, layers=tuple(
                    replace(group.layers[0], name=f"{i:03d}" + "x" * 509)
                    for i in range(128)
                ))
                layout = HMALayout(100, 1, tuple(groups))
                mover = cpu_mover(layout)
                with tempfile.TemporaryDirectory() as directory, open_store(
                    Path(directory) / "rank", layout=layout,
                ) as store, patch.object(
                    mover, "_capture_packed", side_effect=AssertionError("premature capture"),
                ):
                    with self.assertRaisesRegex(ManifestError, "header length"):
                        mover.commit_keys(store, prefixes=prefixes, block_tables=((0,), (0,)))
                    self.assertEqual(list(store.iter_manifest_paths()), [])

    def test_chunk_floor_rounds_up_to_a_whole_runtime_alignment(self):
        for alignment, expected in (
            (32, 256), (64, 256), (96, 288), (128, 256),
            (192, 384), (256, 256), (320, 320), (960, 960), (1664, 1664),
        ):
            with self.subTest(alignment=alignment):
                layout = layout_for(tokens_per_page=alignment, page_bytes=64)
                TokenPageMover.validate_layout(layout, 4096)
                with tempfile.TemporaryDirectory() as directory, open_store(
                    Path(directory) / "rank", layout=layout, slot_bytes=4096
                ) as store:
                    self.assertEqual(store.chunk_tokens, expected)
                    self.assertEqual(store._binding["chunk_tokens"], expected)
                    self.assertEqual(
                        store._segments(expected)[0].page_count,
                        expected // alignment,
                    )
                    prefixes = prefix_digests(
                        range(expected), deployment_digest="b" * 64,
                        chunk_tokens=expected,
                    )
                    pages = tuple(range(1, expected // alignment + 1))
                    snapshot = cpu_mover(layout, 4096).commit_keys(
                        store, prefixes=prefixes, block_tables=(pages,),
                    )
                    self.assertEqual(len(snapshot.objects), 1)
                    payload = b"".join(bytes([page]) * 64 for page in pages)
                    self.assertEqual(snapshot.objects[0].sha256, hashlib.sha256(payload).hexdigest())
                    self.assertTrue(store.lookup(snapshot.entry_id, verify_payloads=True).is_hit)

    def test_common_alignment_accepts_non_power_of_two_pages(self):
        first = layout_for(tokens_per_page=192, page_bytes=8192).groups[0]
        second = replace(
            layout_for(tokens_per_page=320, page_bytes=12288).groups[0],
            group_index=1,
            layers=(replace(first.layers[0], name="other", page_size_bytes=12288),),
        )
        layout = HMALayout(100, 1, (first, second))
        self.assertEqual(layout.alignment_tokens, 960)
        TokenPageMover.validate_layout(layout, 4096)
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096
        ) as store:
            self.assertEqual(store.chunk_tokens, 960)
            self.assertEqual(
                [(s.page_count, s.byte_length) for s in store._segments(960)],
                [(5, 40960), (3, 36864)],
            )
            with self.assertRaises(ManifestError):
                store.validate_prefix_headers(256)

    def test_large_page_is_one_file_streamed_through_one_cpu_credit(self):
        layout = layout_for(tokens_per_page=960, page_bytes=96 * 1024)
        key = prefix_digests(
            range(960), deployment_digest="b" * 64, chunk_tokens=960
        )[0].digest
        expected = hashlib.sha256()
        yielded = []

        def capture():
            # A single reused producer buffer, deliberately larger than neither
            # the reader credit nor the file. Retaining yielded views is wrong.
            data = bytearray(4096)
            for index in range(24):
                data[:] = bytes([index]) * len(data)
                expected.update(data)
                yielded.append(index)
                yield memoryview(data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with open_store(root, layout=layout, slot_bytes=4096) as store:
                desc = store.commit_stream(key, None, 960, capture())
                self.assertEqual(len(list(store.iter_manifest_paths())), 1)
                self.assertEqual(desc.byte_length, 96 * 1024)
                self.assertEqual(desc.sha256, expected.hexdigest())
                self.assertEqual(yielded, list(range(24)))
                self.assertEqual(store._pool.budget_bytes, 4096)
                self.assertTrue(store.lookup(key, verify_payloads=True).is_hit)
                received = []

                @contextmanager
                def receiver(descriptor):
                    self.assertEqual(descriptor, desc)
                    yield lambda data: received.append(
                        (len(data), data[0], hashlib.sha256(data).hexdigest())
                    )

                with store.restore_view(key) as lease:
                    self.assertFalse(store.evict(key))
                    store.stream_objects(lease.descriptors, receiver, lease=lease)
                self.assertEqual(
                    [(size, value) for size, value, _ in received],
                    [(4096, i) for i in range(24)],
                )
                self.assertEqual(store._pool.borrowed, 0)
            with open_store(root, layout=layout, slot_bytes=4096) as store:
                self.assertTrue(store.lookup(key, verify_payloads=True).is_hit)
                self.assertEqual(len(store.scan_offers(10)), 1)

    def test_late_corruption_never_delivers_bad_block_or_leaks_credit(self):
        layout = layout_for(tokens_per_page=960, page_bytes=12288)
        key = "a" * 64
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096
        ) as store:
            desc = store.commit_stream(
                key, None, 960, (bytes([i]) * 4096 for i in range(3))
            )
            received = []

            @contextmanager
            def receiver(_):
                yield lambda data: received.append(data[0])

            with store.restore_view(key) as lease:
                with store._manifest_path(key).open("r+b") as stream:
                    stream.seek(desc.payload_offset + 4096)
                    stream.write(b"!")
                with self.assertRaises(ManifestError):
                    store.stream_objects(lease.descriptors, receiver, lease=lease)
                self.assertEqual(received, [0])
                self.assertTrue(store._inventory_is_withdrawn(key))
                self.assertEqual(store._pool.borrowed, 0)
            self.assertFalse(store.lookup(key).is_hit)

    def test_incomplete_stream_is_never_published(self):
        layout = layout_for(tokens_per_page=960, page_bytes=12288)
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096
        ) as store:
            for pieces in ((b"x" * 4096,), (b"x" * 4097,), (b"x" * 4096,) * 4):
                with self.subTest(lengths=tuple(map(len, pieces))):
                    with self.assertRaises(ManifestError):
                        store.commit_stream("a" * 64, None, 960, iter(pieces))
                    self.assertEqual(store.scan_offers(10), ())
                    self.assertFalse(list((store.root / "tmp").iterdir()))

    def test_stream_durability_cost_does_not_grow_with_transfer_count(self):
        counts = []
        original_fsync = os.fsync
        for blocks in (2, 24):
            layout = layout_for(tokens_per_page=960, page_bytes=4096 * blocks)
            with tempfile.TemporaryDirectory() as directory, open_store(
                Path(directory) / "rank", layout=layout, slot_bytes=4096
            ) as store:
                with patch("os.fsync", wraps=original_fsync) as sync:
                    store.commit_stream("a" * 64, None, 960, (b"x" * 4096 for _ in range(blocks)))
                    counts.append(sync.call_count)
                self.assertTrue(store.lookup("a" * 64, verify_payloads=True).is_hit)
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[0], counts[1])

    def test_failure_after_written_block_cleans_temporary_and_retry_succeeds(self):
        layout = layout_for(tokens_per_page=960, page_bytes=12288)
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096
        ) as store:
            written = 0

            def fail(point):
                nonlocal written
                if point == "token_after_payload_block":
                    written += 1
                    if written == 2:
                        raise OSError("interrupted payload stream")

            store._fault_hook = fail
            with self.assertRaisesRegex(OSError, "interrupted payload"):
                store.commit_stream("a" * 64, None, 960, (b"x" * 4096 for _ in range(3)))
            self.assertEqual(written, 2)
            self.assertFalse(list((store.root / "tmp").iterdir()))
            self.assertEqual(store.scan_offers(10), ())
            store._fault_hook = None
            store.commit_stream("a" * 64, None, 960, (b"x" * 4096 for _ in range(3)))
            self.assertTrue(store.lookup("a" * 64, verify_payloads=True).is_hit)


if __name__ == "__main__":
    unittest.main()
