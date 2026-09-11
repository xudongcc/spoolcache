"""Installed-wheel CUDA byte oracles for buffered token files; no model needed."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from spoolcache.config import aligned_chunk_tokens
from spoolcache.errors import FatalRestoreError
from spoolcache.hma import HMALayout
from spoolcache.prefix import prefix_digests
from spoolcache.token_files import TokenFileStore
from spoolcache.token_mover import TokenPageMover
from tests.token_fixtures import layout_for, open_store
from tests.test_token_files_hma import mixed_layout

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class TokenFileCUDATests(unittest.TestCase):
    def test_aligned_stream_crosses_pages_layers_and_state_with_fixed_credits(self):
        groups = []
        for index, (tokens, size, policy) in enumerate(
            ((192, 3072, "full"), (320, 5120, "full"), (960, 10000, "recurrent_align"))
        ):
            group = layout_for(tokens_per_page=tokens, page_bytes=size).groups[0]
            groups.append(replace(
                group, group_index=index, reuse_policy=policy,
                layers=(replace(group.layers[0], name=f"layer{index}"),),
            ))
        layout = HMALayout(100, 1, tuple(groups))
        self.assertEqual(layout.alignment_tokens, 960)
        source = (tuple(1 + i * 3 for i in range(10)), (3, 8, 2, 5, 9, 1), (0, 0, 4))
        target = (tuple(51 + i * 2 for i in range(10)), (61, 64, 62, 67, 69, 60), (0, 0, 74))
        keys = prefix_digests(range(1920), deployment_digest="b" * 64, chunk_tokens=960)
        for strided in (False, True):
            with self.subTest(strided=strided), tempfile.TemporaryDirectory() as directory:
                caches, expected = {}, {}
                for group in groups:
                    name, size = group.layers[0].name, group.manager_page_size_bytes
                    backing = torch.full((100, size * (2 if strided else 1)), 253,
                                         dtype=torch.uint8, device="cuda")
                    cache = backing[:, :size]
                    values = torch.arange(100 * size, dtype=torch.int64, device="cuda")
                    cache.copy_((values % 251).to(torch.uint8).view(100, size))
                    caches[name] = cache
                    ids = group.select_physical_pages(source[group.group_index], 1920)
                    expected[name] = cache[list(ids)].clone()
                with open_store(Path(directory) / "rank", layout=layout, slot_bytes=4096) as store:
                    mover = TokenPageMover(layout, caches, slot_bytes=4096, slot_count=2)
                    try:
                        snapshot = mover.commit_keys(store, prefixes=keys, block_tables=source)
                        self.assertEqual([d.byte_length for d in snapshot.objects], [30720, 30720, 10000])
                        self.assertEqual(len(list(store.iter_manifest_paths())), 3)
                        self.assertEqual(mover.pinned_budget_bytes, 8192)
                        for cache in caches.values():
                            cache.zero_()
                        mover.restore_entry(store, entry_id=keys[-1].digest,
                                            span_tokens=1920, block_tables=target)
                        for group in groups:
                            name = group.layers[0].name
                            ids = group.select_physical_pages(target[group.group_index], 1920)
                            self.assertTrue(torch.equal(caches[name][list(ids)], expected[name]))
                            others = sorted(set(range(100)) - set(ids))
                            self.assertEqual(int(caches[name][others].count_nonzero()), 0)
                        self.assertEqual(mover._pool.borrowed, 0)
                        state = snapshot.objects[-1]
                        with store._manifest_path(state.key).open("r+b") as stream:
                            stream.seek(state.payload_offset + 8192)
                            stream.write(b"!")
                        with self.assertRaises(FatalRestoreError):
                            mover.restore_entry(store, entry_id=keys[-1].digest,
                                                span_tokens=1920, block_tables=target)
                        self.assertEqual(mover._pool.borrowed, 0)
                        self.assertTrue(all(not s.pending for s in mover._pool._slots))
                        self.assertTrue(store._inventory_is_withdrawn(state.key))
                        self.assertFalse(store._pins.busy(state.key))
                    finally:
                        mover.close()

    def test_native_hma_restores_exact_window_and_drains_state_corruption(self):
        layout = mixed_layout()
        values = (
            torch.arange(6400, device="cuda", dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
            .reshape(100, 64)
        )
        full = values.clone()
        window = values.flip(0).clone()
        caches = {"layer": full, "window": window}
        full_ids = tuple(range(1, 17))
        window_ids = tuple(range(31, 39))
        expected_full = full[list(full_ids)].clone()
        expected_window = window[list(window_ids)].clone()
        keys = prefix_digests(range(1024), deployment_digest="b" * 64, chunk_tokens=aligned_chunk_tokens(layout.alignment_tokens))
        original_open = os.open

        def ordinary(path, flags, *args, **kwargs):
            self.assertFalse(flags & os.O_DIRECT)
            return original_open(path, flags, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("os.open", side_effect=ordinary),
        ):
            store = TokenFileStore(
                Path(directory) / "rank",
                layout=layout,
                expected_deployment_digest="b" * 64,
                expected_rank_digest="c" * 64,
                expected_rank=0,
                expected_topology_digest="d" * 64,
                expected_profile=layout.profile,
                expected_layout_digest=layout.digest,
            )
            mover = TokenPageMover(
                layout, caches, slot_bytes=64 * 1024 * 1024, slot_count=2
            )
            try:
                snapshot = mover.commit_keys(
                    store, prefixes=keys, block_tables=(full_ids, (0,) * 8 + window_ids)
                )
                full.zero_()
                window.zero_()
                mover.restore_entry(
                    store,
                    entry_id=keys[-1].digest,
                    span_tokens=1024,
                    block_tables=(
                        tuple(range(50, 66)),
                        (0,) * 8 + tuple(range(70, 78)),
                    ),
                )
                self.assertTrue(torch.equal(full[50:66], expected_full))
                self.assertTrue(torch.equal(window[70:78], expected_window))
                self.assertEqual(int(window[:70].count_nonzero()), 0)
                self.assertEqual(mover._pool.borrowed, 0)
                state = snapshot.objects[-1]
                with store._manifest_path(state.key).open("r+b") as f:
                    f.seek(-1, 2)
                    f.write(b"!")
                with self.assertRaises(FatalRestoreError):
                    mover.restore_entry(
                        store,
                        entry_id=keys[-1].digest,
                        span_tokens=1024,
                        block_tables=(full_ids, (0,) * 8 + window_ids),
                    )
                self.assertEqual(mover._pool.borrowed, 0)
                self.assertTrue(all(not s.pending for s in mover._pool._slots))
                self.assertTrue(store._inventory_is_withdrawn(state.key))
            finally:
                mover.close()
                store.close()

    def test_save_restore_reuse_and_failure_drain_without_native_io(self):
        layout = layout_for()
        size = layout.num_manager_blocks * 4096
        for scattered in (False, True):
            for strided in (False, True):
                with (
                    self.subTest(scattered=scattered, strided=strided),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    values = (
                        torch.arange(size, device="cuda", dtype=torch.int64) % 251
                    ).to(torch.uint8)
                    if strided:
                        backing = torch.zeros(
                            (layout.num_manager_blocks, 8192),
                            dtype=torch.uint8,
                            device="cuda",
                        )
                        cache = backing[:, :4096]
                        cache.copy_(values.reshape(layout.num_manager_blocks, 4096))
                    else:
                        cache = values.reshape(layout.num_manager_blocks, 4096)
                    ids = (
                        tuple(1 + (i * 7) % 97 for i in range(80))
                        if scattered
                        else tuple(range(1, 81))
                    )
                    original = cache[list(ids)].clone()
                    keys = prefix_digests(range(5120), deployment_digest="b" * 64, chunk_tokens=aligned_chunk_tokens(layout.alignment_tokens))
                    original_open = os.open

                    def ordinary(path, flags, *args, _open=original_open, **kwargs):
                        self.assertFalse(flags & os.O_DIRECT)
                        return _open(path, flags, *args, **kwargs)

                    with patch("os.open", side_effect=ordinary):
                        mover = TokenPageMover(
                            layout,
                            {"layer": cache},
                            slot_bytes=64 * 1024 * 1024,
                            slot_count=2,
                        )
                        self.assertEqual(mover.pinned_budget_bytes, 128 * 1024 * 1024)
                        store = TokenFileStore(
                            Path(directory) / "rank",
                            layout=layout,
                            expected_deployment_digest="b" * 64,
                            expected_rank_digest="c" * 64,
                            expected_rank=0,
                            expected_topology_digest="d" * 64,
                            expected_profile=layout.profile,
                            expected_layout_digest=layout.digest,
                        )
                        try:
                            with patch.object(
                                mover, "_capture_packed", wraps=mover._capture_packed
                            ) as capture:
                                mover.commit_keys(
                                    store, prefixes=keys, block_tables=(ids,)
                                )
                                self.assertEqual(
                                    [len(c.args[0]) for c in capture.call_args_list],
                                    [16, 4],
                                )
                            with patch.object(
                                mover,
                                "_capture_packed",
                                side_effect=AssertionError("recapture"),
                            ):
                                mover.commit_keys(
                                    store, prefixes=keys, block_tables=(ids,)
                                )
                            cache.zero_()
                            mover.restore_entry(
                                store,
                                entry_id=keys[-1].digest,
                                span_tokens=5120,
                                block_tables=(tuple(range(1, 81)),),
                            )
                            self.assertTrue(torch.equal(cache[1:81], original))
                            self.assertEqual(int(cache[81:].count_nonzero()), 0)
                            self.assertEqual(mover._pool.borrowed, 0)
                            # Failure after an earlier batch was submitted must
                            # drain CUDA and return all credits before unwind.
                            with store._manifest_path(keys[-1].digest).open(
                                "r+b"
                            ) as stream:
                                stream.seek(-1, 2)
                                stream.write(b"!")
                            with self.assertRaises(FatalRestoreError):
                                mover.restore_entry(
                                    store,
                                    entry_id=keys[-1].digest,
                                    span_tokens=5120,
                                    block_tables=(tuple(range(1, 81)),),
                                )
                            self.assertEqual(mover._pool.borrowed, 0)
                            self.assertTrue(
                                all(not s.pending for s in mover._pool._slots)
                            )
                        finally:
                            mover.close()
                            store.close()
