from __future__ import annotations

import contextlib
import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from spoolcache.errors import FatalRestoreError, LayoutError
from spoolcache.gpu import (
    TorchPageMover,
    bind_group_owned_kv_caches,
    manager_page_view,
)
from spoolcache.hma import build_hma_layout as _build_hma_layout
from spoolcache.store import ManifestStore

try:
    import torch
except ImportError:
    torch = None


class FullAttentionSpec:
    public_kind = "full_attention"
    block_size = 256
    storage_block_size = 256
    page_size_bytes = 4


class SlidingWindowSpec:
    public_kind = "sliding_window"

    def __init__(self, block_size: int, window: int) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.sliding_window = window
        self.page_size_bytes = 4


class UniformTypeKVCacheSpecs:
    def __init__(self, specs: dict[str, SlidingWindowSpec], block_size: int) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.kv_cache_specs = specs
        self.page_size_bytes = sum(spec.page_size_bytes for spec in specs.values())


def build_hma_layout(*args, **kwargs):
    kwargs.setdefault(
        "spec_kind_resolver",
        lambda spec: getattr(spec, "public_kind", "unknown"),
    )
    return _build_hma_layout(*args, **kwargs)


def cache_config(num_blocks: int = 80) -> types.SimpleNamespace:
    # A large mixed-layout fixture with synthetic names and page geometry.
    signatures = (
        (62, 256, None),
        (23, 64, 128),
        (23, 64, 128),
        (42, 4, 8),
        (20, 8, 128),
    )
    groups = []
    for index, (count, block_size, window) in enumerate(signatures):
        names = tuple(f"g{index}.l{layer}" for layer in range(count))
        if window is None:
            spec = FullAttentionSpec()
        else:
            spec = UniformTypeKVCacheSpecs(
                {name: SlidingWindowSpec(block_size, window) for name in names},
                block_size,
            )
        groups.append(
            types.SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=names,
                is_eagle_group=False,
            )
        )
    return types.SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=tuple(groups),
    )


def entry_id(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class RegisteredKVCacheOwnershipTests(unittest.TestCase):
    def _layout(self):
        spec = FullAttentionSpec()
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            layer_names=("owner",),
            is_eagle_group=False,
        )
        return build_hma_layout(
            types.SimpleNamespace(num_blocks=4, kv_cache_groups=(group,))
        )

    def test_excludes_registered_shared_layer_alias(self) -> None:
        owner = object()
        selected, aliases = bind_group_owned_kv_caches(
            self._layout(),
            {"owner": owner, "shared": owner},
        )
        self.assertEqual(selected, {"owner": owner})
        self.assertEqual(aliases, (("shared", ("owner",)),))

    def test_rejects_unowned_registered_layer(self) -> None:
        with self.assertRaisesRegex(
            LayoutError,
            "neither group-owned nor an exact alias",
        ):
            bind_group_owned_kv_caches(
                self._layout(),
                {"owner": object(), "independent": object()},
            )

    def test_rejects_missing_group_owned_layer(self) -> None:
        with self.assertRaisesRegex(LayoutError, "missing=.*owner"):
            bind_group_owned_kv_caches(self._layout(), {"shared": object()})

    @unittest.skipIf(torch is None, "Torch required")
    def test_accepts_only_an_exact_shared_storage_view(self) -> None:
        owner = torch.zeros((4, 4))
        exact_view = owner.as_strided(owner.shape, owner.stride())
        selected, aliases = bind_group_owned_kv_caches(
            self._layout(),
            {"owner": owner, "shared": exact_view},
        )
        self.assertIs(selected["owner"], owner)
        self.assertEqual(aliases, (("shared", ("owner",)),))

        shifted_view = owner.flatten()[1:]
        with self.assertRaisesRegex(LayoutError, "exact alias"):
            bind_group_owned_kv_caches(
                self._layout(),
                {"owner": owner, "shifted": shifted_view},
            )


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA Torch required")
class TorchPageMoverTests(unittest.TestCase):
    def _mover(self):
        config = cache_config()
        layout = build_hma_layout(config)
        caches = {}
        for group in layout.groups:
            for ordinal, layer in enumerate(group.layers):
                values = torch.arange(
                    layout.num_manager_blocks * layer.page_size_bytes,
                    dtype=torch.int64,
                    device="cuda",
                )
                values = (values + group.group_index * 31 + ordinal) % 251
                caches[layer.name] = values.to(torch.uint8).view(
                    layout.num_manager_blocks,
                    layer.page_size_bytes,
                )
        mover = TorchPageMover(
            layout,
            caches,
            slot_bytes=4096,
            slot_count=2,
        )
        return layout, caches, mover

    def test_manager_view_includes_split_rows_and_physical_tail(self) -> None:
        storage = torch.arange(32, dtype=torch.uint8, device="cuda")
        tensor = torch.as_strided(storage, size=(4, 4), stride=(8, 1))
        pages = manager_page_view(
            tensor,
            manager_blocks=4,
            page_size_bytes=8,
            layer_name="padded",
        )
        self.assertEqual(tuple(pages.shape), (4, 1, 8))
        self.assertEqual(pages[2].cpu().view(-1).tolist(), list(range(16, 24)))

    def test_rank_geometry_digest_tracks_tensor_layout_not_pool_capacity(self) -> None:
        def one_layer_layout(num_blocks: int):
            spec = FullAttentionSpec()
            spec.page_size_bytes = 8
            group = types.SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=("layer",),
                is_eagle_group=False,
            )
            return build_hma_layout(
                types.SimpleNamespace(
                    num_blocks=num_blocks,
                    kv_cache_groups=(group,),
                )
            )

        first_layout = one_layer_layout(4)
        first_tensor = torch.empty((4, 8), dtype=torch.uint8, device="cuda")
        first = TorchPageMover(
            first_layout,
            {"layer": first_tensor},
            slot_bytes=4096,
            slot_count=1,
        )

        resized_layout = one_layer_layout(8)
        resized_tensor = torch.empty((8, 8), dtype=torch.uint8, device="cuda")
        resized = TorchPageMover(
            resized_layout,
            {"layer": resized_tensor},
            slot_bytes=4096,
            slot_count=1,
        )
        self.assertEqual(first.geometry_digest, resized.geometry_digest)

        padded_storage = torch.empty(32, dtype=torch.uint8, device="cuda")
        padded_tensor = torch.as_strided(
            padded_storage,
            size=(4, 4),
            stride=(8, 1),
        )
        padded = TorchPageMover(
            first_layout,
            {"layer": padded_tensor},
            slot_bytes=4096,
            slot_count=1,
        )
        self.assertNotEqual(first.geometry_digest, padded.geometry_digest)
        first.close()
        resized.close()
        padded.close()

    def test_all_groups_round_trip_through_manifest_store(self) -> None:
        layout, caches, mover = self._mover()
        boundary_counts = (1, 4, 4, 64, 32)
        tables = tuple(
            tuple(range(1, count + 1)) for count in boundary_counts
        )
        selected = layout.select_physical_pages(tables, 256)
        expected = {
            (group.group_index, layer.name): mover._bound[
                (group.group_index, layer.name)
            ].pages[list(selected[group.group_index])].clone()
            for group in layout.groups
            for layer in group.layers
        }
        with tempfile.TemporaryDirectory() as directory:
            with ManifestStore(
                Path(directory) / "cache",
                slot_bytes=4096,
                slot_count=1,
                expected_deployment_digest="a" * 64,
                expected_rank_digest="b" * 64,
                expected_rank=0,
            ) as store:
                manifest = mover.commit(
                    store,
                    entry_id=entry_id("round-trip"),
                    deployment_identity_digest="a" * 64,
                    rank_identity_digest="b" * 64,
                    span_tokens=256,
                    physical_rank=0,
                    topology_digest="c" * 64,
                    block_tables=tables,
                    created_at_unix_ns=1,
                )
                layout.validate_manifest_coverage(manifest)
                for group in layout.groups:
                    indexes = torch.tensor(
                        selected[group.group_index],
                        dtype=torch.long,
                        device="cuda",
                    )
                    for layer in group.layers:
                        mover._bound[(group.group_index, layer.name)].pages.index_fill_(
                            0, indexes, 0
                        )
                acquired_slots: list[int] = []
                original_acquire = mover._pool.acquire

                @contextlib.contextmanager
                def tracked_acquire(*, block: bool = False, timeout=None):
                    with original_acquire(block=block, timeout=timeout) as slot:
                        acquired_slots.append(slot.index)
                        yield slot

                with (
                    mock.patch.object(
                        mover._pool,
                        "acquire",
                        new=tracked_acquire,
                    ),
                    mock.patch.object(
                        mover,
                        "_synchronize_restore_stream",
                        wraps=mover._synchronize_restore_stream,
                    ) as synchronize,
                ):
                    mover.restore_entry(
                        store,
                        entry_id=manifest.entry_id,
                        span_tokens=256,
                        block_tables=tables,
                    )
                self.assertEqual(synchronize.call_count, 1)
                self.assertGreater(len(acquired_slots), 2)
                self.assertTrue(
                    all(
                        first != second
                        for first, second in zip(
                            acquired_slots,
                            acquired_slots[1:],
                            strict=False,
                        )
                    )
                )
                self.assertTrue(
                    all(not slot.pending for slot in mover._pool._slots)
                )
                for key, wanted in expected.items():
                    group_index, layer_name = key
                    actual = mover._bound[key].pages[
                        list(selected[group_index])
                    ]
                    self.assertTrue(torch.equal(actual, wanted), layer_name)
        mover.close()

    def test_corruption_fails_before_any_gpu_page_is_placed(self) -> None:
        layout, _, mover = self._mover()
        tables = tuple(
            tuple(range(1, count + 1)) for count in (1, 4, 4, 64, 32)
        )
        selected = layout.select_physical_pages(tables, 256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            with ManifestStore(
                root,
                slot_bytes=4096,
                slot_count=1,
                expected_deployment_digest="a" * 64,
                expected_rank_digest="b" * 64,
                expected_rank=0,
            ) as store:
                manifest = mover.commit(
                    store,
                    entry_id=entry_id("corrupt"),
                    deployment_identity_digest="a" * 64,
                    rank_identity_digest="b" * 64,
                    span_tokens=256,
                    physical_rank=0,
                    topology_digest="c" * 64,
                    block_tables=tables,
                    created_at_unix_ns=1,
                )
                first = root / manifest.objects[0].relative_path
                with first.open("r+b") as handle:
                    original = handle.read(1)
                    handle.seek(0)
                    handle.write(bytes((original[0] ^ 0xFF,)))
                for group in layout.groups:
                    indexes = torch.tensor(
                        selected[group.group_index],
                        dtype=torch.long,
                        device="cuda",
                    )
                    for layer in group.layers:
                        mover._bound[(group.group_index, layer.name)].pages.index_fill_(
                            0, indexes, 0
                        )
                with self.assertRaisesRegex(
                    FatalRestoreError, "POST_ADMISSION_RESTORE_FAILED"
                ):
                    mover.restore_entry(
                        store,
                        entry_id=manifest.entry_id,
                        span_tokens=256,
                        block_tables=tables,
                    )
                for group in layout.groups:
                    for layer in group.layers:
                        actual = mover._bound[
                            (group.group_index, layer.name)
                        ].pages[list(selected[group.group_index])]
                        self.assertEqual(int(actual.count_nonzero()), 0)
        mover.close()

    def test_late_corruption_drains_submitted_cuda_work(self) -> None:
        """A late disk failure must not leave pinned slots owned by CUDA.

        This is deliberately different from the first-object corruption test:
        several valid objects are allowed to reach request-private GPU pages
        before the final object fails authentication.  The engine treats that
        partial restore as fatal, but the mover must still execute its terminal
        barrier so its fixed staging pool is safe to close or reuse.
        """

        layout, _, mover = self._mover()
        tables = tuple(
            tuple(range(1, count + 1)) for count in (1, 4, 4, 64, 32)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            with ManifestStore(
                root,
                slot_bytes=4096,
                slot_count=1,
                expected_deployment_digest="a" * 64,
                expected_rank_digest="b" * 64,
                expected_rank=0,
            ) as store:
                manifest = mover.commit(
                    store,
                    entry_id=entry_id("late-corrupt"),
                    deployment_identity_digest="a" * 64,
                    rank_identity_digest="b" * 64,
                    span_tokens=256,
                    physical_rank=0,
                    topology_digest="c" * 64,
                    block_tables=tables,
                    created_at_unix_ns=1,
                )
                # Damage the last object so earlier H2D/scatter work has
                # definitely been queued when authentication fails.
                last = root / manifest.objects[-1].relative_path
                with last.open("r+b") as handle:
                    original = handle.read(1)
                    handle.seek(0)
                    handle.write(bytes((original[0] ^ 0xFF,)))

                with (
                    mock.patch.object(
                        mover,
                        "_synchronize_restore_stream",
                        wraps=mover._synchronize_restore_stream,
                    ) as synchronize,
                    self.assertRaisesRegex(
                        FatalRestoreError,
                        "POST_ADMISSION_RESTORE_FAILED",
                    ),
                ):
                    mover.restore_entry(
                        store,
                        entry_id=manifest.entry_id,
                        span_tokens=256,
                        block_tables=tables,
                    )

                self.assertEqual(synchronize.call_count, 1)
                self.assertTrue(
                    all(not slot.pending for slot in mover._pool._slots)
                )
        mover.close()


if __name__ == "__main__":
    unittest.main()
