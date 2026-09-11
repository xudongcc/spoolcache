from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import types
import unittest
from unittest import mock

from spoolcache.errors import LayoutError
from spoolcache.gpu import (
    TorchPageMover,
    PinnedTensorPool,
    _PinnedSlot,
    bind_group_owned_kv_caches,
    manager_page_view,
)
from spoolcache.hma import build_hma_layout as _build_hma_layout
from spoolcache.manifest import PageSlice

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
    def test_restore_dense_targets_avoid_temporary_scatter_and_preserve_other_pages(self):
        layout, _, mover = self._mover()
        layer = layout.groups[0].layers[0]
        bound = mover._bound[(0, layer.name)]
        payload = bytes(range(3 * layer.page_size_bytes))
        descriptor = types.SimpleNamespace(byte_length=len(payload), segments=(
            PageSlice(0, layer.name, 0, 3, len(payload)),))
        stream = torch.cuda.current_stream()
        original_copy = torch.Tensor.index_copy_
        backing = torch.full((*bound.pages.shape[:-1], bound.pages.shape[-1]*2), 255,
                             dtype=torch.uint8, device=bound.pages.device)
        strided = backing[..., ::2]
        for target_pages, pages, scatters in ((bound.pages, (1, 2, 3), 0),
                (bound.pages, (1, 3, 5), 1), (bound.pages, (5, 3, 1), 1),
                (strided, (1, 2, 3), 1)):
            mover._bound[(0, layer.name)] = dataclasses.replace(bound, pages=target_pages)
            calls = []

            def copy(tensor, *args, **kwargs):
                calls.append(True)
                return original_copy(tensor, *args, **kwargs)

            target_pages.fill_(255)
            with self.subTest(pages=pages), mover._pool.acquire() as slot:
                with slot.view[:len(payload)] as view:
                    view[:] = payload
                    with mock.patch.object(torch.Tensor, 'index_copy_', new=copy):
                        with mover._restore_object_receiver(descriptor, (pages,), stream=stream) as receive:
                            receive(view)
                    slot.record_use(stream)
                    slot.wait_until_reusable()
                self.assertEqual(len(calls), scatters)
            expected = bytearray([255]) * bound.pages.numel()
            for index, page in enumerate(pages):
                start = index * layer.page_size_bytes
                expected[page*layer.page_size_bytes:(page+1)*layer.page_size_bytes] = payload[start:start+layer.page_size_bytes]
            self.assertEqual(target_pages.cpu().numpy().tobytes(), expected)
        self.assertTrue(bool((backing[..., 1::2] == 255).all()))

    def test_capture_contiguous_view_and_noncontiguous_gather_have_same_bytes(self):
        layout, _, mover = self._mover()
        layer = layout.groups[0].layers[0]
        bound = mover._bound[(0, layer.name)]
        segments = (PageSlice(0, layer.name, 0, 3, 3 * layer.page_size_bytes),)
        for pages, gathers in (((1, 2, 3), 0), ((1, 3, 5), 1), ((5, 3, 1), 1)):
            expected = bound.pages[list(pages)].cpu().numpy().tobytes()
            with self.subTest(pages=pages), mock.patch.object(
                    torch, 'index_select', wraps=torch.index_select) as select:
                with contextlib.closing(mover._capture_packed(segments, (pages,))) as source:
                    self.assertEqual(bytes(next(source)), expected)
                self.assertEqual(select.call_count, gathers)
        # Consecutive page IDs do not prove a dense storage view. A strided
        # opaque page tensor must keep the gather and produce the same bytes.
        backing = torch.empty((*bound.pages.shape[:-1], bound.pages.shape[-1] * 2),
                              dtype=torch.uint8, device=bound.pages.device)
        strided = backing[..., ::2]
        strided.copy_(bound.pages)
        mover._bound[(0, layer.name)] = dataclasses.replace(bound, pages=strided)
        with mock.patch.object(torch, 'index_select', wraps=torch.index_select) as select:
            with contextlib.closing(mover._capture_packed(segments, ((1, 2, 3),))) as source:
                self.assertEqual(bytes(next(source)), bound.pages[1:4].cpu().numpy().tobytes())
            self.assertEqual(select.call_count, 1)

    def test_failed_cuda_event_prevents_reuse_until_terminal_drain(self):
        pool = PinnedTensorPool(slot_bytes=4096, slot_count=1)
        self.addCleanup(pool.close)
        with pool.acquire() as slot:
            slot.completion_event = types.SimpleNamespace(
                record=mock.Mock(side_effect=RuntimeError('injected event failure')))
            with self.assertRaisesRegex(RuntimeError, 'event failure'):
                slot.record_use(None)
        with self.assertRaisesRegex(RuntimeError, 'terminal drain'):
            with pool.acquire():
                self.fail('unsafe credit reuse')
        self.assertEqual(pool.borrowed, 0)
        # The injected fixture enqueues no GPU work; production synchronizes
        # its real current stream before clearing the ownership failure.
        pool.mark_stream_synchronized()
        with pool.acquire():
            pass

    def test_registered_pool_is_aligned_and_drains_before_unregister(self):
        pool = PinnedTensorPool(slot_bytes=4096, slot_count=2)
        events = []
        with pool.acquire() as slot:
            self.assertTrue(slot.tensor.is_pinned())
            self.assertEqual(slot.tensor.data_ptr() % 4096, 0)
            self.assertEqual(pool.budget_bytes, 8192)
            slot.completion_event = types.SimpleNamespace(
                record=lambda _: events.append('record'),
                synchronize=lambda: events.append('drain'),
            )
            slot.record_use(None)
            mapping = slot.mapping._mapping
            with self.assertRaisesRegex(RuntimeError, 'borrowed'):
                pool.close()
        pool.close()
        self.assertEqual(events, ['record', 'drain'])
        self.assertTrue(mapping.closed)

    def test_packed_capture_waits_once_and_drains_a_mid_pack_failure(self) -> None:
        layout, _, mover = self._mover()
        tables = tuple(tuple(range(1, count + 1)) for count in (1, 4, 4, 64, 32))
        selected = layout.select_physical_pages(tables, 256)
        segments = tuple(
            PageSlice(group.group_index, layer.name, 0, len(pages),
                      len(pages) * layer.page_size_bytes)
            for group, pages in zip(layout.groups, selected)
            if group.reuse_policy == 'full'
            for layer in group.layers
        )
        self.assertGreater(len(segments), 1)
        records = []
        original_record = _PinnedSlot.record_use

        def record(slot, stream):
            records.append(slot.index)
            return original_record(slot, stream)

        with mock.patch.object(_PinnedSlot, "record_use", new=record):
            source = mover._capture_packed(segments, selected)
            payload = bytes(next(source))
            self.assertEqual(len(records), 1)
            self.assertTrue(all(not slot.pending for slot in mover._pool._slots))
            source.close()
        expected = b"".join(
            mover._bound[(s.group_index, s.layer_name)].pages[
                list(selected[s.group_index][s.page_start:s.page_start+s.page_count])
            ].cpu().numpy().tobytes() for s in segments
        )
        self.assertEqual(payload, expected)

        records.clear()
        original_copy = torch.Tensor.copy_
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("injected mid-capture failure")
            return original_copy(*args, **kwargs)

        with (
            mock.patch.object(torch.Tensor, "copy_", new=fail_second),
            mock.patch.object(_PinnedSlot, "record_use", new=record),
            self.assertRaisesRegex(RuntimeError, "mid-capture"),
        ):
            next(mover._capture_packed(segments, selected))
        self.assertEqual(len(records), 1)
        self.assertEqual(mover._pool.borrowed, 0)
        self.assertTrue(all(not slot.pending for slot in mover._pool._slots))
        mover.close()

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
        self.addCleanup(mover.close)
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







if __name__ == "__main__":
    unittest.main()
