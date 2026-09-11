"""Bounded Torch mover for opaque HMA manager pages.

Torch is imported lazily so the storage/control modules remain usable in the
CPU-only repository CI. The mover treats every manager page as opaque bytes
and never interprets a model's internal cache representation.
"""

from __future__ import annotations

import contextlib
import math
import threading
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from .buffers import AlignedBuffer
from .errors import FatalRestoreError, LayoutError, StoreBusyError
from .hma import HMALayout
from .identity import sha256_json
from .manifest import PageSlice, TokenSnapshot, TokenFileDescriptor
if TYPE_CHECKING:
    from .token_files import TokenFileStore

@dataclass
class _PinnedSlot:
    index: int
    size: int
    tensor: Any
    completion_event: Any
    pending: bool = False
    mapping: Any = None
    runtime: Any = None
    record_failed: bool = False

    @classmethod
    def allocate(cls, index, size, torch, runtime):
        slot = cls(index, size, None, None,
                   mapping=AlignedBuffer.allocate(index, size), runtime=runtime)
        registered = False
        try:
            slot.tensor = torch.frombuffer(slot.mapping.view, dtype=torch.uint8)
            result = runtime.cudaHostRegister(slot.tensor.data_ptr(), size, 0)
            if int(result) != 0:
                raise RuntimeError(f"CUDA host registration failed: {result}")
            registered = True
            slot.completion_event = torch.cuda.Event(blocking=False, interprocess=False)
            if not slot.tensor.is_pinned():
                raise LayoutError("registered staging memory is not CUDA pinned")
            return slot
        except BaseException:
            if registered:
                slot.close()
            else:
                slot.tensor = None
                slot.mapping.close()
            raise

    @property
    def view(self):
        return self.mapping.view

    def close(self):
        if self.mapping is None:
            return
        self.wait_until_reusable()
        result = self.runtime.cudaHostUnregister(self.tensor.data_ptr())
        if int(result) != 0:
            raise RuntimeError(f"CUDA host unregister failed: {result}")
        self.tensor = None
        self.mapping.close()
        self.mapping = None

    def wait_until_reusable(self) -> None:
        if self.record_failed:
            raise RuntimeError("CUDA staging event recording failed; terminal drain required")
        if self.pending:
            self.completion_event.synchronize()
            self.pending = False

    def record_use(self, stream: Any) -> None:
        if self.pending:
            raise RuntimeError("pinned staging slot is still owned by CUDA")
        self.pending = True
        self.record_failed = True
        self.completion_event.record(stream)
        self.record_failed = False


class PinnedTensorPool:
    """Fixed page-aligned mappings registered with CUDA, without a CPU copy.

    Ordinary file reads fill these credits directly. A terminal CUDA event
    guards every disk/CPU reuse and unregister.
    """

    def __init__(self, *, slot_bytes: int, slot_count: int) -> None:
        if slot_bytes <= 0 or slot_bytes % 4096:
            raise ValueError("pinned slot size must be a positive 4096-byte multiple")
        if slot_count <= 0:
            raise ValueError("pinned slot count must be positive")
        torch = _torch()
        self.slot_bytes = slot_bytes
        self.slot_count = slot_count
        slots = []
        runtime = torch.cuda.cudart()
        try:
            for index in range(slot_count):
                slots.append(_PinnedSlot.allocate(index, slot_bytes, torch, runtime))
        except BaseException:
            for slot in reversed(slots):
                slot.close()
            raise
        self._slots = tuple(slots)
        # FIFO reuse alternates slots.  While CUDA consumes slot N, the CPU can
        # fill slot N+1 from NVMe; a LIFO queue would immediately reacquire the
        # just-released slot and serialize the two stages.
        self._available = queue.Queue(slot_count)
        for slot in self._slots:
            self._available.put_nowait(slot)
        self._lock = threading.Lock()
        self._borrowed = 0
        self._closed = False

    @property
    def budget_bytes(self) -> int:
        return self.slot_bytes * self.slot_count

    @property
    def borrowed(self) -> int:
        with self._lock:
            return self._borrowed

    @contextlib.contextmanager
    def acquire(
        self,
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> Iterator[_PinnedSlot]:
        with self._lock:
            if self._closed:
                raise RuntimeError("pinned tensor pool is closed")
        try:
            if block:
                slot = self._available.get(block=True, timeout=timeout)
            else:
                slot = self._available.get_nowait()
        except queue.Empty as error:
            raise StoreBusyError("no pinned GPU staging slot is available") from error
        with self._lock:
            if self._closed:
                self._available.put_nowait(slot)
                raise RuntimeError("pinned tensor pool closed during acquisition")
            self._borrowed += 1
        try:
            slot.wait_until_reusable()
            yield slot
        finally:
            with self._lock:
                self._borrowed -= 1
            self._available.put_nowait(slot)

    def mark_stream_synchronized(self) -> None:
        """Release CUDA ownership after the caller synchronized its stream."""

        with self._lock:
            if self._borrowed:
                raise RuntimeError("cannot clear events while a slot is borrowed")
            for slot in self._slots:
                slot.pending = False
                slot.record_failed = False

    def close(self) -> None:
        with self._lock:
            if self._borrowed:
                raise RuntimeError("cannot close a pool with borrowed pinned tensors")
            if self._closed:
                return
            for slot in self._slots:
                slot.close()
            self._slots = ()
            self._closed = True


def bind_group_owned_kv_caches(
    layout: HMALayout,
    kv_caches: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Select physical group owners and prove any extra names are aliases.

    vLLM may add cross-layer KV-sharing names after it constructs the public
    ``KVCacheConfig`` passed to a connector.  Those names are absent from
    ``kv_cache_groups`` because they have no independent block table; the
    final ``register_kv_caches`` mapping points them at an owner's exact tensor
    view.  Persisting them as another group would duplicate one physical cache
    under the wrong block-id space.

    LMCache likewise excludes registered names not covered by an engine group.
    SpoolCache additionally proves that every excluded name is an exact alias
    of at least one group-owned view before omitting it from the manifest.
    """

    owned_names = tuple(
        layer.name for group in layout.groups for layer in group.layers
    )
    registered_names = tuple(kv_caches)
    if any(not isinstance(name, str) or not name for name in registered_names):
        raise LayoutError("registered KV layer names must be non-empty strings")
    owned_set = set(owned_names)
    registered_set = set(registered_names)
    missing = sorted(owned_set - registered_set)
    if missing:
        raise LayoutError(
            "registered KV layers differ from the discovered HMA layout: "
            f"missing={missing}, extra={sorted(registered_set - owned_set)}"
        )

    selected = {name: kv_caches[name] for name in owned_names}
    aliases: list[tuple[str, tuple[str, ...]]] = []
    for name in sorted(registered_set - owned_set):
        tensor = kv_caches[name]
        targets = tuple(
            owner
            for owner in owned_names
            if _same_tensor_view(tensor, selected[owner])
        )
        if not targets:
            raise LayoutError(
                f"registered KV layer {name!r} is neither group-owned nor an "
                "exact alias of a group-owned tensor"
            )
        aliases.append((name, targets))
    return selected, tuple(aliases)


def _same_tensor_view(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_identity = _tensor_view_identity(left)
    return left_identity is not None and left_identity == _tensor_view_identity(right)


def _tensor_view_identity(tensor: Any) -> tuple[object, ...] | None:
    """Return the public Torch facts that identify one exact storage view."""

    try:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.element_size()),
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
            int(tensor.storage_offset()),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


@dataclass(frozen=True)
class BoundPageLayer:
    group_index: int
    layer_name: str
    page_size_bytes: int
    pages: Any


def _contiguous_page_span(pages, physical_pages):
    """Borrow a proven dense GPU span while request blocks remain owned.

    A continuous page-id range over dense byte pages needs no index tensor or
    temporary GPU gather/scatter. Other layouts keep opaque indexed copies.
    The caller's CUDA completion still guards source/destination ownership.
    """
    if not physical_pages or not pages.is_contiguous():
        return None
    first = physical_pages[0]
    if first < 0 or any(value != first + index for index, value in enumerate(physical_pages)):
        return None
    return pages.narrow(0, first, len(physical_pages)).view(-1)


def _segment_byte_ranges(segments, start, length):
    """Intersect a file byte range with its ordered complete-page metadata."""
    if start < 0 or length <= 0 or start + length > sum(s.byte_length for s in segments):
        raise LayoutError("file byte range differs from page coverage")
    for segment in segments:
        if start >= segment.byte_length:
            start -= segment.byte_length
            continue
        size = min(length, segment.byte_length - start)
        yield segment, start, size
        length -= size
        if not length:
            return
        start = 0


class TorchPageMover:
    """Capture and restore strict HMA layouts with bounded staging memory."""

    def __init__(
        self,
        layout: HMALayout,
        kv_caches: Mapping[str, Any],
        *,
        slot_bytes: int,
        slot_count: int,
    ) -> None:
        self.layout = layout
        self._torch = _torch()
        expected = {
            layer.name for group in layout.groups for layer in group.layers
        }
        if set(kv_caches) != expected:
            missing = sorted(expected - set(kv_caches))
            extra = sorted(set(kv_caches) - expected)
            raise LayoutError(
                "registered KV layers differ from the discovered HMA layout: "
                f"missing={missing}, extra={extra}"
            )
        bound: dict[tuple[int, str], BoundPageLayer] = {}
        geometry: list[dict[str, object]] = []
        device_indexes: set[int] = set()
        for group in layout.groups:
            for layer in group.layers:
                tensor = kv_caches[layer.name]
                device = getattr(tensor, "device", None)
                if getattr(device, "type", None) != "cuda":
                    raise LayoutError(f"KV layer {layer.name} is not CUDA memory")
                device_indexes.add(int(device.index or 0))
                pages = manager_page_view(
                    tensor,
                    manager_blocks=layout.num_manager_blocks,
                    page_size_bytes=layer.page_size_bytes,
                    layer_name=layer.name,
                )
                bound[(group.group_index, layer.name)] = BoundPageLayer(
                    group_index=group.group_index,
                    layer_name=layer.name,
                    page_size_bytes=layer.page_size_bytes,
                    pages=pages,
                )
                element_size = int(tensor.element_size())
                tensor_shape = tuple(int(size) for size in tensor.shape)
                tensor_stride = tuple(int(stride) for stride in tensor.stride())
                page_stride = tuple(int(stride) for stride in pages.stride())
                rows_per_page = tensor_shape[0] // layout.num_manager_blocks
                geometry.append(
                    {
                        "group_index": group.group_index,
                        "layer_name": layer.name,
                        "dtype": str(tensor.dtype),
                        "element_size_bytes": element_size,
                        # Allocation capacity is intentionally normalized out;
                        # the remaining dimensions and byte strides fully
                        # describe how one persistent manager page is read.
                        "rows_per_manager_page": rows_per_page,
                        "tensor_tail_shape": tensor_shape[1:],
                        "tensor_row_stride_bytes": tensor_stride[0]
                        * element_size,
                        "tensor_tail_strides_bytes": tuple(
                            stride * element_size for stride in tensor_stride[1:]
                        ),
                        "tensor_storage_offset_bytes": int(
                            tensor.storage_offset()
                        )
                        * element_size,
                        "manager_page_size_bytes": layer.page_size_bytes,
                        "manager_page_stride_bytes": page_stride[0],
                        "manager_page_tail_shape": tuple(
                            int(size) for size in pages.shape[1:]
                        ),
                        "manager_page_tail_strides_bytes": page_stride[1:],
                    }
                )
        if len(device_indexes) != 1:
            raise LayoutError("all HMA page sources must share one CUDA device")
        self.device_index = next(iter(device_indexes))
        self._bound = bound
        self.geometry_digest = sha256_json(
            {
                "schema": "spoolcache-registered-tensor-layout/v1",
                "hma_layout_sha256": layout.digest,
                "layers": geometry,
            }
        )
        self._pool = PinnedTensorPool(
            slot_bytes=slot_bytes,
            slot_count=slot_count,
        )
        self._closed = False

    @property
    def pinned_budget_bytes(self) -> int:
        return self._pool.budget_bytes


    def _copy_page_bytes(self, layer, physical_pages, byte_start, host, *, to_device,
                         index_cache=None):
        """Copy a bounded byte range, including partial or oversized GPU pages."""
        page_bytes = layer.page_size_bytes
        first, offset = divmod(byte_start, page_bytes)
        count = (offset + host.numel() + page_bytes - 1) // page_bytes
        pages = physical_pages[first : first + count]
        if len(pages) != count:
            raise LayoutError("transfer page range is incomplete")
        dense = _contiguous_page_span(layer.pages, pages)
        if dense is not None:
            target = dense[offset : offset + host.numel()]
            if to_device:
                target.copy_(host, non_blocking=True)
            else:
                host.copy_(target, non_blocking=True)
            return

        # Partial head/tail pages are direct views. Only complete intervening
        # pages need indexed gather/scatter, bounded by this one host credit.
        cursor, page_index = 0, 0
        if offset:
            length = min(page_bytes - offset, host.numel())
            target = layer.pages[pages[0]].view(-1)[offset : offset + length]
            if to_device:
                target.copy_(host[:length], non_blocking=True)
            else:
                host[:length].copy_(target, non_blocking=True)
            cursor, page_index = length, 1
        whole = (host.numel() - cursor) // page_bytes
        if whole:
            ids = pages[page_index : page_index + whole]
            cache_key = (ids, layer.pages.device)
            if index_cache is not None and index_cache[0] == cache_key:
                indexes = index_cache[1]
            else:
                indexes = self._torch.tensor(ids, dtype=self._torch.long,
                                             device=layer.pages.device)
                if index_cache is not None:
                    index_cache[:] = [cache_key, indexes]
            length = whole * page_bytes
            piece = host[cursor : cursor + length]
            if to_device:
                staged = piece.to(device=layer.pages.device, non_blocking=True)
                layer.pages.index_copy_(0, indexes, staged.view(whole, 1, page_bytes))
            else:
                gathered = self._torch.index_select(layer.pages, 0, indexes).view(-1)
                piece.copy_(gathered, non_blocking=True)
            cursor += length
            page_index += whole
        if cursor < host.numel():
            target = layer.pages[pages[page_index]].view(-1)[:host.numel() - cursor]
            if to_device:
                target.copy_(host[cursor:], non_blocking=True)
            else:
                host[cursor:].copy_(target, non_blocking=True)

    def _capture_packed(self, segments, selected, *, byte_start=0, byte_length=None):
        """Borrow one pinned credit for a bounded range of one or more files."""
        total = sum(segment.byte_length for segment in segments)
        length = total - byte_start if byte_length is None else byte_length
        if not 0 < length <= self._pool.slot_bytes:
            raise LayoutError("capture range exceeds the fixed staging credit")
        with self._pool.acquire(block=True) as slot:
            cursor = 0
            stream = self._torch.cuda.current_stream(self.device_index)
            submitted = False
            index_cache = [None, None]
            try:
                for segment, offset, size in _segment_byte_ranges(segments, byte_start, length):
                    layer = self._bound[(segment.group_index, segment.layer_name)]
                    pages = selected[segment.group_index][
                        segment.page_start : segment.page_start + segment.page_count
                    ]
                    submitted = True
                    self._copy_page_bytes(
                        layer, pages, offset, slot.tensor[cursor : cursor + size],
                        to_device=False, index_cache=index_cache,
                    )
                    cursor += size
            finally:
                # No producer bytes or credit may escape before D2H completes,
                # including a failure after a previous fragment was submitted.
                if submitted:
                    slot.record_use(stream)
                    slot.wait_until_reusable()
            view = memoryview(slot.tensor[:cursor].numpy()).cast("B")
            try:
                yield view
            finally:
                view.release()

    def restore_entry(
        self,
        store: TokenFileStore,
        *,
        entry_id: str,
        span_tokens: int,
        block_tables: Sequence[Sequence[int]],
        expected_manifest_digest: str | None = None,
    ) -> TokenSnapshot:
        # Pin every object before releasing the namespace lock. The view stays
        # live through the final CUDA drain, while unrelated GC may progress.
        try:
            with store.restore_view(entry_id) as view:
                return self._restore_entry_view(
                    store, view=view, span_tokens=span_tokens,
                    block_tables=block_tables,
                    expected_manifest_digest=expected_manifest_digest)
        except FatalRestoreError:
            raise
        except Exception as error:
            raise FatalRestoreError("SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED") from error

    def _restore_entry_view(
        self,
        store: TokenFileStore,
        *,
        view: Any,
        span_tokens: int,
        block_tables: Sequence[Sequence[int]],
        expected_manifest_digest: str | None = None,
    ) -> TokenSnapshot:
        """Restore one authenticated object at a time with a single disk pass.

        The manifest and every object's immutable metadata are checked before
        placement starts. A bounded batch reuses pinned staging; each payload
        is authenticated before GPU placement.
        A failure after an earlier object was placed is fatal by
        contract, so the engine cannot continue with a partial transaction.
        """

        self._ensure_open()
        try:
            # Do not pre-read payloads here. The streaming receiver authenticates the
            # same bytes it transfers, avoiding a full verify pass followed by
            # a second full transfer pass.
            result = view.result
            if not result.is_hit or result.manifest is None:
                raise FatalRestoreError(
                    f"SPOOLCACHE_POST_ADMISSION_LOOKUP_FAILED:{result.reason}"
                )
            if result.manifest.span_tokens != span_tokens:
                raise FatalRestoreError("SPOOLCACHE_POST_ADMISSION_SPAN_MISMATCH")
            if (
                expected_manifest_digest is not None
                and result.manifest_digest != expected_manifest_digest
            ):
                raise FatalRestoreError("SPOOLCACHE_POST_ADMISSION_MANIFEST_CHANGED")
            self.layout.validate_manifest_coverage(result.manifest)
            selected = self.layout.select_physical_pages(block_tables, span_tokens)
            stream = self._torch.cuda.current_stream(self.device_index)
            submitted = False
            try:
                # A receiver can enqueue some layers before a later failure.
                # Always drain the whole attempt, including failed receivers.
                submitted = True
                store.stream_objects(
                    result.manifest.objects,
                    lambda descriptor: self._restore_object_receiver(
                        descriptor, selected, stream=stream), lease=view,
                    buffer_pool=self._pool,
                    on_batch_complete=lambda slot: slot.record_use(stream))
            finally:
                if submitted:
                    self._synchronize_restore_stream(stream)
            return result.manifest
        except FatalRestoreError:
            raise
        except Exception as error:
            raise FatalRestoreError(
                "SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED"
            ) from error

    @contextlib.contextmanager
    def _restore_object_receiver(
        self,
        descriptor: TokenFileDescriptor,
        selected: Sequence[Sequence[int]],
        *,
        stream: Any,
    ):
        payload_bytes = descriptor.byte_length
        received = 0
        index_cache = [None, None]

        def receive(chunk: memoryview) -> None:
            nonlocal received
            if not 0 < len(chunk) <= min(self._pool.slot_bytes, payload_bytes - received):
                raise LayoutError("token restore range differs from authenticated object")
            source = self._torch.frombuffer(chunk, dtype=self._torch.uint8)
            if not source.is_pinned():
                raise LayoutError("token restore source is not CUDA pinned")
            cursor = 0
            for segment, offset, size in _segment_byte_ranges(
                descriptor.segments, received, len(chunk)
            ):
                layer = self._bound[(segment.group_index, segment.layer_name)]
                pages = selected[segment.group_index][
                    segment.page_start : segment.page_start + segment.page_count
                ]
                self._copy_page_bytes(
                    layer, pages, offset, source[cursor : cursor + size],
                    to_device=True, index_cache=index_cache,
                )
                cursor += size
            received += len(chunk)

        # Each range has its own authenticated checksum in the immutable header.
        # The store marks the containing credit used before returning it; final
        # whole-file authentication and the outer CUDA drain precede completion.
        yield receive
        if received != payload_bytes:
            raise LayoutError("token restore received an incomplete object")

    def _synchronize_restore_stream(self, stream: Any) -> None:
        """One terminal barrier for every object submitted by one restore."""

        stream.synchronize()
        self._pool.mark_stream_synchronized()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Torch page mover is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._pool.close()
        self._closed = True

    def __enter__(self) -> "TorchPageMover":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def manager_page_view(
    tensor: Any,
    *,
    manager_blocks: int,
    page_size_bytes: int,
    layer_name: str,
) -> Any:
    """Expose complete physical manager pages on dimension zero."""

    torch = _torch()
    if manager_blocks <= 0 or page_size_bytes <= 0:
        raise LayoutError("manager page geometry must be positive")
    if tensor.dim() < 2:
        raise LayoutError(
            f"KV layer {layer_name} has unexpected shape {tuple(tensor.shape)}"
        )
    kernel_rows = int(tensor.shape[0])
    if kernel_rows % manager_blocks:
        raise LayoutError(
            f"KV layer {layer_name} rows do not divide manager block count"
        )
    rows_per_page = kernel_rows // manager_blocks
    element_size = int(tensor.element_size())
    logical_page_bytes = (
        rows_per_page
        * math.prod(int(size) for size in tensor.shape[1:])
        * element_size
    )
    manager_page_stride = int(tensor.stride(0)) * element_size * rows_per_page
    if manager_page_stride <= 0 or not (
        logical_page_bytes <= page_size_bytes <= manager_page_stride
    ):
        raise LayoutError(
            f"KV layer {layer_name} page geometry differs: "
            f"logical={logical_page_bytes}, physical={page_size_bytes}, "
            f"stride={manager_page_stride}"
        )
    try:
        byte_view = tensor.view(torch.uint8)
        storage_offset = int(byte_view.storage_offset())
        required = (
            storage_offset
            + (manager_blocks - 1) * manager_page_stride
            + page_size_bytes
        )
        if required > int(byte_view.untyped_storage().nbytes()):
            raise LayoutError("physical manager pages exceed tensor storage")
        return torch.as_strided(
            byte_view,
            size=(manager_blocks, 1, page_size_bytes),
            stride=(manager_page_stride, page_size_bytes, 1),
            storage_offset=storage_offset,
        )
    except LayoutError:
        raise
    except RuntimeError as error:
        raise LayoutError(
            f"KV layer {layer_name} physical page view is invalid"
        ) from error


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("TorchPageMover requires PyTorch") from error
    return torch


__all__ = [
    "BoundPageLayer",
    "PinnedTensorPool",
    "TorchPageMover",
    "bind_group_owned_kv_caches",
    "manager_page_view",
]
