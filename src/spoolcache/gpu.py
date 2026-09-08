"""Bounded Torch mover for opaque HMA manager pages.

Torch is imported lazily so the storage/control modules remain usable in the
CPU-only repository CI. The mover treats every manager page as opaque bytes
and never interprets a model's internal cache representation.
"""

from __future__ import annotations

import contextlib
import math
import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .errors import FatalRestoreError, LayoutError, StoreBusyError
from .hma import HMALayout
from .identity import sha256_json
from .manifest import ObjectDescriptor, RankManifest
from .store import ManifestStore, ObjectSource


@dataclass
class _PinnedSlot:
    index: int
    size: int
    tensor: Any
    completion_event: Any
    pending: bool = False

    def wait_until_reusable(self) -> None:
        if self.pending:
            self.completion_event.synchronize()
            self.pending = False

    def record_use(self, stream: Any) -> None:
        if self.pending:
            raise RuntimeError("pinned staging slot is still owned by CUDA")
        self.completion_event.record(stream)
        self.pending = True


class PinnedTensorPool:
    """One fixed allocation of Torch pinned host tensors.

    CUDA only requires these tensors to be pinned.  O_DIRECT alignment belongs
    to the separate I/O pool; a future shared-pool fast path must prove both
    properties rather than assuming Torch's pinned allocator is page aligned.
    """

    def __init__(self, *, slot_bytes: int, slot_count: int) -> None:
        if slot_bytes <= 0 or slot_bytes % 4096:
            raise ValueError("pinned slot size must be a positive 4096-byte multiple")
        if slot_count <= 0:
            raise ValueError("pinned slot count must be positive")
        torch = _torch()
        self.slot_bytes = slot_bytes
        self.slot_count = slot_count
        self._slots = tuple(
            _PinnedSlot(
                index=index,
                size=slot_bytes,
                tensor=torch.empty(
                    slot_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                ),
                completion_event=torch.cuda.Event(
                    blocking=False,
                    interprocess=False,
                ),
            )
            for index in range(slot_count)
        )
        # FIFO reuse alternates slots.  While CUDA consumes slot N, the CPU can
        # fill slot N+1 from NVMe; a LIFO queue would immediately reacquire the
        # just-released slot and serialize the two stages.
        self._available: queue.Queue[_PinnedSlot] = queue.Queue(slot_count)
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

    def close(self) -> None:
        with self._lock:
            if self._borrowed:
                raise RuntimeError("cannot close a pool with borrowed pinned tensors")
            if self._closed:
                return
            self._closed = True
        self._slots = ()


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
            "registered KV layers differ from the HMA profile: "
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
                "registered KV layers differ from the HMA profile: "
                f"missing={missing}, extra={extra}"
            )
        largest_page = max(
            layer.page_size_bytes
            for group in layout.groups
            for layer in group.layers
        )
        if slot_bytes < largest_page:
            raise LayoutError(
                f"staging slot {slot_bytes} is smaller than page {largest_page}"
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

    def capture_sources(
        self,
        block_tables: Sequence[Sequence[int]],
        span_tokens: int,
    ) -> Iterator[ObjectSource]:
        """Yield bounded object sources in manifest order.

        ``ManifestStore.commit`` consumes each source before requesting the
        next one, so a slot remains borrowed only through one object write.
        """

        self._ensure_open()
        selected = self.layout.select_physical_pages(block_tables, span_tokens)
        for group, physical_pages in zip(
            self.layout.groups, selected, strict=True
        ):
            for layer in group.layers:
                pages_per_object = self._pool.slot_bytes // layer.page_size_bytes
                for page_start in range(0, len(physical_pages), pages_per_object):
                    page_ids = physical_pages[
                        page_start : page_start + pages_per_object
                    ]
                    yield ObjectSource(
                        group_index=group.group_index,
                        layer_name=layer.name,
                        page_start=page_start,
                        page_count=len(page_ids),
                        chunks=self._capture_chunks(
                            self._bound[(group.group_index, layer.name)],
                            page_ids,
                        ),
                    )

    def _capture_chunks(
        self,
        layer: BoundPageLayer,
        physical_pages: Sequence[int],
    ) -> Iterator[memoryview]:
        torch = self._torch
        payload_bytes = len(physical_pages) * layer.page_size_bytes
        with self._pool.acquire(block=True) as slot:
            indexes = torch.tensor(
                physical_pages,
                dtype=torch.long,
                device=layer.pages.device,
            )
            gathered = torch.index_select(layer.pages, 0, indexes).contiguous()
            flattened = gathered.view(-1)
            if flattened.numel() != payload_bytes:
                raise LayoutError("captured HMA payload has an unexpected size")
            slot.tensor[:payload_bytes].copy_(flattened, non_blocking=False)
            array = slot.tensor[:payload_bytes].numpy()
            view = memoryview(array).cast("B")
            try:
                yield view
            finally:
                view.release()

    def commit(
        self,
        store: ManifestStore,
        *,
        entry_id: str,
        deployment_identity_digest: str,
        rank_identity_digest: str,
        span_tokens: int,
        physical_rank: int,
        topology_digest: str,
        block_tables: Sequence[Sequence[int]],
        created_at_unix_ns: int | None = None,
    ) -> RankManifest:
        manifest = store.commit(
            entry_id=entry_id,
            deployment_identity_digest=deployment_identity_digest,
            rank_identity_digest=rank_identity_digest,
            span_tokens=span_tokens,
            physical_rank=physical_rank,
            topology_digest=topology_digest,
            profile=self.layout.profile,
            layout_digest=self.layout.digest,
            sources=self.capture_sources(block_tables, span_tokens),
            created_at_unix_ns=created_at_unix_ns,
        )
        self.layout.validate_manifest_coverage(manifest)
        return manifest

    def restore_entry(
        self,
        store: ManifestStore,
        *,
        entry_id: str,
        span_tokens: int,
        block_tables: Sequence[Sequence[int]],
        expected_manifest_digest: str | None = None,
    ) -> RankManifest:
        """Restore one authenticated object at a time with a single disk pass.

        The manifest and every object's immutable metadata are checked before
        placement starts.  ``_restore_object`` then reads each payload exactly
        once into request-owned staging, authenticates it, and only then places
        that object.  A failure after an earlier object was placed is fatal by
        contract, so the engine cannot continue with a partial transaction.
        """

        self._ensure_open()
        try:
            # Do not pre-read payloads here.  _restore_object authenticates the
            # same bytes it transfers, avoiding a full verify pass followed by
            # a second full transfer pass.
            result = store.lookup(entry_id, verify_payloads=False)
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
                for descriptor in result.manifest.objects:
                    targets = selected[descriptor.group_index][
                        descriptor.page_start : descriptor.page_start
                        + descriptor.page_count
                    ]
                    if len(targets) != descriptor.page_count:
                        raise LayoutError("restore target page range is incomplete")
                    self._restore_object(store, descriptor, targets, stream=stream)
                    submitted = True
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

    def _restore_object(
        self,
        store: ManifestStore,
        descriptor: ObjectDescriptor,
        physical_pages: Sequence[int],
        *,
        stream: Any,
    ) -> None:
        layer = self._bound[(descriptor.group_index, descriptor.layer_name)]
        payload_bytes = descriptor.byte_length
        if payload_bytes > self._pool.slot_bytes:
            raise LayoutError("manifest object exceeds the fixed staging slot")
        with self._pool.acquire(block=True) as slot:
            array = slot.tensor[:payload_bytes].numpy()
            destination = memoryview(array).cast("B")
            cursor = 0

            def receive(chunk: memoryview) -> None:
                nonlocal cursor
                end = cursor + len(chunk)
                if end > payload_bytes:
                    raise LayoutError("object stream exceeds its manifest length")
                destination[cursor:end] = chunk
                cursor = end

            try:
                store.stream_object(descriptor, receive)
                if cursor != payload_bytes:
                    raise LayoutError("object stream is shorter than its manifest")
            finally:
                destination.release()
            torch = self._torch
            indexes = torch.tensor(
                physical_pages,
                dtype=torch.long,
                device=layer.pages.device,
            )
            staged = slot.tensor[:payload_bytes].to(
                device=layer.pages.device,
                non_blocking=True,
            )
            staged = staged.view(
                descriptor.page_count,
                *tuple(int(size) for size in layer.pages.shape[1:]),
            )
            layer.pages.index_copy_(0, indexes, staged)
            # Keep the pinned source immutable until this stream reaches the
            # event.  FIFO pool reuse gives the other slot to the CPU first.
            slot.record_use(stream)

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
