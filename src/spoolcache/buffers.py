"""Bounded aligned buffers used by the NVMe data path."""

from __future__ import annotations

import contextlib
import mmap
import queue
import threading
from dataclasses import dataclass
from typing import Iterator

from .errors import StoreBusyError


@dataclass
class AlignedBuffer:
    """One anonymous-mmap buffer whose address is page aligned."""

    index: int
    size: int
    _mapping: mmap.mmap

    @classmethod
    def allocate(cls, index: int, size: int) -> "AlignedBuffer":
        return cls(index=index, size=size, _mapping=mmap.mmap(-1, size))

    @property
    def view(self) -> memoryview:
        return memoryview(self._mapping)

    def zero(self, start: int = 0, end: int | None = None) -> None:
        stop = self.size if end is None else end
        if not 0 <= start <= stop <= self.size:
            raise ValueError("buffer zero range is invalid")
        zeros = b"\x00" * min(1024 * 1024, stop - start)
        cursor = start
        while cursor < stop:
            count = min(len(zeros), stop - cursor)
            self._mapping[cursor : cursor + count] = zeros[:count]
            cursor += count

    def close(self) -> None:
        self._mapping.close()


class AlignedBufferPool:
    """A fixed-size, non-growing pool.

    Acquiring without credit raises ``StoreBusyError`` by default. Callers may
    explicitly request blocking behavior for a transaction which already owns
    the corresponding vLLM request blocks.
    """

    def __init__(self, *, slot_bytes: int, slot_count: int) -> None:
        if slot_bytes <= 0 or slot_bytes % mmap.PAGESIZE:
            raise ValueError("slot_bytes must be a positive page multiple")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self.slot_bytes = slot_bytes
        self.slot_count = slot_count
        self._slots = tuple(
            AlignedBuffer.allocate(index, slot_bytes) for index in range(slot_count)
        )
        self._available: queue.LifoQueue[AlignedBuffer] = queue.LifoQueue(slot_count)
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
    ) -> Iterator[AlignedBuffer]:
        with self._lock:
            if self._closed:
                raise RuntimeError("buffer pool is closed")
        try:
            if block:
                slot = self._available.get(block=True, timeout=timeout)
            else:
                slot = self._available.get_nowait()
        except queue.Empty as error:
            raise StoreBusyError("no aligned I/O buffer is available") from error
        with self._lock:
            if self._closed:
                self._available.put_nowait(slot)
                raise RuntimeError("buffer pool closed during acquisition")
            self._borrowed += 1
        try:
            yield slot
        finally:
            with self._lock:
                self._borrowed -= 1
            self._available.put_nowait(slot)

    def close(self) -> None:
        with self._lock:
            if self._borrowed:
                raise RuntimeError("cannot close a pool with borrowed buffers")
            if self._closed:
                return
            self._closed = True
        for slot in self._slots:
            slot.close()

    def __enter__(self) -> "AlignedBufferPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
