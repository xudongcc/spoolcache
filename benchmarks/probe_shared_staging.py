#!/usr/bin/env python3
"""Prove whether one fixed host pool can serve O_DIRECT and CUDA together.

This is a qualification probe, not production allocation code.  SpoolCache
currently owns two independent 128 MiB pools per rank: anonymous mmap buffers
for aligned direct I/O and Torch pinned tensors for CUDA.  The experiment below
allocates only ``slot_count * slot_mib`` bytes, registers those same page-aligned
mappings with the CUDA runtime, reads/writes them with O_DIRECT, and copies them
to the GPU through Torch.

The probe intentionally fails loudly.  Falling back to buffered I/O or silently
allocating an extra pinned tensor would make the memory-saving conclusion false.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import json
import mmap
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
DIRECT_ALIGNMENT = 4096


@dataclass
class RegisteredSlot:
    """Resources sharing exactly one underlying anonymous mmap allocation."""

    index: int
    size: int
    mapping: mmap.mmap
    address: int
    tensor: Any
    registered: bool = True


def _load_cuda_runtime() -> ctypes.CDLL:
    """Load cudart without relying on one CUDA-major-specific filename."""

    candidates = (
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
    )
    last_error: OSError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            runtime = ctypes.CDLL(candidate)
            runtime.cudaHostRegister.argtypes = (
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            )
            runtime.cudaHostRegister.restype = ctypes.c_int
            runtime.cudaHostUnregister.argtypes = (ctypes.c_void_p,)
            runtime.cudaHostUnregister.restype = ctypes.c_int
            runtime.cudaGetErrorString.argtypes = (ctypes.c_int,)
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            return runtime
        except OSError as error:
            last_error = error
    raise RuntimeError("unable to load the CUDA runtime") from last_error


def _cuda_error(runtime: ctypes.CDLL, status: int) -> str:
    raw = runtime.cudaGetErrorString(status)
    return raw.decode("utf-8", "replace") if raw else f"CUDA error {status}"


def _register_slot(runtime: ctypes.CDLL, torch: Any, index: int, size: int) -> RegisteredSlot:
    mapping = mmap.mmap(-1, size)
    # from_buffer exposes the mmap's real base address without allocating or
    # copying.  Anonymous mmap is page aligned on Linux, but prove the exact
    # O_DIRECT requirement here instead of assuming it.
    anchor = ctypes.c_char.from_buffer(mapping)
    address = ctypes.addressof(anchor)
    del anchor
    if address % DIRECT_ALIGNMENT:
        mapping.close()
        raise RuntimeError(
            f"slot {index} address 0x{address:x} is not {DIRECT_ALIGNMENT}-byte aligned"
        )

    # cudaHostRegisterDefault=0.  Registration page-locks the existing mmap;
    # no second host allocation is allowed in this experiment.
    status = runtime.cudaHostRegister(ctypes.c_void_p(address), size, 0)
    if status:
        mapping.close()
        raise RuntimeError(
            f"cudaHostRegister(slot={index}) failed: {_cuda_error(runtime, status)}"
        )
    try:
        tensor = torch.frombuffer(mapping, dtype=torch.uint8, count=size)
    except BaseException:
        runtime.cudaHostUnregister(ctypes.c_void_p(address))
        mapping.close()
        raise
    return RegisteredSlot(index, size, mapping, address, tensor)


def _write_exact(descriptor: int, view: memoryview) -> None:
    cursor = 0
    while cursor < len(view):
        written = os.write(descriptor, view[cursor:])
        if written <= 0:
            raise OSError("direct write made no progress")
        cursor += written
        # A short, non-aligned direct write would make the next memory address
        # and file offset invalid.  Report it rather than changing I/O mode.
        if cursor < len(view) and cursor % DIRECT_ALIGNMENT:
            raise OSError("direct write stopped at a non-aligned offset")


def _read_exact(descriptor: int, view: memoryview) -> None:
    cursor = 0
    while cursor < len(view):
        count = os.readv(descriptor, [view[cursor:]])
        if count <= 0:
            raise EOFError("direct read ended before the staging slot was full")
        cursor += count
        if cursor < len(view) and cursor % DIRECT_ALIGNMENT:
            raise OSError("direct read stopped at a non-aligned offset")


def _repeated_byte_digest(value: int, size: int) -> str:
    """Compute the expected digest with at most one MiB of helper memory."""

    digest = hashlib.sha256()
    chunk = bytes((value,)) * min(MIB, size)
    remaining = size
    while remaining:
        count = min(len(chunk), remaining)
        digest.update(chunk[:count])
        remaining -= count
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Existing cache-rank root whose tmp/ directory is on the target NVMe",
    )
    parser.add_argument("--slot-mib", type=int, default=64)
    parser.add_argument("--slot-count", type=int, default=2)
    args = parser.parse_args()

    if args.slot_mib <= 0 or args.slot_count <= 0:
        raise SystemExit("slot size and count must be positive")
    slot_bytes = args.slot_mib * MIB
    if slot_bytes % DIRECT_ALIGNMENT:
        raise SystemExit("slot size must be 4096-byte aligned")
    temporary_directory = args.root.resolve() / "tmp"
    if not temporary_directory.is_dir():
        raise SystemExit(f"cache tmp directory does not exist: {temporary_directory}")
    if not hasattr(os, "O_DIRECT"):
        raise SystemExit("this platform does not expose O_DIRECT")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    # Initialize the primary context before calling cudaHostRegister through
    # ctypes, so both cudart and Torch refer to the same current CUDA device.
    torch.cuda.init()
    runtime = _load_cuda_runtime()
    slots: list[RegisteredSlot] = []
    probe_path = temporary_directory / f"shared-staging-{uuid.uuid4().hex}.part"
    # Keep every pattern representable as uint8 even when callers experiment
    # with more than the production default of two slots.
    patterns = tuple(1 + ((60 + index * 53) % 255) for index in range(args.slot_count))
    write_seconds = 0.0
    read_seconds = 0.0
    h2d_seconds: list[float] = []
    read_digests: list[str] = []

    try:
        for index in range(args.slot_count):
            slots.append(_register_slot(runtime, torch, index, slot_bytes))

        # Fill through the Torch view.  O_DIRECT below observes the same mmap
        # bytes; there is no hidden handoff buffer between these two APIs.
        for slot, pattern in zip(slots, patterns, strict=True):
            slot.tensor.fill_(pattern)

        descriptor = os.open(
            probe_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT,
            0o600,
        )
        try:
            started = time.perf_counter()
            for slot in slots:
                view = memoryview(slot.mapping)
                try:
                    _write_exact(descriptor, view)
                finally:
                    view.release()
            os.fsync(descriptor)
            write_seconds = time.perf_counter() - started
        finally:
            os.close(descriptor)

        for slot in slots:
            slot.tensor.zero_()

        descriptor = os.open(probe_path, os.O_RDONLY | os.O_DIRECT)
        try:
            started = time.perf_counter()
            for slot in slots:
                view = memoryview(slot.mapping)
                try:
                    _read_exact(descriptor, view)
                    read_digests.append(hashlib.sha256(view).hexdigest())
                finally:
                    view.release()
            read_seconds = time.perf_counter() - started
        finally:
            os.close(descriptor)

        expected_digests = [
            _repeated_byte_digest(pattern, slot_bytes) for pattern in patterns
        ]
        if read_digests != expected_digests:
            raise RuntimeError("O_DIRECT round-trip changed staging bytes")

        destination = torch.empty(slot_bytes, dtype=torch.uint8, device="cuda")
        # A small warm-up keeps CUDA context/JIT initialization out of the
        # measured copies.  It still reads from the externally registered mmap.
        destination[:DIRECT_ALIGNMENT].copy_(
            slots[0].tensor[:DIRECT_ALIGNMENT],
            non_blocking=True,
        )
        torch.cuda.current_stream().synchronize()
        for slot, pattern in zip(slots, patterns, strict=True):
            started = time.perf_counter()
            destination.copy_(slot.tensor, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            elapsed = time.perf_counter() - started
            h2d_seconds.append(elapsed)
            # Reduction checks all GPU bytes without materializing a second
            # slot-sized comparison tensor.
            actual_sum = int(destination.sum(dtype=torch.int64).item())
            if actual_sum != pattern * slot_bytes:
                raise RuntimeError(f"CUDA copy verification failed for slot {slot.index}")
        del destination

        total_mib = args.slot_count * args.slot_mib
        result = {
            "schema": "spoolcache-shared-staging-probe/v1",
            "qualified": True,
            "root": str(args.root.resolve()),
            "slot_mib": args.slot_mib,
            "slot_count": args.slot_count,
            "total_host_mib": total_mib,
            "addresses": [f"0x{slot.address:x}" for slot in slots],
            "alignment_bytes": DIRECT_ALIGNMENT,
            # Torch's allocator metadata may report False for externally
            # registered memory.  cudaHostRegister success is authoritative;
            # expose both facts so later implementation does not confuse them.
            "torch_is_pinned": [bool(slot.tensor.is_pinned()) for slot in slots],
            "direct_write_seconds": write_seconds,
            "direct_write_mib_s": total_mib / write_seconds,
            "direct_read_and_sha256_seconds": read_seconds,
            "direct_read_and_sha256_mib_s": total_mib / read_seconds,
            "h2d_seconds": h2d_seconds,
            "h2d_mib_s": [args.slot_mib / value for value in h2d_seconds],
            "sha256_verified": True,
            "cuda_sum_verified": True,
        }
        print(json.dumps(result, sort_keys=True))
    finally:
        # Never leave a 128 MiB probe file or page-locked mapping behind, even
        # when direct I/O, hashing, CUDA copy, or validation raises.
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass
        torch.cuda.synchronize()
        for slot in reversed(slots):
            if slot.registered:
                status = runtime.cudaHostUnregister(ctypes.c_void_p(slot.address))
                if status:
                    raise RuntimeError(
                        f"cudaHostUnregister(slot={slot.index}) failed: "
                        f"{_cuda_error(runtime, status)}"
                    )
                slot.registered = False
            slot.tensor = None
        gc.collect()
        for slot in slots:
            slot.mapping.close()


if __name__ == "__main__":
    main()
