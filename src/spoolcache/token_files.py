"""Runtime-aligned prefix-key files: buffered I/O and bounded payload staging.

The embedded header is the publication record. Data files own full-KV intervals;
a separate state key owns one exact HMA boundary. Parent keys are validated by
following the chain; there is no child index, arena, or reference database.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import os
import stat
import struct
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict

from .config import STAGING_SLOT_BYTES, aligned_chunk_tokens
from .errors import ManifestError, StoreBusyError
from .identity import canonical_json
from .manifest import TokenFileDescriptor, TokenSegments, TokenSnapshot
from .leases import RestoreView
from .rank_store import (
    CapacityBatchReport,
    LookupResult,
    ManagedDiskUsage,
    ManifestOffer,
    RankStore,
    _cleanup_temporary_file,
    _fsync_directory,
    _is_digest,
    _managed_tree_bytes,
)

TOKEN_FILE_SCHEMA = "spoolcache-token-key-file/v3"
MAX_BATCH_CHUNKS = 16
MAX_HEADER_BYTES = 64 * 1024
MAX_TRANSFER_BLOCKS = 512
_FRAME = struct.Struct("<8sI32s")
_MAGIC = b"SPCTOK02"


def has_boundary_state(layout):
    return any(group.reuse_policy != "full" for group in layout.groups)


def boundary_state_key(prefix_key):
    """A separately inventoried, pinned file for one exact HMA boundary.

    The data key still identifies one runtime-aligned full-KV interval. This disjoint
    domain lets quorum/GC track the boundary state without pretending an
    earlier or later window snapshot belongs to that interval.
    """
    if not _is_digest(prefix_key):
        raise ManifestError("boundary prefix key is malformed")
    return hashlib.sha256(
        b"spoolcache-hma-boundary/v1\0" + bytes.fromhex(prefix_key)
    ).hexdigest()


def _positive(value, maximum, label):
    if type(value) is not int or not 0 < value <= maximum:
        raise ManifestError(f"invalid token file {label}")
    return value


def _write_vector(fd, parts):
    views = [memoryview(part) for part in parts]
    try:
        while views:
            count = os.writev(fd, views)
            if count <= 0:
                raise OSError("short token file write")
            while views and count >= len(views[0]):
                count -= len(views[0])
                views.pop(0).release()
            if views and count:
                previous = views[0]
                views[0] = previous[count:]
                previous.release()
    finally:
        for view in views:
            view.release()


def _read_into(fd, view):
    position = 0
    while position < len(view):
        with view[position:] as target:
            count = os.readv(fd, [target])
        if count <= 0:
            raise ManifestError("truncated token file")
        position += count


def _read_small(fd, size):
    data = bytearray(size)
    with memoryview(data) as view:
        _read_into(fd, view)
    return data


class TokenFileStore(RankStore):
    """Reuse rank control leases/tombstones; replace the complete data plane.

    One fixed CPU slot serves dedup authentication and scrub. GPU restore reads
    directly into the mover's two pinned slots. No whole-prefix payload list or
    write queue is retained; the OS page cache is outside these process bounds.
    """

    def __init__(
        self, root, *, layout, slot_bytes=STAGING_SLOT_BYTES, slot_count=1, **kwargs
    ):
        from .token_mover import TokenPageMover

        TokenPageMover.validate_layout(layout, slot_bytes)
        required = (
            "expected_deployment_digest",
            "expected_rank_digest",
            "expected_rank",
            "expected_topology_digest",
            "expected_profile",
            "expected_layout_digest",
        )
        if any(kwargs.get(name) is None for name in required):
            raise ValueError("token files require a complete rank identity")
        if (
            kwargs["expected_profile"] != layout.profile
            or kwargs["expected_layout_digest"] != layout.digest
        ):
            raise ValueError("token file layout binding differs")
        if not 0 < slot_bytes <= STAGING_SLOT_BYTES or slot_count != 1:
            raise ValueError("token file CPU staging is one bounded slot")
        self.layout = layout
        self.chunk_tokens = aligned_chunk_tokens(layout.alignment_tokens)
        self._binding = {
            name.removeprefix("expected_"): kwargs[name] for name in required
        }
        self._binding["storage_schema"] = TOKEN_FILE_SCHEMA
        self._binding["chunk_tokens"] = self.chunk_tokens
        self._binding["transfer_bytes"] = slot_bytes
        super().__init__(root, slot_bytes=slot_bytes, slot_count=1, **kwargs)

    def _initialize_payload_backend(self):
        # Deliberately no O_DIRECT probe, slot database, or io_uring engine.
        marker = self.root / "state" / "token-format.json"
        if marker.exists():
            from .rank_store import _read_small_json

            if _read_small_json(marker, maximum=4096) != self._binding:
                raise ValueError("token file root identity differs")
        else:
            # Do not adopt a legacy data root, even if caller reused its path.
            for name in ("manifests", "objects"):
                if next((self.root / name).iterdir(), None) is not None:
                    raise ValueError(
                        "token files cannot adopt an existing data namespace"
                    )
            self._atomic_small_write(marker, canonical_json(self._binding))

    def _manifest_path(self, entry_id):
        if not _is_digest(entry_id):
            raise ValueError("token key is malformed")
        return self.root / "manifests" / entry_id[:2] / f"{entry_id}.kv"

    def _segments(self, span, kind="data"):
        if kind not in ("data", "state"):
            raise ManifestError("unknown token file kind")
        return TokenSegments(self.layout, span, self.chunk_tokens, kind)

    def _header_payload(self, key, parent, span, digest, kind="data", block_sha256=None):
        _positive(span, (1 << 63) - 1, "span")
        if span % self.chunk_tokens or not _is_digest(key) or not _is_digest(digest):
            raise ManifestError("token key/interval/hash is invalid")
        if kind == "state":
            if (
                not has_boundary_state(self.layout)
                or not _is_digest(parent)
                or key != boundary_state_key(parent)
                or span % self.layout.alignment_tokens
            ):
                raise ManifestError("boundary state identity/alignment differs")
        elif kind == "data":
            if (span == self.chunk_tokens and parent is not None) or (
                span > self.chunk_tokens and not _is_digest(parent)
            ):
                raise ManifestError("token parent is invalid")
        else:
            raise ManifestError("unknown token file kind")
        segments = self._segments(span, kind)
        length = sum(x.byte_length for x in segments)
        blocks = (length + self._pool.slot_bytes - 1) // self._pool.slot_bytes
        _positive(blocks, MAX_TRANSFER_BLOCKS, "transfer block count")
        if block_sha256 is None:
            block_sha256 = ["0" * 64] * blocks
        if (
            not isinstance(block_sha256, (tuple, list))
            or len(block_sha256) != blocks
            or any(not _is_digest(value) for value in block_sha256)
            or (blocks == 1 and block_sha256[0] != digest)
        ):
            raise ManifestError("token transfer checksums differ from geometry")
        return {
            "schema": TOKEN_FILE_SCHEMA,
            "kind": kind,
            "binding": self._binding,
            "key": key,
            "parent": parent,
            "span_tokens": span,
            "segments": [asdict(x) for x in segments],
            "byte_length": length,
            "sha256": digest,
            "block_sha256": list(block_sha256),
        }

    def _encode_header(self, payload):
        encoded = canonical_json(payload)
        _positive(len(encoded), MAX_HEADER_BYTES, "header length")
        return (
            _FRAME.pack(_MAGIC, len(encoded), hashlib.sha256(encoded).digest())
            + encoded
        )

    def validate_prefix_headers(self, span):
        # Before GPU capture, prove the largest data header and required state
        # fit the file format. Chain length has no separate metadata budget.
        kinds = ("data", "state") if has_boundary_state(self.layout) else ("data",)
        for kind in kinds:
            payload = self._header_payload(
                "0" * 64 if kind == "data" else boundary_state_key("0" * 64),
                (None if span == self.chunk_tokens else "0" * 64)
                if kind == "data" else "0" * 64, span, "0" * 64, kind)
            self._encode_header(payload)

    def _open_chunk(self, key, *, allow_withdrawn=False):
        path = self._manifest_path(key)
        if not allow_withdrawn and self._inventory_is_withdrawn(key):
            raise ManifestError("token key withdrawn")
        if path.parent.is_symlink():
            raise ManifestError("token shard is a symlink")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ManifestError("token file is not regular")
            magic, size, checksum = _FRAME.unpack(_read_small(fd, _FRAME.size))
            if magic != _MAGIC:
                raise ManifestError("token file magic differs")
            _positive(size, MAX_HEADER_BYTES, "header length")
            encoded = _read_small(fd, size)
            if hashlib.sha256(encoded).digest() != checksum:
                raise ManifestError("token header checksum differs")
            try:
                payload = json.loads(encoded)
                if not isinstance(payload, dict):
                    raise TypeError("token header is not an object")
                expected = self._header_payload(
                    key,
                    payload["parent"],
                    payload["span_tokens"],
                    payload["sha256"],
                    payload["kind"],
                    payload["block_sha256"],
                )
                # Canonical comparison also distinguishes booleans from integers.
                if canonical_json(payload) != canonical_json(expected):
                    raise ValueError("token metadata/identity/geometry differs")
            except (
                ValueError,
                KeyError,
                TypeError,
                RecursionError,
                OverflowError,
            ) as error:
                raise ManifestError("invalid token metadata") from error
            length = _positive(
                payload["byte_length"], self._pool.slot_bytes * MAX_TRANSFER_BLOCKS,
                "payload length"
            )
            offset = _FRAME.size + size
            if metadata.st_size != offset + length:
                raise ManifestError("token stored length differs")
            desc = TokenFileDescriptor(
                key,
                payload["parent"],
                payload["span_tokens"],
                self._segments(payload["span_tokens"], payload["kind"]),
                length,
                payload["sha256"],
                offset,
                checksum.hex(),
                payload["kind"],
                tuple(payload["block_sha256"]),
            )
            return fd, desc
        except BaseException:
            os.close(fd)
            raise

    def _authenticate(self, fd, desc, slot):
        checksum = hashlib.sha256()
        remaining = desc.byte_length
        with slot.view as full:
            for expected in desc.block_sha256:
                length = min(remaining, self._pool.slot_bytes)
                with full[:length] as view:
                    _read_into(fd, view)
                    block_checksum = hashlib.sha256(view)
                    if block_checksum.hexdigest() != expected:
                        raise ManifestError("token transfer block checksum differs")
                    if len(desc.block_sha256) == 1:
                        checksum = block_checksum
                    else:
                        checksum.update(view)
                remaining -= length
        if remaining or checksum.hexdigest() != desc.sha256:
            raise ManifestError("token payload checksum differs")

    def valid_chunk(self, key, parent, span, kind="data"):
        """Full authentication before allowing a GPU-capture skip."""
        with self._exclusive():
            try:
                fd, desc = self._open_chunk(key)
                try:
                    if (desc.parent, desc.span_tokens, desc.kind) != (
                        parent,
                        span,
                        kind,
                    ):
                        raise ManifestError("token chain differs")
                    with self._pool.acquire(block=True) as slot:
                        self._authenticate(fd, desc, slot)
                finally:
                    os.close(fd)
                return desc
            except FileNotFoundError:
                return None
            except (OSError, ManifestError):
                self._quarantine_bad(key)
                return None

    def _quarantine_bad(self, key):
        if self._pins.busy(key):
            self._mark_inventory_withdrawn(key)
            self._record_withdrawal(key)
            return
        self._quarantine_manifest(self._manifest_path(key), reason="payload_checksum")

    def commit_chunk(self, key, parent, span, data, kind="data"):
        """Publish a borrowed in-memory file with one normal-path writev."""
        with self._exclusive(), memoryview(data) as view:
            hashes = []
            for start in range(0, len(view), self._pool.slot_bytes):
                with view[start : start + self._pool.slot_bytes] as piece:
                    hashes.append(hashlib.sha256(piece).hexdigest())
            payload = self._header_payload(
                key, parent, span,
                hashes[0] if len(hashes) == 1 else hashlib.sha256(view).hexdigest(),
                kind, hashes,
            )
            if view.nbytes != payload["byte_length"]:
                raise ManifestError("captured token payload differs from geometry")
            return self._publish_chunk(
                key, payload,
                lambda fd, header: _write_vector(fd, (header, view)),
            )

    def commit_stream(self, key, parent, span, pieces, kind="data"):
        """Consume fixed-size borrowed ranges without retaining the file payload.

        Reserve the bounded header, stream bytes once, then fill the checksums
        before the sole file fsync. A partial header/file is never published.
        The checksum table authenticates each later read before GPU placement.
        """
        with self._exclusive():
            payload = self._header_payload(key, parent, span, "0" * 64, kind)

            def write(fd, header):
                _write_vector(fd, (header,))
                checksum = hashlib.sha256()
                hashes = []
                remaining = payload["byte_length"]
                for piece in pieces:
                    with memoryview(piece) as view:
                        length = min(remaining, self._pool.slot_bytes)
                        if length <= 0 or view.nbytes != length:
                            raise ManifestError("captured transfer range differs from geometry")
                        hashes.append(hashlib.sha256(view).hexdigest())
                        if payload["byte_length"] > self._pool.slot_bytes:
                            checksum.update(view)
                        _write_vector(fd, (view,))
                        remaining -= length
                        self._checkpoint("token_after_payload_block")
                if remaining:
                    raise ManifestError("captured token stream is incomplete")
                payload["sha256"] = hashes[0] if len(hashes) == 1 else checksum.hexdigest()
                payload["block_sha256"] = hashes
                final_header = self._encode_header(payload)
                if len(final_header) != len(header):
                    raise ManifestError("streamed token header changed length")
                os.lseek(fd, 0, os.SEEK_SET)
                _write_vector(fd, (final_header,))

            return self._publish_chunk(key, payload, write)

    def _publish_chunk(self, key, payload, write):
        # Caller holds the namespace lock and all borrowed producer buffers.
        self._ensure_open()
        header = self._encode_header(payload)
        if self._manifest_path(key).exists():
            raise ManifestError("capture requires an absent or quarantined key")
        temporary = self.root / "tmp" / f"token-{uuid.uuid4().hex}.part"
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        published = False
        temporary_retired = False
        try:
            write(fd, header)
            os.fsync(fd)
            self._checkpoint("token_before_publish")
            path = self._manifest_path(key)
            self._ensure_shard_directory(path.parent)
            os.link(temporary, path)
            published = True
            self._checkpoint("token_after_link")
            _fsync_directory(path.parent)
            temporary.unlink()
            _fsync_directory(temporary.parent)
            temporary_retired = True
            self._clear_inventory_withdrawal(key)
        except BaseException as error:
            if published:
                try:
                    self._mark_inventory_withdrawn(key)
                    self._record_withdrawal(key)
                except Exception as secondary:  # noqa: BLE001 - preserve publication failure
                    error.add_note(f"token withdrawal failed: {type(secondary).__name__}")
            raise
        finally:
            os.close(fd)
            import sys

            if not temporary_retired:
                _cleanup_temporary_file(temporary, primary_error=sys.exception())
        fd, desc = self._open_chunk(key)
        os.close(fd)
        return desc

    def _chain(self, key):
        entry_id = key
        objects, expected_span = [], None
        while key is not None:
            try:
                fd, desc = self._open_chunk(key)
                os.close(fd)
            except FileNotFoundError:
                raise
            except (OSError, ManifestError):
                self._quarantine_bad(key)
                raise
            if desc.kind != "data":
                # A valid state key is not a restorable prefix by itself.
                # Reject the query/reference without quarantining that file.
                raise ManifestError("state file is not a token-chain node")
            if expected_span is not None and desc.span_tokens != expected_span:
                self._quarantine_bad(key)
                raise ManifestError("token chain is not consecutive")
            objects.append(desc)
            expected_span = desc.span_tokens - self.chunk_tokens
            key = desc.parent
        objects.reverse()
        if not objects or objects[0].span_tokens != self.chunk_tokens:
            raise ManifestError("token chain has no beginning")
        span = objects[-1].span_tokens
        if has_boundary_state(self.layout):
            state_key = boundary_state_key(entry_id)
            try:
                fd, state = self._open_chunk(state_key)
                os.close(fd)
                if (state.kind, state.parent, state.span_tokens) != (
                    "state",
                    entry_id,
                    span,
                ):
                    raise ManifestError("boundary state differs from token chain")
            except FileNotFoundError:
                raise
            except (OSError, ManifestError):
                self._quarantine_bad(state_key)
                raise
            objects.append(state)
        snapshot = TokenSnapshot(
            entry_id,
            span,
            self.layout.profile,
            self.layout.digest,
            tuple(objects),
        )
        self.layout.validate_manifest_coverage(snapshot)
        return snapshot

    def lookup(self, entry_id, *, verify_payloads=False):
        # Metadata-only queries do not refresh recency or release a tombstone.
        with self._exclusive():
            self._ensure_open()
            self._manifest_path(entry_id)
            try:
                snapshot = self._chain(entry_id)
                if verify_payloads:
                    for desc in snapshot.objects:
                        if (
                            self.valid_chunk(
                                desc.key, desc.parent, desc.span_tokens, desc.kind
                            )
                            is None
                        ):
                            return LookupResult(False, "token_payload")
                return LookupResult(
                    True, "hit", snapshot, snapshot.objects[-1].metadata_digest
                )
            except (OSError, ManifestError):
                return LookupResult(False, "token_chain")

    def touch(self, descriptors):
        with self._exclusive():
            # Like LMCache's reverse batch touch, the head becomes most recent.
            # File mtime is the recency source for GC, including after restart.
            for desc in reversed(descriptors):
                os.utime(
                    self._manifest_path(desc.key),
                    ns=(time.time_ns(), time.time_ns()),
                    follow_symlinks=False,
                )

    @contextmanager
    def restore_view(self, entry_id):
        if not self._restore_credits.acquire(blocking=False):
            raise StoreBusyError("token restore credits exhausted")
        pin, view = None, None
        try:
            with self._exclusive():
                result = self.lookup(entry_id)
                objects = result.manifest.objects if result.is_hit else ()
                # Locks name prefix keys, not payload hashes: identical bytes at
                # distinct token positions are independent file lifetimes.
                pin = self._pins.acquire(
                    tuple(x.key for x in objects)
                )
                self._active_restores += 1
                view = RestoreView(owner=self, descriptors=objects, result=result)
            yield view
            if result.is_hit:
                self.touch(objects)
        finally:
            if view is not None:
                view.active = False
                with self._exclusive():
                    self._active_restores -= 1
            if pin is not None:
                os.close(pin)
            self._restore_credits.release()

    def stream_objects(
        self, descriptors, receiver, *, lease, buffer_pool=None, on_batch_complete=None
    ):
        if (
            lease.owner is not self
            or not lease.active
            or tuple(descriptors) != lease.descriptors
        ):
            raise ValueError("token read requires its live, complete chain lease")
        pool = buffer_pool if buffer_pool is not None else self._pool
        if any(min(d.byte_length, self._pool.slot_bytes) > pool.slot_bytes for d in descriptors):
            raise ValueError("read credit is smaller than the file authentication unit")
        # A transfer batch contains at most 16 key descriptors. It does not
        # materialize 16 payloads. Two FIFO CUDA slots alternate and their events
        # gate reuse. Small files share one credit; larger files stream through
        # fixed authenticated ranges. Geometry can make a batch less than 16.
        cursor = 0
        while cursor < len(descriptors):
            if descriptors[cursor].byte_length > pool.slot_bytes:
                self._stream_large_object(
                    descriptors[cursor], receiver, pool, on_batch_complete
                )
                cursor += 1
                continue
            batch, batch_bytes = [], 0
            while cursor < len(descriptors) and len(batch) < MAX_BATCH_CHUNKS:
                desc = descriptors[cursor]
                if batch and batch_bytes + desc.byte_length > pool.slot_bytes:
                    break
                batch.append(desc)
                batch_bytes += desc.byte_length
                cursor += 1
            failed_key = None
            try:
                with pool.acquire(block=True) as slot:
                    try:
                        with slot.view as full:
                            offset = 0
                            for desc in batch:
                                fd = None
                                try:
                                    fd, current = self._open_chunk(desc.key)
                                    if current != desc:
                                        raise ManifestError("pinned token metadata changed")
                                    with full[offset : offset + desc.byte_length] as view:
                                        _read_into(fd, view)
                                        self._verify_buffer(desc, view)
                                        with receiver(desc) as receive:
                                            receive(view)
                                    offset += desc.byte_length
                                except (OSError, ManifestError):
                                    failed_key = desc.key
                                    raise
                                finally:
                                    if fd is not None:
                                        os.close(fd)
                    finally:
                        # Even a failed receiver may have submitted H2D. Its buffer
                        # cannot be reused until completion; outer mover drains too.
                        if on_batch_complete is not None:
                            on_batch_complete(slot)
            finally:
                if failed_key is not None:
                    # A writer may hold the namespace lock while waiting for a
                    # staging credit. Return our credit before waiting for that
                    # lock; the pool's CUDA event still gates its next reuse.
                    # The complete-chain pins remain owned by the restore view.
                    with self._exclusive():
                        self._quarantine_bad(failed_key)

    def _verify_buffer(self, desc, view):
        for index, expected in enumerate(desc.block_sha256):
            start = index * self._pool.slot_bytes
            with view[start : start + self._pool.slot_bytes] as piece:
                if hashlib.sha256(piece).hexdigest() != expected:
                    raise ManifestError("token transfer block checksum differs")
        if len(desc.block_sha256) > 1 and hashlib.sha256(view).hexdigest() != desc.sha256:
            raise ManifestError("token payload checksum differs")

    def _stream_large_object(self, desc, receiver, pool, on_batch_complete):
        fd, failed = None, False
        try:
            fd, current = self._open_chunk(desc.key)
            if current != desc:
                raise ManifestError("pinned token metadata changed")
            checksum = hashlib.sha256()
            remaining = desc.byte_length
            with receiver(desc) as receive:
                for expected in desc.block_sha256:
                    length = min(remaining, self._pool.slot_bytes)
                    with pool.acquire(block=True) as slot:
                        try:
                            with slot.view as full, full[:length] as view:
                                _read_into(fd, view)
                                if hashlib.sha256(view).hexdigest() != expected:
                                    raise ManifestError("token transfer block checksum differs")
                                checksum.update(view)
                                receive(view)
                        finally:
                            if on_batch_complete is not None:
                                on_batch_complete(slot)
                    remaining -= length
                if remaining or checksum.hexdigest() != desc.sha256:
                    raise ManifestError("token payload checksum differs")
        except (OSError, ManifestError):
            failed = True
            raise
        finally:
            if fd is not None:
                os.close(fd)
            # Every borrowed credit has returned before taking this lock, even
            # on a late checksum/receiver failure. Complete-chain pins remain.
            if failed:
                with self._exclusive():
                    self._quarantine_bad(desc.key)

    def iter_manifest_paths(self):
        # Fixed shard traversal never follows links and never retains all keys.
        for index in range(256):
            shard = self.root / "manifests" / f"{index:02x}"
            if shard.is_symlink():
                raise ManifestError("token shard is a symlink")
            try:
                with os.scandir(shard) as entries:
                    for entry in entries:
                        if (
                            entry.name.endswith(".kv")
                            and _is_digest(entry.name[:-3])
                            and entry.name.startswith(f"{index:02x}")
                        ):
                            yield shard / entry.name
            except FileNotFoundError:
                continue

    def scan_offers(self, limit):
        _positive(limit, (1 << 63) - 1, "inventory selection size")
        with self._exclusive():
            # Validate before selecting the bounded heap. Unknown/bad files do
            # not starve older valid keys. No quarantine during active scandir.
            def offers():
                for path in self.iter_manifest_paths():
                    try:
                        fd, desc = self._open_chunk(path.stem)
                        modified = os.fstat(fd).st_mtime_ns
                        os.close(fd)
                    except (OSError, ManifestError):
                        continue
                    yield ManifestOffer(
                        desc.key, desc.span_tokens, desc.metadata_digest, modified
                    )

            return tuple(
                heapq.nsmallest(
                    limit, offers(), key=lambda x: (x.span_tokens, x.entry_id)
                )
            )

    def evict(self, key):
        with self._exclusive():
            if self._pins.busy(key):
                return False
            path = self._manifest_path(key)
            if not path.exists():
                return False
            self._mark_inventory_withdrawn(key)
            self._record_withdrawal(key)
            path.unlink()
            _fsync_directory(path.parent)
            return True

    def maintain_capacity_batch(
        self, *, trigger_bytes, remaining_candidates=0, **kwargs
    ):
        if kwargs:
            raise TypeError(f"unsupported token GC arguments: {tuple(kwargs)}")
        _positive(trigger_bytes, (1 << 63) - 1, "GC trigger")
        if type(remaining_candidates) is not int or remaining_candidates < 0:
            raise ValueError("invalid token GC quota")
        with self._exclusive(blocking=False) as acquired:
            if not acquired:
                return None
            before = self.disk_usage_bytes()
            if not remaining_candidates and before < trigger_bytes:
                return CapacityBatchReport(before, before, 0, 0, 0, 0, 0)
            count = 0

            def candidates():
                nonlocal count
                for path in self.iter_manifest_paths():
                    try:
                        modified = path.lstat().st_mtime_ns
                    except FileNotFoundError:
                        continue
                    count += 1
                    if not self._pins.busy(path.stem):
                        yield (modified, path.stem)

            selected = heapq.nsmallest(4, candidates())
            quota = remaining_candidates or max(1, count // 5)
            attempted = removed = 0
            for _, key in selected[:quota]:
                attempted += 1
                removed += bool(self.evict(key))
            after = self.disk_usage_bytes()
            remaining = (
                max(0, quota - attempted) if len(selected) >= min(4, quota) else 0
            )
            return CapacityBatchReport(
                before,
                after,
                removed,
                removed,
                max(0, before - after),
                attempted,
                remaining,
            )

    def disk_usage_bytes(self):
        return _managed_tree_bytes(self.root / "manifests")

    def managed_disk_usage(self):
        return ManagedDiskUsage(
            self.disk_usage_bytes(),
            _managed_tree_bytes(self.root / "quarantine"),
            _managed_tree_bytes(self.root / "tmp"),
            _managed_tree_bytes(self.root / "state"),
            _managed_tree_bytes(self.root / "objects"),
            _managed_tree_bytes(self.root / ".spoolcache-root"),
        )

    def recover(self):
        # Only our own abandoned temporary names, serialized against writers.
        removed = 0
        with self._exclusive():
            while True:
                with os.scandir(self.root / "tmp") as entries:
                    batch = [
                        entry.name
                        for entry in itertools.islice(
                            (
                                x
                                for x in entries
                                if x.name.startswith("token-")
                                and x.name.endswith(".part")
                            ),
                            64,
                        )
                    ]
                if not batch:
                    break
                for name in batch:
                    (self.root / "tmp" / name).unlink(missing_ok=True)
                    removed += 1
            if removed:
                _fsync_directory(self.root / "tmp")
        return removed
