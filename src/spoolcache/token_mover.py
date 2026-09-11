"""Bounded, ordinary-I/O mover for the prefix-key file experiment."""

from contextlib import closing

from .config import STAGING_SLOT_BYTES, aligned_chunk_tokens
from .errors import LayoutError, ManifestError
from .gpu import TorchPageMover
from .prefix import PrefixDigest
from .token_files import (
    MAX_BATCH_CHUNKS,
    MAX_TRANSFER_BLOCKS,
    TokenSnapshot,
    boundary_state_key,
    has_boundary_state,
)


class TokenPageMover(TorchPageMover):
    @staticmethod
    def validate_layout(layout, slot_bytes=STAGING_SLOT_BYTES):
        if type(slot_bytes) is not int or not 0 < slot_bytes <= STAGING_SLOT_BYTES:
            raise LayoutError("invalid fixed transfer credit")
        quantum = aligned_chunk_tokens(layout.alignment_tokens)
        if not layout.groups:
            raise LayoutError("HMA layout is empty")
        size = sum(
            (quantum // group.logical_tokens_per_page) * layer.page_size_bytes
            for group in layout.groups
            if group.reuse_policy == "full"
            for layer in group.layers
        )
        if not 0 < size <= slot_bytes * MAX_TRANSFER_BLOCKS:
            raise LayoutError(
                "aligned full-KV file exceeds the transfer metadata bound"
            )
        if has_boundary_state(layout):
            # State does not scale with full-prefix length. Bound the largest
            # boundary representable by this file protocol before allocating.
            # Sliding state saturates at its window; recurrent and scratch
            # state each retain one page, independently of total prefix length.
            largest_window = max(g.reuse_window_tokens or g.logical_tokens_per_page
                                 for g in layout.groups)
            maximum = ((largest_window + quantum - 1) // quantum) * quantum
            state_size = sum(
                group.selected_page_count(maximum) * layer.page_size_bytes
                for group in layout.groups
                if group.reuse_policy != "full"
                for layer in group.layers
            )
            if not 0 < state_size <= slot_bytes * MAX_TRANSFER_BLOCKS:
                raise LayoutError("HMA boundary state exceeds the transfer metadata bound")

    def __init__(self, layout, kv_caches, *, slot_bytes, slot_count):
        self.validate_layout(layout, slot_bytes)
        super().__init__(
            layout, kv_caches, slot_bytes=slot_bytes, slot_count=slot_count
        )

    def commit_keys(self, store, *, prefixes, block_tables):
        self._ensure_open()
        quantum = aligned_chunk_tokens(self.layout.alignment_tokens)
        prefixes = tuple(prefixes)
        if (
            not prefixes
            or any(
                not isinstance(p, PrefixDigest)
                or p.span_tokens != (i + 1) * quantum
                for i, p in enumerate(prefixes)
            )
        ):
            raise ManifestError("save requires every consecutive runtime-aligned key")
        if store.layout != self.layout:
            raise LayoutError("store and mover layouts differ")
        self.validate_layout(self.layout, self._pool.slot_bytes)
        store.validate_prefix_headers(prefixes[-1].span_tokens)
        selected = self.layout.select_physical_pages(
            block_tables, prefixes[-1].span_tokens
        )
        objects = []
        size = sum(x.byte_length for x in store._segments(quantum))
        batch_count = max(1, min(MAX_BATCH_CHUNKS, self._pool.slot_bytes // size))
        for start in range(0, len(prefixes), batch_count):
            batch = prefixes[start : start + batch_count]
            missing, resolved = [], {}
            with store._exclusive():
                for index, prefix in enumerate(batch, start):
                    parent = prefixes[index - 1].digest if index else None
                    existing = store.valid_chunk(
                        prefix.digest, parent, prefix.span_tokens
                    )
                    if existing is not None:
                        resolved[prefix.digest] = existing
                    else:
                        missing.append(
                            (prefix, parent, store._segments(prefix.span_tokens))
                        )
                if missing and size > self._pool.slot_bytes:
                    prefix, parent, parts = missing[0]
                    resolved[prefix.digest] = self._commit_streamed(
                        store, prefix.digest, parent, prefix.span_tokens,
                        parts, selected, "data",
                    )
                elif missing:
                    segments = tuple(
                        segment for _, _, parts in missing for segment in parts
                    )
                    # Only absent chunks enter D2H. The yielded view owns one
                    # fixed credit through all file publications in this batch.
                    generator = self._capture_packed(segments, selected)
                    with closing(generator):
                        data = next(generator)
                        offset = 0
                        with memoryview(data) as view:
                            for prefix, parent, parts in missing:
                                length = sum(x.byte_length for x in parts)
                                with view[offset : offset + length] as payload:
                                    resolved[prefix.digest] = store.commit_chunk(
                                        prefix.digest,
                                        parent,
                                        prefix.span_tokens,
                                        payload,
                                    )
                                offset += length
                            if offset != len(view):
                                raise LayoutError(
                                    "captured token batch differs from geometry"
                                )
                objects.extend(resolved[p.digest] for p in batch)
        if has_boundary_state(self.layout):
            tail = prefixes[-1]
            key = boundary_state_key(tail.digest)
            with store._exclusive():
                state = store.valid_chunk(key, tail.digest, tail.span_tokens, "state")
                if state is None:
                    parts = store._segments(tail.span_tokens, "state")
                    # Only the live boundary's non-full pages are captured.
                    # Earlier data keys do not imply earlier window/state hits.
                    if sum(p.byte_length for p in parts) > self._pool.slot_bytes:
                        state = self._commit_streamed(
                            store, key, tail.digest, tail.span_tokens, parts, selected, "state"
                        )
                    else:
                        generator = self._capture_packed(parts, selected)
                        with closing(generator):
                            state = store.commit_chunk(
                                key, tail.digest, tail.span_tokens, next(generator), "state"
                            )
                objects.append(state)
        snapshot = TokenSnapshot(
            prefixes[-1].digest,
            prefixes[-1].span_tokens,
            self.layout.profile,
            self.layout.digest,
            tuple(objects),
        )
        self.layout.validate_manifest_coverage(snapshot)
        store.touch(objects)
        return snapshot

    def _commit_streamed(self, store, key, parent, span, parts, selected, kind):
        step = store._pool.slot_bytes
        if step > self._pool.slot_bytes:
            raise LayoutError("file authentication unit exceeds GPU staging credit")

        def capture():
            length = sum(part.byte_length for part in parts)
            for start in range(0, length, step):
                # The nested generator holds exactly one borrowed credit until
                # commit_stream has hashed and written this byte range.
                with closing(self._capture_packed(
                    parts, selected, byte_start=start,
                    byte_length=min(step, length - start),
                )) as iterator:
                    yield next(iterator)

        with closing(capture()) as iterator:
            return store.commit_stream(key, parent, span, iterator, kind)
