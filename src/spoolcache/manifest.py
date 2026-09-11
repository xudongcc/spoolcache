"""Shared metadata for authenticated token files and complete HMA coverage."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import itertools
from typing import Any, Mapping
from .errors import ManifestError

@dataclass(frozen=True, slots=True)
class PageSlice:
    """One contiguous layer/page range within an authenticated object."""

    group_index: int
    layer_name: str
    page_start: int
    page_count: int
    byte_length: int

    def __post_init__(self) -> None:
        values = (self.group_index, self.page_start, self.page_count, self.byte_length)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise ManifestError("page slice integers are malformed")
        if (
            self.group_index < 0
            or self.page_start < 0
            or self.page_count <= 0
            or self.byte_length <= 0
        ):
            raise ManifestError("page slice range is invalid")
        if not isinstance(self.layer_name, str) or not 0 < len(self.layer_name) <= 512:
            raise ManifestError("page slice layer is invalid")

    @property
    def sort_key(self) -> tuple[int, str, int]:
        return self.group_index, self.layer_name, self.page_start

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PageSlice":
        if set(raw) != {
            "group_index",
            "layer_name",
            "page_start",
            "page_count",
            "byte_length",
        }:
            raise ManifestError("page slice fields differ from schema")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class TokenSegments(Sequence):
    """Derive page slices from one shared immutable layout as they are consumed.

    Retaining a long chain must not retain one layer descriptor per file/layer.
    The file still authenticates the exact serialized geometry before this view
    is admitted; only its in-memory representation is shared.
    """

    layout: Any
    span_tokens: int
    chunk_tokens: int
    kind: str

    def __iter__(self):
        for group in self.layout.groups:
            if (group.reuse_policy == "full") != (self.kind == "data"):
                continue
            if self.kind == "data":
                start = (self.span_tokens - self.chunk_tokens) // group.logical_tokens_per_page
                count = self.chunk_tokens // group.logical_tokens_per_page
            else:
                start, count = 0, group.selected_page_count(self.span_tokens)
            for layer in group.layers:
                yield PageSlice(group.group_index, layer.name, start, count,
                                count * layer.page_size_bytes)

    def __len__(self):
        return sum(len(g.layers) for g in self.layout.groups
                   if (g.reuse_policy == "full") == (self.kind == "data"))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return next(itertools.islice(self, index, None))


@dataclass(frozen=True, slots=True)
class TokenFileDescriptor:
    key: str
    parent: str | None
    span_tokens: int
    segments: Sequence[PageSlice]
    byte_length: int
    sha256: str
    payload_offset: int
    metadata_digest: str
    kind: str = "data"
    block_sha256: tuple[str, ...] = ()

    @property
    def stored_length(self):
        return self.payload_offset + self.byte_length

@dataclass(frozen=True)
class TokenSnapshot:
    entry_id: str
    span_tokens: int
    profile: str
    layout_digest: str
    objects: tuple[TokenFileDescriptor, ...]
