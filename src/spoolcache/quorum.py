"""Bounded worker inventory reporting and scheduler all-rank quorum."""

from __future__ import annotations

import collections
import functools
import itertools
import threading
from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import ManifestError
from .config import inventory_capacity, INVENTORY_ENTRY_BYTES


_MAX_GENERATION_LENGTH = 128
_MAX_COUNTER = (1 << 63) - 1
MAX_INVENTORY_REPORTS = 4096
_GENERATION_HISTORY_LIMIT = 64


def _reporter_locked(method):
    @functools.wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


@dataclass(frozen=True)
class InventoryCheckpoint:
    sequence: int
    cycle: int
    index: int
    count: int
    held_count: int
    entries: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class InventoryDelta:
    sequence: int
    base_sequence: int
    added: tuple[tuple[str, int], ...]
    removed: tuple[str, ...]


@dataclass(frozen=True)
class WorkerInventoryReport:
    rank: int
    generation: str
    generation_epoch: int
    checkpoint: InventoryCheckpoint
    delta: InventoryDelta | None = None


class InventoryReporter:
    """Produces bounded rolling checkpoints and state deltas for one worker."""

    def __init__(
        self,
        *,
        rank: int,
        generation: str,
        generation_epoch: int,
        max_bytes: int,
        max_report_entries: int,
    ) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= MAX_INVENTORY_REPORTS
            or not isinstance(generation, str)
            or not generation
            or len(generation) > _MAX_GENERATION_LENGTH
            or isinstance(generation_epoch, bool)
            or not isinstance(generation_epoch, int)
            or generation_epoch < 0
            or generation_epoch > _MAX_COUNTER
        ):
            raise ValueError("worker inventory identity is invalid")
        for label, value in (
            ("inventory", max_bytes),
            ("report", max_report_entries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"worker {label} bound must be positive")
        self.rank = rank
        self.generation = generation
        self.generation_epoch = generation_epoch
        self.max_bytes = max_bytes
        inventory_capacity(max_bytes)
        self.max_report_entries = max_report_entries
        self._held: dict[str, int] = {}
        self._observed: dict[str, int] = {}
        self._dirty = False
        self._sequence = 0
        self._history: collections.deque[InventoryDelta] = collections.deque(maxlen=64)
        self._delta_cursor = 0
        self._checkpoint_items: tuple[tuple[str, int], ...] = ()
        self._checkpoint_sequence = 0
        self._checkpoint_cycle = 0
        self._checkpoint_index = 0
        self._lock = threading.RLock()

    @_reporter_locked
    def replace(self, entries: Mapping[str, int]) -> None:
        # ``scan_offers`` supplies newest entries first. Keep that bounded
        # subset while storing it oldest-to-newest so later additions can evict
        # one oldest reportable entry in O(1). Entries outside this catalog are
        # safe false negatives; they are never advertised to the scheduler.
        selected: list[tuple[str, int]] = []
        for item in entries.items():
            if (len(selected) + 1) * INVENTORY_ENTRY_BYTES > self.max_bytes:
                break
            selected.append(item)
        _validate_entries(selected)
        self._held = dict(reversed(selected))
        self._dirty = True

    @_reporter_locked
    def add(self, entry_id: str, span_tokens: int) -> None:
        _validate_entries(((entry_id, span_tokens),))
        previous = self._held.pop(entry_id, None)
        if (len(self._held) + 1) * INVENTORY_ENTRY_BYTES > self.max_bytes:
            self._held.pop(next(iter(self._held)))
        self._held[entry_id] = span_tokens
        self._dirty |= previous != span_tokens

    @_reporter_locked
    def remove(self, entry_id: str) -> None:
        self._dirty |= self._held.pop(entry_id, None) is not None

    @_reporter_locked
    def held_entry_ids(self) -> tuple[str, ...]:
        """Return the reporter's already-bounded local inventory keys."""

        return tuple(self._held)

    @_reporter_locked
    def startup(self) -> tuple[tuple[str, int], ...]:
        # The catalog is already memory-bounded. Never slice this handshake:
        # dropping one intermediate key can make a complete long prefix miss.
        inventory = tuple(sorted(self._held.items()))
        self._observed = dict(self._held)
        self._dirty = False
        self._checkpoint_items = inventory
        self._checkpoint_sequence = self._sequence
        return inventory

    @_reporter_locked
    def next_report(self, batch_size: int) -> WorkerInventoryReport:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
            or batch_size > self.max_report_entries
        ):
            raise ValueError("inventory batch size exceeds its fixed bound")
        if self._dirty and self._held != self._observed:
            # Withdraw every obsolete advertised value immediately. New keys
            # can wait in the bounded held catalog: their absence is a safe
            # false negative, and must not withdraw unrelated existing hits.
            # A changed span is removed now and re-added on a later report.
            removed = tuple(sorted(
                key for key, span in self._observed.items()
                if self._held.get(key) != span
            ))
            base = self._sequence
            if len(removed) <= batch_size:
                added = tuple(sorted(
                    (key, span) for key, span in self._held.items()
                    if key not in self._observed
                )[:batch_size - len(removed)])
                self._sequence += 1
                delta = InventoryDelta(
                    sequence=self._sequence,
                    base_sequence=base,
                    added=added,
                    removed=removed,
                )
                for key in removed:
                    self._observed.pop(key)
                self._observed.update(added)
            else:
                # Too many removals cannot wait while stale entries remain
                # admitted. A deliberate gap withdraws the entire rank until
                # a complete rolling checkpoint proves its replacement.
                self._sequence += 2
                delta = InventoryDelta(
                    sequence=self._sequence,
                    base_sequence=self._sequence - 1,
                    added=(),
                    removed=(),
                )
                self._history.clear()
                self._delta_cursor = 0
                self._observed = dict(self._held)
            self._history.append(delta)
            # The report which first observes a mutation must carry that
            # newest delta. Replaying an older retained delta here could leave
            # a just-withdrawn corrupt entry admitted for up to an entire
            # history rotation. Older deltas remain available on later calls;
            # if one was actually lost, the sequence gap safely withdraws the
            # rank until its rolling checkpoint completes.
            self._delta_cursor = len(self._history) - 1
            # A checkpoint represents exactly the emitted sequence. Pending
            # additions enter it only after their own bounded delta is sent.
            self._checkpoint_items = tuple(sorted(self._observed.items()))
            self._checkpoint_sequence = self._sequence
            self._checkpoint_cycle += 1
            self._checkpoint_index = 0

        # After removals, observed is an identical-value subset of held.
        # Keep draining queued additions without scanning unchanged inventory
        # on every idle report. Cancelled mutations need no new sequence.
        self._dirty = len(self._held) != len(self._observed)

        count = max(
            1,
            (len(self._checkpoint_items) + batch_size - 1) // batch_size,
        )
        index = self._checkpoint_index
        start = index * batch_size
        checkpoint = InventoryCheckpoint(
            sequence=self._checkpoint_sequence,
            cycle=self._checkpoint_cycle,
            index=index,
            count=count,
            held_count=len(self._checkpoint_items),
            entries=self._checkpoint_items[start : start + batch_size],
        )
        self._checkpoint_index += 1
        if self._checkpoint_index >= count:
            self._checkpoint_index = 0
            self._checkpoint_cycle += 1
        delta = None
        if self._history:
            history = tuple(self._history)
            delta = history[self._delta_cursor % len(history)]
            self._delta_cursor = (self._delta_cursor + 1) % len(history)
        return WorkerInventoryReport(
            rank=self.rank,
            generation=self.generation,
            generation_epoch=self.generation_epoch,
            checkpoint=checkpoint,
            delta=delta,
        )


@dataclass
class _PendingCheckpoint:
    sequence: int
    cycle: int
    count: int
    held_count: int
    pages: dict[int, tuple[tuple[str, int], ...]]
    entry_count: int = 0


class QuorumCatalog:
    """Scheduler catalog which exposes only entries held by every rank."""

    def __init__(
        self,
        *,
        expected_ranks: Iterable[int],
        max_bytes: int,
        max_report_entries: int,
    ) -> None:
        materialized_ranks = tuple(
            itertools.islice(expected_ranks, MAX_INVENTORY_REPORTS + 1)
        )
        if not materialized_ranks or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in materialized_ranks
        ) or (
            len(materialized_ranks) > MAX_INVENTORY_REPORTS
            or len(set(materialized_ranks)) != len(materialized_ranks)
            or sorted(materialized_ranks) != list(range(len(materialized_ranks)))
        ):
            raise ValueError("expected ranks are invalid")
        ranks = frozenset(materialized_ranks)
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("catalog max_bytes must be positive")
        if (
            isinstance(max_report_entries, bool)
            or not isinstance(max_report_entries, int)
            or max_report_entries <= 0
        ):
            raise ValueError("catalog report bound must be positive")
        self.expected_ranks = ranks
        self.max_bytes = max_bytes
        inventory_capacity(max_bytes)
        self.max_report_entries = max_report_entries
        self._generation: dict[int, tuple[str, int]] = {}
        self._generation_history: dict[
            int, collections.deque[tuple[str, int]]
        ] = {
            rank: collections.deque(maxlen=_GENERATION_HISTORY_LIMIT)
            for rank in ranks
        }
        self._sequences: dict[int, int] = {}
        self._rank_entries: dict[int, dict[str, int]] = {rank: {} for rank in ranks}
        self._entry_ranks: dict[str, set[int]] = {}
        self._pending: dict[int, _PendingCheckpoint] = {}
        self._desynchronized: set[int] = set(ranks)
        # An equal epoch with a different UUID cannot be ordered. Keep that
        # rank withdrawn until a strictly newer epoch arrives; otherwise a
        # delayed packet from either process could re-admit stale entries.
        self._generation_conflicts: set[int] = set()
        self._recency: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._generation_changes_total = 0
        self._quorum_count: int | None = None

    def _withdraw_rank(self, rank: int) -> None:
        self._quorum_count = None
        for entry_id in tuple(self._rank_entries[rank]):
            ranks = self._entry_ranks.get(entry_id)
            if ranks is not None:
                ranks.discard(rank)
                if not ranks:
                    self._entry_ranks.pop(entry_id, None)
                    self._recency.pop(entry_id, None)
        self._rank_entries[rank].clear()

    def _replace_rank(self, rank: int, entries: Mapping[str, int]) -> None:
        _validate_entries(entries.items())
        self._withdraw_rank(rank)
        self._rank_entries[rank] = dict(entries)
        for entry_id in entries:
            self._entry_ranks.setdefault(entry_id, set()).add(rank)
            self._touch(entry_id)
        self._desynchronized.discard(rank)
        self._enforce_bound()

    def _touch(self, entry_id: str) -> None:
        self._recency.pop(entry_id, None)
        self._recency[entry_id] = None

    def _enforce_bound(self) -> None:
        while len(self._entry_ranks) * INVENTORY_ENTRY_BYTES > self.max_bytes and self._recency:
            self._quorum_count = None
            entry_id, _ = self._recency.popitem(last=False)
            ranks = self._entry_ranks.pop(entry_id, set())
            for rank in ranks:
                self._rank_entries[rank].pop(entry_id, None)

    @staticmethod
    def _validate_generation_identity(generation: object, epoch: object) -> None:
        if (
            not isinstance(generation, str)
            or not generation
            or len(generation) > _MAX_GENERATION_LENGTH
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or epoch > _MAX_COUNTER
        ):
            raise ValueError("inventory generation identity is invalid")

    def _generation_disposition(
        self,
        rank: int,
        generation: str,
        generation_epoch: int,
    ) -> str:
        previous = self._generation.get(rank)
        if previous is None:
            return "initial"
        previous_generation, previous_epoch = previous
        if generation_epoch < previous_epoch:
            identity = (generation, generation_epoch)
            # A lower epoch is safely ignorable only when this catalog itself
            # previously observed that exact process identity. An unknown UUID
            # at a lower value can instead be a live worker after state loss or
            # snapshot rollback; retaining the old image would create ghost
            # quorum, so ambiguity withdraws the rank.
            if identity in self._generation_history[rank]:
                return "stale"
            return "conflict"
        if generation_epoch > previous_epoch:
            return "newer"
        if generation != previous_generation or rank in self._generation_conflicts:
            return "conflict"
        return "current"

    def _adopt_generation(
        self,
        rank: int,
        generation: str,
        generation_epoch: int,
    ) -> None:
        previous = self._generation.get(rank)
        identity = (generation, generation_epoch)
        if previous is not None and previous != identity:
            self._generation_changes_total += 1
            history = self._generation_history[rank]
            if previous not in history:
                history.append(previous)
        self._withdraw_rank(rank)
        self._generation[rank] = identity
        self._sequences[rank] = 0
        self._pending.pop(rank, None)
        self._desynchronized.add(rank)
        self._generation_conflicts.discard(rank)

    def _mark_generation_conflict(self, rank: int) -> None:
        if rank not in self._generation_conflicts:
            self._generation_changes_total += 1
        self._withdraw_rank(rank)
        self._pending.pop(rank, None)
        self._desynchronized.add(rank)
        self._generation_conflicts.add(rank)

    def apply_startup(
        self,
        *,
        rank: int,
        generation: str,
        generation_epoch: int,
        entries: Iterable[tuple[str, int]],
    ) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in self.expected_ranks
        ):
            raise ValueError("inventory report came from an unexpected rank")
        try:
            self._validate_generation_identity(generation, generation_epoch)
        except ValueError:
            # A packet attributable to this rank but carrying an unorderable
            # process identity cannot leave the previous worker image live.
            self._mark_generation_conflict(rank)
            raise
        disposition = self._generation_disposition(
            rank,
            generation,
            generation_epoch,
        )
        if disposition == "stale":
            return
        if disposition == "conflict":
            self._mark_generation_conflict(rank)
            return
        # A startup image is emitted once for a process generation. Replaying
        # it after deltas/checkpoints could roll the catalog backwards, so an
        # exact duplicate generation is idempotently ignored.
        if disposition == "current":
            return
        self._adopt_generation(rank, generation, generation_epoch)
        items = tuple(itertools.islice(entries, inventory_capacity(self.max_bytes) + 1))
        _validate_entries(items)
        if len(items) * INVENTORY_ENTRY_BYTES > self.max_bytes:
            raise ManifestError("startup inventory exceeds the catalog bound")
        materialized = dict(items)
        if len(materialized) != len(items):
            raise ManifestError("startup inventory contains duplicate entry IDs")
        self._replace_rank(rank, materialized)

    def apply_report(self, report: WorkerInventoryReport) -> None:
        rank = report.rank
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in self.expected_ranks
        ):
            raise ValueError("inventory report came from an unexpected rank")
        try:
            self._validate_generation_identity(
                report.generation,
                report.generation_epoch,
            )
        except ValueError:
            self._mark_generation_conflict(rank)
            raise
        disposition = self._generation_disposition(
            rank,
            report.generation,
            report.generation_epoch,
        )
        if disposition == "stale":
            return
        if disposition == "conflict":
            self._mark_generation_conflict(rank)
            return
        if disposition in {"initial", "newer"}:
            # Merely observing a newer, well-formed generation identity makes
            # the old worker image stale. Adopt it before validating the rest
            # of the report so a malformed first packet cannot let the older
            # process reappear later.
            self._adopt_generation(
                rank,
                report.generation,
                report.generation_epoch,
            )
        try:
            validate_inventory_report(
                report,
                max_bytes=self.max_bytes,
                max_report_entries=self.max_report_entries,
            )
        except ManifestError:
            self._withdraw_rank(rank)
            self._pending.pop(rank, None)
            self._desynchronized.add(rank)
            return
        delta = report.delta
        # A delta can advance only a synchronized base.  Once a sequence gap
        # or worker generation change withdraws a rank, accepting a later
        # delta would reconstruct state from a knowingly incomplete catalog.
        # Only a complete rolling checkpoint may re-admit that rank.
        if delta is not None and rank not in self._desynchronized:
            current = self._sequences.get(rank, 0)
            if delta.base_sequence == current and delta.sequence == current + 1:
                updated = dict(self._rank_entries[rank])
                for entry_id in delta.removed:
                    updated.pop(entry_id, None)
                for entry_id, span in delta.added:
                    updated[entry_id] = span
                self._sequences[rank] = delta.sequence
                self._replace_rank(rank, updated)
            elif delta.sequence > current:
                self._withdraw_rank(rank)
                self._desynchronized.add(rank)

        checkpoint = report.checkpoint
        if (
            rank not in self._desynchronized
            and checkpoint.sequence > self._sequences.get(rank, 0)
        ):
            # A checkpoint from a newer worker state without every intervening
            # delta proves that the scheduler catalog may contain stale
            # entries. Withdraw immediately; completing this checkpoint is the
            # only safe way to reconstruct the rank image.
            self._withdraw_rank(rank)
            self._desynchronized.add(rank)
        pending = self._pending.get(rank)
        if (
            pending is None
            or pending.cycle != checkpoint.cycle
            or pending.sequence != checkpoint.sequence
        ):
            pending = _PendingCheckpoint(
                sequence=checkpoint.sequence,
                cycle=checkpoint.cycle,
                count=checkpoint.count,
                held_count=checkpoint.held_count,
                pages={},
            )
            self._pending[rank] = pending
        if (
            checkpoint.count <= 0
            or not 0 <= checkpoint.index < checkpoint.count
            or checkpoint.count != pending.count
            or checkpoint.held_count != pending.held_count
        ):
            self._pending.pop(rank, None)
            self._withdraw_rank(rank)
            self._desynchronized.add(rank)
            return
        previous_page = pending.pages.get(checkpoint.index, ())
        projected_count = (
            pending.entry_count - len(previous_page) + len(checkpoint.entries)
        )
        if projected_count > pending.held_count:
            self._pending.pop(rank, None)
            self._withdraw_rank(rank)
            self._desynchronized.add(rank)
            return
        pending.pages[checkpoint.index] = checkpoint.entries
        pending.entry_count = projected_count
        if len(pending.pages) == pending.count:
            flattened = tuple(
                item for index in range(pending.count) for item in pending.pages[index]
            )
            current = self._sequences.get(rank, 0)
            if pending.sequence < current:
                # A newer valid delta already superseded this rolling image.
                pass
            elif len(flattened) != pending.held_count or len(dict(flattened)) != len(flattened):
                self._withdraw_rank(rank)
                self._desynchronized.add(rank)
            else:
                self._sequences[rank] = pending.sequence
                self._replace_rank(rank, dict(flattened))
            self._pending.pop(rank, None)

    def has_quorum(self, entry_id: str, span_tokens: int | None = None) -> bool:
        ranks = self._entry_ranks.get(entry_id, set())
        if ranks != self.expected_ranks:
            return False
        spans = {self._rank_entries[rank].get(entry_id) for rank in ranks}
        if len(spans) != 1:
            return False
        only = next(iter(spans))
        return only == span_tokens if span_tokens is not None else only is not None

    def longest(self, candidates: Iterable[tuple[int, str]]) -> tuple[int, str] | None:
        for span, entry_id in sorted(candidates, reverse=True):
            if self.has_quorum(entry_id, span):
                self._touch(entry_id)
                return span, entry_id
        return None

    @property
    def quorum_count(self) -> int:
        if self._quorum_count is None:
            self._quorum_count = sum(
                self.has_quorum(entry_id) for entry_id in self._entry_ranks
            )
        return self._quorum_count

    @property
    def desynchronized_ranks(self) -> frozenset[int]:
        return frozenset(self._desynchronized)

    @property
    def generation_changes_total(self) -> int:
        return self._generation_changes_total

    @property
    def ready_rank_count(self) -> int:
        return len(self.expected_ranks - self._desynchronized)

    @property
    def has_all_rank_identities(self) -> bool:
        return set(self._generation) == self.expected_ranks

    @property
    def is_ready(self) -> bool:
        return self.has_all_rank_identities and not self._desynchronized


def _validate_entries(entries: Iterable[tuple[str, int]]) -> None:
    for item in entries:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ManifestError("inventory entry is malformed")
        entry_id, span = item
        if (
            not isinstance(entry_id, str)
            or len(entry_id) != 64
            or any(char not in "0123456789abcdef" for char in entry_id)
        ):
            raise ManifestError("inventory entry ID is malformed")
        if (
            isinstance(span, bool)
            or not isinstance(span, int)
            or span <= 0
            or span > _MAX_COUNTER
        ):
            raise ManifestError("inventory token span is invalid")


def validate_inventory_report(
    report: WorkerInventoryReport,
    *,
    max_bytes: int,
    max_report_entries: int,
) -> None:
    for label, value in (
        ("inventory", max_bytes),
        ("report", max_report_entries),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} validation bound must be positive")
    if not isinstance(report, WorkerInventoryReport):
        raise ManifestError("inventory report type is malformed")
    if (
        isinstance(report.rank, bool)
        or not isinstance(report.rank, int)
        or report.rank < 0
        or report.rank >= MAX_INVENTORY_REPORTS
        or not isinstance(report.generation, str)
        or not report.generation
        or len(report.generation) > _MAX_GENERATION_LENGTH
        or isinstance(report.generation_epoch, bool)
        or not isinstance(report.generation_epoch, int)
        or report.generation_epoch < 0
        or report.generation_epoch > _MAX_COUNTER
    ):
        raise ManifestError("inventory generation is malformed")
    checkpoint = report.checkpoint
    if not isinstance(checkpoint, InventoryCheckpoint) or not isinstance(
        checkpoint.entries, tuple
    ):
        raise ManifestError("inventory checkpoint type is malformed")
    integer_fields = (
        checkpoint.sequence,
        checkpoint.cycle,
        checkpoint.index,
        checkpoint.count,
        checkpoint.held_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_fields
    ):
        raise ManifestError("checkpoint counters are malformed")
    if (
        checkpoint.sequence < 0
        or checkpoint.sequence > _MAX_COUNTER
        or checkpoint.cycle < 0
        or checkpoint.cycle > _MAX_COUNTER
        or checkpoint.count <= 0
        or checkpoint.count > max(1, checkpoint.held_count)
        or not 0 <= checkpoint.index < checkpoint.count
        or not 0 <= checkpoint.held_count <= inventory_capacity(max_bytes)
        or len(checkpoint.entries) > max_report_entries
        or len(checkpoint.entries) > checkpoint.held_count
    ):
        raise ManifestError("checkpoint bounds are invalid")
    _validate_entries(checkpoint.entries)
    delta = report.delta
    if delta is None:
        return
    if (
        not isinstance(delta, InventoryDelta)
        or not isinstance(delta.added, tuple)
        or not isinstance(delta.removed, tuple)
    ):
        raise ManifestError("inventory delta type is malformed")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (delta.sequence, delta.base_sequence)
    ) or (
        delta.base_sequence < 0
        or delta.base_sequence > _MAX_COUNTER
        or delta.sequence <= 0
        or delta.sequence > _MAX_COUNTER
        or delta.sequence != delta.base_sequence + 1
    ):
        raise ManifestError("delta counters are malformed")
    _validate_entries(delta.added)
    _validate_entries((entry_id, 1) for entry_id in delta.removed)
    added_ids = tuple(entry_id for entry_id, _ in delta.added)
    if (
        len(delta.added) + len(delta.removed) > max_report_entries
        or len(added_ids) != len(set(added_ids))
        or len(delta.removed) != len(set(delta.removed))
        or set(added_ids).intersection(delta.removed)
    ):
        raise ManifestError("inventory delta contains duplicate entry IDs")


__all__ = [
    "InventoryCheckpoint",
    "InventoryDelta",
    "InventoryReporter",
    "MAX_INVENTORY_REPORTS",
    "QuorumCatalog",
    "WorkerInventoryReport",
    "validate_inventory_report",
]
