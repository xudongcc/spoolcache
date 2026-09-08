"""Crash-consistent counters for failures that terminate their worker."""

from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from pathlib import Path
from typing import Iterable

from .identity import canonical_json
from .telemetry import MAX_METRIC_VALUE, METRIC_DEFINITIONS, MetricKind


EVENT_JOURNAL_SCHEMA = "spoolcache-events/v1"
EVENT_JOURNAL_NAME = "spoolcache-events.json"
_MAX_EVENT_JOURNAL_BYTES = 64 * 1024
_PERSISTENT_METRICS = frozenset(
    {
        "spoolcache_post_admission_failure_total",
        "spoolcache_quarantined_entries_total",
        "spoolcache_scrub_shutdown_failures_total",
    }
)


class PersistentEventJournal:
    """Persist a finite set of crash-path counter totals in one rank root.

    A post-admission failure calls ``os._exit`` and cannot rely on another
    vLLM stats collection cycle.  Persisting just its bounded phase counter
    lets the replacement TP group export the event after restart.  No request
    or cache-entry identity is accepted by this format.
    """

    def __init__(self, state_directory: str | os.PathLike[str]) -> None:
        self.state_directory = Path(state_directory)
        if not self.state_directory.is_absolute():
            raise ValueError("event journal state directory must be absolute")
        if self.state_directory.is_symlink():
            raise ValueError("event journal state directory cannot be a symlink")
        try:
            metadata = self.state_directory.stat()
        except OSError as error:
            raise ValueError("event journal state directory is unavailable") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("event journal state path is not a directory")
        self.path = self.state_directory / EVENT_JOURNAL_NAME
        if self.path.is_symlink():
            raise ValueError("event journal cannot be a symlink")
        self._lock = threading.Lock()
        # The rank has one journal owner. Recover only this component's exact
        # atomic-write names before opening its live file; never sweep state/.
        _cleanup_abandoned_temporaries(self.state_directory)
        self._totals = self._load()

    @staticmethod
    def _metric_key(
        name: object, labels: Iterable[object]
    ) -> tuple[str, tuple[str, ...]]:
        if not isinstance(name, str) or name not in _PERSISTENT_METRICS:
            raise ValueError("event journal metric is not crash-persistent")
        definition = METRIC_DEFINITIONS[name]
        if definition.kind is not MetricKind.COUNTER:
            raise ValueError("event journal metric is not a counter")
        materialized = tuple(labels)
        if (
            any(not isinstance(label, str) for label in materialized)
            or materialized not in definition.allowed_labels
        ):
            raise ValueError("event journal labels are outside the bounded vocabulary")
        return name, materialized  # type: ignore[return-value]

    def _load(self) -> dict[tuple[str, tuple[str, ...]], int]:
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise ValueError("event journal cannot be opened") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("event journal is not a regular file")
            size = metadata.st_size
            if size <= 0 or size > _MAX_EVENT_JOURNAL_BYTES:
                raise ValueError("event journal size is invalid")
            encoded = bytearray()
            while len(encoded) < size:
                chunk = os.read(descriptor, size - len(encoded))
                if not chunk:
                    break
                encoded.extend(chunk)
            if len(encoded) != size:
                raise ValueError("event journal is truncated")
        finally:
            os.close(descriptor)
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("event journal is malformed") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "totals"}
            or payload.get("schema") != EVENT_JOURNAL_SCHEMA
            or not isinstance(payload.get("totals"), list)
        ):
            raise ValueError("event journal schema is malformed")
        totals: dict[tuple[str, tuple[str, ...]], int] = {}
        for raw in payload["totals"]:
            if not isinstance(raw, dict) or set(raw) != {"name", "labels", "value"}:
                raise ValueError("event journal record is malformed")
            labels = raw["labels"]
            if not isinstance(labels, list):
                raise ValueError("event journal labels are malformed")
            key = self._metric_key(raw["name"], labels)
            value = raw["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MAX_METRIC_VALUE
            ):
                raise ValueError("event journal counter is malformed")
            if key in totals:
                raise ValueError("event journal contains duplicate counters")
            totals[key] = value
        return totals

    def _write(self) -> None:
        payload = {
            "schema": EVENT_JOURNAL_SCHEMA,
            "totals": [
                {"name": name, "labels": list(labels), "value": value}
                for (name, labels), value in sorted(self._totals.items())
            ],
        }
        encoded = canonical_json(payload) + b"\n"
        if len(encoded) > _MAX_EVENT_JOURNAL_BYTES:
            raise ValueError("event journal exceeds its fixed byte bound")
        temporary = self.state_directory / f".{EVENT_JOURNAL_NAME}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            cursor = 0
            view = memoryview(encoded)
            try:
                while cursor < len(view):
                    written = os.write(descriptor, view[cursor:])
                    if written <= 0:
                        raise OSError("event journal write made no progress")
                    cursor += written
            finally:
                view.release()
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        try:
            os.replace(temporary, self.path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        directory_descriptor = os.open(
            self.state_directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def increment(
        self,
        name: str,
        labels: Iterable[str] = (),
        *,
        amount: int = 1,
    ) -> None:
        key = self._metric_key(name, labels)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("event journal increment must be a positive integer")
        with self._lock:
            previous = self._totals.get(key, 0)
            if amount > MAX_METRIC_VALUE - previous:
                raise ValueError("event journal counter exceeds its fixed bound")
            self._totals[key] = previous + amount
            try:
                self._write()
            except BaseException:
                if previous:
                    self._totals[key] = previous
                else:
                    self._totals.pop(key, None)
                raise

    def totals(self) -> dict[tuple[str, tuple[str, ...]], int]:
        with self._lock:
            return dict(self._totals)


def _cleanup_abandoned_temporaries(state_directory: Path) -> int:
    prefix = f".{EVENT_JOURNAL_NAME}."
    suffix = ".tmp"
    removed = 0
    with os.scandir(state_directory) as entries:
        for entry in entries:
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            nonce = name[len(prefix) : -len(suffix)]
            if (
                len(nonce) != 32
                or any(character not in "0123456789abcdef" for character in nonce)
            ):
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("event journal temporary is not a regular file")
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                continue
            removed += 1
    if removed:
        directory_descriptor = os.open(
            state_directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return removed


__all__ = [
    "EVENT_JOURNAL_NAME",
    "EVENT_JOURNAL_SCHEMA",
    "PersistentEventJournal",
]
