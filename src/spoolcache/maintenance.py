"""Resumable, rate-bounded deep scrub for one rank-local cache store."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .identity import canonical_json
from .errors import ManifestError
from .manifest import ObjectDescriptor, RankManifest
from .store import (
    ManifestStore,
    _fsync_directory,
    _is_canonical_object_relative_path,
    _is_digest,
)


SCRUB_STATE_SCHEMA = "spoolcache-scrub-state/v1"
SCRUB_REQUEST_SCHEMA = "spoolcache-scrub-request/v1"
SCRUB_DATABASE_NAME = "deep-scrub.sqlite3"
SCRUB_REQUEST_NAME = "deep-scrub-request.json"

# These are internal production bounds rather than model/profile options.
# One step can authenticate at most one 64 MiB production object beyond its
# exact budget, and the scheduler spaces steps by the bytes actually read.
SCRUB_BYTES_PER_SECOND = 64 * 1024 * 1024
SCRUB_STEP_BYTES = 64 * 1024 * 1024
SCRUB_STEP_ITEMS = 64
SCRUB_POLL_SECONDS = 1.0
SCRUB_STARTUP_DELAY_SECONDS = 60.0
SCRUB_CYCLE_INTERVAL_SECONDS = 6 * 60 * 60.0
SCRUB_SHUTDOWN_TIMEOUT_SECONDS = 5.0
SCRUB_SHUTDOWN_SCHEMA = "spoolcache-scrub-shutdown/v1"

_MAX_STATE_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 4096
_MAX_COUNTER = (1 << 63) - 1
_PHASES = frozenset(
    {
        "idle",
        "snapshot_manifests",
        "manifests",
        "snapshot_objects",
        "objects",
        "snapshot_tmp",
        "tmp",
    }
)
_SNAPSHOT_PHASES = frozenset(
    {"snapshot_manifests", "snapshot_objects", "snapshot_tmp"}
)
_SNAPSHOT_QUARANTINE_BATCH = 64
_REQUEST_STATUSES = frozenset({"", "authenticated", "absent", "quarantined"})
logger = logging.getLogger(__name__)


class _UnsupportedScrubDatabase(ValueError):
    """A newer or foreign database must survive for an explicit upgrade."""


class _ScrubCancelled(Exception):
    """Internal cooperative stop at a fixed staging-chunk boundary."""


@dataclasses.dataclass(frozen=True)
class ScrubStepReport:
    namespace_items_scanned: int = 0
    payload_bytes: int = 0
    manifests_examined: int = 0
    manifests_authenticated: int = 0
    objects_authenticated: int = 0
    entries_quarantined: int = 0
    objects_quarantined: int = 0
    orphan_objects_removed: int = 0
    orphan_bytes_removed: int = 0
    temporary_files_removed: int = 0
    inventory_released: int = 0
    cycle_completed: bool = False
    request_completed: bool = False
    request_status: str = ""


@dataclasses.dataclass(frozen=True)
class ScrubStatus:
    schema: str
    cycle: int
    phase: str
    cycle_started_unix_ns: int
    manifest_cursor: str
    active_manifest: str
    next_object_index: int
    object_cursor: str
    last_completed_unix_ns: int
    last_cycle_manifests_authenticated: int
    last_cycle_objects_authenticated: int
    last_cycle_payload_bytes: int
    last_cycle_orphan_objects_removed: int
    last_cycle_orphan_bytes_removed: int
    last_cycle_temporary_files_removed: int
    target_entry: str
    target_next_object_index: int
    last_request_entry: str
    last_request_status: str
    last_request_completed_unix_ns: int


@dataclasses.dataclass(frozen=True)
class ScrubShutdownReport:
    schema: str
    status: str
    thread_alive: bool
    waited_seconds: float


@dataclasses.dataclass
class _State:
    schema: str = SCRUB_STATE_SCHEMA
    cycle: int = 0
    phase: str = "idle"
    cycle_started_unix_ns: int = 0
    manifest_cursor: str = ""
    active_manifest: str = ""
    active_manifest_digest: str = ""
    next_object_index: int = 0
    object_cursor: str = ""
    current_manifests_authenticated: int = 0
    current_objects_authenticated: int = 0
    current_payload_bytes: int = 0
    current_orphan_objects_removed: int = 0
    current_orphan_bytes_removed: int = 0
    current_temporary_files_removed: int = 0
    last_completed_unix_ns: int = 0
    last_cycle_manifests_authenticated: int = 0
    last_cycle_objects_authenticated: int = 0
    last_cycle_payload_bytes: int = 0
    last_cycle_orphan_objects_removed: int = 0
    last_cycle_orphan_bytes_removed: int = 0
    last_cycle_temporary_files_removed: int = 0
    target_nonce: str = ""
    target_entry: str = ""
    target_manifest_observed: bool = False
    target_manifest_digest: str = ""
    target_next_object_index: int = 0
    target_requested_unix_ns: int = 0
    last_request_nonce: str = ""
    last_request_entry: str = ""
    last_request_status: str = ""
    last_request_completed_unix_ns: int = 0

    def status(self) -> ScrubStatus:
        return ScrubStatus(
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(ScrubStatus)
            }
        )


class DeepScrubber:
    """Incrementally authenticate manifests and immutable objects.

    Progress and a disk-backed reference set are committed after every object.
    A process restart therefore resumes the active manifest without trusting a
    partial in-memory digest.  Orphan deletion begins only after a complete
    manifest namespace pass and rechecks current references under the store's
    cross-process maintenance lock.
    """

    def __init__(self, store: ManifestStore) -> None:
        self.store = store
        required_identity = (
            store.expected_deployment_digest,
            store.expected_rank_digest,
            store.expected_rank,
            store.expected_topology_digest,
            store.expected_profile,
            store.expected_layout_digest,
        )
        if any(value is None for value in required_identity):
            raise ValueError(
                "deep scrub requires a complete expected identity for one rank"
            )
        self.state_directory = store.root / "state"
        self.database_path = self.state_directory / SCRUB_DATABASE_NAME
        self.request_path = self.state_directory / SCRUB_REQUEST_NAME
        self._lock = threading.RLock()
        self._snapshot_phase = ""
        self._snapshot_cycle = 0
        self._snapshot_cutoff_unix_ns = 0
        self._snapshot_iterator: Iterator[Path] | None = None
        self._snapshot_quarantines: list[tuple[Path, str]] = []
        # Always acquire the in-process scrub lock before the store-wide file
        # lock. Target completion uses the same order; reversing it in status
        # polling would deadlock a scheduler thread at request acknowledgement.
        with self._lock, self.store._exclusive():
            _cleanup_atomic_temporaries(
                self.state_directory,
                target_name=SCRUB_REQUEST_NAME,
            )
            connection: sqlite3.Connection | None = None
            try:
                try:
                    connection = self._connect()
                    self._initialize(connection)
                    self._validate_resume_work(
                        connection,
                        self._load_state(connection),
                    )
                except _UnsupportedScrubDatabase:
                    raise
                except (sqlite3.DatabaseError, ValueError) as error:
                    # Scrub metadata never makes a cache entry visible. Losing
                    # its cursor can only postpone authentication/collection,
                    # so malformed current-schema state is safe to quarantine
                    # and rebuild. A foreign user_version is preserved above
                    # because silently downgrading it could destroy an
                    # upgrade's state.
                    if connection is not None:
                        connection.close()
                        connection = None
                    logger.warning(
                        "spoolcache: rebuilding invalid deep scrub state: %s",
                        type(error).__name__,
                    )
                    self._quarantine_database()
                    connection = self._connect()
                    self._initialize(connection)
            finally:
                if connection is not None:
                    connection.close()

    def status(self) -> ScrubStatus:
        with self._lock, self.store._exclusive():
            connection = self._connect()
            try:
                return self._load_state(connection).status()
            finally:
                connection.close()

    def repair_invalid_state(self) -> bool:
        """Rebuild only malformed current-schema auxiliary state.

        The method never repairs cache manifests or payloads and never accepts
        an unknown database version. It is safe after an arbitrary scrub-step
        exception: a healthy database is left untouched and returns ``False``.
        """

        with self._lock, self.store._exclusive():
            connection: sqlite3.Connection | None = None
            try:
                try:
                    connection = self._connect()
                    self._initialize(connection)
                    self._validate_resume_work(
                        connection,
                        self._load_state(connection),
                    )
                    return False
                except _UnsupportedScrubDatabase:
                    raise
                except (sqlite3.DatabaseError, ValueError) as error:
                    if connection is not None:
                        connection.close()
                        connection = None
                    logger.warning(
                        "spoolcache: quarantining invalid live scrub state: %s",
                        type(error).__name__,
                    )
                    self._quarantine_database()
                    connection = self._connect()
                    self._initialize(connection)
                    self._reset_snapshot_stream()
                    return True
            finally:
                if connection is not None:
                    connection.close()

    def _validate_resume_work(
        self,
        connection: sqlite3.Connection,
        state: _State,
    ) -> None:
        if state.phase == "manifests" and not state.active_manifest:
            _next_work_path(
                connection,
                table="manifest_work",
                root=self.store.root,
                after=state.manifest_cursor,
                namespace="manifests",
            )
        elif state.phase == "objects":
            _next_work_path(
                connection,
                table="object_work",
                root=self.store.root,
                after=state.object_cursor,
                namespace="objects",
            )
        elif state.phase == "tmp":
            _next_temporary_work_path(
                connection,
                self.store.root / "tmp",
                after=state.object_cursor,
            )

    def start_cycle(self) -> int:
        with self._lock, self.store._exclusive():
            connection = self._connect()
            try:
                state = self._load_state(connection)
                if state.phase != "idle":
                    return state.cycle
                self._reset_snapshot_stream()
                state.cycle += 1
                state.phase = "snapshot_manifests"
                state.cycle_started_unix_ns = time.time_ns()
                state.manifest_cursor = ""
                state.active_manifest = ""
                state.active_manifest_digest = ""
                state.next_object_index = 0
                state.object_cursor = ""
                state.current_manifests_authenticated = 0
                state.current_objects_authenticated = 0
                state.current_payload_bytes = 0
                state.current_orphan_objects_removed = 0
                state.current_orphan_bytes_removed = 0
                state.current_temporary_files_removed = 0
                connection.execute("DELETE FROM refs")
                connection.execute("DELETE FROM manifest_work")
                connection.execute("DELETE FROM object_work")
                connection.execute("DELETE FROM temporary_work")
                self._save_state(connection, state)
                connection.commit()
                return state.cycle
            finally:
                connection.close()

    def request(self, entry_id: str) -> str:
        return request_deep_scrub(self.store.root, entry_id)

    def close(self) -> None:
        """Release an in-progress directory stream without changing progress."""

        with self._lock:
            self._reset_snapshot_stream()

    def has_pending_request(self) -> bool:
        try:
            self.request_path.lstat()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return True

    def step(
        self,
        *,
        payload_budget_bytes: int,
        item_budget: int = SCRUB_STEP_ITEMS,
        on_payload_read: Callable[[int], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ScrubStepReport:
        _positive_int(payload_budget_bytes, "scrub payload budget")
        _positive_int(item_budget, "scrub item budget")
        if on_payload_read is not None and not callable(on_payload_read):
            raise TypeError("scrub payload observer must be callable")
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("scrub cancellation probe must be callable")
        with self._lock:
            connection = self._connect()
            try:
                state = self._load_state(connection)
                if self._snapshot_iterator is not None and (
                    self._snapshot_phase != state.phase
                    or self._snapshot_cycle != state.cycle
                    or self._snapshot_cutoff_unix_ns
                    != state.cycle_started_unix_ns
                ):
                    # Another cooperating scrubber may have advanced or
                    # replaced the durable cycle between local steps.
                    self._reset_snapshot_stream()
                self._ingest_request(connection, state)
                if state.target_entry:
                    return self._step_target(
                        connection,
                        state,
                        payload_budget_bytes=payload_budget_bytes,
                        item_budget=item_budget,
                        on_payload_read=on_payload_read,
                    )
                if state.phase == "idle":
                    return ScrubStepReport()
                return self._step_cycle(
                    connection,
                    state,
                    payload_budget_bytes=payload_budget_bytes,
                    item_budget=item_budget,
                    on_payload_read=on_payload_read,
                    cancel_requested=cancel_requested,
                )
            finally:
                connection.close()

    def _step_target(
        self,
        connection: sqlite3.Connection,
        state: _State,
        *,
        payload_budget_bytes: int,
        item_budget: int,
        on_payload_read: Callable[[int], None] | None,
    ) -> ScrubStepReport:
        path = self.store._manifest_path(state.target_entry)
        if not state.target_manifest_observed:
            try:
                path.lstat()
            except FileNotFoundError:
                return self._complete_request(connection, state, "absent")
            state.target_manifest_observed = True
            self._save_state(connection, state)
            connection.commit()
        try:
            manifest, encoded = self.store._read_manifest_file(
                path,
                expected_entry_id=state.target_entry,
                probe_objects=False,
            )
        except FileNotFoundError:
            # Once the manifest was durably observed, disappearance means a
            # previous scrub attempt or concurrent safe maintenance removed
            # it. Preserve that fail-closed result across a crash between
            # quarantine and request acknowledgement.
            return self._complete_request(connection, state, "quarantined")
        except OSError:
            quarantined = self.store._quarantine_manifest(
                path,
                reason="manifest_io",
            )
            if not quarantined and path.exists():
                raise RuntimeError("target manifest could not be quarantined")
            return self._complete_request(
                connection,
                state,
                "quarantined",
                entries_quarantined=int(quarantined),
            )
        except ManifestError:
            quarantined = self.store._quarantine_manifest(
                path,
                reason="manifest_validation",
            )
            if not quarantined and path.exists():
                raise RuntimeError("target manifest could not be quarantined")
            return self._complete_request(
                connection,
                state,
                "quarantined",
                entries_quarantined=int(quarantined),
            )
        manifest_digest = hashlib.sha256(encoded).hexdigest()
        if state.target_manifest_digest != manifest_digest:
            state.target_manifest_digest = manifest_digest
            state.target_next_object_index = 0
            self._save_state(connection, state)
            connection.commit()
        report, completed, corrupted = self._authenticate_objects(
            connection,
            state,
            manifest,
            target=True,
            payload_budget_bytes=payload_budget_bytes,
            item_budget=item_budget,
            on_payload_read=on_payload_read,
        )
        if corrupted:
            completion = self._complete_request(
                connection,
                state,
                "quarantined",
            )
            return _merge_reports(report, completion)
        if completed:
            # Linearize the receipt against cooperating invalidate, GC, store,
            # and quarantine operations. If the manifest changed while its
            # objects were read, leave the cursor in place; the next step will
            # observe the new digest and restart at object zero.
            with self.store._exclusive():
                try:
                    path.lstat()
                except FileNotFoundError:
                    completion = self._complete_request(
                        connection,
                        state,
                        "absent",
                    )
                    return _merge_reports(report, completion)
                if not self._manifest_is_current(
                    state,
                    manifest,
                    target=True,
                ):
                    return report
                # A failed atomic object repair can leave a live manifest
                # deliberately tombstoned. Full byte authentication plus the
                # manifest-version check above is the only non-publication
                # path allowed to make it reportable again.
                inventory_released = int(
                    self.store.release_authenticated_inventory(manifest)
                )
                completion = self._complete_request(
                    connection,
                    state,
                    "authenticated",
                )
            return _merge_reports(
                report,
                completion,
                ScrubStepReport(
                    manifests_authenticated=1,
                    inventory_released=inventory_released,
                ),
            )
        return report

    def _step_cycle(
        self,
        connection: sqlite3.Connection,
        state: _State,
        *,
        payload_budget_bytes: int,
        item_budget: int,
        on_payload_read: Callable[[int], None] | None,
        cancel_requested: Callable[[], bool] | None,
    ) -> ScrubStepReport:
        aggregate = ScrubStepReport()
        remaining_bytes = payload_budget_bytes
        remaining_items = item_budget
        while remaining_items > 0:
            if cancel_requested is not None and cancel_requested():
                raise _ScrubCancelled()
            # A single immutable object may be larger than the byte budget, but
            # only that first object may overshoot. Without this cross-manifest
            # guard, a step containing many one-object manifests could burst by
            # ``item_budget * object_size`` before the scheduler sleeps.
            if aggregate.payload_bytes >= payload_budget_bytes:
                break
            if state.phase in _SNAPSHOT_PHASES:
                report = self._step_namespace_snapshot(
                    connection,
                    state,
                    item_budget=remaining_items,
                    cancel_requested=cancel_requested,
                )
                aggregate = _merge_reports(aggregate, report)
                # A namespace batch is the complete unit of lock-bounded scan
                # work for one step. Processing its durable work queue begins
                # on a later step, even when this batch reached end-of-stream.
                break

            if state.phase == "manifests":
                if not state.active_manifest:
                    path = _next_work_path(
                        connection,
                        table="manifest_work",
                        root=self.store.root,
                        after=state.manifest_cursor,
                        namespace="manifests",
                    )
                    if path is None:
                        state.phase = "snapshot_objects"
                        state.object_cursor = ""
                        self._save_state(connection, state)
                        connection.commit()
                        continue
                    relative = path.relative_to(self.store.root).as_posix()
                    if not _is_canonical_manifest_relative_path(relative):
                        quarantined = (
                            self.store.quarantine_unrecognized_managed_path(
                                path,
                                category="manifest",
                                not_newer_than_unix_ns=(
                                    state.cycle_started_unix_ns
                                ),
                            )
                        )
                        state.manifest_cursor = relative
                        self._save_state(connection, state)
                        connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(
                                entries_quarantined=int(quarantined),
                            ),
                        )
                        remaining_items -= 1
                        continue
                    try:
                        manifest, encoded = self.store._read_manifest_file(
                            path,
                            expected_entry_id=path.stem,
                            probe_objects=False,
                        )
                    except FileNotFoundError:
                        state.manifest_cursor = relative
                        self._save_state(connection, state)
                        connection.commit()
                        continue
                    except OSError:
                        quarantined = self.store._quarantine_manifest(
                            path,
                            reason="manifest_io",
                        )
                        if not quarantined and path.exists():
                            raise RuntimeError("manifest could not be quarantined")
                        state.manifest_cursor = relative
                        self._save_state(connection, state)
                        connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(
                                manifests_examined=1,
                                entries_quarantined=int(quarantined),
                            ),
                        )
                        remaining_items -= 1
                        continue
                    except ManifestError:
                        quarantined = self.store._quarantine_manifest(
                            path,
                            reason="manifest_validation",
                        )
                        if not quarantined and path.exists():
                            raise RuntimeError("manifest could not be quarantined")
                        state.manifest_cursor = relative
                        self._save_state(connection, state)
                        connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(
                                manifests_examined=1,
                                entries_quarantined=int(quarantined),
                            ),
                        )
                        remaining_items -= 1
                        continue
                    state.active_manifest = relative
                    state.active_manifest_digest = hashlib.sha256(encoded).hexdigest()
                    state.next_object_index = 0
                    connection.executemany(
                        "INSERT OR IGNORE INTO refs(relative_path) VALUES (?)",
                        ((item.relative_path,) for item in manifest.objects),
                    )
                    self._save_state(connection, state)
                    connection.commit()
                    aggregate = _merge_reports(
                        aggregate,
                        ScrubStepReport(manifests_examined=1),
                    )
                    remaining_items -= 1
                    if remaining_items <= 0:
                        break
                else:
                    path = self.store.root / state.active_manifest
                    try:
                        manifest, encoded = self.store._read_manifest_file(
                            path,
                            expected_entry_id=path.stem,
                            probe_objects=False,
                        )
                    except FileNotFoundError:
                        state.manifest_cursor = state.active_manifest
                        _clear_active_manifest(state)
                        self._save_state(connection, state)
                        connection.commit()
                        continue
                    except OSError:
                        quarantined = self.store._quarantine_manifest(
                            path,
                            reason="manifest_io",
                        )
                        if not quarantined and path.exists():
                            raise RuntimeError("manifest could not be quarantined")
                        state.manifest_cursor = state.active_manifest
                        _clear_active_manifest(state)
                        self._save_state(connection, state)
                        connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(entries_quarantined=int(quarantined)),
                        )
                        continue
                    except ManifestError:
                        quarantined = self.store._quarantine_manifest(
                            path,
                            reason="manifest_validation",
                        )
                        if not quarantined and path.exists():
                            raise RuntimeError("manifest could not be quarantined")
                        state.manifest_cursor = state.active_manifest
                        _clear_active_manifest(state)
                        self._save_state(connection, state)
                        connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(entries_quarantined=int(quarantined)),
                        )
                        continue
                    digest = hashlib.sha256(encoded).hexdigest()
                    if digest != state.active_manifest_digest:
                        state.active_manifest_digest = digest
                        state.next_object_index = 0
                        connection.executemany(
                            "INSERT OR IGNORE INTO refs(relative_path) VALUES (?)",
                            ((item.relative_path,) for item in manifest.objects),
                        )
                        self._save_state(connection, state)
                        connection.commit()
                    report, completed, corrupted = self._authenticate_objects(
                        connection,
                        state,
                        manifest,
                        target=False,
                        payload_budget_bytes=remaining_bytes,
                        item_budget=remaining_items,
                        on_payload_read=on_payload_read,
                    )
                    aggregate = _merge_reports(aggregate, report)
                    remaining_bytes -= report.payload_bytes
                    remaining_items -= max(
                        report.objects_authenticated,
                        int(corrupted),
                    )
                    if corrupted:
                        state.manifest_cursor = state.active_manifest
                        _clear_active_manifest(state)
                        self._save_state(connection, state)
                        connection.commit()
                        continue
                    if completed:
                        # Do not count an old manifest image after a concurrent
                        # replacement. Holding the store lock through the state
                        # commit provides the cycle's manifest linearization
                        # point; a changed image is retried on the next step.
                        with self.store._exclusive():
                            if not self._manifest_is_current(
                                state,
                                manifest,
                                target=False,
                            ):
                                break
                            inventory_released = int(
                                self.store.release_authenticated_inventory(
                                    manifest
                                )
                            )
                            state.current_manifests_authenticated += 1
                            state.manifest_cursor = state.active_manifest
                            _clear_active_manifest(state)
                            self._save_state(connection, state)
                            connection.commit()
                        aggregate = _merge_reports(
                            aggregate,
                            ScrubStepReport(
                                manifests_authenticated=1,
                                inventory_released=inventory_released,
                            ),
                        )
                        continue
                    break

            elif state.phase == "objects":
                path = _next_work_path(
                    connection,
                    table="object_work",
                    root=self.store.root,
                    after=state.object_cursor,
                    namespace="objects",
                )
                if path is None:
                    state.phase = "snapshot_tmp"
                    state.object_cursor = ""
                    self._save_state(connection, state)
                    connection.commit()
                    continue
                relative = path.relative_to(self.store.root).as_posix()
                try:
                    metadata = path.lstat()
                except FileNotFoundError:
                    state.object_cursor = relative
                    self._save_state(connection, state)
                    connection.commit()
                    continue
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or not _is_canonical_object_relative_path(relative)
                ):
                    quarantined = self.store.quarantine_unrecognized_managed_path(
                        path,
                        category="object",
                        not_newer_than_unix_ns=state.cycle_started_unix_ns,
                    )
                    state.object_cursor = relative
                    self._save_state(connection, state)
                    connection.commit()
                    aggregate = _merge_reports(
                        aggregate,
                        ScrubStepReport(objects_quarantined=int(quarantined)),
                    )
                    remaining_items -= 1
                    continue
                if remaining_bytes < metadata.st_size and aggregate.payload_bytes:
                    break
                referenced = connection.execute(
                    "SELECT 1 FROM refs WHERE relative_path = ?",
                    (relative,),
                ).fetchone()
                reclaimed = 0
                if referenced is None:
                    reclaimed = self.store.remove_orphan_object(
                        path,
                        not_newer_than_unix_ns=state.cycle_started_unix_ns,
                    )
                state.object_cursor = relative
                if reclaimed:
                    state.current_orphan_objects_removed += 1
                    state.current_orphan_bytes_removed += reclaimed
                self._save_state(connection, state)
                connection.commit()
                aggregate = _merge_reports(
                    aggregate,
                    ScrubStepReport(
                        orphan_objects_removed=int(bool(reclaimed)),
                        orphan_bytes_removed=reclaimed,
                    ),
                )
                remaining_items -= 1

            elif state.phase == "tmp":
                path = _next_temporary_work_path(
                    connection,
                    self.store.root / "tmp",
                    after=state.object_cursor,
                )
                if path is None:
                    return _merge_reports(
                        aggregate,
                        self._complete_cycle(connection, state),
                    )
                removed = self.store.remove_abandoned_temporary(
                    path,
                    not_newer_than_unix_ns=state.cycle_started_unix_ns,
                )
                quarantined = False
                if not removed:
                    try:
                        metadata = path.lstat()
                    except FileNotFoundError:
                        metadata = None
                    if metadata is not None and metadata.st_mtime_ns <= (
                        state.cycle_started_unix_ns
                    ):
                        quarantined = (
                            self.store.quarantine_unrecognized_managed_path(
                                path,
                                category="temporary",
                                not_newer_than_unix_ns=(
                                    state.cycle_started_unix_ns
                                ),
                            )
                        )
                state.object_cursor = path.name
                if removed:
                    state.current_temporary_files_removed += 1
                self._save_state(connection, state)
                connection.commit()
                aggregate = _merge_reports(
                    aggregate,
                    ScrubStepReport(
                        temporary_files_removed=int(removed),
                        objects_quarantined=int(quarantined),
                    ),
                )
                remaining_items -= 1
            else:
                raise RuntimeError("deep scrub state has an unsupported phase")
        return aggregate

    def _step_namespace_snapshot(
        self,
        connection: sqlite3.Connection,
        state: _State,
        *,
        item_budget: int,
        cancel_requested: Callable[[], bool] | None,
    ) -> ScrubStepReport:
        """Persist at most ``item_budget`` raw directory entries for a cycle.

        The live scandir iterator is process-local. After a process restart the
        current namespace phase is rescanned from its beginning; INSERT OR
        IGNORE makes already committed work idempotent. This trades bounded
        re-read work after a crash for constant memory and a fixed lock-hold
        batch without trusting unstable directory offsets.
        """

        phase = state.phase
        if phase == "snapshot_manifests":
            iterator_factory = self.store.iter_managed_manifest_paths
            table = "manifest_work"
            column = "relative_path"
            namespace = "manifests"
            category = "manifest"
            next_phase = "manifests"
        elif phase == "snapshot_objects":
            iterator_factory = self.store.iter_managed_object_paths
            table = "object_work"
            column = "relative_path"
            namespace = "objects"
            category = "object"
            next_phase = "objects"
        elif phase == "snapshot_tmp":
            iterator_factory = lambda: _iter_direct_paths(self.store.root / "tmp")
            table = "temporary_work"
            column = "name"
            namespace = "tmp"
            category = "temporary"
            next_phase = "tmp"
        else:
            raise RuntimeError("deep scrub is not in a namespace snapshot phase")

        if self._snapshot_iterator is None:
            self._reset_snapshot_stream()
            self._snapshot_phase = phase
            self._snapshot_cycle = state.cycle
            self._snapshot_cutoff_unix_ns = state.cycle_started_unix_ns
            self._snapshot_iterator = iter(iterator_factory())

        scanned = 0
        rows: list[tuple[str]] = []
        exhausted = False
        flush_quarantines = False
        try:
            with self.store._exclusive():
                while scanned < item_budget:
                    if cancel_requested is not None and cancel_requested():
                        raise _ScrubCancelled()
                    try:
                        path = next(self._snapshot_iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    scanned += 1
                    try:
                        metadata = path.lstat()
                    except FileNotFoundError:
                        continue
                    if metadata.st_mtime_ns > state.cycle_started_unix_ns:
                        continue
                    if namespace == "tmp":
                        value = path.name
                        if (
                            value in {".", ".."}
                            or "/" in value
                            or len(value) > 255
                        ):
                            raise ValueError(
                                "deep scrub temporary work name is invalid"
                            )
                    else:
                        try:
                            value = path.relative_to(self.store.root).as_posix()
                        except ValueError as error:
                            raise ValueError(
                                "deep scrub work item is outside the cache root"
                            ) from error
                        if not _is_safe_managed_relative_path(
                            value,
                            namespace=namespace,
                        ):
                            raise ValueError(
                                "deep scrub work item is not a safe managed path"
                            )
                    if not _sqlite_text_safe(value):
                        self._snapshot_quarantines.append((path, category))
                        if (
                            len(self._snapshot_quarantines)
                            >= _SNAPSHOT_QUARANTINE_BATCH
                        ):
                            flush_quarantines = True
                            break
                        continue
                    rows.append((value,))

                if rows:
                    connection.executemany(
                        f"INSERT OR IGNORE INTO {table}({column}) VALUES (?)",
                        rows,
                    )

                if exhausted or flush_quarantines:
                    deferred = tuple(self._snapshot_quarantines)
                    self._reset_snapshot_stream()
                    for path, deferred_category in deferred:
                        self.store.quarantine_unrecognized_managed_path(
                            path,
                            category=deferred_category,
                            not_newer_than_unix_ns=state.cycle_started_unix_ns,
                        )
                if exhausted:
                    state.phase = next_phase
                self._save_state(connection, state)
                connection.commit()
        except BaseException:
            connection.rollback()
            self._reset_snapshot_stream()
            raise
        return ScrubStepReport(namespace_items_scanned=scanned)

    def _reset_snapshot_stream(self) -> None:
        iterator = self._snapshot_iterator
        self._snapshot_iterator = None
        self._snapshot_phase = ""
        self._snapshot_cycle = 0
        self._snapshot_cutoff_unix_ns = 0
        self._snapshot_quarantines.clear()
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def _authenticate_objects(
        self,
        connection: sqlite3.Connection,
        state: _State,
        manifest: RankManifest,
        *,
        target: bool,
        payload_budget_bytes: int,
        item_budget: int,
        on_payload_read: Callable[[int], None] | None,
    ) -> tuple[ScrubStepReport, bool, bool]:
        index = (
            state.target_next_object_index if target else state.next_object_index
        )
        if index > len(manifest.objects):
            raise ValueError("deep scrub object cursor exceeds its manifest")
        payload_bytes = 0
        authenticated = 0
        while index < len(manifest.objects) and authenticated < item_budget:
            item = manifest.objects[index]
            if payload_bytes and payload_bytes + item.stored_length > payload_budget_bytes:
                break
            object_path = self.store.root / item.relative_path
            try:
                metadata = object_path.lstat()
            except FileNotFoundError:
                withdrawn = self._quarantine_current_object(
                    state,
                    manifest,
                    item,
                    target=target,
                    reason="payload_size",
                )
                if withdrawn is None:
                    return ScrubStepReport(payload_bytes=payload_bytes), False, False
                return (
                    ScrubStepReport(
                        payload_bytes=payload_bytes,
                        objects_authenticated=authenticated,
                        entries_quarantined=withdrawn,
                        objects_quarantined=0,
                    ),
                    False,
                    True,
                )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != item.stored_length
            ):
                withdrawn = self._quarantine_current_object(
                    state,
                    manifest,
                    item,
                    target=target,
                    reason="payload_size",
                )
                if withdrawn is None:
                    return ScrubStepReport(payload_bytes=payload_bytes), False, False
                return (
                    ScrubStepReport(
                        payload_bytes=payload_bytes,
                        objects_authenticated=authenticated,
                        entries_quarantined=withdrawn,
                        objects_quarantined=1,
                    ),
                    False,
                    True,
                )
            try:
                object_payload_bytes = 0

                def observe_payload(byte_count: int) -> None:
                    nonlocal object_payload_bytes
                    object_payload_bytes += byte_count
                    if on_payload_read is not None:
                        on_payload_read(byte_count)

                self.store.stream_object(
                    item,
                    lambda _view: None,
                    on_bytes_read=observe_payload,
                )
            except (OSError, ManifestError):
                withdrawn = self._quarantine_current_object(
                    state,
                    manifest,
                    item,
                    target=target,
                    reason="payload_checksum",
                )
                if withdrawn is None:
                    return ScrubStepReport(payload_bytes=payload_bytes), False, False
                return (
                    ScrubStepReport(
                        payload_bytes=payload_bytes + object_payload_bytes,
                        objects_authenticated=authenticated,
                        entries_quarantined=withdrawn,
                        objects_quarantined=1,
                    ),
                    False,
                    True,
                )
            if object_payload_bytes != item.stored_length:
                raise RuntimeError("deep scrub observed an incomplete object read")
            payload_bytes += object_payload_bytes
            authenticated += 1
            index += 1
            if target:
                state.target_next_object_index = index
            else:
                state.next_object_index = index
                state.current_objects_authenticated += 1
                state.current_payload_bytes += item.stored_length
            self._save_state(connection, state)
            connection.commit()
            if payload_bytes >= payload_budget_bytes:
                break
        return (
            ScrubStepReport(
                payload_bytes=payload_bytes,
                objects_authenticated=authenticated,
            ),
            index == len(manifest.objects),
            False,
        )

    def _quarantine_current_object(
        self,
        state: _State,
        manifest: RankManifest,
        descriptor: ObjectDescriptor,
        *,
        target: bool,
        reason: str,
    ) -> int | None:
        # Close the check→quarantine gap. A concurrent replacement must not
        # cause bytes from an old manifest to quarantine a new entry image.
        with self.store._exclusive():
            if not self._manifest_is_current(state, manifest, target=target):
                return None
            return self.store.quarantine_object(descriptor, reason=reason)

    def _manifest_is_current(
        self,
        state: _State,
        manifest: RankManifest,
        *,
        target: bool,
    ) -> bool:
        path = (
            self.store._manifest_path(state.target_entry)
            if target
            else self.store.root / state.active_manifest
        )
        expected_digest = (
            state.target_manifest_digest
            if target
            else state.active_manifest_digest
        )
        try:
            current, encoded = self.store._read_manifest_file(
                path,
                expected_entry_id=manifest.entry_id,
                probe_objects=False,
            )
        except (OSError, ManifestError):
            return False
        return (
            current == manifest
            and hashlib.sha256(encoded).hexdigest() == expected_digest
        )

    def _complete_request(
        self,
        connection: sqlite3.Connection,
        state: _State,
        status: str,
        *,
        entries_quarantined: int = 0,
        objects_quarantined: int = 0,
    ) -> ScrubStepReport:
        if status not in _REQUEST_STATUSES - {""}:
            raise ValueError("target scrub completion status is invalid")
        completed_nonce = state.target_nonce
        state.last_request_nonce = completed_nonce
        state.last_request_entry = state.target_entry
        state.last_request_status = status
        state.last_request_completed_unix_ns = time.time_ns()
        state.target_nonce = ""
        state.target_entry = ""
        state.target_manifest_observed = False
        state.target_manifest_digest = ""
        state.target_next_object_index = 0
        state.target_requested_unix_ns = 0
        self._save_state(connection, state)
        connection.commit()
        self._unlink_matching_request(completed_nonce)
        return ScrubStepReport(
            entries_quarantined=entries_quarantined,
            objects_quarantined=objects_quarantined,
            request_completed=True,
            request_status=status,
        )

    def _complete_cycle(
        self,
        connection: sqlite3.Connection,
        state: _State,
    ) -> ScrubStepReport:
        state.phase = "idle"
        state.last_completed_unix_ns = time.time_ns()
        state.last_cycle_manifests_authenticated = (
            state.current_manifests_authenticated
        )
        state.last_cycle_objects_authenticated = state.current_objects_authenticated
        state.last_cycle_payload_bytes = state.current_payload_bytes
        state.last_cycle_orphan_objects_removed = (
            state.current_orphan_objects_removed
        )
        state.last_cycle_orphan_bytes_removed = state.current_orphan_bytes_removed
        state.last_cycle_temporary_files_removed = (
            state.current_temporary_files_removed
        )
        state.manifest_cursor = ""
        state.active_manifest = ""
        state.active_manifest_digest = ""
        state.next_object_index = 0
        state.object_cursor = ""
        # The reference index is a per-cycle proof, not durable cache state.
        # FULL auto-vacuum returns its pages here so maintenance metadata does
        # not retain the high-water size of an old cache population.
        connection.execute("DELETE FROM refs")
        connection.execute("DELETE FROM manifest_work")
        connection.execute("DELETE FROM object_work")
        connection.execute("DELETE FROM temporary_work")
        self._save_state(connection, state)
        connection.commit()
        return ScrubStepReport(cycle_completed=True)

    def _ingest_request(
        self,
        connection: sqlite3.Connection,
        state: _State,
    ) -> None:
        with self.store._exclusive():
            try:
                payload = _read_bounded_json(
                    self.request_path,
                    _MAX_REQUEST_BYTES,
                )
                request = _validate_request(payload)
            except FileNotFoundError:
                return
            except (OSError, ValueError) as error:
                quarantine = (
                    self.store.root
                    / "quarantine"
                    / f"control-{self.request_path.name}.{uuid.uuid4().hex}.bad"
                )
                try:
                    os.replace(self.request_path, quarantine)
                except FileNotFoundError:
                    pass
                else:
                    _fsync_directory(self.state_directory)
                    _fsync_directory(self.store.root / "quarantine")
                raise ValueError(
                    "deep scrub request control file is malformed"
                ) from error
        nonce = request["nonce"]
        if state.last_request_nonce == nonce:
            self._unlink_matching_request(nonce)
            return
        if state.target_nonce:
            if state.target_nonce != nonce:
                raise RuntimeError("a different deep scrub request is already active")
            return
        state.target_nonce = nonce
        state.target_entry = request["entry_id"]
        state.target_manifest_observed = False
        state.target_requested_unix_ns = request["requested_at_unix_ns"]
        state.target_manifest_digest = ""
        state.target_next_object_index = 0
        self._save_state(connection, state)
        connection.commit()

    def _unlink_matching_request(self, nonce: str) -> None:
        with self.store._exclusive():
            try:
                payload = _validate_request(
                    _read_bounded_json(self.request_path, _MAX_REQUEST_BYTES)
                )
            except FileNotFoundError:
                return
            if payload["nonce"] != nonce:
                return
            self.request_path.unlink()
            _fsync_directory(self.state_directory)

    def _connect(self) -> sqlite3.Connection:
        if self.database_path.is_symlink():
            raise ValueError("deep scrub database cannot be a symlink")
        if self.database_path.exists() and not self.database_path.is_file():
            raise ValueError("deep scrub database must be a regular file")
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-2048")
            return connection
        except BaseException:
            connection.close()
            raise

    def _quarantine_database(self) -> None:
        incident = uuid.uuid4().hex
        artifacts: list[tuple[Path, str]] = []
        for suffix, label in (
            ("", "database"),
            ("-journal", "journal"),
            ("-wal", "wal"),
            ("-shm", "shm"),
        ):
            path = Path(f"{self.database_path}{suffix}")
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("deep scrub database artifact is not regular")
            artifacts.append((path, label))
        for path, label in artifacts:
            target = (
                self.store.root
                / "quarantine"
                / f"control-deep-scrub-{label}.{incident}.bad"
            )
            os.replace(path, target)
        if artifacts:
            _fsync_directory(self.state_directory)
            _fsync_directory(self.store.root / "quarantine")

    def _initialize(self, connection: sqlite3.Connection) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            connection.execute("PRAGMA auto_vacuum=FULL")
            connection.execute("PRAGMA user_version=1")
        elif version != 1:
            raise _UnsupportedScrubDatabase(
                "deep scrub database schema version is unsupported"
            )
        if connection.execute("PRAGMA auto_vacuum").fetchone()[0] != 1:
            raise ValueError("deep scrub database is missing bounded auto-vacuum")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta("
            "key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS refs("
            "relative_path TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS manifest_work("
            "relative_path TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS object_work("
            "relative_path TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS temporary_work("
            "name TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'state'"
        ).fetchone()
        if row is None:
            self._save_state(connection, _State())
        else:
            self._decode_state(row[0])
        connection.commit()
        _fsync_directory(self.state_directory)

    def _load_state(self, connection: sqlite3.Connection) -> _State:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'state'"
        ).fetchone()
        if row is None:
            raise ValueError("deep scrub database has no state record")
        return self._decode_state(row[0])

    def _save_state(self, connection: sqlite3.Connection, state: _State) -> None:
        _validate_state(state)
        encoded = canonical_json(dataclasses.asdict(state))
        if len(encoded) > _MAX_STATE_BYTES:
            raise ValueError("deep scrub state exceeds its fixed bound")
        connection.execute(
            "INSERT INTO meta(key, value) VALUES('state', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (encoded,),
        )

    @staticmethod
    def _decode_state(encoded: object) -> _State:
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise ValueError("deep scrub state encoding is invalid")
        raw_bytes = bytes(encoded)
        if not raw_bytes or len(raw_bytes) > _MAX_STATE_BYTES:
            raise ValueError("deep scrub state length is invalid")
        try:
            raw = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("deep scrub state JSON is invalid") from error
        expected = {field.name for field in dataclasses.fields(_State)}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("deep scrub state fields differ from schema")
        try:
            state = _State(**raw)
        except TypeError as error:
            raise ValueError("deep scrub state types are invalid") from error
        _validate_state(state)
        return state


class ScheduledDeepScrubber:
    """Low-priority periodic driver using the fixed internal I/O rate."""

    def __init__(
        self,
        scrubber: DeepScrubber,
        *,
        bytes_per_second: int = SCRUB_BYTES_PER_SECOND,
        step_bytes: int = SCRUB_STEP_BYTES,
        item_budget: int = SCRUB_STEP_ITEMS,
        poll_seconds: float = SCRUB_POLL_SECONDS,
        startup_delay_seconds: float = SCRUB_STARTUP_DELAY_SECONDS,
        cycle_interval_seconds: float = SCRUB_CYCLE_INTERVAL_SECONDS,
    ) -> None:
        _positive_int(bytes_per_second, "scrub byte rate")
        _positive_int(step_bytes, "scrub step bytes")
        _positive_int(item_budget, "scrub step items")
        for value, label in (
            (poll_seconds, "scrub poll interval"),
            (startup_delay_seconds, "scrub startup delay"),
            (cycle_interval_seconds, "scrub cycle interval"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (label == "scrub poll interval" and value == 0)
            ):
                raise ValueError(f"{label} is invalid")
        self.scrubber = scrubber
        self.bytes_per_second = bytes_per_second
        self.step_bytes = step_bytes
        self.item_budget = item_budget
        self.poll_seconds = float(poll_seconds)
        self.startup_delay_seconds = float(startup_delay_seconds)
        self.cycle_interval_seconds = float(cycle_interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._metrics_lock = threading.RLock()
        self._pending_metrics: dict[str, int] = {}
        self._inventory_rescan_epoch = 0
        self._inventory_rescan_acknowledged = 0
        self._inventory_force_withdrawal_epoch = 0
        self._shutdown_timeout_reported = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("deep scrub scheduler is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="spoolcache-deep-scrub",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> ScrubShutdownReport:
        thread = self._thread
        if thread is None:
            return ScrubShutdownReport(
                schema=SCRUB_SHUTDOWN_SCHEMA,
                status="not_started",
                thread_alive=False,
                waited_seconds=0.0,
            )
        self._stop.set()
        started = time.monotonic()
        thread.join(timeout=SCRUB_SHUTDOWN_TIMEOUT_SECONDS)
        waited = max(0.0, time.monotonic() - started)
        alive = thread.is_alive()
        if alive:
            with self._metrics_lock:
                if not self._shutdown_timeout_reported:
                    self._pending_metrics["shutdown_failures"] = (
                        self._pending_metrics.get("shutdown_failures", 0) + 1
                    )
                    self._shutdown_timeout_reported = True
            return ScrubShutdownReport(
                schema=SCRUB_SHUTDOWN_SCHEMA,
                status="timeout",
                thread_alive=True,
                waited_seconds=waited,
            )
        self._thread = None
        close_scrubber = getattr(self.scrubber, "close", None)
        if callable(close_scrubber):
            close_scrubber()
        return ScrubShutdownReport(
            schema=SCRUB_SHUTDOWN_SCHEMA,
            status="stopped",
            thread_alive=False,
            waited_seconds=waited,
        )

    def drain_metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            result = dict(self._pending_metrics)
            self._pending_metrics.clear()
            return result

    def pending_inventory_rescan_epoch(self) -> int | None:
        with self._metrics_lock:
            if self._inventory_rescan_epoch <= self._inventory_rescan_acknowledged:
                return None
            return self._inventory_rescan_epoch

    def require_inventory_rescan(self, *, force_withdrawal: bool = False) -> None:
        if not isinstance(force_withdrawal, bool):
            raise ValueError("deep scrub force-withdrawal flag is invalid")
        with self._metrics_lock:
            if self._inventory_rescan_epoch >= _MAX_COUNTER:
                raise RuntimeError("deep scrub inventory rescan epoch overflowed")
            self._inventory_rescan_epoch += 1
            if force_withdrawal:
                self._inventory_force_withdrawal_epoch = (
                    self._inventory_rescan_epoch
                )

    def inventory_rescan_requires_withdrawal(self, epoch: int) -> bool:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("deep scrub inventory rescan epoch is invalid")
        with self._metrics_lock:
            if epoch > self._inventory_rescan_epoch:
                raise ValueError("deep scrub inventory rescan epoch is from the future")
            return (
                self._inventory_rescan_acknowledged
                < self._inventory_force_withdrawal_epoch
                <= epoch
            )

    def acknowledge_inventory_rescan(self, epoch: int) -> None:
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0
        ):
            raise ValueError("deep scrub inventory rescan epoch is invalid")
        with self._metrics_lock:
            if epoch > self._inventory_rescan_epoch:
                raise ValueError("deep scrub inventory rescan epoch is from the future")
            self._inventory_rescan_acknowledged = max(
                self._inventory_rescan_acknowledged,
                epoch,
            )

    def _run(self) -> None:
        try:
            initial = self.scrubber.status()
            completed_age = max(
                0.0,
                (time.time_ns() - initial.last_completed_unix_ns) / 1_000_000_000,
            ) if initial.last_completed_unix_ns else self.cycle_interval_seconds
            next_cycle = time.monotonic() + max(
                self.startup_delay_seconds,
                self.cycle_interval_seconds - completed_age,
            )
        except Exception:
            logger.exception("spoolcache: could not read initial deep scrub state")
            try:
                self.scrubber.repair_invalid_state()
            except Exception:
                logger.exception("spoolcache: could not repair initial scrub state")
            with self._metrics_lock:
                self._pending_metrics["failures"] = (
                    self._pending_metrics.get("failures", 0) + 1
                )
                self.require_inventory_rescan(force_withdrawal=True)
            next_cycle = time.monotonic() + self.startup_delay_seconds
        while not self._stop.is_set():
            try:
                status = self.scrubber.status()
                requested = self.scrubber.has_pending_request() or bool(
                    status.target_entry
                )
                now = time.monotonic()
                if status.phase == "idle" and not requested and now < next_cycle:
                    self._stop.wait(min(self.poll_seconds, next_cycle - now))
                    continue
                if status.phase == "idle" and not requested:
                    self.scrubber.start_cycle()
                report = self.scrubber.step(
                    payload_budget_bytes=self.step_bytes,
                    item_budget=self.item_budget,
                    on_payload_read=self._pace_payload,
                    cancel_requested=self._stop.is_set,
                )
                self._accumulate(report)
                if report.cycle_completed:
                    next_cycle = time.monotonic() + self.cycle_interval_seconds
                self._stop.wait(self.poll_seconds)
            except _ScrubCancelled:
                return
            except Exception:
                logger.exception("spoolcache: scheduled deep scrub step failed")
                try:
                    self.scrubber.repair_invalid_state()
                except Exception:
                    logger.exception("spoolcache: scrub state repair failed")
                with self._metrics_lock:
                    self._pending_metrics["failures"] = (
                        self._pending_metrics.get("failures", 0) + 1
                    )
                    # Reconcile from durable manifests after any unexpected
                    # maintenance error. This is a safe false-negative path
                    # and prevents an already-moved manifest remaining held by
                    # the worker reporter.
                    self.require_inventory_rescan(force_withdrawal=True)
                self._stop.wait(max(self.poll_seconds, 1.0))

    def _pace_payload(self, byte_count: int) -> None:
        _positive_int(byte_count, "scrub observed byte count")
        if self._stop.wait(byte_count / self.bytes_per_second):
            raise _ScrubCancelled()

    def _accumulate(self, report: ScrubStepReport) -> None:
        values = {
            "namespace_items_scanned": report.namespace_items_scanned,
            "payload_bytes": report.payload_bytes,
            "objects_authenticated": report.objects_authenticated,
            "manifests_authenticated": report.manifests_authenticated,
            "cycles": int(report.cycle_completed),
            "objects_quarantined": report.objects_quarantined,
            "orphan_objects_removed": report.orphan_objects_removed,
            "orphan_bytes_removed": report.orphan_bytes_removed,
            "temporary_files_removed": report.temporary_files_removed,
        }
        with self._metrics_lock:
            if report.entries_quarantined or report.inventory_released:
                self.require_inventory_rescan()
            for key, value in values.items():
                if value:
                    self._pending_metrics[key] = (
                        self._pending_metrics.get(key, 0) + value
                    )


def request_deep_scrub(root: str | os.PathLike[str], entry_id: str) -> str:
    if not _is_digest(entry_id):
        raise ValueError("target scrub entry ID is malformed")
    state_directory = _validated_state_directory(root)
    request_path = state_directory / SCRUB_REQUEST_NAME
    lock_path = state_directory / "maintenance.lock"
    descriptor = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("deep scrub maintenance lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        _cleanup_atomic_temporaries(
            state_directory,
            target_name=SCRUB_REQUEST_NAME,
        )
        try:
            existing = _validate_request(
                _read_bounded_json(request_path, _MAX_REQUEST_BYTES)
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing["entry_id"] != entry_id:
                raise RuntimeError("a different deep scrub request is already pending")
            return existing["nonce"]
        nonce = uuid.uuid4().hex
        payload = {
            "schema": SCRUB_REQUEST_SCHEMA,
            "nonce": nonce,
            "entry_id": entry_id,
            "requested_at_unix_ns": time.time_ns(),
        }
        _atomic_write(request_path, canonical_json(payload) + b"\n")
        return nonce
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def read_scrub_status(root: str | os.PathLike[str]) -> Mapping[str, Any]:
    state_directory = _validated_state_directory(root)
    database = state_directory / SCRUB_DATABASE_NAME
    try:
        metadata = database.lstat()
    except FileNotFoundError:
        return {"schema": SCRUB_STATE_SCHEMA, "initialized": False}
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("deep scrub database is not a regular owned file")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        auto_vacuum = connection.execute("PRAGMA auto_vacuum").fetchone()[0]
        if version != 1 or auto_vacuum != 1:
            raise ValueError("deep scrub database schema is unsupported")
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'state'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("deep scrub database has no state record")
    state = DeepScrubber._decode_state(row[0])
    return {
        "schema": SCRUB_STATE_SCHEMA,
        "initialized": True,
        **dataclasses.asdict(state.status()),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    request_parser = subparsers.add_parser(
        "request",
        help="request a resumable authenticated scrub of one entry",
    )
    request_parser.add_argument("--root", required=True)
    request_parser.add_argument("--entry", required=True)
    status_parser = subparsers.add_parser(
        "status",
        help="print bounded machine-readable deep scrub progress",
    )
    status_parser.add_argument("--root", required=True)
    arguments = parser.parse_args(tuple(argv) if argv is not None else None)
    if arguments.command == "request":
        nonce = request_deep_scrub(arguments.root, arguments.entry)
        print(json.dumps({"schema": SCRUB_REQUEST_SCHEMA, "nonce": nonce}))
        return 0
    print(json.dumps(read_scrub_status(arguments.root), sort_keys=True))
    return 0


def _validate_state(state: _State) -> None:
    if (
        not isinstance(state.schema, str)
        or not isinstance(state.phase, str)
        or state.schema != SCRUB_STATE_SCHEMA
        or state.phase not in _PHASES
    ):
        raise ValueError("deep scrub state schema or phase is invalid")
    integer_fields = (
        field.name
        for field in dataclasses.fields(_State)
        if field.name.endswith("_ns")
        or field.name.startswith("cycle")
        or "index" in field.name
        or field.name.startswith("current_")
        or field.name.startswith("last_cycle_")
        or field.name.startswith("target_requested_")
        or field.name.startswith("last_request_completed_")
    )
    for name in integer_fields:
        value = getattr(state, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_COUNTER
        ):
            raise ValueError(f"deep scrub state {name} is invalid")
    for name in (
        "manifest_cursor",
        "active_manifest",
        "object_cursor",
    ):
        value = getattr(state, name)
        if not isinstance(value, str) or len(value) > 1024:
            raise ValueError(f"deep scrub state {name} is invalid")
    if state.manifest_cursor and not _is_safe_managed_relative_path(
        state.manifest_cursor,
        namespace="manifests",
    ):
        raise ValueError("deep scrub manifest cursor is not canonical")
    if state.active_manifest and not _is_canonical_manifest_relative_path(
        state.active_manifest
    ):
        raise ValueError("deep scrub active manifest is not canonical")
    if state.phase == "objects" and state.object_cursor and not (
        _is_safe_managed_relative_path(
            state.object_cursor,
            namespace="objects",
        )
    ):
        raise ValueError("deep scrub object cursor is not canonical")
    if state.phase == "tmp" and state.object_cursor and (
        "/" in state.object_cursor
        or state.object_cursor in {".", ".."}
        or len(state.object_cursor) > 255
    ):
        raise ValueError("deep scrub temporary cursor is not canonical")
    if state.phase in {"idle", "manifests"} and state.object_cursor:
        raise ValueError("deep scrub object cursor is invalid for its phase")
    for name in (
        "active_manifest_digest",
        "target_manifest_digest",
    ):
        value = getattr(state, name)
        if not isinstance(value, str) or (value and not _is_digest(value)):
            raise ValueError(f"deep scrub state {name} is invalid")
    for name in ("target_entry", "last_request_entry"):
        value = getattr(state, name)
        if not isinstance(value, str) or (value and not _is_digest(value)):
            raise ValueError(f"deep scrub state {name} is invalid")
    for name in ("target_nonce", "last_request_nonce"):
        value = getattr(state, name)
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError(f"deep scrub state {name} is invalid")
    if (
        not isinstance(state.last_request_status, str)
        or state.last_request_status not in _REQUEST_STATUSES
    ):
        raise ValueError("deep scrub last request status is invalid")
    if bool(state.active_manifest) != bool(state.active_manifest_digest):
        raise ValueError("deep scrub active manifest receipt is incomplete")
    if not state.active_manifest and state.next_object_index:
        raise ValueError("deep scrub object index has no active manifest")
    if state.phase != "manifests" and state.active_manifest:
        raise ValueError("deep scrub active manifest is outside manifest phase")
    if state.phase != "idle" and (
        state.cycle <= 0 or state.cycle_started_unix_ns <= 0
    ):
        raise ValueError("deep scrub active cycle identity is incomplete")
    if bool(state.target_nonce) != bool(state.target_entry):
        raise ValueError("deep scrub target identity is incomplete")
    if not isinstance(state.target_manifest_observed, bool):
        raise ValueError("deep scrub target manifest observation is invalid")
    if state.target_manifest_digest and not state.target_manifest_observed:
        raise ValueError("deep scrub target manifest receipt is inconsistent")
    if state.target_next_object_index and not state.target_manifest_digest:
        raise ValueError("deep scrub target object cursor has no manifest receipt")
    if state.target_nonce and state.target_requested_unix_ns <= 0:
        raise ValueError("deep scrub target timestamp is missing")
    if not state.target_nonce and (
        state.target_manifest_observed
        or state.target_manifest_digest
        or state.target_next_object_index
        or state.target_requested_unix_ns
    ):
        raise ValueError("deep scrub inactive target retains progress")
    if bool(state.last_request_nonce) != bool(state.last_request_entry):
        raise ValueError("deep scrub last request identity is incomplete")
    if state.last_request_nonce and (
        not state.last_request_status
        or state.last_request_completed_unix_ns <= 0
    ):
        raise ValueError("deep scrub last request receipt is incomplete")


def _validate_request(payload: object) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema", "nonce", "entry_id", "requested_at_unix_ns"}
        or payload.get("schema") != SCRUB_REQUEST_SCHEMA
        or not isinstance(payload.get("nonce"), str)
        or len(payload["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in payload["nonce"])
        or not _is_digest(payload.get("entry_id"))
    ):
        raise ValueError("deep scrub request is malformed")
    requested = payload["requested_at_unix_ns"]
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
        or requested > _MAX_COUNTER
    ):
        raise ValueError("deep scrub request timestamp is malformed")
    return payload


def _validated_state_directory(root: str | os.PathLike[str]) -> Path:
    path = Path(root)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.abspath(path)):
        raise ValueError("deep scrub root must be an absolute narrow path")
    for component in reversed((path, *path.parents)):
        if not component.exists():
            continue
        if stat.S_ISLNK(component.lstat().st_mode):
            raise ValueError("deep scrub root contains a symlink")
    marker = path / ".spoolcache-root"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("deep scrub root is not owned by SpoolCache")
    try:
        marker_payload = _read_bounded_json(marker, 4096)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("deep scrub root marker is invalid") from error
    if marker_payload != {"schema": "spoolcache-root/v1"}:
        raise ValueError("deep scrub root marker schema is unsupported")
    state_directory = path / "state"
    if state_directory.is_symlink():
        raise ValueError("deep scrub state directory cannot be a symlink")
    metadata = state_directory.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("deep scrub state directory is invalid")
    return state_directory


def _read_bounded_json(path: Path, maximum: int) -> object:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError("bounded JSON file size is invalid")
        encoded = bytearray()
        while len(encoded) < metadata.st_size:
            chunk = os.read(descriptor, metadata.st_size - len(encoded))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) != metadata.st_size:
            raise ValueError("bounded JSON file is truncated")
    finally:
        os.close(descriptor)
    try:
        return json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bounded JSON file is malformed") from error


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        try:
            cursor = 0
            while cursor < len(view):
                written = os.write(descriptor, view[cursor:])
                if written <= 0:
                    raise OSError("deep scrub state write made no progress")
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
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(path.parent)


def _sqlite_text_safe(value: str) -> bool:
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def _cleanup_atomic_temporaries(directory: Path, *, target_name: str) -> int:
    prefix = f".{target_name}."
    suffix = ".tmp"
    removed = 0
    for path in _iter_direct_paths(directory):
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        nonce = name[len(prefix) : -len(suffix)]
        if (
            len(nonce) != 32
            or any(character not in "0123456789abcdef" for character in nonce)
        ):
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("atomic state temporary is not a regular file")
        path.unlink()
        removed += 1
    if removed:
        _fsync_directory(directory)
    return removed


def _next_work_path(
    connection: sqlite3.Connection,
    *,
    table: str,
    root: Path,
    after: str,
    namespace: str,
) -> Path | None:
    if table not in {"manifest_work", "object_work"}:
        raise ValueError("deep scrub work table is invalid")
    row = connection.execute(
        f"SELECT relative_path FROM {table} "
        "WHERE relative_path > ? ORDER BY relative_path LIMIT 1",
        (after,),
    ).fetchone()
    if row is None:
        return None
    relative = row[0]
    if not isinstance(relative, str) or not _is_safe_managed_relative_path(
        relative,
        namespace=namespace,
    ):
        raise ValueError("deep scrub persisted work path is invalid")
    return root / relative


def _is_canonical_manifest_relative_path(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) != 3 or parts[0] != "manifests":
        return False
    filename = parts[2]
    if not filename.endswith(".json"):
        return False
    entry_id = filename[: -len(".json")]
    return _is_digest(entry_id) and parts[1] == entry_id[:2]


def _is_safe_managed_relative_path(relative: str, *, namespace: str) -> bool:
    parts = relative.split("/")
    return (
        len(parts) in {2, 3}
        and parts[0] == namespace
        and all(
            part not in {"", ".", ".."} and len(part) <= 255
            for part in parts[1:]
        )
    )


def _iter_direct_paths(root: Path) -> Iterable[Path]:
    try:
        entries = os.scandir(root)
    except FileNotFoundError:
        return
    with entries:
        for entry in entries:
            if (
                entry.name in {".", ".."}
                or "/" in entry.name
                or len(entry.name) > 255
            ):
                raise ValueError("deep scrub temporary work name is invalid")
            yield Path(entry.path)


def _next_temporary_work_path(
    connection: sqlite3.Connection,
    root: Path,
    *,
    after: str,
) -> Path | None:
    row = connection.execute(
        "SELECT name FROM temporary_work WHERE name > ? ORDER BY name LIMIT 1",
        (after,),
    ).fetchone()
    if row is None:
        return None
    name = row[0]
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or "/" in name
        or len(name) > 255
    ):
        raise ValueError("deep scrub persisted temporary path is invalid")
    return root / name


def _clear_active_manifest(state: _State) -> None:
    state.active_manifest = ""
    state.active_manifest_digest = ""
    state.next_object_index = 0


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _merge_reports(*reports: ScrubStepReport) -> ScrubStepReport:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(ScrubStepReport):
        if field.type is bool or field.name in {"cycle_completed", "request_completed"}:
            values[field.name] = any(getattr(report, field.name) for report in reports)
        elif field.name == "request_status":
            values[field.name] = next(
                (
                    getattr(report, field.name)
                    for report in reversed(reports)
                    if getattr(report, field.name)
                ),
                "",
            )
        else:
            values[field.name] = sum(getattr(report, field.name) for report in reports)
    return ScrubStepReport(**values)


if __name__ == "__main__":  # pragma: no cover - exercised by CLI integration
    raise SystemExit(main())


__all__ = [
    "DeepScrubber",
    "SCRUB_BYTES_PER_SECOND",
    "SCRUB_CYCLE_INTERVAL_SECONDS",
    "SCRUB_REQUEST_SCHEMA",
    "SCRUB_SHUTDOWN_SCHEMA",
    "SCRUB_STATE_SCHEMA",
    "ScheduledDeepScrubber",
    "ScrubShutdownReport",
    "ScrubStatus",
    "ScrubStepReport",
    "read_scrub_status",
    "request_deep_scrub",
]
