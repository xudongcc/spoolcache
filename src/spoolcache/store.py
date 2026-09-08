"""Crash-consistent immutable objects and manifest-last publication."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import heapq
import json
import logging
import os
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .buffers import AlignedBufferPool
from .errors import ManifestError, ObjectCorruptionError
from .identity import canonical_json
from .manifest import (
    MAX_MANIFEST_BYTES,
    ObjectDescriptor,
    RankManifest,
    decode_manifest,
    encode_manifest,
    ordered_descriptors,
)

_ROOT_MARKER_SCHEMA = "spoolcache-root/v1"
GENERATION_STATE_SCHEMA = "spoolcache-generation/v1"
GENERATION_STATE_NAME = "generation.json"
GENERATION_REQUIRED_SCHEMA = "spoolcache-generation-required/v1"
GENERATION_REQUIRED_NAME = "generation.required.json"
_MAX_GENERATION_STATE_BYTES = 4096
_MAX_GENERATION_EPOCH = (1 << 63) - 1
# Legacy SpoolCache workers advertised raw ``time.time_ns()`` values. Reserve
# the upper half of the signed protocol counter for crash-consistent epochs so
# the first upgraded worker is newer even when the wall clock moved backwards.
_PERSISTENT_GENERATION_EPOCH_BASE = 1 << 62
_INVENTORY_WITHDRAWAL_DIRECTORY = "inventory-withdrawn"
_OBJECT_WITHDRAWAL_DIRECTORY = "object-withdrawn"
_INVENTORY_OWNER_LOCK_NAME = "inventory-owner.lock"
_SCAN_QUARANTINE_BATCH = 64
_ALIGNMENT = 4096
QUARANTINE_REASONS = frozenset(
    {
        "manifest_io",
        "manifest_validation",
        "object_collision",
        "payload_checksum",
        "payload_size",
        "unknown",
    }
)
logger = logging.getLogger(__name__)


class _NamespaceMutationError(RuntimeError):
    """A live path changed, but its durability/withdrawal receipt failed."""

    def __init__(self, message: str, cause: Exception) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class ObjectSource:
    group_index: int
    layer_name: str
    page_start: int
    page_count: int
    chunks: Iterable[bytes | bytearray | memoryview]


@dataclass(frozen=True)
class ManifestOffer:
    entry_id: str
    span_tokens: int
    manifest_digest: str
    modified_at_unix_ns: int


@dataclass(frozen=True)
class LookupResult:
    is_hit: bool
    reason: str
    manifest: RankManifest | None = None
    manifest_digest: str | None = None


@dataclass(frozen=True)
class MaintenanceReport:
    bytes_before: int
    bytes_after: int
    manifests_removed: int
    objects_removed: int
    bytes_reclaimed: int


@dataclass(frozen=True)
class ManagedDiskUsage:
    cache_bytes: int
    quarantine_bytes: int
    temporary_bytes: int
    state_bytes: int
    unrecognized_bytes: int
    control_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.cache_bytes
            + self.quarantine_bytes
            + self.temporary_bytes
            + self.state_bytes
            + self.unrecognized_bytes
            + self.control_bytes
        )


class ManifestStore:
    """One physical rank's narrow, owned cache root.

    Payload objects may be arbitrarily large, but reads and writes use only a
    fixed number of fixed-size aligned buffers with mandatory O_DIRECT. Small
    metadata uses ordinary I/O. A manifest becomes visible only after every
    referenced immutable object has been fsynced.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        slot_bytes: int = 64 * 1024 * 1024,
        slot_count: int = 2,
        expected_deployment_digest: str | None = None,
        expected_rank_digest: str | None = None,
        expected_rank: int | None = None,
        expected_topology_digest: str | None = None,
        expected_profile: str | None = None,
        expected_layout_digest: str | None = None,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.expected_deployment_digest = expected_deployment_digest
        self.expected_rank_digest = expected_rank_digest
        self.expected_rank = expected_rank
        self.expected_topology_digest = expected_topology_digest
        self.expected_profile = expected_profile
        self.expected_layout_digest = expected_layout_digest
        self._fault_hook = fault_hook
        self._quarantine_hook: Callable[[str], None] | None = None
        self._withdraw_hook: Callable[[str], None] | None = None
        self._thread_lock = threading.RLock()
        self._lock_state = threading.local()
        self._inventory_owner_descriptor: int | None = None
        self._closed = False
        for label, value in (
            ("deployment", expected_deployment_digest),
            ("rank", expected_rank_digest),
            ("topology", expected_topology_digest),
            ("layout", expected_layout_digest),
        ):
            if value is not None and not _is_digest(value):
                raise ValueError(f"expected {label} digest is malformed")
        if expected_rank is not None and (
            isinstance(expected_rank, bool)
            or not isinstance(expected_rank, int)
            or expected_rank < 0
        ):
            raise ValueError("expected physical rank is malformed")
        if expected_profile is not None and (
            not isinstance(expected_profile, str)
            or not expected_profile
            or len(expected_profile) > 128
        ):
            raise ValueError("expected profile is malformed")
        self._prepare_root()
        self._pool = AlignedBufferPool(
            slot_bytes=slot_bytes,
            slot_count=slot_count,
        )
        try:
            if not hasattr(os, "O_DIRECT") or not self._probe_direct_io():
                raise OSError("O_DIRECT is required but unavailable for the cache root")
            self.recover()
        except BaseException:
            self._pool.close()
            raise

    @property
    def buffer_budget_bytes(self) -> int:
        return self._pool.budget_bytes

    def set_quarantine_hook(
        self, hook: Callable[[str], None] | None
    ) -> None:
        if hook is not None and not callable(hook):
            raise TypeError("quarantine hook must be callable")
        self._quarantine_hook = hook

    def set_withdraw_hook(
        self, hook: Callable[[str], None] | None
    ) -> None:
        if hook is not None and not callable(hook):
            raise TypeError("withdraw hook must be callable")
        self._withdraw_hook = hook

    def _record_quarantine(self, reason: str) -> None:
        if reason not in QUARANTINE_REASONS:
            raise ValueError("quarantine reason is outside the bounded vocabulary")
        hook = self._quarantine_hook
        if hook is None:
            return
        try:
            hook(reason)
        except Exception:
            # A telemetry sink must never convert an authenticated cache miss
            # into a request failure. Fatal restore still takes its separate
            # fail-stop path even if this best-effort notification fails.
            logger.exception("spoolcache: quarantine telemetry hook failed")

    def _record_withdrawal(self, entry_id: str) -> None:
        hook = self._withdraw_hook
        if hook is None:
            return
        try:
            hook(entry_id)
        except Exception:
            # A stale offer is unsafe, so a worker callback failure must be
            # visible to the caller rather than downgraded to best effort.
            logger.exception("spoolcache: cache-offer withdrawal hook failed")
            raise

    def _prepare_root(self) -> None:
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ValueError("SpoolCache root must be an absolute narrow path")
        normalized = Path(os.path.abspath(self.root))
        if normalized != self.root:
            raise ValueError("SpoolCache root must not contain relative components")
        for component in reversed((self.root, *self.root.parents)):
            if not component.exists():
                continue
            mode = component.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"SpoolCache root contains a symlink: {component}")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("SpoolCache root is not a directory")
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / ".spoolcache-root"
        has_existing = any(path != marker for path in self.root.iterdir())
        if marker.is_symlink():
            raise ValueError("SpoolCache root marker must not be a symlink")
        if not marker.exists() and has_existing:
            raise ValueError("refusing to adopt a non-empty unowned cache root")
        if marker.exists():
            try:
                payload = _read_small_json(marker, maximum=4096)
            except (OSError, UnicodeDecodeError, ValueError) as error:
                raise ValueError("SpoolCache root marker is invalid") from error
            if payload != {"schema": _ROOT_MARKER_SCHEMA}:
                raise ValueError("SpoolCache root marker schema is unsupported")
        else:
            self._atomic_small_write(
                marker,
                canonical_json({"schema": _ROOT_MARKER_SCHEMA}) + b"\n",
            )
        for name in ("objects", "manifests", "quarantine", "tmp", "state"):
            path = self.root / name
            created = False
            try:
                path.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                pass
            if path.is_symlink():
                raise ValueError(f"SpoolCache managed directory is a symlink: {path}")
            if not path.is_dir():
                raise ValueError(f"SpoolCache managed path is not a directory: {path}")
            if created:
                _fsync_directory(self.root)
        # Complete any earlier mkdir whose parent receipt failed. Reopening an
        # otherwise valid store must make these namespace links durable before
        # publishing data or state beneath them.
        _fsync_directory(self.root)
        lock = self.root / "state" / "maintenance.lock"
        if lock.is_symlink():
            raise ValueError("SpoolCache maintenance lock must not be a symlink")
        if not lock.exists():
            descriptor = os.open(
                lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _o_nofollow(),
                0o600,
            )
            os.close(descriptor)
            _fsync_directory(lock.parent)
        if not stat.S_ISREG(lock.lstat().st_mode):
            raise ValueError("SpoolCache maintenance lock is not a regular file")
        owner_lock = self.root / "state" / _INVENTORY_OWNER_LOCK_NAME
        if owner_lock.is_symlink():
            raise ValueError("SpoolCache inventory owner lock must not be a symlink")
        if not owner_lock.exists():
            descriptor = os.open(
                owner_lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _o_nofollow(),
                0o600,
            )
            os.close(descriptor)
            _fsync_directory(owner_lock.parent)
        if not stat.S_ISREG(owner_lock.lstat().st_mode):
            raise ValueError("SpoolCache inventory owner lock is not a regular file")
        for name, label in (
            (_INVENTORY_WITHDRAWAL_DIRECTORY, "inventory-withdrawal"),
            (_OBJECT_WITHDRAWAL_DIRECTORY, "object-withdrawal"),
        ):
            withdrawn = self.root / "state" / name
            created = False
            try:
                withdrawn.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                pass
            if withdrawn.is_symlink() or not withdrawn.is_dir():
                raise ValueError(
                    f"SpoolCache {label} state is not a directory"
                )
            if created:
                _fsync_directory(withdrawn.parent)
        # As with an individual marker, an existing namespace may be the
        # residue of a successful mkdir followed by a failed state-directory
        # fsync. Reassert the ancestor receipt on every open.
        _fsync_directory(self.root / "state")
        # A second process can be finishing a failed quarantine while this
        # store instance opens. Validate the namespace under the same
        # cross-process lock used to create and clear individual markers.
        with self._exclusive():
            self._validate_inventory_withdrawal_namespace()
            self._validate_object_withdrawal_namespace()

    def _checkpoint(self, stage: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage)

    def acquire_inventory_owner(self) -> None:
        """Hold the rank-root inventory lease until this store is closed.

        The maintenance lock serializes individual namespace operations, but
        it does not prevent two worker generations from retaining independent
        in-memory reporters. Only one vLLM worker may therefore own the durable
        rank inventory at a time. Standalone maintenance processes deliberately
        do not take this lifetime lease.
        """

        self._ensure_open()
        with self._thread_lock:
            if self._inventory_owner_descriptor is not None:
                return
            path = self.root / "state" / _INVENTORY_OWNER_LOCK_NAME
            descriptor = os.open(path, os.O_RDWR | _o_nofollow())
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        "SpoolCache inventory owner lock is not a regular file"
                    )
                try:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as error:
                    raise RuntimeError(
                        "SpoolCache rank inventory already has a live owner"
                    ) from error
            except BaseException:
                os.close(descriptor)
                raise
            self._inventory_owner_descriptor = descriptor

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock:
            depth = getattr(self._lock_state, "depth", 0)
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock_state.depth = depth
                return
            lock_path = self.root / "state" / "maintenance.lock"
            descriptor = os.open(lock_path, os.O_RDWR | _o_nofollow())
            locked = False
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        "SpoolCache maintenance lock is not a regular file"
                    )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                self._lock_state.depth = 1
                yield
            finally:
                self._lock_state.depth = 0
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _probe_direct_io(self) -> bool:
        probe = self.root / "tmp" / f"direct-probe-{uuid.uuid4().hex}.part"
        descriptor = -1
        try:
            descriptor = os.open(
                probe,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_DIRECT
                | _o_nofollow(),
                0o600,
            )
            with self._pool.acquire(block=True) as slot:
                view = slot.view
                try:
                    view[:_ALIGNMENT] = b"\x00" * _ALIGNMENT
                    written = os.write(descriptor, view[:_ALIGNMENT])
                    if written != _ALIGNMENT:
                        return False
                finally:
                    view.release()
            os.fsync(descriptor)
            return True
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,
                errno.EPERM,
            }:
                raise
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

    def _atomic_small_write(self, target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _o_nofollow(),
            0o600,
        )
        try:
            _write_all(descriptor, memoryview(payload))
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
            os.replace(temporary, target)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        _fsync_directory(target.parent)

    def reserve_inventory_generation_epoch(self) -> int:
        """Durably reserve a rank-local epoch before advertising inventory.

        Legacy wall-clock epochs are migrated into a disjoint high domain.
        Once an epoch has been advertised, the fsynced state makes every later
        worker generation strictly larger even if the clock moves backwards.
        A durable sentry distinguishes first migration from lost/rolled-back
        state; ambiguity fails startup instead of recreating an unsafe epoch.
        """

        self._ensure_open()
        path = self.root / "state" / GENERATION_STATE_NAME
        required_path = self.root / "state" / GENERATION_REQUIRED_NAME
        with self._exclusive():
            _cleanup_atomic_write_temporaries(path)
            _cleanup_atomic_write_temporaries(required_path)
            required_exists = False
            try:
                required_metadata = required_path.lstat()
            except FileNotFoundError:
                pass
            else:
                required_exists = True
                if not stat.S_ISREG(required_metadata.st_mode):
                    raise ValueError(
                        "inventory generation sentry is not a regular file"
                    )
                try:
                    required_payload = _read_small_json(
                        required_path,
                        maximum=_MAX_GENERATION_STATE_BYTES,
                    )
                except (OSError, UnicodeDecodeError, ValueError) as error:
                    raise ValueError(
                        "inventory generation sentry is invalid"
                    ) from error
                if required_payload != {"schema": GENERATION_REQUIRED_SCHEMA}:
                    raise ValueError(
                        "inventory generation sentry schema is unsupported"
                    )
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                if required_exists:
                    raise ValueError(
                        "inventory generation state is missing after initialization"
                    )
                previous = 0
            else:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        "inventory generation state is not a regular file"
                    )
                try:
                    payload = _read_small_json(
                        path,
                        maximum=_MAX_GENERATION_STATE_BYTES,
                    )
                except (OSError, UnicodeDecodeError, ValueError) as error:
                    raise ValueError(
                        "inventory generation state is invalid"
                    ) from error
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"schema", "epoch"}
                    or payload.get("schema") != GENERATION_STATE_SCHEMA
                ):
                    raise ValueError(
                        "inventory generation state schema is unsupported"
                    )
                previous = payload.get("epoch")
                if (
                    isinstance(previous, bool)
                    or not isinstance(previous, int)
                    or not 0 < previous <= _MAX_GENERATION_EPOCH
                ):
                    raise ValueError("inventory generation epoch is invalid")
            if previous >= _MAX_GENERATION_EPOCH:
                raise RuntimeError("inventory generation epoch overflowed")
            clock_epoch = time.time_ns()
            if (
                isinstance(clock_epoch, bool)
                or not isinstance(clock_epoch, int)
                or not 0 < clock_epoch <= _MAX_GENERATION_EPOCH
            ):
                raise RuntimeError("system clock cannot seed inventory generation")
            if clock_epoch > (
                _MAX_GENERATION_EPOCH - _PERSISTENT_GENERATION_EPOCH_BASE
            ):
                raise RuntimeError("system clock exceeds inventory generation domain")
            persistent_clock = _PERSISTENT_GENERATION_EPOCH_BASE + clock_epoch
            if previous < _PERSISTENT_GENERATION_EPOCH_BASE:
                # ``previous == 0`` is a fresh or legacy root. A low nonzero
                # state was produced by the pre-domain implementation. Both
                # migrate above every raw wall-clock epoch it could advertise.
                epoch = max(
                    persistent_clock,
                    _PERSISTENT_GENERATION_EPOCH_BASE + previous,
                )
            else:
                epoch = max(previous + 1, persistent_clock)
            self._atomic_small_write(
                path,
                canonical_json(
                    {
                        "schema": GENERATION_STATE_SCHEMA,
                        "epoch": epoch,
                    }
                )
                + b"\n",
            )
            if not required_exists:
                self._atomic_small_write(
                    required_path,
                    canonical_json({"schema": GENERATION_REQUIRED_SCHEMA})
                    + b"\n",
                )
            return epoch

    def _object_path(self, digest: str) -> Path:
        return self.root / "objects" / digest[:2] / f"{digest}.spool"

    def object_path(self, digest: str) -> Path:
        """Return the canonical path for an already validated object digest."""

        if not _is_digest(digest):
            raise ValueError("object digest is malformed")
        return self._object_path(digest)

    def _manifest_path(self, entry_id: str) -> Path:
        return self.root / "manifests" / entry_id[:2] / f"{entry_id}.json"

    def _inventory_withdrawal_path(self, entry_id: str) -> Path:
        return (
            self.root
            / "state"
            / _INVENTORY_WITHDRAWAL_DIRECTORY
            / entry_id
        )

    def _object_withdrawal_path(self, object_digest: str) -> Path:
        return (
            self.root
            / "state"
            / _OBJECT_WITHDRAWAL_DIRECTORY
            / object_digest
        )

    def _validate_inventory_withdrawal_namespace(self) -> None:
        self._validate_digest_marker_namespace(
            self.root / "state" / _INVENTORY_WITHDRAWAL_DIRECTORY,
            label="inventory-withdrawal",
        )

    def _validate_object_withdrawal_namespace(self) -> None:
        self._validate_digest_marker_namespace(
            self.root / "state" / _OBJECT_WITHDRAWAL_DIRECTORY,
            label="object-withdrawal",
        )

    @staticmethod
    def _validate_digest_marker_namespace(root: Path, *, label: str) -> None:
        with os.scandir(root) as entries:
            for entry in entries:
                if not _is_digest(entry.name):
                    raise ValueError(
                        f"{label} state contains an invalid entry"
                    )
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A marker removed before this instance acquired the lock
                    # cannot happen for cooperating writers, but treating the
                    # already-absent name as harmless keeps startup robust to a
                    # completed pre-upgrade operation.
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(
                        f"{label} entry is not a directory"
                    )
                with os.scandir(entry.path) as children:
                    if next(children, None) is not None:
                        raise ValueError(
                            f"{label} entry is not empty"
                        )

    @staticmethod
    def _digest_marker_is_present(path: Path, *, label: str) -> bool:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} entry is not a directory")
        try:
            children = os.scandir(path)
        except FileNotFoundError:
            return False
        with children:
            if next(children, None) is not None:
                raise ValueError(f"{label} entry is not empty")
        return True

    @staticmethod
    def _mark_digest_marker(path: Path, *, label: str) -> None:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not ManifestStore._digest_marker_is_present(path, label=label):
            raise ValueError(f"{label} marker is invalid")
        # A previous mkdir may have succeeded while its parent-directory fsync
        # failed. Reassert the receipt even for an existing valid marker so an
        # idempotent retry cannot advance into namespace mutation with a fence
        # that would disappear after power loss.
        _fsync_directory(path.parent)

    @staticmethod
    def _clear_digest_marker(path: Path, *, label: str) -> None:
        if not ManifestStore._digest_marker_is_present(path, label=label):
            return
        path.rmdir()
        _fsync_directory(path.parent)

    def _inventory_is_withdrawn(self, entry_id: str) -> bool:
        return self._digest_marker_is_present(
            self._inventory_withdrawal_path(entry_id),
            label="inventory-withdrawal",
        )

    def _mark_inventory_withdrawn(self, entry_id: str) -> None:
        self._mark_digest_marker(
            self._inventory_withdrawal_path(entry_id),
            label="inventory-withdrawal",
        )

    def _clear_inventory_withdrawal(self, entry_id: str) -> None:
        self._clear_digest_marker(
            self._inventory_withdrawal_path(entry_id),
            label="inventory-withdrawal",
        )

    def _object_is_withdrawn(self, object_digest: str) -> bool:
        if not _is_digest(object_digest):
            raise ValueError("object withdrawal digest is malformed")
        return self._digest_marker_is_present(
            self._object_withdrawal_path(object_digest),
            label="object-withdrawal",
        )

    def _mark_object_withdrawn(self, object_digest: str) -> None:
        if not _is_digest(object_digest):
            raise ValueError("object withdrawal digest is malformed")
        self._mark_digest_marker(
            self._object_withdrawal_path(object_digest),
            label="object-withdrawal",
        )

    def _clear_object_withdrawal(self, object_digest: str) -> None:
        if not _is_digest(object_digest):
            raise ValueError("object withdrawal digest is malformed")
        self._clear_digest_marker(
            self._object_withdrawal_path(object_digest),
            label="object-withdrawal",
        )

    def _object_withdrawals_present(self) -> bool:
        root = self.root / "state" / _OBJECT_WITHDRAWAL_DIRECTORY
        present = False
        with os.scandir(root) as entries:
            for entry in entries:
                present = True
                if not _is_digest(entry.name):
                    raise ValueError(
                        "object-withdrawal state contains an invalid entry"
                    )
                if not self._digest_marker_is_present(
                    Path(entry.path),
                    label="object-withdrawal",
                ):
                    raise RuntimeError("object-withdrawal marker disappeared")
        return present

    def _manifest_has_withdrawn_object(self, manifest: RankManifest) -> bool:
        return any(
            self._object_is_withdrawn(item.sha256)
            for item in manifest.objects
        )

    def pending_inventory_withdrawals(
        self,
        entry_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Filter an already-bounded reporter image through durable signals.

        An object fence is written before a corrupt shared object's manifest
        references are traversed. If a maintenance process stops midway, this
        bounded reporter pass still withdraws every held manifest which names
        the fenced content address.
        """

        with self._exclusive():
            pending: list[str] = []
            object_withdrawals_present = self._object_withdrawals_present()
            for entry_id in entry_ids:
                if not _is_digest(entry_id):
                    raise ValueError("inventory withdrawal entry is malformed")
                if self._inventory_is_withdrawn(entry_id):
                    pending.append(entry_id)
                    continue
                if not object_withdrawals_present:
                    continue
                try:
                    manifest, _ = self._read_manifest_file(
                        self._manifest_path(entry_id),
                        expected_entry_id=entry_id,
                        probe_objects=False,
                    )
                except (OSError, ManifestError):
                    # A held offer whose manifest cannot be inspected while an
                    # object fence is active cannot prove it is unrelated.
                    pending.append(entry_id)
                    continue
                if self._manifest_has_withdrawn_object(manifest):
                    pending.append(entry_id)
            return tuple(pending)

    def inventory_withdrawal_marker_batch(
        self,
        limit: int,
        *,
        after: str | None = None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Return one bounded, eventually fair entry-marker reconciliation page."""

        with self._exclusive():
            return self._digest_marker_batch(
                self.root / "state" / _INVENTORY_WITHDRAWAL_DIRECTORY,
                limit,
                after=after,
                label="inventory-withdrawal",
            )

    def object_withdrawal_marker_batch(
        self,
        limit: int,
        *,
        after: str | None = None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Return one bounded, eventually fair object-fence reconciliation page."""

        with self._exclusive():
            return self._digest_marker_batch(
                self.root / "state" / _OBJECT_WITHDRAWAL_DIRECTORY,
                limit,
                after=after,
                label="object-withdrawal",
            )

    @staticmethod
    def _digest_marker_batch(
        root: Path,
        limit: int,
        *,
        after: str | None,
        label: str,
    ) -> tuple[tuple[str, ...], str | None]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 0 < limit <= 100_000
        ):
            raise ValueError(f"{label} batch limit must be positive")
        if after is not None and not _is_digest(after):
            raise ValueError(f"{label} batch cursor is malformed")

        def names(*, following: str | None) -> Iterator[str]:
            with os.scandir(root) as entries:
                for entry in entries:
                    name = entry.name
                    if not _is_digest(name):
                        raise ValueError(
                            f"{label} state contains an invalid entry"
                        )
                    if not ManifestStore._digest_marker_is_present(
                        Path(entry.path),
                        label=label,
                    ):
                        continue
                    if following is None or name > following:
                        yield name

        selected = heapq.nsmallest(limit, names(following=after))
        if not selected and after is not None:
            selected = heapq.nsmallest(limit, names(following=None))
        # A short final page resets the cursor so the next call wraps around.
        # ``heapq.nsmallest`` retains at most ``limit`` names even if the marker
        # namespace is much larger than the in-memory reporter.
        cursor = selected[-1] if len(selected) == limit else None
        return tuple(selected), cursor

    def acknowledge_absent_inventory_withdrawals(
        self,
        entry_ids: Iterable[str],
    ) -> None:
        """Retire signals whose live manifest is durably absent.

        The active reporter calls this only after applying every signal to its
        in-memory image. Live-manifest tombstones remain until a replacement
        commit or a full payload authentication proves the entry safe again.
        """

        with self._exclusive():
            for entry_id in entry_ids:
                if not _is_digest(entry_id):
                    raise ValueError("inventory withdrawal entry is malformed")
                manifest_path = self._manifest_path(entry_id)
                try:
                    manifest_path.lstat()
                except FileNotFoundError:
                    # A prior invalidate/quarantine can have removed the live
                    # name but failed its shard-directory fsync. Persist that
                    # absence before durably removing the fail-closed marker;
                    # otherwise a crash could restore the manifest without its
                    # withdrawal receipt.
                    self._fsync_manifest_absence_locked(manifest_path)
                    try:
                        manifest_path.lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        continue
                else:
                    continue
                self._clear_inventory_withdrawal(entry_id)

    def acknowledge_unreferenced_object_withdrawals(
        self,
        object_digests: Iterable[str],
    ) -> None:
        """Retire bounded object fences which no live manifest can reference."""

        with self._exclusive():
            for object_digest in object_digests:
                if not _is_digest(object_digest):
                    raise ValueError("object withdrawal digest is malformed")
                relative_path = (
                    f"objects/{object_digest[:2]}/{object_digest}.spool"
                )
                if self.object_is_referenced(relative_path):
                    continue
                # The last referencing manifest may have been unlinked by an
                # operation whose directory receipt failed. Make every fixed
                # canonical manifest shard durable, then recheck the namespace,
                # before retiring the object-wide crash fence.
                self._fsync_manifest_namespace_locked()
                if self.object_is_referenced(relative_path):
                    continue
                self._clear_object_withdrawal(object_digest)

    def release_authenticated_inventory(self, manifest: RankManifest) -> bool:
        """Release a live tombstone after deep scrub authenticated its image."""

        entry_id = manifest.entry_id
        if not _is_digest(entry_id):
            raise ValueError("authenticated inventory entry is malformed")
        with self._exclusive():
            released = self._inventory_is_withdrawn(entry_id)
            try:
                metadata = self._manifest_path(entry_id).lstat()
            except FileNotFoundError:
                raise RuntimeError("authenticated inventory manifest is absent")
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("authenticated inventory manifest is invalid")
            # Authentication proves the currently visible bytes. Complete any
            # previously uncertain atomic-replacement directory receipt before
            # retiring the fail-closed marker.
            object_directories = {
                (self.root / item.relative_path).parent
                for item in manifest.objects
            }
            for directory in sorted(object_directories):
                _fsync_directory(directory)
            # A fence may have appeared after the scrub streamed this object
            # but before it acquired the final linearization lock. Re-hash only
            # fenced objects under this lock before releasing the shared
            # content address for every referring manifest.
            for item in manifest.objects:
                if not self._object_is_withdrawn(item.sha256):
                    continue
                released = True
                if not self._verify_path(
                    self.root / item.relative_path,
                    digest=item.sha256,
                    logical_length=item.byte_length,
                    stored_length=item.stored_length,
                    hash_payload=True,
                ):
                    raise ObjectCorruptionError(
                        "fenced object did not pass final authentication"
                    )
                _fsync_directory((self.root / item.relative_path).parent)
                self._clear_object_withdrawal(item.sha256)
            self._clear_inventory_withdrawal(entry_id)
            return released

    def _fsync_manifest_absence_locked(self, manifest_path: Path) -> None:
        """Persist one absent manifest name before clearing its tombstone."""

        shard = manifest_path.parent
        try:
            metadata = shard.lstat()
        except FileNotFoundError:
            _fsync_directory(self.root / "manifests")
            return
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("manifest shard is not a directory")
        _fsync_directory(shard)

    def _fsync_manifest_namespace_locked(self) -> None:
        """Persist every possible canonical manifest unlink in fixed space."""

        root = self.root / "manifests"
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("manifest namespace is not a directory")
        for shard_index in range(256):
            shard = root / f"{shard_index:02x}"
            try:
                shard_metadata = shard.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(shard_metadata.st_mode):
                raise ValueError("manifest shard is not a directory")
            _fsync_directory(shard)
        _fsync_directory(root)

    def _withdraw_object_references_locked(
        self,
        relative_path: str,
    ) -> int:
        """Persistently fence an object and stream-withdraw all live references."""

        object_digest = _object_digest_from_relative_path(relative_path)
        # This single durable fence precedes the potentially unbounded manifest
        # traversal. Startup scan, lookup, and bounded reporter reconciliation
        # all honor it, so process loss after any reference cannot expose a
        # later, not-yet-visited manifest.
        self._mark_object_withdrawn(object_digest)
        self._checkpoint("object-withdrawal-after-fence")
        withdrawn = 0
        deferred_error: Exception | None = None
        for manifest_path in self.iter_manifest_paths():
            entry_id = manifest_path.stem
            references_object = False
            try:
                manifest, _ = self._read_manifest_file(
                    manifest_path,
                    expected_entry_id=entry_id,
                    probe_objects=False,
                )
            except (OSError, ManifestError):
                # A malformed manifest cannot prove that it does not reference
                # the known-bad content address, and is unsafe independently.
                references_object = _is_digest(entry_id)
            else:
                references_object = any(
                    item.relative_path == relative_path
                    for item in manifest.objects
                )
            if not references_object:
                continue
            withdrawn += 1
            try:
                self._mark_inventory_withdrawn(entry_id)
            except Exception as error:
                deferred_error = deferred_error or error
            try:
                self._record_withdrawal(entry_id)
            except Exception as error:
                deferred_error = deferred_error or error
            self._checkpoint("object-withdrawal-after-reference")
        if deferred_error is not None:
            raise _NamespaceMutationError(
                "object collision withdrawal did not get a complete receipt",
                deferred_error,
            ) from deferred_error
        return withdrawn

    def _ensure_shard_directory(self, path: Path) -> None:
        parent = path.parent
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("SpoolCache shard path is not a directory")
        # Retrying after a successful mkdir plus failed parent fsync must not
        # publish content beneath an unreceipted shard link.
        _fsync_directory(parent)

    def put_object(
        self,
        chunks: Iterable[bytes | bytearray | memoryview],
    ) -> tuple[str, int, int]:
        """Write one immutable object and return digest/logical/stored lengths."""

        self._ensure_open()
        # Public callers may publish an object without going through
        # ``commit``. Cover the complete temporary-file lifetime so recovery,
        # deep scrub, and capacity GC cannot mistake an in-flight object for
        # abandoned state. ``commit`` is already locked and the maintenance
        # lock is deliberately reentrant for that nested path.
        with self._exclusive():
            return self._put_object_locked(chunks)

    def _put_object_locked(
        self,
        chunks: Iterable[bytes | bytearray | memoryview],
    ) -> tuple[str, int, int]:
        temporary = self.root / "tmp" / f"object-{uuid.uuid4().hex}.part"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT | _o_nofollow()
        descriptor = os.open(temporary, flags, 0o600)
        logical_length = 0
        stored_length = 0
        digest = hashlib.sha256()
        try:
            with self._pool.acquire(block=True) as slot:
                target = slot.view
                try:
                    filled = 0
                    for raw_chunk in chunks:
                        source = memoryview(raw_chunk).cast("B")
                        try:
                            cursor = 0
                            while cursor < len(source):
                                count = min(slot.size - filled, len(source) - cursor)
                                target[filled : filled + count] = source[
                                    cursor : cursor + count
                                ]
                                digest.update(source[cursor : cursor + count])
                                logical_length += count
                                filled += count
                                cursor += count
                                if filled == slot.size:
                                    self._checkpoint("object-before-write")
                                    _write_exact(
                                        descriptor,
                                        target[:filled],
                                    )
                                    self._checkpoint("object-after-write")
                                    stored_length += filled
                                    filled = 0
                        finally:
                            source.release()
                    if logical_length <= 0:
                        raise ValueError("immutable objects cannot be empty")
                    if filled:
                        write_length = _round_up(filled, _ALIGNMENT)
                        target[filled:write_length] = b"\x00" * (
                            write_length - filled
                        )
                        self._checkpoint("object-before-write")
                        _write_exact(
                            descriptor,
                            target[:write_length],
                        )
                        self._checkpoint("object-after-write")
                        stored_length += write_length
                finally:
                    target.release()
            self._checkpoint("object-before-fsync")
            os.fsync(descriptor)
            self._checkpoint("object-after-fsync")
        except BaseException:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        object_digest = digest.hexdigest()
        destination = self._object_path(object_digest)
        try:
            # Shard creation is part of the temporary object's cleanup scope.
            # A failed parent-directory durability receipt must not strand the
            # already-written full payload in tmp on every retry.
            self._ensure_shard_directory(destination.parent)
            self._checkpoint("object-before-link")
            os.link(temporary, destination, follow_symlinks=False)
            self._checkpoint("object-after-link")
            self._checkpoint("object-before-directory-fsync")
            _fsync_directory(destination.parent)
            self._checkpoint("object-after-directory-fsync")
        except FileExistsError:
            if not self._verify_path(
                destination,
                digest=object_digest,
                logical_length=logical_length,
                stored_length=stored_length,
            ):
                relative_path = destination.relative_to(self.root).as_posix()
                # Persist the fail-closed decision before touching the live
                # content address. A process loss at any later boundary leaves
                # shallow scans unable to advertise its referring manifests.
                try:
                    self._withdraw_object_references_locked(relative_path)
                except Exception as error:
                    if isinstance(error, _NamespaceMutationError):
                        raise error
                    raise _NamespaceMutationError(
                        "object collision repair did not get a complete receipt",
                        error,
                    ) from error
                quarantine = (
                    self.root
                    / "quarantine"
                    / f"{destination.name}.{uuid.uuid4().hex}.bad"
                )
                try:
                    # Keep forensic evidence on the old inode, then atomically
                    # switch the live name from corrupt data to the already
                    # fsynced temporary object. There is never a missing-object
                    # gap beneath an existing manifest.
                    self._checkpoint("object-collision-before-evidence-link")
                    os.link(destination, quarantine, follow_symlinks=False)
                    self._checkpoint("object-collision-after-evidence-link")
                    self._checkpoint(
                        "object-collision-before-quarantine-directory-fsync"
                    )
                    _fsync_directory(self.root / "quarantine")
                    self._checkpoint(
                        "object-collision-after-quarantine-directory-fsync"
                    )
                    self._record_quarantine("object_collision")
                    self._checkpoint("object-collision-before-replace")
                    os.replace(temporary, destination)
                    self._checkpoint("object-collision-after-replace")
                    self._checkpoint(
                        "object-collision-before-object-directory-fsync"
                    )
                    _fsync_directory(destination.parent)
                    self._checkpoint(
                        "object-collision-after-object-directory-fsync"
                    )
                    # The new live object is content-authenticated and durable.
                    # Object repair proves only this digest. Entry tombstones
                    # deliberately retain no inferred provenance and remain
                    # until each complete manifest is authenticated or replaced.
                    self._checkpoint("object-collision-before-fence-release")
                    self._clear_object_withdrawal(object_digest)
                    self._checkpoint("object-collision-after-fence-release")
                except Exception as error:
                    raise _NamespaceMutationError(
                        "object collision repair did not get a complete receipt",
                        error,
                    ) from error
            elif self._object_is_withdrawn(object_digest):
                # Recover a process loss after the atomic replacement and its
                # directory fsync. A fresh full hash above proves that the live
                # content address is now safe to release.
                _fsync_directory(destination.parent)
                self._clear_object_withdrawal(object_digest)
        finally:
            _cleanup_temporary_file(
                temporary,
                primary_error=sys.exc_info()[1],
            )
        return object_digest, logical_length, stored_length

    def put_source(self, source: ObjectSource) -> ObjectDescriptor:
        digest, logical_length, stored_length = self.put_object(source.chunks)
        return ObjectDescriptor(
            group_index=source.group_index,
            layer_name=source.layer_name,
            page_start=source.page_start,
            page_count=source.page_count,
            byte_length=logical_length,
            stored_length=stored_length,
            sha256=digest,
            relative_path=f"objects/{digest[:2]}/{digest}.spool",
        )

    def publish_manifest(self, manifest: RankManifest) -> str:
        """Atomically expose a manifest after checking all referenced objects."""

        self._ensure_open()
        with self._exclusive():
            return self._publish_manifest_locked(manifest)

    def _publish_manifest_locked(self, manifest: RankManifest) -> str:
        self._validate_expected_identity(manifest)
        for descriptor in manifest.objects:
            path = self.root / descriptor.relative_path
            if not self._verify_path(
                path,
                digest=descriptor.sha256,
                logical_length=descriptor.byte_length,
                stored_length=descriptor.stored_length,
                hash_payload=False,
            ):
                raise ObjectCorruptionError(
                    f"cannot publish missing or invalid object {descriptor.sha256}"
                )
        encoded = encode_manifest(manifest)
        manifest_digest = hashlib.sha256(encoded).hexdigest()
        destination = self._manifest_path(manifest.entry_id)
        self._ensure_shard_directory(destination.parent)
        temporary = self.root / "tmp" / f"manifest-{uuid.uuid4().hex}.part"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _o_nofollow(),
            0o600,
        )
        try:
            self._checkpoint("manifest-before-write")
            _write_all(descriptor, memoryview(encoded))
            self._checkpoint("manifest-after-write")
            self._checkpoint("manifest-before-fsync")
            os.fsync(descriptor)
            self._checkpoint("manifest-after-fsync")
        except BaseException:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        with self._exclusive():
            try:
                self._checkpoint("manifest-before-link")
                os.link(temporary, destination, follow_symlinks=False)
                self._checkpoint("manifest-after-link")
                self._checkpoint("manifest-before-directory-fsync")
                _fsync_directory(destination.parent)
                self._checkpoint("manifest-after-directory-fsync")
            except FileExistsError:
                try:
                    existing, existing_encoded = self._read_manifest_file(
                        destination,
                        expected_entry_id=manifest.entry_id,
                        probe_objects=False,
                    )
                except (OSError, ManifestError):
                    self._quarantine_manifest(
                        destination,
                        reason="manifest_validation",
                    )
                    os.link(temporary, destination, follow_symlinks=False)
                    _fsync_directory(destination.parent)
                else:
                    if not _manifests_equivalent(existing, manifest):
                        raise ManifestError(
                            "an existing entry has incompatible immutable content"
                        )
                    manifest_digest = hashlib.sha256(existing_encoded).hexdigest()
            finally:
                _cleanup_temporary_file(
                    temporary,
                    primary_error=sys.exc_info()[1],
                )
        return manifest_digest

    def commit(
        self,
        *,
        entry_id: str,
        deployment_identity_digest: str,
        rank_identity_digest: str,
        span_tokens: int,
        physical_rank: int,
        topology_digest: str,
        profile: str,
        layout_digest: str,
        sources: Iterable[ObjectSource],
        created_at_unix_ns: int | None = None,
    ) -> RankManifest:
        # Hold the cross-process maintenance lock for the whole transaction.
        # Otherwise an orphan collector could unlink a newly published object
        # in the gap before its manifest becomes the visibility point.
        with self._exclusive():
            descriptors = ordered_descriptors(
                tuple(self.put_source(source) for source in sources)
            )
            manifest = RankManifest(
                entry_id=entry_id,
                deployment_identity_digest=deployment_identity_digest,
                rank_identity_digest=rank_identity_digest,
                span_tokens=span_tokens,
                physical_rank=physical_rank,
                topology_digest=topology_digest,
                profile=profile,
                layout_digest=layout_digest,
                objects=descriptors,
                created_at_unix_ns=(
                    time.time_ns()
                    if created_at_unix_ns is None
                    else created_at_unix_ns
                ),
            )
            self.publish_manifest(manifest)
            # A previous failed quarantine can leave a durable tombstone.
            # Only a complete replacement commit may make that entry
            # reportable again.
            self._clear_inventory_withdrawal(entry_id)
            return manifest

    def lookup(self, entry_id: str, *, verify_payloads: bool = False) -> LookupResult:
        return self._lookup(
            entry_id,
            verify_payloads=verify_payloads,
            update_recency=True,
        )

    def _lookup(
        self,
        entry_id: str,
        *,
        verify_payloads: bool,
        update_recency: bool,
        defer_quarantine: Callable[[Path, str], None] | None = None,
    ) -> LookupResult:
        self._ensure_open()
        if not _is_digest(entry_id):
            return LookupResult(False, "malformed-key")
        if self._inventory_is_withdrawn(entry_id):
            return LookupResult(False, "withdrawn")
        path = self._manifest_path(entry_id)
        try:
            manifest, encoded = self._read_manifest_file(
                path,
                expected_entry_id=entry_id,
                probe_objects=True,
            )
        except FileNotFoundError:
            return LookupResult(False, "absent")
        except OSError as error:
            if defer_quarantine is None:
                self._quarantine_manifest(path, reason="manifest_io")
            else:
                defer_quarantine(path, "manifest_io")
            return LookupResult(False, f"corrupt:{type(error).__name__}")
        except ManifestError as error:
            if defer_quarantine is None:
                self._quarantine_manifest(path, reason="manifest_validation")
            else:
                defer_quarantine(path, "manifest_validation")
            return LookupResult(False, f"corrupt:{type(error).__name__}")
        if self._manifest_has_withdrawn_object(manifest):
            return LookupResult(False, "withdrawn")
        if verify_payloads:
            for item in manifest.objects:
                object_path = self.root / item.relative_path
                if not self._verify_path(
                    object_path,
                    digest=item.sha256,
                    logical_length=item.byte_length,
                    stored_length=item.stored_length,
                    hash_payload=True,
                ):
                    self.quarantine_object(
                        item,
                        reason="payload_checksum",
                    )
                    return LookupResult(False, "corrupt:ObjectCorruptionError")
        if update_recency:
            try:
                os.utime(path, None, follow_symlinks=False)
            except OSError:
                pass
        return LookupResult(
            True,
            "hit",
            manifest=manifest,
            manifest_digest=hashlib.sha256(encoded).hexdigest(),
        )

    def _read_manifest_file(
        self,
        path: Path,
        *,
        expected_entry_id: str | None,
        probe_objects: bool,
    ) -> tuple[RankManifest, bytes]:
        descriptor = os.open(path, os.O_RDONLY | _o_nofollow())
        try:
            stat_result = os.fstat(descriptor)
            if not stat.S_ISREG(stat_result.st_mode):
                raise ManifestError("manifest is not a regular file")
            if stat_result.st_size <= 0 or stat_result.st_size > MAX_MANIFEST_BYTES:
                raise ManifestError("manifest file size is invalid")
            encoded = _read_exact_file(descriptor, stat_result.st_size)
        finally:
            os.close(descriptor)
        manifest = decode_manifest(encoded).manifest
        if expected_entry_id is not None and manifest.entry_id != expected_entry_id:
            raise ManifestError("manifest path and entry ID disagree")
        self._validate_expected_identity(manifest)
        if probe_objects:
            for item in manifest.objects:
                object_path = self.root / item.relative_path
                try:
                    metadata = object_path.lstat()
                except FileNotFoundError as error:
                    raise ObjectCorruptionError(
                        f"object is missing: {item.sha256}"
                    ) from error
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != item.stored_length
                ):
                    raise ObjectCorruptionError(
                        f"object metadata differs: {item.sha256}"
                    )
        return manifest, encoded

    def stream_object(
        self,
        descriptor: ObjectDescriptor,
        sink: Callable[[memoryview], None],
        *,
        on_bytes_read: Callable[[int], None] | None = None,
    ) -> None:
        """Read, authenticate, and synchronously pass bounded views to ``sink``.

        The sink must consume each view before returning and must not retain it.
        Authentication completes only after the last sink call. A connector
        therefore writes only request-private GPU blocks and reports completion
        only after this function returns successfully.
        """

        if on_bytes_read is not None and not callable(on_bytes_read):
            raise TypeError("object read observer must be callable")
        path = self.root / descriptor.relative_path
        flags = os.O_RDONLY | os.O_DIRECT | _o_nofollow()
        file_descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        logical_remaining = descriptor.byte_length
        stored_remaining = descriptor.stored_length
        try:
            actual_size = os.fstat(file_descriptor).st_size
            if actual_size != descriptor.stored_length:
                raise ObjectCorruptionError("object stored length differs")
            with self._pool.acquire(block=True) as slot:
                target = slot.view
                try:
                    while stored_remaining:
                        count = min(slot.size, stored_remaining)
                        if count % _ALIGNMENT:
                            raise ObjectCorruptionError(
                                "direct-I/O object length is not aligned"
                            )
                        read = _read_into_exact(
                            file_descriptor,
                            target[:count],
                        )
                        if on_bytes_read is not None:
                            on_bytes_read(read)
                        logical = min(logical_remaining, read)
                        if logical:
                            digest.update(target[:logical])
                            sink(target[:logical])
                            logical_remaining -= logical
                        if read > logical and any(target[logical:read]):
                            raise ObjectCorruptionError("object padding is not zero")
                        stored_remaining -= read
                finally:
                    target.release()
            if logical_remaining or digest.hexdigest() != descriptor.sha256:
                raise ObjectCorruptionError("object SHA-256 differs")
        finally:
            os.close(file_descriptor)

    def read_object_bytes(self, descriptor: ObjectDescriptor) -> bytes:
        """Testing/offline helper. The production connector must stream instead."""

        parts: list[bytes] = []
        self.stream_object(descriptor, lambda view: parts.append(bytes(view)))
        return b"".join(parts)

    def scan_offers(self, limit: int) -> tuple[ManifestOffer, ...]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 0 < limit <= 100_000
        ):
            raise ValueError("offer scan limit must be positive")
        with self._exclusive():
            deferred_quarantines: list[tuple[Path, str]] = []

            def defer_quarantine(path: Path, reason: str) -> None:
                if len(deferred_quarantines) < _SCAN_QUARANTINE_BATCH:
                    deferred_quarantines.append((path, reason))

            def candidates() -> Iterator[tuple[int, str, ManifestOffer]]:
                for path in self.iter_manifest_paths():
                    try:
                        metadata = path.lstat()
                    except FileNotFoundError:
                        continue
                    # Validate before competing for the bounded newest-entry
                    # heap. Otherwise newer withdrawn/corrupt manifests can
                    # fill the raw candidate window and starve older healthy
                    # cache entries from startup/rescan inventory forever.
                    result = self._lookup(
                        path.stem,
                        verify_payloads=False,
                        update_recency=False,
                        defer_quarantine=defer_quarantine,
                    )
                    if not result.is_hit or result.manifest is None:
                        continue
                    yield (
                        metadata.st_mtime_ns,
                        path.as_posix(),
                        ManifestOffer(
                            entry_id=result.manifest.entry_id,
                            span_tokens=result.manifest.span_tokens,
                            manifest_digest=result.manifest_digest or "",
                            modified_at_unix_ns=metadata.st_mtime_ns,
                        ),
                    )

            # Validate the namespace as a stream, retaining only the caller's
            # bounded number of newest healthy offers. Holding the maintenance
            # lock keeps the snapshot, validation, and mtimes consistent with
            # cooperating scrub/GC operations.
            selected = heapq.nlargest(
                limit,
                candidates(),
                key=lambda item: item[:2],
            )
            # ``candidates`` is exhausted here, so every nested scandir stream
            # has closed. POSIX does not promise snapshot iteration semantics
            # when the same directory is mutated during readdir; quarantine a
            # fixed batch only after the healthy-offer pass has completed.
            for path, reason in deferred_quarantines:
                self._quarantine_manifest(path, reason=reason)
            return tuple(item[2] for item in selected)

    def invalidate(self, entry_id: str) -> bool:
        if not _is_digest(entry_id):
            return False
        path = self._manifest_path(entry_id)
        with self._exclusive():
            try:
                path.lstat()
            except FileNotFoundError:
                return False
            # Persist a cross-process signal before removing the visibility
            # point. A standalone invalidator cannot directly mutate the live
            # worker's in-memory reporter.
            self._mark_inventory_withdrawn(entry_id)
            withdrawal_error: Exception | None = None
            try:
                self._record_withdrawal(entry_id)
            except Exception as error:
                withdrawal_error = error
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            durability_error = withdrawal_error
            try:
                _fsync_directory(path.parent)
            except Exception as error:
                durability_error = durability_error or error
            if durability_error is not None:
                raise _NamespaceMutationError(
                    "manifest invalidation changed namespace without a complete receipt",
                    durability_error,
                ) from durability_error
            if (
                self._inventory_owner_descriptor is not None
                and self._withdraw_hook is not None
            ):
                self._clear_inventory_withdrawal(entry_id)
            return True

    def recover(self) -> int:
        """Remove incomplete temporary files left by a terminated process."""

        removed = 0
        with self._exclusive():
            for path in (self.root / "tmp").iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
                    removed += 1
            _fsync_directory(self.root / "tmp")
        return removed

    def quarantine_object(
        self,
        descriptor: ObjectDescriptor,
        *,
        reason: str,
    ) -> int:
        """Quarantine a bad object and every manifest that references it.

        The manifest set is withdrawn first while holding the rank-wide
        maintenance lock.  Moving the immutable object afterwards prevents a
        second entry which shares the same content address from remaining
        advertised with a known-bad payload.
        """

        if reason not in {"payload_checksum", "payload_size"}:
            raise ValueError("object quarantine reason is invalid")
        withdrawn = 0
        deferred_error: Exception | None = None
        with self._exclusive():
            # One durable object fence makes the following streaming manifest
            # traversal crash-atomic without retaining an unbounded reference
            # list. Shallow startup scans and worker reconciliation both honor
            # it until every reference and namespace receipt is complete.
            self._mark_object_withdrawn(descriptor.sha256)
            self._checkpoint("object-quarantine-after-fence")
            for manifest_path in self.iter_manifest_paths():
                try:
                    manifest, _ = self._read_manifest_file(
                        manifest_path,
                        expected_entry_id=manifest_path.stem,
                        probe_objects=False,
                    )
                except (OSError, ManifestError):
                    try:
                        self._quarantine_manifest(
                            manifest_path,
                            reason="manifest_validation",
                        )
                    except Exception as error:
                        # A durable withdrawal marker is written before the
                        # rename. Continue even when the rename itself failed,
                        # so every other manifest sharing a known-bad object is
                        # also withdrawn before this operation reports failure.
                        deferred_error = deferred_error or error
                    continue
                if any(
                    item.relative_path == descriptor.relative_path
                    for item in manifest.objects
                ):
                    try:
                        quarantined = self._quarantine_manifest(
                            manifest_path,
                            reason=reason,
                        )
                    except Exception as error:
                        deferred_error = deferred_error or error
                        quarantined = True
                    if quarantined:
                        withdrawn += 1
                    self._checkpoint("object-quarantine-after-reference")

            object_path = self.root / descriptor.relative_path
            try:
                object_path.lstat()
            except FileNotFoundError:
                pass
            else:
                quarantine = (
                    self.root
                    / "quarantine"
                    / f"object-{descriptor.sha256}.{uuid.uuid4().hex}.bad"
                )
                try:
                    os.replace(object_path, quarantine)
                except Exception as error:
                    deferred_error = deferred_error or error
                else:
                    for directory in (
                        object_path.parent,
                        self.root / "quarantine",
                    ):
                        try:
                            _fsync_directory(directory)
                        except Exception as error:
                            deferred_error = deferred_error or error
            if deferred_error is not None:
                if isinstance(deferred_error, _NamespaceMutationError):
                    raise deferred_error
                raise _NamespaceMutationError(
                    "object quarantine changed namespace without a complete receipt",
                    deferred_error,
                ) from deferred_error
            self._checkpoint("object-quarantine-before-fence-release")
            self._clear_object_withdrawal(descriptor.sha256)
            self._checkpoint("object-quarantine-after-fence-release")
        return withdrawn

    def iter_manifest_paths(self) -> Iterator[Path]:
        """Yield manifest paths without materializing an unbounded directory."""

        for path in self.iter_managed_manifest_paths():
            relative = path.relative_to(self.root).as_posix()
            if _is_canonical_manifest_relative_path(relative):
                yield path

    def iter_managed_manifest_paths(self) -> Iterator[Path]:
        """Yield every leaf or unexpected shard in the manifest namespace."""

        yield from _iter_two_level_managed_paths(self.root / "manifests")

    def iter_object_paths(self) -> Iterator[Path]:
        """Yield immutable-object paths without materializing the namespace."""

        for path in self.iter_managed_object_paths():
            relative = path.relative_to(self.root).as_posix()
            if _is_canonical_object_relative_path(relative):
                yield path

    def iter_managed_object_paths(self) -> Iterator[Path]:
        """Yield every leaf or unexpected shard in the object namespace."""

        yield from _iter_two_level_managed_paths(self.root / "objects")

    def object_is_referenced(self, relative_path: str) -> bool:
        """Recheck an object's live references under the maintenance lock."""

        with self._exclusive():
            for manifest_path in self.iter_manifest_paths():
                try:
                    manifest, _ = self._read_manifest_file(
                        manifest_path,
                        expected_entry_id=manifest_path.stem,
                        probe_objects=False,
                    )
                except (OSError, ManifestError):
                    # Unknown is not unreferenced. A transient manifest read
                    # failure must retain the object until a later scrub can
                    # quarantine or authenticate that manifest.
                    return True
                if any(
                    item.relative_path == relative_path
                    for item in manifest.objects
                ):
                    return True
        return False

    def remove_orphan_object(
        self,
        path: Path,
        *,
        not_newer_than_unix_ns: int,
    ) -> int:
        """Delete one proven orphan, returning reclaimed logical disk bytes."""

        with self._exclusive():
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                return 0
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mtime_ns > not_newer_than_unix_ns
            ):
                return 0
            try:
                relative = path.relative_to(self.root).as_posix()
            except ValueError:
                raise ValueError("orphan candidate is outside the cache root")
            if not _is_canonical_object_relative_path(relative):
                return 0
            if self.object_is_referenced(relative):
                return 0
            path.unlink()
            _fsync_directory(path.parent)
            return metadata.st_size

    def remove_abandoned_temporary(
        self,
        path: Path,
        *,
        not_newer_than_unix_ns: int,
    ) -> bool:
        with self._exclusive():
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                return False
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mtime_ns > not_newer_than_unix_ns
                or path.parent != self.root / "tmp"
                or not path.name.endswith(".part")
            ):
                return False
            path.unlink()
            _fsync_directory(path.parent)
            return True

    def quarantine_unrecognized_managed_path(
        self,
        path: Path,
        *,
        category: str,
        not_newer_than_unix_ns: int,
    ) -> bool:
        """Move an old unexpected managed entry aside without following it."""

        if category not in {"manifest", "object", "temporary"}:
            raise ValueError("managed quarantine category is invalid")
        with self._exclusive():
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                return False
            if metadata.st_mtime_ns > not_newer_than_unix_ns:
                return False
            try:
                relative = path.relative_to(self.root)
            except ValueError as error:
                raise ValueError("managed quarantine path is outside the root") from error
            namespace = {
                "manifest": "manifests",
                "object": "objects",
                "temporary": "tmp",
            }[category]
            maximum_parts = 2 if category == "temporary" else 3
            allowed = (
                relative.parts
                and relative.parts[0] == namespace
                and 2 <= len(relative.parts) <= maximum_parts
            )
            if not allowed:
                raise ValueError("managed quarantine path is outside its namespace")
            path_digest = hashlib.sha256(
                relative.as_posix().encode("utf-8", "surrogateescape")
            ).hexdigest()
            quarantine = (
                self.root
                / "quarantine"
                / f"unknown-{category}-{path_digest}.{uuid.uuid4().hex}.bad"
            )
            os.replace(path, quarantine)
            _fsync_directory(path.parent)
            _fsync_directory(self.root / "quarantine")
            self._record_quarantine(
                "manifest_validation" if category == "manifest" else "unknown"
            )
            return True

    def maintain_capacity(
        self,
        *,
        max_bytes: int,
        low_watermark_bytes: int,
    ) -> MaintenanceReport:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or isinstance(low_watermark_bytes, bool)
            or not isinstance(low_watermark_bytes, int)
            or not 0 < low_watermark_bytes < max_bytes
        ):
            raise ValueError("capacity watermarks are invalid")
        with self._exclusive():
            before = self.disk_usage_bytes()
            if before <= max_bytes:
                return MaintenanceReport(before, before, 0, 0, 0)
            manifests_removed = 0
            objects_removed = 0
            current = before
            # Failed or interrupted manifest-last publication can leave
            # durable immutable objects with no owner. Reclaim those first so
            # a disk-full recovery does not evict healthy reusable entries to
            # make room for proven garbage.
            for object_path in self.iter_object_paths():
                reclaimed = self.remove_orphan_object(
                    object_path,
                    not_newer_than_unix_ns=time.time_ns(),
                )
                if reclaimed:
                    objects_removed += 1
                    current -= reclaimed
                    if current <= low_watermark_bytes:
                        break
            while current > low_watermark_bytes:
                manifest_path = self._oldest_manifest_path()
                if manifest_path is None:
                    # A crashed writer can leave only immutable objects. The
                    # full commit is serialized by this same lock, so a live
                    # unpublished transaction cannot be mistaken for an
                    # orphan here.
                    for object_path in self.iter_object_paths():
                        reclaimed = self.remove_orphan_object(
                            object_path,
                            not_newer_than_unix_ns=time.time_ns(),
                        )
                        if reclaimed:
                            objects_removed += 1
                            current = self.disk_usage_bytes()
                            if current <= low_watermark_bytes:
                                break
                    break
                manifest: RankManifest | None = None
                try:
                    manifest, _ = self._read_manifest_file(
                        manifest_path,
                        expected_entry_id=manifest_path.stem,
                        probe_objects=False,
                    )
                except (OSError, ManifestError):
                    quarantined = self._quarantine_manifest(
                        manifest_path,
                        reason="manifest_validation",
                    )
                    if not quarantined:
                        break
                    current = self.disk_usage_bytes()
                    continue
                self._mark_inventory_withdrawn(manifest.entry_id)
                withdrawal_error: Exception | None = None
                try:
                    self._record_withdrawal(manifest.entry_id)
                except Exception as error:
                    withdrawal_error = error
                try:
                    manifest_path.unlink()
                except FileNotFoundError:
                    continue
                durability_error = withdrawal_error
                try:
                    _fsync_directory(manifest_path.parent)
                except Exception as error:
                    durability_error = durability_error or error
                manifests_removed += 1
                if durability_error is not None:
                    raise _NamespaceMutationError(
                        "capacity GC changed namespace without a complete receipt",
                        durability_error,
                    ) from durability_error
                if (
                    self._inventory_owner_descriptor is not None
                    and self._withdraw_hook is not None
                ):
                    self._clear_inventory_withdrawal(manifest.entry_id)
                for item in manifest.objects:
                    object_path = self.root / item.relative_path
                    if self.object_is_referenced(item.relative_path):
                        continue
                    try:
                        object_path.unlink()
                        _fsync_directory(object_path.parent)
                        objects_removed += 1
                    except FileNotFoundError:
                        pass
                current = self.disk_usage_bytes()
            _fsync_directory(self.root / "manifests")
            _fsync_directory(self.root / "objects")
            return MaintenanceReport(
                bytes_before=before,
                bytes_after=current,
                manifests_removed=manifests_removed,
                objects_removed=objects_removed,
                bytes_reclaimed=max(0, before - current),
            )

    def disk_usage_bytes(self) -> int:
        total = 0
        for paths in (self.iter_object_paths(), self.iter_manifest_paths()):
            for path in paths:
                try:
                    metadata = path.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
        return total

    def managed_disk_usage(self) -> ManagedDiskUsage:
        """Account for live cache and non-evictable rank-root occupancy.

        ``max_bytes`` continues to bound reusable objects/manifests. Quarantine
        is deliberately not auto-deleted, so operators need a separate exact
        gauge for evidence which can otherwise fill the underlying filesystem.
        """

        cache_bytes = 0
        unrecognized_bytes = 0
        for paths, validator in (
            (
                self.iter_managed_object_paths(),
                _is_canonical_object_relative_path,
            ),
            (
                self.iter_managed_manifest_paths(),
                _is_canonical_manifest_relative_path,
            ),
        ):
            for path in paths:
                try:
                    relative = path.relative_to(self.root).as_posix()
                    metadata = path.lstat()
                except FileNotFoundError:
                    continue
                if validator(relative) and stat.S_ISREG(metadata.st_mode):
                    cache_bytes += metadata.st_size
                else:
                    unrecognized_bytes += _managed_tree_bytes(path)
        marker = self.root / ".spoolcache-root"
        control_bytes = _managed_tree_bytes(marker)
        return ManagedDiskUsage(
            cache_bytes=cache_bytes,
            quarantine_bytes=_managed_tree_bytes(self.root / "quarantine"),
            temporary_bytes=_managed_tree_bytes(self.root / "tmp"),
            state_bytes=_managed_tree_bytes(self.root / "state"),
            unrecognized_bytes=unrecognized_bytes,
            control_bytes=control_bytes,
        )

    def _oldest_manifest_path(self) -> Path | None:
        oldest: tuple[int, str, Path] | None = None
        for path in self.iter_manifest_paths():
            try:
                candidate = (path.lstat().st_mtime_ns, path.as_posix(), path)
            except FileNotFoundError:
                continue
            if oldest is None or candidate[:2] < oldest[:2]:
                oldest = candidate
        return None if oldest is None else oldest[2]

    def _validate_expected_identity(self, manifest: RankManifest) -> None:
        if (
            self.expected_deployment_digest is not None
            and manifest.deployment_identity_digest
            != self.expected_deployment_digest
        ):
            raise ManifestError("deployment identity differs")
        if (
            self.expected_rank_digest is not None
            and manifest.rank_identity_digest != self.expected_rank_digest
        ):
            raise ManifestError("rank identity differs")
        if self.expected_rank is not None and manifest.physical_rank != self.expected_rank:
            raise ManifestError("physical rank differs")
        if (
            self.expected_topology_digest is not None
            and manifest.topology_digest != self.expected_topology_digest
        ):
            raise ManifestError("topology identity differs")
        if (
            self.expected_profile is not None
            and manifest.profile != self.expected_profile
        ):
            raise ManifestError("profile identity differs")
        if (
            self.expected_layout_digest is not None
            and manifest.layout_digest != self.expected_layout_digest
        ):
            raise ManifestError("layout identity differs")

    def _verify_path(
        self,
        path: Path,
        *,
        digest: str,
        logical_length: int,
        stored_length: int,
        hash_payload: bool = True,
    ) -> bool:
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != stored_length:
                return False
            if not hash_payload:
                return True
            descriptor = ObjectDescriptor(
                group_index=0,
                layer_name="verification",
                page_start=0,
                page_count=1,
                byte_length=logical_length,
                stored_length=stored_length,
                sha256=digest,
                relative_path=f"objects/{digest[:2]}/{digest}.spool",
            )
            self.stream_object(descriptor, lambda _: None)
            return True
        except (OSError, ManifestError):
            return False

    def _quarantine_manifest(self, path: Path, *, reason: str) -> bool:
        if reason not in QUARANTINE_REASONS:
            raise ValueError("quarantine reason is outside the bounded vocabulary")
        with self._exclusive():
            entry_id = path.stem
            withdrawal_error: Exception | None = None
            if _is_digest(entry_id):
                # Persist the fail-closed decision before attempting rename.
                # If rename itself fails, shallow startup/rescan probes must
                # still be unable to advertise the known-bad entry.
                try:
                    self._mark_inventory_withdrawn(entry_id)
                except Exception as error:
                    withdrawal_error = error
                try:
                    self._record_withdrawal(entry_id)
                except Exception as error:
                    withdrawal_error = withdrawal_error or error
            try:
                quarantine = (
                    self.root
                    / "quarantine"
                    / f"manifest-{path.name}.{uuid.uuid4().hex}.bad"
                )
                os.replace(path, quarantine)
            except FileNotFoundError:
                return False
            except OSError:
                logger.exception("spoolcache: failed to quarantine manifest")
                raise
            durability_error = withdrawal_error
            for directory in (path.parent, self.root / "quarantine"):
                try:
                    _fsync_directory(directory)
                except Exception as error:
                    durability_error = durability_error or error
            self._record_quarantine(reason)
            if durability_error is not None:
                raise _NamespaceMutationError(
                    "manifest quarantine changed namespace without a complete receipt",
                    durability_error,
                ) from durability_error
            if (
                _is_digest(entry_id)
                and self._inventory_owner_descriptor is not None
                and self._withdraw_hook is not None
            ):
                # Both namespace directories and the withdrawal callback are
                # complete for the sole live reporter. A standalone process
                # leaves the marker for that owner to observe and acknowledge.
                self._clear_inventory_withdrawal(entry_id)
            return True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("manifest store is closed")

    def close(self) -> None:
        if self._closed:
            return
        descriptor = self._inventory_owner_descriptor
        self._inventory_owner_descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self._pool.close()
        self._closed = True

    def __enter__(self) -> "ManifestStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _o_nofollow() -> int:
    return getattr(os, "O_NOFOLLOW", 0)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _write_all(descriptor: int, view: memoryview) -> None:
    cursor = 0
    while cursor < len(view):
        written = os.write(descriptor, view[cursor:])
        if written <= 0:
            raise OSError("file write made no progress")
        cursor += written


def _write_exact(descriptor: int, view: memoryview) -> None:
    written = os.write(descriptor, view)
    if written != len(view):
        raise OSError("direct-I/O write was short")


def _read_exact_file(descriptor: int, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise ManifestError("manifest read was truncated")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _read_into_exact(descriptor: int, target: memoryview) -> int:
    read = os.readv(descriptor, [target])
    if read <= 0:
        raise ObjectCorruptionError("object read was truncated")
    if read != len(target):
        # Retrying a short read could use an unaligned pointer or length.
        raise ObjectCorruptionError("direct-I/O object read was short")
    return read


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_temporary_file(
    path: Path,
    *,
    primary_error: BaseException | None,
) -> None:
    """Remove one publication temporary without masking its primary failure."""

    cleanup_errors: list[Exception] = []
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as error:
        cleanup_errors.append(error)
    try:
        _fsync_directory(path.parent)
    except Exception as error:
        cleanup_errors.append(error)
    if not cleanup_errors:
        return
    if primary_error is None:
        first = cleanup_errors[0]
        for extra in cleanup_errors[1:]:
            if hasattr(first, "add_note"):
                first.add_note(
                    f"additional temporary cleanup failure: {type(extra).__name__}: "
                    f"{extra}"
                )
        raise first
    for error in cleanup_errors:
        if hasattr(primary_error, "add_note"):
            primary_error.add_note(
                f"temporary cleanup also failed: {type(error).__name__}: {error}"
            )
        try:
            logger.error(
                "spoolcache: temporary cleanup failed while preserving primary error",
                exc_info=(type(error), error, error.__traceback__),
            )
        except Exception:
            # Diagnostics must never replace either the primary state failure
            # or the cleanup failure selected above.
            pass


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifests_equivalent(left: RankManifest, right: RankManifest) -> bool:
    """Compare immutable entry facts while ignoring diagnostic creation time."""

    return (
        left.entry_id == right.entry_id
        and left.deployment_identity_digest
        == right.deployment_identity_digest
        and left.rank_identity_digest == right.rank_identity_digest
        and left.span_tokens == right.span_tokens
        and left.physical_rank == right.physical_rank
        and left.topology_digest == right.topology_digest
        and left.profile == right.profile
        and left.layout_digest == right.layout_digest
        and left.objects == right.objects
        and left.schema == right.schema
    )


def _read_small_json(path: Path, *, maximum: int) -> object:
    descriptor = os.open(path, os.O_RDONLY | _o_nofollow())
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError("small JSON file size is invalid")
        encoded = _read_exact_file(descriptor, metadata.st_size)
    finally:
        os.close(descriptor)
    return json.loads(encoded)


def _cleanup_atomic_write_temporaries(target: Path) -> int:
    """Remove only abandoned temporary files for one exact state target."""

    prefix = f".{target.name}."
    suffix = ".tmp"
    removed = 0
    with os.scandir(target.parent) as entries:
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
                raise ValueError(
                    "inventory generation temporary is not a regular file"
                )
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                continue
            removed += 1
    if removed:
        _fsync_directory(target.parent)
    return removed


def _managed_tree_bytes(path: Path) -> int:
    """Count regular-file lengths below ``path`` without following symlinks."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISREG(metadata.st_mode):
        return metadata.st_size
    if not stat.S_ISDIR(metadata.st_mode):
        return 0
    total = 0
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return 0
    with entries:
        for entry in entries:
            total += _managed_tree_bytes(Path(entry.path))
    return total


def _iter_two_level_managed_paths(root: Path) -> Iterator[Path]:
    """Walk one sharded namespace without following links or retaining it."""

    try:
        first_level = os.scandir(root)
    except FileNotFoundError:
        return
    with first_level:
        for shard in first_level:
            shard_path = Path(shard.path)
            try:
                canonical_shard = (
                    len(shard.name) == 2
                    and all(character in "0123456789abcdef" for character in shard.name)
                    and shard.is_dir(follow_symlinks=False)
                )
                if not canonical_shard:
                    yield shard_path
                    continue
                second_level = os.scandir(shard.path)
            except FileNotFoundError:
                continue
            with second_level:
                for entry in second_level:
                    yield Path(entry.path)


def _is_canonical_manifest_relative_path(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) != 3 or parts[0] != "manifests":
        return False
    filename = parts[2]
    if not filename.endswith(".json"):
        return False
    entry_id = filename[: -len(".json")]
    return _is_digest(entry_id) and parts[1] == entry_id[:2]


def _is_canonical_object_relative_path(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) != 3 or parts[0] != "objects":
        return False
    filename = parts[2]
    if not filename.endswith(".spool"):
        return False
    digest = filename[: -len(".spool")]
    return _is_digest(digest) and parts[1] == digest[:2]


def _object_digest_from_relative_path(relative: str) -> str:
    if not _is_canonical_object_relative_path(relative):
        raise ValueError("object withdrawal path is not canonical")
    return relative.rsplit("/", 1)[1][: -len(".spool")]
