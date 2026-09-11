"""Rank-local ownership, durable inventory generations and key withdrawal."""

from __future__ import annotations
import contextlib
import fcntl
import heapq
import json
import logging
import os
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator
from .buffers import AlignedBufferPool
from .errors import ManifestError, StoreBusyError
from .identity import canonical_json
from .manifest import TokenSnapshot
from .leases import KeyPins, open_regular
from .paths import CachePath as Path

_ROOT_MARKER_SCHEMA = "spoolcache-root/v1"


GENERATION_STATE_SCHEMA = "spoolcache-generation/v1"


GENERATION_STATE_NAME = "generation.json"


GENERATION_REQUIRED_SCHEMA = "spoolcache-generation-required/v1"


GENERATION_REQUIRED_NAME = "generation.required.json"


_MAX_GENERATION_STATE_BYTES = 4096


_MAX_GENERATION_EPOCH = (1 << 63) - 1


_PERSISTENT_GENERATION_EPOCH_BASE = 1 << 62


_INVENTORY_WITHDRAWAL_DIRECTORY = "inventory-withdrawn"


_INVENTORY_OWNER_LOCK_NAME = "inventory-owner.lock"


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
class ManifestOffer:
    entry_id: str
    span_tokens: int
    manifest_digest: str
    modified_at_unix_ns: int


@dataclass(frozen=True)
class LookupResult:
    is_hit: bool
    reason: str
    manifest: TokenSnapshot | None = None
    manifest_digest: str | None = None


@dataclass(frozen=True)
class MaintenanceReport:
    bytes_before: int
    bytes_after: int
    manifests_removed: int
    objects_removed: int
    bytes_reclaimed: int


@dataclass(frozen=True)
class CapacityBatchReport(MaintenanceReport):
    candidates_attempted: int
    candidates_remaining: int


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


class RankStore:
    """Control state shared by token-file publication, restore and maintenance."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        slot_bytes: int = 64 * 1024 * 1024,
        slot_count: int = 1,
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
        self._pins = None
        self._restore_credits = threading.BoundedSemaphore(2)
        self._active_restores = 0
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
            with self._exclusive():
                self._pins = KeyPins(self.root / "state" / "object-reads.lock")
                self._initialize_payload_backend()
            self.recover()
        except BaseException:
            if self._pins is not None:
                self._pins.close()
            self._pool.close()
            raise

    @property
    def buffer_budget_bytes(self) -> int:
        return self._pool.budget_bytes

    def set_quarantine_hook(self, hook: Callable[[str], None] | None) -> None:
        if hook is not None and not callable(hook):
            raise TypeError("quarantine hook must be callable")
        self._quarantine_hook = hook

    def set_withdraw_hook(self, hook: Callable[[str], None] | None) -> None:
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
        for name in ("namespace.lock", "object-reads.lock"):
            path = self.root / "state" / name
            try:
                fd = os.open(
                    path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _o_nofollow(), 0o600
                )
            except FileExistsError:
                fd = open_regular(path)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("SpoolCache lock is not a regular file")
            finally:
                os.close(fd)
        _fsync_directory(lock.parent)
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
        for name, label in ((_INVENTORY_WITHDRAWAL_DIRECTORY, "inventory-withdrawal"),):
            withdrawn = self.root / "state" / name
            created = False
            try:
                withdrawn.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                pass
            if withdrawn.is_symlink() or not withdrawn.is_dir():
                raise ValueError(f"SpoolCache {label} state is not a directory")
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
    def _exclusive(self, *, blocking: bool = True) -> Iterator[bool]:
        acquired = self._thread_lock.acquire(blocking=blocking)
        if not acquired:
            yield False
            return
        try:
            depth = getattr(self._lock_state, "depth", 0)
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield True
                finally:
                    self._lock_state.depth = depth
                return
            lock_path = self.root / "state" / "maintenance.lock"
            descriptor = os.open(lock_path, os.O_RDWR | _o_nofollow())
            locked = False
            namespace_descriptor = None
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError(
                        "SpoolCache maintenance lock is not a regular file"
                    )
                try:
                    # Old packages use EX on maintenance.lock. SH gates those
                    # writers while new packages serialize on namespace.lock.
                    # A restore keeps a separate SH gate through its CUDA drain.
                    fcntl.flock(
                        descriptor, fcntl.LOCK_SH | (0 if blocking else fcntl.LOCK_NB)
                    )
                    locked = True
                    namespace_descriptor = open_regular(
                        self.root / "state" / "namespace.lock"
                    )
                    fcntl.flock(
                        namespace_descriptor,
                        fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
                    )
                except BlockingIOError:
                    if blocking:
                        raise
                    yield False
                    return
                locked = True
                self._lock_state.depth = 1
                yield True
            finally:
                self._lock_state.depth = 0
                if locked:
                    if namespace_descriptor is not None:
                        os.close(namespace_descriptor)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        finally:
            self._thread_lock.release()

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
                    raise ValueError("inventory generation state is not a regular file")
                try:
                    payload = _read_small_json(
                        path,
                        maximum=_MAX_GENERATION_STATE_BYTES,
                    )
                except (OSError, UnicodeDecodeError, ValueError) as error:
                    raise ValueError("inventory generation state is invalid") from error
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"schema", "epoch"}
                    or payload.get("schema") != GENERATION_STATE_SCHEMA
                ):
                    raise ValueError("inventory generation state schema is unsupported")
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
                    canonical_json({"schema": GENERATION_REQUIRED_SCHEMA}) + b"\n",
                )
            return epoch

    def _inventory_withdrawal_path(self, entry_id: str) -> Path:
        return self.root / "state" / _INVENTORY_WITHDRAWAL_DIRECTORY / entry_id

    def _validate_inventory_withdrawal_namespace(self) -> None:
        self._validate_digest_marker_namespace(
            self.root / "state" / _INVENTORY_WITHDRAWAL_DIRECTORY,
            label="inventory-withdrawal",
        )

    @staticmethod
    def _validate_digest_marker_namespace(root: Path, *, label: str) -> None:
        with os.scandir(root) as entries:
            for entry in entries:
                if not _is_digest(entry.name):
                    raise ValueError(f"{label} state contains an invalid entry")
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A marker removed before this instance acquired the lock
                    # cannot happen for cooperating writers, but treating the
                    # already-absent name as harmless keeps startup robust to a
                    # completed pre-upgrade operation.
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"{label} entry is not a directory")
                with os.scandir(entry.path) as children:
                    if next(children, None) is not None:
                        raise ValueError(f"{label} entry is not empty")

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
        if not RankStore._digest_marker_is_present(path, label=label):
            raise ValueError(f"{label} marker is invalid")
        # A previous mkdir may have succeeded while its parent-directory fsync
        # failed. Reassert the receipt even for an existing valid marker so an
        # idempotent retry cannot advance into namespace mutation with a fence
        # that would disappear after power loss.
        _fsync_directory(path.parent)

    @staticmethod
    def _clear_digest_marker(path: Path, *, label: str) -> None:
        if not RankStore._digest_marker_is_present(path, label=label):
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

    def inventory_withdrawal_marker_batch(
        self,
        limit: int,
        *,
        after: str | None = None,
        withdraw: Callable[[str], None] | None = None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Withdraw every marked key, selecting a bounded acknowledgement page.

        The callback only updates memory; namespace mutation must wait until
        the directory stream closes. Scanning actual markers avoids probing
        every healthy held key on each worker stats report.
        """
        root = self.root / "state" / _INVENTORY_WITHDRAWAL_DIRECTORY
        label = "inventory-withdrawal"
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
                        raise ValueError(f"{label} state contains an invalid entry")
                    if not RankStore._digest_marker_is_present(
                        Path(entry.path),
                        label=label,
                    ):
                        continue
                    if withdraw is not None:
                        withdraw(name)
                    if following is None or name > following:
                        yield name

        with self._exclusive():
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
        with self._thread_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        if self._active_restores:
            raise StoreBusyError("cannot close a store with active restores")
        descriptor = self._inventory_owner_descriptor
        self._inventory_owner_descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self._pins.close()
        self._pool.close()
        self._closed = True

    def __enter__(self) -> "RankStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _o_nofollow() -> int:
    return getattr(os, "O_NOFOLLOW", 0)


def _write_all(descriptor: int, view: memoryview) -> None:
    cursor = 0
    while cursor < len(view):
        written = os.write(descriptor, view[cursor:])
        if written <= 0:
            raise OSError("file write made no progress")
        cursor += written


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
            if len(nonce) != 32 or any(
                character not in "0123456789abcdef" for character in nonce
            ):
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("inventory generation temporary is not a regular file")
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
