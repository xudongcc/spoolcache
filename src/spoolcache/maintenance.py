"""Rate-bounded scheduling for token-file scrub and configuration CLI."""
from __future__ import annotations
import argparse
import dataclasses
import logging
import math
import threading
import time
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from .token_scrub import TokenFileScrubber

SCRUB_BYTES_PER_SECOND = 64 * 1024 * 1024


SCRUB_STEP_BYTES = 64 * 1024 * 1024


SCRUB_STEP_ITEMS = 64


SCRUB_POLL_SECONDS = 1.0


SCRUB_STARTUP_DELAY_SECONDS = 60.0


SCRUB_CYCLE_INTERVAL_SECONDS = 6 * 60 * 60.0


SCRUB_SHUTDOWN_TIMEOUT_SECONDS = 5.0


SCRUB_SHUTDOWN_SCHEMA = "spoolcache-scrub-shutdown/v1"


_MAX_COUNTER = (1 << 63) - 1


logger = logging.getLogger(__name__)


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
class ScrubShutdownReport:
    schema: str
    status: str
    thread_alive: bool
    waited_seconds: float


class ScheduledDeepScrubber:
    """Low-priority periodic driver using the fixed internal I/O rate."""

    def __init__(
        self,
        scrubber: TokenFileScrubber,
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


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SpoolCache configuration tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("config", help="print validated vLLM connector JSON")
    parser.parse_args(tuple(argv) if argv is not None else None)
    from .vllm.config_json import main as render_config
    try:
        render_config()
    except ValueError as error:
        parser.error(str(error))
    return 0
