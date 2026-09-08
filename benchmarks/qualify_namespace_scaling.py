#!/usr/bin/env python3
"""Qualify bounded scrub startup, step, and shutdown on a large namespace."""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__:
    from .soak_storage_maintenance import (
        _open_store,
        _start_rss_monitor,
        _state_database_bytes,
        _stop_rss_monitor,
        _tree_usage,
    )
else:
    from soak_storage_maintenance import (  # type: ignore[no-redef]
        _open_store,
        _start_rss_monitor,
        _state_database_bytes,
        _stop_rss_monitor,
        _tree_usage,
    )
from spoolcache.maintenance import DeepScrubber, ScheduledDeepScrubber


SCHEMA = "spoolcache-namespace-scaling/v1"
DEFAULT_NAMESPACE_ITEMS = 1_000_000
SNAPSHOT_ITEM_BUDGET = 64
MAX_START_SECONDS = 5.0
MAX_STEP_SECONDS = 5.0
MAX_CLOSE_SECONDS = 6.0
MAX_RSS_GROWTH_KIB = 128 * 1024


def _populate_manifest_namespace(root: Path, item_count: int) -> None:
    namespace = root / "manifests"
    shards = tuple(namespace / f"{index:02x}" for index in range(256))
    for shard in shards:
        shard.mkdir(exist_ok=True)
    for index in range(item_count):
        shard = index & 0xFF
        entry_id = f"{shard:02x}{index:062x}"
        path = shards[shard] / f"{entry_id}.json"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)


def _populate_manifest_namespace_process(
    root_text: str,
    item_count: int,
    connection: Any,
) -> None:
    try:
        root = Path(root_text)
        _populate_manifest_namespace(root, item_count)
        namespace_bytes, namespace_inodes = _tree_usage(root / "manifests")
        connection.send(("ok", namespace_bytes, namespace_inodes))
    except BaseException as error:
        try:
            connection.send(("error", type(error).__name__, str(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _populate_in_fresh_process(
    root: Path,
    item_count: int,
) -> tuple[float, int, int]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_populate_manifest_namespace_process,
        args=(str(root), item_count, child_connection),
    )
    started = time.monotonic()
    process.start()
    child_connection.close()
    process.join(600)
    if process.is_alive():
        process.terminate()
        process.join(30)
        parent_connection.close()
        raise RuntimeError("namespace population exceeded its fixed timeout")
    message = parent_connection.recv() if parent_connection.poll() else None
    parent_connection.close()
    if process.exitcode != 0:
        raise RuntimeError(
            "namespace population process failed with "
            f"exit {process.exitcode}: {message!r}"
        )
    if (
        not isinstance(message, tuple)
        or len(message) != 3
        or message[0] != "ok"
    ):
        raise RuntimeError(f"namespace population receipt is invalid: {message!r}")
    namespace_bytes, namespace_inodes = message[1:]
    if (
        isinstance(namespace_bytes, bool)
        or not isinstance(namespace_bytes, int)
        or namespace_bytes < 0
        or isinstance(namespace_inodes, bool)
        or not isinstance(namespace_inodes, int)
        or namespace_inodes < item_count
    ):
        raise RuntimeError("namespace population receipt values are invalid")
    return time.monotonic() - started, namespace_bytes, namespace_inodes


def run_qualification(*, namespace_items: int) -> dict[str, Any]:
    if (
        isinstance(namespace_items, bool)
        or not isinstance(namespace_items, int)
        or namespace_items < SNAPSHOT_ITEM_BUDGET
    ):
        raise ValueError(
            f"namespace_items must be an integer >= {SNAPSHOT_ITEM_BUDGET}"
        )

    overall_started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="spoolcache-namespace-scaling-"
    ) as directory:
        root = Path(directory) / "rank-0000"
        store = _open_store(root, lambda _entry_id: None)
        rss_monitor = None
        rss_connection = None
        scrubber: DeepScrubber | None = None
        scheduler: ScheduledDeepScrubber | None = None
        try:
            population_seconds, namespace_bytes, namespace_inodes = (
                _populate_in_fresh_process(root, namespace_items)
            )
            rss_monitor, rss_connection, baseline_rss_kib = _start_rss_monitor()

            scrubber = DeepScrubber(store)
            started = time.monotonic()
            cycle = scrubber.start_cycle()
            start_seconds = time.monotonic() - started

            started = time.monotonic()
            step_report = scrubber.step(
                payload_budget_bytes=64 * 1024,
                item_budget=SNAPSHOT_ITEM_BUDGET,
            )
            step_seconds = time.monotonic() - started
            phase_after_step = scrubber.status().phase
            scrubber.close()
            scrubber = None

            scheduler = ScheduledDeepScrubber(
                DeepScrubber(store),
                poll_seconds=0.001,
                startup_delay_seconds=3600.0,
                cycle_interval_seconds=3600.0,
            )
            scheduler.start()
            time.sleep(0.01)
            started = time.monotonic()
            shutdown_report = scheduler.close()
            close_seconds = time.monotonic() - started
            scheduler = None

            max_rss_kib, rss_sample_count = _stop_rss_monitor(
                rss_monitor,
                rss_connection,
            )
            rss_monitor = None
            rss_connection = None
            rss_growth_kib = max(0, max_rss_kib - baseline_rss_kib)
            state_database_bytes = _state_database_bytes(root)
            state_tree_bytes, state_inodes = _tree_usage(root / "state")

            if cycle <= 0:
                raise RuntimeError("deep scrub did not start a valid cycle")
            if step_report.namespace_items_scanned != SNAPSHOT_ITEM_BUDGET:
                raise RuntimeError("snapshot step exceeded or missed its item budget")
            if phase_after_step != "snapshot_manifests":
                raise RuntimeError("bounded snapshot unexpectedly consumed the namespace")
            if start_seconds > MAX_START_SECONDS:
                raise RuntimeError("deep scrub cycle startup exceeded its fixed bound")
            if step_seconds > MAX_STEP_SECONDS:
                raise RuntimeError("deep scrub namespace step exceeded its fixed bound")
            if close_seconds > MAX_CLOSE_SECONDS or shutdown_report.thread_alive:
                raise RuntimeError("scheduled deep scrub shutdown exceeded its bound")
            if rss_growth_kib > MAX_RSS_GROWTH_KIB:
                raise RuntimeError("namespace qualification RSS exceeded its bound")

            return {
                "schema": SCHEMA,
                "result": "passed",
                "namespace_items": namespace_items,
                "namespace_inodes": namespace_inodes,
                "namespace_bytes": namespace_bytes,
                "population_seconds": population_seconds,
                "snapshot_item_budget": SNAPSHOT_ITEM_BUDGET,
                "snapshot_items_scanned": step_report.namespace_items_scanned,
                "phase_after_step": phase_after_step,
                "start_cycle_seconds": start_seconds,
                "step_seconds": step_seconds,
                "close_seconds": close_seconds,
                "max_start_seconds": MAX_START_SECONDS,
                "max_step_seconds": MAX_STEP_SECONDS,
                "max_close_seconds": MAX_CLOSE_SECONDS,
                "shutdown": dataclasses.asdict(shutdown_report),
                "state_database_bytes": state_database_bytes,
                "state_tree_bytes": state_tree_bytes,
                "state_inodes": state_inodes,
                "baseline_rss_kib": baseline_rss_kib,
                "max_rss_kib": max_rss_kib,
                "rss_growth_kib": rss_growth_kib,
                "rss_growth_bound_kib": MAX_RSS_GROWTH_KIB,
                "rss_sample_count": rss_sample_count,
                "rss_sampling": "external-process-vmrss-poll-v1",
                "rss_scope": "maintenance-after-fresh-process-population-v1",
                "elapsed_seconds_before_cleanup": (
                    time.monotonic() - overall_started
                ),
            }
        finally:
            if scheduler is not None:
                scheduler.close()
            if scrubber is not None:
                scrubber.close()
            if rss_monitor is not None and rss_connection is not None:
                _stop_rss_monitor(rss_monitor, rss_connection)
            store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace-items",
        type=int,
        default=DEFAULT_NAMESPACE_ITEMS,
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_qualification(namespace_items=arguments.namespace_items),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
