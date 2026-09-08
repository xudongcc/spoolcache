#!/usr/bin/env python3
"""Run a deterministic bounded storage/GC/reopen soak and emit one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from spoolcache.maintenance import DeepScrubber
from spoolcache.store import ManifestStore, ObjectSource


SCHEMA = "spoolcache-storage-soak/v1"
DEPLOYMENT = "a" * 64
RANK_IDENTITY = "b" * 64
TOPOLOGY = "c" * 64
LAYOUT = "d" * 64
PROFILE = "vllm-runtime-kv-v1"
_MIB = 1024 * 1024
_RSS_MONITOR_INTERVAL_SECONDS = 0.002


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _soak_payload(index: int, byte_length: int) -> bytes:
    """Return deterministic high-entropy content unique to this iteration."""

    seed = f"spoolcache-storage-soak-payload-v1:{index}".encode("ascii")
    return hashlib.shake_256(seed).digest(byte_length)


def _maximum_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB while macOS reports bytes. SpoolCache deploys on Linux,
    # but keeping the standalone receipt normalized makes local qualification
    # meaningful too.
    if sys.platform == "darwin":
        return (value + 1023) // 1024
    return value


def _process_rss_kib(process_id: int) -> int:
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("RSS process ID must be a positive integer")
    with Path(f"/proc/{process_id}/status").open(
        "r", encoding="ascii"
    ) as status:
        for line in status:
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1])
                raise RuntimeError("Linux VmRSS has an unexpected format")
    raise RuntimeError("Linux process status does not expose VmRSS")


def _current_rss_kib() -> int:
    """Return current resident memory instead of a process-lifetime high-water."""

    if sys.platform.startswith("linux"):
        return _process_rss_kib(os.getpid())
    # The standalone qualification target is Linux. This fallback keeps the
    # helper usable elsewhere, where getrusage is the best portable evidence.
    return _maximum_rss_kib()


def _rss_monitor_main(process_id: int, connection: Any) -> None:
    """Continuously sample another process, independent of its Python GIL."""

    try:
        baseline = _process_rss_kib(process_id)
        maximum = baseline
        samples = 1
        connection.send(("ready", baseline))
        while True:
            if connection.poll(_RSS_MONITOR_INTERVAL_SECONDS):
                command = connection.recv()
                if command != "stop":
                    raise RuntimeError("RSS monitor received an invalid command")
                current = _process_rss_kib(process_id)
                maximum = max(maximum, current)
                samples += 1
                connection.send(("result", baseline, maximum, samples))
                return
            current = _process_rss_kib(process_id)
            maximum = max(maximum, current)
            samples += 1
    except BaseException as error:
        try:
            connection.send(("error", type(error).__name__, str(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _start_rss_monitor() -> tuple[Any, Any, int]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("external RSS qualification requires Linux /proc")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_rss_monitor_main,
        args=(os.getpid(), child_connection),
        daemon=True,
    )
    process.start()
    child_connection.close()
    if not parent_connection.poll(10):
        process.terminate()
        process.join(10)
        parent_connection.close()
        raise RuntimeError("RSS monitor did not become ready")
    message = parent_connection.recv()
    if not isinstance(message, tuple) or len(message) != 2 or message[0] != "ready":
        process.join(10)
        parent_connection.close()
        raise RuntimeError(f"RSS monitor failed during startup: {message!r}")
    return process, parent_connection, int(message[1])


def _stop_rss_monitor(process: Any, connection: Any) -> tuple[int, int]:
    try:
        connection.send("stop")
        if not connection.poll(10):
            raise RuntimeError("RSS monitor did not return its peak")
        message = connection.recv()
        if (
            not isinstance(message, tuple)
            or len(message) != 4
            or message[0] != "result"
        ):
            raise RuntimeError(f"RSS monitor failed: {message!r}")
        _kind, _baseline, maximum, samples = message
        return int(maximum), int(samples)
    finally:
        connection.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)


def _state_database_bytes(root: Path) -> int:
    state = root / "state"
    return sum(
        path.stat().st_size
        for path in (
            state / "deep-scrub.sqlite3",
            state / "deep-scrub.sqlite3-wal",
            state / "deep-scrub.sqlite3-shm",
        )
        if path.exists()
    )


def _tree_usage(root: Path) -> tuple[int, int]:
    """Return regular metadata bytes and inode count without following links."""

    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return 0, 0
    total_bytes = root_metadata.st_size
    total_inodes = 1
    directories = [root] if root.is_dir() and not root.is_symlink() else []
    while directories:
        directory = directories.pop()
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                total_bytes += metadata.st_size
                total_inodes += 1
                if entry.is_dir(follow_symlinks=False):
                    directories.append(Path(entry.path))
    return total_bytes, total_inodes


def _open_store(
    root: Path,
    withdraw_hook: Callable[[str], None],
) -> ManifestStore:
    store = ManifestStore(
        root,
        slot_bytes=4096,
        slot_count=2,
        expected_deployment_digest=DEPLOYMENT,
        expected_rank_digest=RANK_IDENTITY,
        expected_rank=0,
        expected_topology_digest=TOPOLOGY,
        expected_profile=PROFILE,
        expected_layout_digest=LAYOUT,
    )
    store.set_withdraw_hook(withdraw_hook)
    return store


def _finish_cycle(
    scrubber: DeepScrubber,
    *,
    maximum_steps: int,
    observe: Callable[[], None] | None = None,
) -> int:
    for step in range(1, maximum_steps + 1):
        report = scrubber.step(
            payload_budget_bytes=64 * 1024,
            item_budget=32,
        )
        if observe is not None:
            observe()
        if report.cycle_completed:
            return step
    raise RuntimeError("storage soak scrub exceeded its bounded step allowance")


def run_soak(
    *,
    iterations: int,
    payload_bytes: int,
    max_cache_bytes: int,
    scrub_every: int,
    reopen_every: int,
) -> dict[str, Any]:
    for label, value in (
        ("iterations", iterations),
        ("payload_bytes", payload_bytes),
        ("max_cache_bytes", max_cache_bytes),
        ("scrub_every", scrub_every),
        ("reopen_every", reopen_every),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if max_cache_bytes <= payload_bytes:
        raise ValueError("max_cache_bytes must exceed one payload")

    started = time.monotonic()
    rss_monitor, rss_connection, baseline_rss_kib = _start_rss_monitor()
    baseline_process_peak_rss_kib = _maximum_rss_kib()
    state_database_bound = max(_MIB, max_cache_bytes * 4)
    state_tree_bound = max(2 * _MIB, max_cache_bytes * 8)
    quarantine_bound = max(_MIB, max_cache_bytes * 2)
    managed_inode_bound = max(
        4096,
        (max_cache_bytes // max(1, payload_bytes)) * 16 + 2048,
    )
    tracemalloc_bound = max(16 * _MIB, max_cache_bytes * 8)
    rss_growth_bound_kib = max(
        128 * 1024,
        (max_cache_bytes * 8 + 1023) // 1024,
    )
    tracemalloc.start()
    withdrawn_entries = 0

    def record_withdrawal(_entry_id: str) -> None:
        nonlocal withdrawn_entries
        withdrawn_entries += 1

    capacity_runs = 0
    scrub_cycles = 0
    scrub_steps = 0
    rank_reopens = 0
    orphans_created = 0
    temporary_files_created = 0
    max_cache_observed = 0
    max_state_database_bytes = 0
    max_state_tree_bytes = 0
    max_quarantine_bytes = 0
    max_managed_inodes = 0
    boundary_max_rss_kib = baseline_rss_kib
    low_watermark = max_cache_bytes * 3 // 4
    inventory_marker_cursor: str | None = None
    object_marker_cursor: str | None = None
    inventory_marker_pages = 0
    object_marker_pages = 0

    try:
        with tempfile.TemporaryDirectory(prefix="spoolcache-storage-soak-") as directory:
            root = Path(directory) / "rank-0000"

            def observe_runtime_bounds() -> None:
                nonlocal boundary_max_rss_kib, max_state_database_bytes
                nonlocal max_state_tree_bytes, max_quarantine_bytes
                nonlocal max_managed_inodes
                boundary_max_rss_kib = max(
                    boundary_max_rss_kib,
                    _current_rss_kib(),
                )
                max_state_database_bytes = max(
                    max_state_database_bytes,
                    _state_database_bytes(root),
                )
                state_bytes, _state_inodes = _tree_usage(root / "state")
                quarantine_bytes, _quarantine_inodes = _tree_usage(
                    root / "quarantine"
                )
                _managed_bytes, managed_inodes = _tree_usage(root)
                max_state_tree_bytes = max(max_state_tree_bytes, state_bytes)
                max_quarantine_bytes = max(
                    max_quarantine_bytes,
                    quarantine_bytes,
                )
                max_managed_inodes = max(max_managed_inodes, managed_inodes)
                if max_state_database_bytes > state_database_bound:
                    raise RuntimeError(
                        "deep scrub state database exceeded its soak bound"
                    )
                if max_state_tree_bytes > state_tree_bound:
                    raise RuntimeError("rank state tree exceeded its soak bound")
                if max_quarantine_bytes > quarantine_bound:
                    raise RuntimeError("quarantine tree exceeded its soak bound")
                if max_managed_inodes > managed_inode_bound:
                    raise RuntimeError("managed inode count exceeded its soak bound")

            store = _open_store(root, record_withdrawal)

            def reconcile_withdrawal_markers() -> None:
                nonlocal inventory_marker_cursor, object_marker_cursor
                nonlocal inventory_marker_pages, object_marker_pages
                inventory_batch, inventory_marker_cursor = (
                    store.inventory_withdrawal_marker_batch(
                        64,
                        after=inventory_marker_cursor,
                    )
                )
                if inventory_batch:
                    store.acknowledge_absent_inventory_withdrawals(
                        inventory_batch
                    )
                    inventory_marker_pages += 1
                object_batch, object_marker_cursor = (
                    store.object_withdrawal_marker_batch(
                        1,
                        after=object_marker_cursor,
                    )
                )
                if object_batch:
                    store.acknowledge_unreferenced_object_withdrawals(
                        object_batch
                    )
                    object_marker_pages += 1

            try:
                observe_runtime_bounds()
                for index in range(iterations):
                    entry_id = _digest(f"soak-entry-{index}")
                    payload = _soak_payload(index, payload_bytes)
                    store.commit(
                        entry_id=entry_id,
                        deployment_identity_digest=DEPLOYMENT,
                        rank_identity_digest=RANK_IDENTITY,
                        span_tokens=1024,
                        physical_rank=0,
                        topology_digest=TOPOLOGY,
                        profile=PROFILE,
                        layout_digest=LAYOUT,
                        sources=(
                            ObjectSource(
                                group_index=0,
                                layer_name="layer",
                                page_start=0,
                                page_count=1,
                                chunks=(payload,),
                            ),
                        ),
                        created_at_unix_ns=index + 1,
                    )
                    manifest_path = (
                        root
                        / "manifests"
                        / entry_id[:2]
                        / f"{entry_id}.json"
                    )
                    os.utime(manifest_path, ns=(index + 1, index + 1))

                    if index % 7 == 0:
                        store.put_object((f"orphan-{index}".encode() * 128,))
                        orphans_created += 1
                    if index % 11 == 0:
                        (root / "tmp" / f"abandoned-{index}.part").write_bytes(
                            b"partial"
                        )
                        temporary_files_created += 1

                    current = store.disk_usage_bytes()
                    max_cache_observed = max(max_cache_observed, current)
                    if current > max_cache_bytes:
                        report = store.maintain_capacity(
                            max_bytes=max_cache_bytes,
                            low_watermark_bytes=low_watermark,
                        )
                        capacity_runs += 1
                        if report.bytes_after > low_watermark:
                            raise RuntimeError(
                                "capacity maintenance did not reach its low watermark"
                            )
                    # A production inventory owner consumes one bounded marker
                    # page before its next report. Model that convergence so
                    # this long-lived store soak does not measure an ownerless
                    # standalone-maintenance artifact as runtime state growth.
                    reconcile_withdrawal_markers()

                    if (index + 1) % scrub_every == 0:
                        scrubber = DeepScrubber(store)
                        scrubber.start_cycle()
                        # Snapshot rows are now materialized incrementally.
                        # Sampling every step captures their pre-vacuum peak.
                        observe_runtime_bounds()
                        scrub_steps += _finish_cycle(
                            scrubber,
                            maximum_steps=max(64, iterations * 4),
                            observe=observe_runtime_bounds,
                        )
                        scrub_cycles += 1

                    if (index + 1) % reopen_every == 0:
                        store.close()
                        store = _open_store(root, record_withdrawal)
                        rank_reopens += 1
                    observe_runtime_bounds()

                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                observe_runtime_bounds()
                scrub_steps += _finish_cycle(
                    scrubber,
                    maximum_steps=max(64, iterations * 4),
                    observe=observe_runtime_bounds,
                )
                scrub_cycles += 1
                if store.disk_usage_bytes() > max_cache_bytes:
                    report = store.maintain_capacity(
                        max_bytes=max_cache_bytes,
                        low_watermark_bytes=low_watermark,
                    )
                    capacity_runs += 1
                    if report.bytes_after > low_watermark:
                        raise RuntimeError(
                            "final capacity pass did not reach its low watermark"
                        )
                while True:
                    pending, _cursor = store.inventory_withdrawal_marker_batch(64)
                    if not pending:
                        break
                    store.acknowledge_absent_inventory_withdrawals(pending)
                    inventory_marker_pages += 1
                while True:
                    pending_objects, _cursor = store.object_withdrawal_marker_batch(1)
                    if not pending_objects:
                        break
                    store.acknowledge_unreferenced_object_withdrawals(
                        pending_objects
                    )
                    object_marker_pages += 1

                offers = store.scan_offers(100_000)
                for offer in offers:
                    if not store.lookup(offer.entry_id, verify_payloads=True).is_hit:
                        raise RuntimeError("a final offered entry failed authentication")
                final_usage = store.managed_disk_usage()
                final_manifests = len(tuple(store.iter_manifest_paths()))
                final_objects = len(tuple(store.iter_object_paths()))
                final_temporary_files = len(tuple((root / "tmp").iterdir()))
                final_state_database_bytes = _state_database_bytes(root)
                final_state_tree_bytes, final_state_inodes = _tree_usage(
                    root / "state"
                )
                final_quarantine_tree_bytes, final_quarantine_inodes = (
                    _tree_usage(root / "quarantine")
                )
                _final_tree_bytes, final_managed_inodes = _tree_usage(root)
                observe_runtime_bounds()
                if final_usage.cache_bytes > max_cache_bytes:
                    raise RuntimeError("final live cache exceeds its high watermark")
                if final_temporary_files:
                    raise RuntimeError("temporary publication files survived final scrub")
            finally:
                store.close()
        _current, peak_traced_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        max_rss_kib, rss_sample_count = _stop_rss_monitor(
            rss_monitor,
            rss_connection,
        )
    process_peak_rss_kib = _maximum_rss_kib()
    sampled_rss_growth_kib = max(0, max_rss_kib - baseline_rss_kib)
    process_peak_rss_growth_kib = max(
        0,
        process_peak_rss_kib - baseline_process_peak_rss_kib,
    )
    # ``ru_maxrss`` is process-lifetime state and can be inherited across
    # fork/exec. Keep it as diagnostics only; pass/fail uses a post-start
    # baseline and an independent sampler which can observe native bursts even
    # while this process holds the Python GIL.
    rss_growth_kib = sampled_rss_growth_kib
    if peak_traced_bytes > tracemalloc_bound:
        raise RuntimeError("tracemalloc peak exceeded its soak bound")
    if rss_growth_kib > rss_growth_bound_kib:
        raise RuntimeError("maximum RSS growth exceeded its soak bound")

    return {
        "schema": SCHEMA,
        "iterations": iterations,
        "payload_bytes": payload_bytes,
        "max_cache_bytes": max_cache_bytes,
        "low_watermark_bytes": low_watermark,
        "scrub_cycles": scrub_cycles,
        "scrub_steps": scrub_steps,
        "capacity_runs": capacity_runs,
        "rank_reopens": rank_reopens,
        "orphans_created": orphans_created,
        "temporary_files_created": temporary_files_created,
        "withdrawn_entries": withdrawn_entries,
        "inventory_marker_pages": inventory_marker_pages,
        "object_marker_pages": object_marker_pages,
        "max_cache_observed_bytes": max_cache_observed,
        "max_state_database_bytes": max_state_database_bytes,
        "final_state_database_bytes": final_state_database_bytes,
        "state_database_bound_bytes": state_database_bound,
        "state_database_sampling": "incremental-snapshot-and-every-step-v2",
        "max_state_tree_bytes": max_state_tree_bytes,
        "final_state_tree_bytes": final_state_tree_bytes,
        "final_state_inodes": final_state_inodes,
        "state_tree_bound_bytes": state_tree_bound,
        "max_quarantine_tree_bytes": max_quarantine_bytes,
        "final_quarantine_tree_bytes": final_quarantine_tree_bytes,
        "final_quarantine_inodes": final_quarantine_inodes,
        "quarantine_tree_bound_bytes": quarantine_bound,
        "max_managed_inodes": max_managed_inodes,
        "final_managed_inodes": final_managed_inodes,
        "managed_inode_bound": managed_inode_bound,
        "final_cache_bytes": final_usage.cache_bytes,
        "final_managed_bytes": final_usage.total_bytes,
        "final_quarantine_bytes": final_usage.quarantine_bytes,
        "final_manifests": final_manifests,
        "final_objects": final_objects,
        "final_temporary_files": final_temporary_files,
        "peak_tracemalloc_bytes": peak_traced_bytes,
        "tracemalloc_bound_bytes": tracemalloc_bound,
        "baseline_rss_kib": baseline_rss_kib,
        "baseline_process_peak_rss_kib": baseline_process_peak_rss_kib,
        "max_rss_kib": max_rss_kib,
        "boundary_max_rss_kib": boundary_max_rss_kib,
        "rss_sample_count": rss_sample_count,
        "sampled_rss_growth_kib": sampled_rss_growth_kib,
        "process_peak_rss_kib": process_peak_rss_kib,
        "process_peak_rss_growth_kib": process_peak_rss_growth_kib,
        "rss_growth_kib": rss_growth_kib,
        "rss_growth_bound_kib": rss_growth_bound_kib,
        "rss_sampling": "external-process-vmrss-poll-v1",
        "payload_pattern": "shake256-indexed-v1",
        "elapsed_seconds": time.monotonic() - started,
        "result": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--max-cache-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--scrub-every", type=int, default=50)
    parser.add_argument("--reopen-every", type=int, default=100)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_soak(
                iterations=arguments.iterations,
                payload_bytes=arguments.payload_bytes,
                max_cache_bytes=arguments.max_cache_bytes,
                scrub_every=arguments.scrub_every,
                reopen_every=arguments.reopen_every,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
