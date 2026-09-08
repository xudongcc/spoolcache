from __future__ import annotations

import json
import mmap
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from benchmarks.soak_storage_maintenance import (
    DEPLOYMENT,
    LAYOUT,
    PROFILE,
    RANK_IDENTITY,
    SCHEMA,
    TOPOLOGY,
    _maximum_rss_kib,
    _soak_payload,
    _start_rss_monitor,
    _state_database_bytes,
    _stop_rss_monitor,
    run_soak,
)
from spoolcache.maintenance import DeepScrubber
from spoolcache.store import ManifestStore


class StorageSoakTests(unittest.TestCase):
    def test_repeated_gc_scrub_and_rank_reopen_emit_a_bounded_receipt(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        python_path = str(repository / "src")
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        completed = subprocess.run(
            (
                sys.executable,
                str(repository / "benchmarks" / "soak_storage_maintenance.py"),
                "--iterations",
                "96",
                "--payload-bytes",
                "256",
                "--max-cache-bytes",
                # Each small payload now occupies a 4096-byte direct-I/O block.
                # Retain enough entries to grow the scrub work tables, while
                # still forcing GC within these 96 iterations.
                str(384 * 1024),
                "--scrub-every",
                "96",
                "--reopen-every",
                "48",
            ),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["result"], "passed")
        self.assertEqual(receipt["rank_reopens"], 2)
        self.assertEqual(receipt["scrub_cycles"], 2)
        self.assertGreater(receipt["capacity_runs"], 0)
        self.assertGreater(receipt["withdrawn_entries"], 0)
        self.assertGreater(receipt["inventory_marker_pages"], 0)
        self.assertEqual(receipt["object_marker_pages"], 0)
        self.assertLessEqual(
            receipt["final_cache_bytes"],
            receipt["max_cache_bytes"],
        )
        self.assertEqual(receipt["final_temporary_files"], 0)
        self.assertLessEqual(receipt["final_objects"], receipt["final_manifests"])
        self.assertEqual(receipt["payload_pattern"], "shake256-indexed-v1")
        self.assertLessEqual(
            receipt["max_state_database_bytes"],
            receipt["state_database_bound_bytes"],
        )
        self.assertGreater(
            receipt["max_state_database_bytes"],
            receipt["final_state_database_bytes"],
        )
        self.assertEqual(
            receipt["state_database_sampling"],
            "incremental-snapshot-and-every-step-v2",
        )
        self.assertLessEqual(
            receipt["max_state_tree_bytes"],
            receipt["state_tree_bound_bytes"],
        )
        self.assertLessEqual(
            receipt["final_state_tree_bytes"],
            receipt["max_state_tree_bytes"],
        )
        self.assertLessEqual(
            receipt["max_quarantine_tree_bytes"],
            receipt["quarantine_tree_bound_bytes"],
        )
        self.assertLessEqual(
            receipt["max_managed_inodes"],
            receipt["managed_inode_bound"],
        )
        self.assertGreaterEqual(
            receipt["max_managed_inodes"],
            receipt["final_managed_inodes"],
        )
        self.assertLessEqual(
            receipt["peak_tracemalloc_bytes"],
            receipt["tracemalloc_bound_bytes"],
        )
        self.assertLessEqual(
            receipt["rss_growth_kib"],
            receipt["rss_growth_bound_kib"],
        )
        self.assertEqual(
            receipt["rss_sampling"],
            "external-process-vmrss-poll-v1",
        )
        self.assertGreater(receipt["rss_sample_count"], 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc")
    def test_external_rss_sampler_catches_native_burst_below_old_process_peak(self) -> None:
        # Raise process-lifetime ru_maxrss, then release it. A smaller later
        # burst is invisible to a ru_maxrss delta but must be visible to the
        # independent current-RSS sampler used by the qualification receipt.
        historical = mmap.mmap(-1, 48 * 1024 * 1024)
        for offset in range(0, len(historical), 4096):
            historical[offset : offset + 1] = b"h"
        historical.close()
        historical_peak = _maximum_rss_kib()

        process, connection, baseline = _start_rss_monitor()
        try:
            burst = mmap.mmap(-1, 24 * 1024 * 1024)
            for offset in range(0, len(burst), 4096):
                burst[offset : offset + 1] = b"b"
            time.sleep(0.1)
            burst.close()
        finally:
            maximum, samples = _stop_rss_monitor(process, connection)
        sampler_growth = maximum - baseline
        process_peak_growth = max(0, _maximum_rss_kib() - historical_peak)
        self.assertGreaterEqual(sampler_growth, 20 * 1024)
        self.assertGreater(samples, 1)
        self.assertLess(process_peak_growth, sampler_growth)

    def test_state_database_peak_is_sampled_before_cycle_vacuum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            with ManifestStore(
                root,
                slot_bytes=4096,
                slot_count=1,
                expected_deployment_digest=DEPLOYMENT,
                expected_rank_digest=RANK_IDENTITY,
                expected_rank=0,
                expected_topology_digest=TOPOLOGY,
                expected_profile=PROFILE,
                expected_layout_digest=LAYOUT,
            ) as store:
                # A deliberately enlarged work queue makes the pre-vacuum
                # database observably larger than its idle low-water size.
                for index in range(400):
                    (root / "tmp" / f"queued-{index:04d}.part").write_bytes(b"x")
                scrubber = DeepScrubber(store)
                scrubber.start_cycle()
                peak_bytes = _state_database_bytes(root)
                for _ in range(40):
                    report = scrubber.step(
                        payload_budget_bytes=4096,
                        item_budget=32,
                    )
                    peak_bytes = max(peak_bytes, _state_database_bytes(root))
                    if report.cycle_completed:
                        break
                else:
                    self.fail("enlarged scrub cycle did not finish")
                idle_bytes = _state_database_bytes(root)
                self.assertGreater(peak_bytes, idle_bytes)

    def test_payloads_do_not_repeat_after_the_old_251_iteration_period(self) -> None:
        payloads = {_soak_payload(index, 64) for index in range(512)}
        self.assertEqual(len(payloads), 512)

    def test_invalid_soak_bounds_fail_before_creating_a_store(self) -> None:
        for values in (
            {"iterations": 0},
            {"payload_bytes": True},
            {"max_cache_bytes": 1024},
            {"scrub_every": -1},
            {"reopen_every": 0},
        ):
            arguments = {
                "iterations": 10,
                "payload_bytes": 2048,
                "max_cache_bytes": 32 * 1024,
                "scrub_every": 5,
                "reopen_every": 5,
            }
            arguments.update(values)
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    run_soak(**arguments)


if __name__ == "__main__":
    unittest.main()
