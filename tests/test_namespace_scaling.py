from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks.qualify_namespace_scaling import SCHEMA


class NamespaceScalingTests(unittest.TestCase):
    def test_independent_process_receipt_enforces_bounded_namespace_work(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        python_path = os.pathsep.join(
            filter(
                None,
                (
                    str(repository),
                    str(repository / "src"),
                    environment.get("PYTHONPATH"),
                ),
            )
        )
        environment["PYTHONPATH"] = python_path
        completed = subprocess.run(
            (
                sys.executable,
                str(repository / "benchmarks" / "qualify_namespace_scaling.py"),
                "--namespace-items",
                "4096",
            ),
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["result"], "passed")
        self.assertEqual(receipt["namespace_items"], 4096)
        self.assertGreaterEqual(receipt["namespace_inodes"], 4096)
        self.assertEqual(
            receipt["snapshot_items_scanned"],
            receipt["snapshot_item_budget"],
        )
        self.assertEqual(receipt["phase_after_step"], "snapshot_manifests")
        self.assertLessEqual(
            receipt["start_cycle_seconds"], receipt["max_start_seconds"]
        )
        self.assertLessEqual(receipt["step_seconds"], receipt["max_step_seconds"])
        self.assertLessEqual(receipt["close_seconds"], receipt["max_close_seconds"])
        self.assertFalse(receipt["shutdown"]["thread_alive"])
        self.assertEqual(receipt["shutdown"]["status"], "stopped")
        self.assertLessEqual(
            receipt["rss_growth_kib"], receipt["rss_growth_bound_kib"]
        )
        self.assertEqual(
            receipt["rss_scope"],
            "maintenance-after-fresh-process-population-v1",
        )
        self.assertGreater(receipt["rss_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
