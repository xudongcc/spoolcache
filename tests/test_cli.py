from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

from spoolcache.maintenance import SCRUB_REQUEST_NAME
from tests.test_storage_maintenance import open_store


class InstalledCliTests(unittest.TestCase):
    def test_config_outputs_valid_json_without_creating_cache_directories(self) -> None:
        command = Path(sysconfig.get_path("scripts")) / "spoolcache"
        with tempfile.TemporaryDirectory() as directory:
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("SPOOLCACHE_")}
            environment["HOME"] = directory
            for overrides, path, size in (
                ({}, Path(directory) / ".cache" / "spoolcache", 200),
                ({"SPOOLCACHE_PATH": "~/custom cache", "SPOOLCACHE_MAX_SIZE": "0.5"},
                 Path(directory) / "custom cache", 0.5),
            ):
                with self.subTest(overrides=overrides):
                    result = subprocess.run(
                        [str(command), "config"], env={**environment, **overrides},
                        capture_output=True, text=True, check=True,
                    )
                    self.assertEqual(result.stderr, "")
                    self.assertEqual(len(result.stdout.splitlines()), 1)
                    self.assertEqual(json.loads(result.stdout), {
                        "kv_connector": "SpoolCacheConnector",
                        "kv_role": "kv_both",
                        "kv_connector_module_path": "spoolcache.vllm.connector",
                        "kv_load_failure_policy": "fail",
                        "kv_connector_extra_config": {
                            "spoolcache_path": str(path), "spoolcache_max_size": size,
                        },
                    })
                    self.assertFalse(path.exists())

    def test_invalid_config_fails_without_partial_json(self) -> None:
        command = Path(sysconfig.get_path("scripts")) / "spoolcache"
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("SPOOLCACHE_")}
        for overrides in (
            {"SPOOLCACHE_MAX_SIZE": "not-a-number"},
            {"SPOOLCACHE_MAX_SIZE": "nan"},
            {"SPOOLCACHE_MAX_SIZE": "0"},
            {"SPOOLCACHE_PATH": "/"},
            {"SPOOLCACHE_PATH": "relative"},
        ):
            with self.subTest(overrides=overrides):
                result = subprocess.run(
                    [str(command), "config"], env={**environment, **overrides},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)
                self.assertNotIn("invalid choice", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_installed_command_reads_status_and_queues_a_request(self) -> None:
        command = Path(sysconfig.get_path("scripts")) / "spoolcache"

        def run(*arguments: str) -> str:
            return subprocess.check_output([str(command), *arguments], text=True)

        self.assertIn("config", run("--help"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank-0000"
            with open_store(root):
                status = json.loads(run("status", "--root", str(root)))
                self.assertFalse(status["initialized"])

                entry = "a" * 64
                receipt = json.loads(
                    run("request", "--root", str(root), "--entry", entry)
                )
                queued = json.loads((root / "state" / SCRUB_REQUEST_NAME).read_text())
                self.assertEqual(queued["entry_id"], entry)
                self.assertEqual(queued["nonce"], receipt["nonce"])
                self.assertEqual(
                    json.loads(run("request", "--root", str(root), "--entry", entry)),
                    receipt,
                )


if __name__ == "__main__":
    unittest.main()
