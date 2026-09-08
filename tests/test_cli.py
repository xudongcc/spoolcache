from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

from spoolcache.maintenance import SCRUB_REQUEST_NAME
from tests.test_storage_maintenance import open_store


class InstalledCliTests(unittest.TestCase):
    def test_installed_command_reads_status_and_queues_a_request(self) -> None:
        command = Path(sysconfig.get_path("scripts")) / "spoolcache"

        def run(*arguments: str) -> str:
            return subprocess.check_output([str(command), *arguments], text=True)

        self.assertIn("{request,status}", run("--help"))
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
