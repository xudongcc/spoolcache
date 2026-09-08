from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from benchmarks.verify_release_install import verify


class ReleaseInstallTests(unittest.TestCase):
    def test_wheel_origin_and_installed_bytes_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "spoolcache/__init__.py"
            module.parent.mkdir()
            module.write_bytes(b"# release\n")
            wheel = root / "spoolcache.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("spoolcache/__init__.py", module.read_bytes())
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            origin = {"archive_info": {"hashes": {"sha256": digest}}}
            distribution = SimpleNamespace(
                locate_file=lambda path: root / path, version="0.1.0",
                read_text=lambda _: json.dumps(origin),
            )
            with (
                patch("importlib.metadata.distribution", return_value=distribution),
                patch("importlib.util.find_spec", return_value=SimpleNamespace(origin=str(module))),
            ):
                self.assertEqual(verify(wheel, digest)["status"], "passed")
                module.write_bytes(b"# altered\n")
                with self.assertRaisesRegex(ValueError, "installed bytes differ"):
                    verify(wheel, digest)
                module.write_bytes(b"# release\n")
                extra = module.parent / "removed_module.py"
                extra.write_text("# stale")
                with self.assertRaisesRegex(ValueError, "extra files"):
                    verify(wheel, digest)
                extra.unlink()
                origin.clear()
                with self.assertRaisesRegex(ValueError, "immutable wheel"):
                    verify(wheel, digest)
                with self.assertRaisesRegex(ValueError, "wheel SHA-256 mismatch"):
                    verify(wheel, "0" * 64)


if __name__ == "__main__":
    unittest.main()
