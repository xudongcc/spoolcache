from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location(
    "container_release", Path(__file__).resolve().parents[1] / "scripts/container-release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContainerReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.commit = "a" * 40
        self.wheel = self.root / "spoolcache-1.2.3-py3-none-any.whl"
        self.write_wheel("1.2.3")

    def write_wheel(self, metadata_version):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("spoolcache-1.2.3.dist-info/METADATA",
                             f"Name: spoolcache\nVersion: {metadata_version}\n")
        self.receipt = {"schema": "spoolcache-release/v1", "commit": self.commit,
                        "wheel": self.wheel.name,
                        "wheel_sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest()}
        self.save_receipt()

    def save_receipt(self):
        (self.root / "release.json").write_text(json.dumps(self.receipt))

    def verify(self, tag="v1.2.3"):
        return module.verify(self.root, tag, self.commit)

    def test_version_is_derived_from_release_tag(self):
        result = self.verify()
        self.assertEqual(result["version"], "1.2.3")
        self.assertEqual(result["wheel_sha256"], self.receipt["wheel_sha256"])

    def test_wrong_tag_commit_or_extra_wheel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "universal wheel"):
            self.verify("v1.2.4")
        for tag in ("main", "1.2.3", "v1.2.3rc1", "v01.2.3", "v1.2.3\n"):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "stable release"):
                self.verify(tag)
        self.receipt["commit"] = "b" * 40
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "commit differs"):
            self.verify()
        self.receipt["commit"] = self.commit
        self.save_receipt()
        (self.root / "other.whl").write_bytes(b"stale wheel")
        with self.assertRaisesRegex(ValueError, "universal wheel"):
            self.verify()

    def test_corruption_and_relabelled_wheel_are_rejected(self):
        self.wheel.write_bytes(self.wheel.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.verify()
        self.write_wheel("1.2.4")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
