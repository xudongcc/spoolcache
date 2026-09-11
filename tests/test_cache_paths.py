"""Non-interning cache paths retain POSIX path and filesystem semantics."""

from pathlib import Path
import tempfile
import unittest

from spoolcache.paths import CachePath


class CachePathTests(unittest.TestCase):
    def test_posix_lexical_operations_match_pathlib(self):
        for value in (
            "",
            ".",
            "..",
            "/",
            "//",
            "///",
            "//a/b",
            "/a//./b/../c",
            "a//b/",
            "a/../b.spool",
            "objects/ab/" + "ab" * 32 + ".spool",
        ):
            with self.subTest(value=value):
                actual, expected = CachePath(value), Path(value)
                for item in ("parts", "anchor", "root", "name", "stem", "suffix"):
                    self.assertEqual(getattr(actual, item), getattr(expected, item))
                self.assertEqual(actual, expected)
                self.assertEqual(str(actual.parent), str(expected.parent))
                self.assertEqual(str(actual / "next"), str(expected / "next"))
                self.assertEqual(str(actual / "/absolute"), str(expected / "/absolute"))
                self.assertIsInstance(actual / "next", CachePath)

    def test_owned_file_operations_and_relative_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = CachePath(directory).resolve()
            child = root / "shard" / "object.spool"
            child.parent.mkdir()
            child.write_bytes(b"payload")
            self.assertEqual(child.read_bytes(), b"payload")
            self.assertEqual(child.relative_to(root).as_posix(), "shard/object.spool")
            self.assertEqual(child.lstat().st_size, 7)
            self.assertEqual(list(child.parent.iterdir()), [child])
            self.assertIsInstance(child.resolve(), CachePath)
