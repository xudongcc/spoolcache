from __future__ import annotations

import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from spoolcache.buffers import AlignedBufferPool
from spoolcache.config import SpoolCacheConfig
from spoolcache.errors import ConfigurationError, ObjectCorruptionError
from spoolcache.store import ManifestStore
from tests.test_manifest_store import commit_one, digest


class DirectIOTests(unittest.TestCase):
    def test_removed_modes_are_rejected(self) -> None:
        for key in ("direct_io", "spoolcache_direct_io"):
            for mode in ("required", "best-effort", "disabled"):
                with self.subTest(key=key, mode=mode):
                    with self.assertRaisesRegex(ConfigurationError, "unknown"):
                        SpoolCacheConfig.from_mapping({"path": "/cache", key: mode})

    def test_unsupported_filesystem_fails_without_recovery_or_buffer_leak(self) -> None:
        real_open = os.open
        for error_number in (errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSPC):
            with (
                self.subTest(error_number=error_number),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory) / "rank"
                pool = AlignedBufferPool(slot_bytes=4096, slot_count=1)

                def fail_probe(path: object, flags: int, *args: object) -> int:
                    if Path(path).name.startswith("direct-probe-"):
                        self.assertTrue(flags & os.O_DIRECT)
                        raise OSError(error_number, "probe failure")
                    return real_open(path, flags, *args)

                with (
                    mock.patch("spoolcache.store.AlignedBufferPool", return_value=pool),
                    mock.patch.object(pool, "close", wraps=pool.close) as close,
                    mock.patch("spoolcache.store.os.open", side_effect=fail_probe),
                    mock.patch.object(ManifestStore, "recover") as recover,
                ):
                    with self.assertRaises(OSError) as caught:
                        ManifestStore(root, slot_bytes=4096, slot_count=1)
                    if error_number == errno.ENOSPC:
                        self.assertEqual(caught.exception.errno, errno.ENOSPC)
                    else:
                        self.assertIn("O_DIRECT is required", str(caught.exception))
                    recover.assert_not_called()
                    close.assert_called_once()
                self.assertFalse(tuple((root / "tmp").iterdir()))

    def test_missing_platform_support_fails_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.__dict__):
            del os.O_DIRECT
            with mock.patch.object(ManifestStore, "_probe_direct_io") as probe:
                with self.assertRaisesRegex(OSError, "O_DIRECT is required"):
                    ManifestStore(
                        Path(directory) / "rank", slot_bytes=4096, slot_count=1
                    )
                probe.assert_not_called()

    def test_payload_round_trip_uses_direct_io_and_aligned_zero_padding(self) -> None:
        real_open = os.open
        payload_opens: list[int] = []

        def observe_open(path: object, flags: int, *args: object) -> int:
            candidate = Path(path)
            if candidate.name.startswith("object-") or "objects" in candidate.parts:
                # Directory fsync is metadata I/O, not a payload transfer.
                if not flags & os.O_DIRECTORY:
                    payload_opens.append(flags)
            return real_open(path, flags, *args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank"
            payload = bytes(range(251)) * 41
            with mock.patch("spoolcache.store.os.open", side_effect=observe_open):
                with ManifestStore(root, slot_bytes=4096, slot_count=1) as store:
                    manifest = commit_one(
                        store, entry=digest("direct-only"), payload=payload
                    )
                    descriptor = manifest.objects[0]
                    self.assertEqual(descriptor.stored_length, 12288)
                with ManifestStore(root, slot_bytes=4096, slot_count=1) as reopened:
                    self.assertTrue(
                        reopened.lookup(manifest.entry_id, verify_payloads=True).is_hit
                    )
                    self.assertEqual(reopened.read_object_bytes(descriptor), payload)
                    self.assertEqual(reopened.buffer_budget_bytes, 4096)
            self.assertGreaterEqual(len(payload_opens), 3)
            self.assertTrue(all(flags & os.O_DIRECT for flags in payload_opens))
            # Offline inspection proves the last direct write zero-filled its tail.
            stored = (root / descriptor.relative_path).read_bytes()
            self.assertEqual(stored[len(payload):], b"\x00" * (12288 - len(payload)))

    def test_short_payload_io_fails_without_unaligned_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with ManifestStore(
                Path(directory) / "rank", slot_bytes=4096, slot_count=1
            ) as store:
                manifest = commit_one(
                    store, entry=digest("short-read"), payload=b"x" * 5000
                )
                with mock.patch("spoolcache.store.os.readv", return_value=512) as read:
                    with self.assertRaisesRegex(ObjectCorruptionError, "short"):
                        store.read_object_bytes(manifest.objects[0])
                    read.assert_called_once()
                    read.reset_mock()  # Release the recorded mmap view before close.
                entry = digest("short-write")
                with mock.patch("spoolcache.store.os.write", return_value=512) as write:
                    with self.assertRaisesRegex(OSError, "short"):
                        commit_one(store, entry=entry, payload=b"z" * 5000)
                    write.assert_called_once()
                    write.reset_mock()
                self.assertFalse(store.lookup(entry).is_hit)
                self.assertFalse(tuple((store.root / "tmp").iterdir()))


if __name__ == "__main__":
    unittest.main()
