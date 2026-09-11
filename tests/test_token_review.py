"""Regression contracts found during the token-file PR review."""

import multiprocessing
import os
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import spoolcache.token_files as token_files
from spoolcache.maintenance import main
from spoolcache.token_files import TokenFileStore
import tests.test_token_files as fixtures


def _hold_restore(root, layout, binding, entry, ready):
    with TokenFileStore(root, layout=layout, slot_bytes=16384, **binding) as store:
        with store.restore_view(entry) as lease:
            ready.send(lease.result.is_hit)
            ready.recv()


def _exit_during_publication(root, layout, binding, key, stage):
    def fault(current):
        if current == stage:
            os._exit(91)

    with TokenFileStore(root, layout=layout, slot_bytes=16384,
                        fault_hook=fault, **binding) as store:
        store.commit_chunk(key, None, 256, b"x" * 16384)


def _corrupt_read_and_authenticate(root, layout, binding, keys, result):
    """A subprocess bounds the test even if the lock-order regression returns."""
    with TokenFileStore(root, layout=layout, slot_bytes=16384, **binding) as store:
        ready, writer_locked, release = (threading.Event() for _ in range(3))
        original = token_files._read_into
        errors = []

        def paused(fd, view):
            original(fd, view)
            if threading.current_thread().name == "token-reader" and len(view) == 16384:
                ready.set()
                if not release.wait(5):
                    raise TimeoutError("reader was not released")

        @contextmanager
        def receiver(_):
            yield lambda data: None

        with store.restore_view(keys[-1].digest) as lease:
            def read():
                try:
                    store.stream_objects(lease.descriptors, receiver, lease=lease)
                except BaseException as error:
                    errors.append(type(error).__name__)

            def authenticate():
                with store._exclusive():
                    writer_locked.set()
                    store.valid_chunk(keys[1].digest, keys[0].digest, 512)

            with patch.object(token_files, "_read_into", side_effect=paused):
                reader = threading.Thread(target=read, name="token-reader", daemon=True)
                reader.start()
                if not ready.wait(5):
                    raise TimeoutError("reader did not acquire its staging credit")
                writer = threading.Thread(target=authenticate, daemon=True)
                writer.start()
                if not writer_locked.wait(5):
                    raise TimeoutError("writer did not acquire the namespace lock")
                release.set()
                reader.join(3)
                writer.join(3)
                blocked = reader.is_alive() or writer.is_alive()
                result.send({"blocked": blocked, "errors": errors})
                if blocked:
                    # Do not enter lock-taking cleanup on the failing path.
                    # The parent owns the temporary root and terminates us.
                    threading.Event().wait(30)
                assert store._inventory_is_withdrawn(keys[0].digest)
        assert store._pool.borrowed == 0


class TokenReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TokenFileTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_cross_process_pins_release_on_process_loss(self):
        fixture = self.fixture
        fixture.save(tuple(range(512)))
        fixture.save((999,) * 256)
        keys = fixture.keys(range(512))
        unrelated = fixture.keys((999,) * 256)[0].digest
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_hold_restore, args=(
            fixture.root, fixture.layout, fixture.binding, keys[-1].digest, child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(10), "child did not acquire its pins")
            self.assertTrue(parent.recv())
            for key in keys:
                self.assertFalse(fixture.store.evict(key.digest))
            self.assertTrue(fixture.store.evict(unrelated))
            process.terminate()
            process.join(5)
            self.assertFalse(process.is_alive())
            self.assertTrue(fixture.store.evict(keys[-1].digest))
            self.assertTrue(fixture.store.lookup(keys[0].digest, verify_payloads=True).is_hit)
        finally:
            if process.is_alive():
                process.kill()
            process.join(5)
            process.close()
            parent.close()

    def test_publication_process_loss_leaves_absent_or_authenticated_file(self):
        fixture = self.fixture
        context = multiprocessing.get_context("spawn")
        for index, stage in enumerate(("token_before_publish", "token_after_link")):
            key = fixture.keys((index + 100,) * 256)[0].digest
            process = context.Process(target=_exit_during_publication, args=(
                fixture.root, fixture.layout, fixture.binding, key, stage))
            process.start()
            try:
                process.join(10)
                self.assertEqual(process.exitcode, 91)
                with fixture.open_store() as reopened:
                    result = reopened.lookup(key, verify_payloads=True)
                    self.assertEqual(result.is_hit, stage == "token_after_link")
                    self.assertFalse(list((fixture.root / "tmp").iterdir()))
            finally:
                if process.is_alive():
                    process.kill()
                process.join(5)
                process.close()

    def test_corruption_releases_credit_before_waiting_for_writer_lock(self):
        fixture = self.fixture
        fixture.save(tuple(range(512)))
        keys = fixture.keys(range(512))
        with fixture.store._manifest_path(keys[0].digest).open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        fixture.store.close()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_corrupt_read_and_authenticate,
            args=(fixture.root, fixture.layout, fixture.binding, keys, child),
        )
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(20), "lock-order probe timed out")
            report = parent.recv()
            self.assertFalse(report["blocked"], report)
            self.assertEqual(report["errors"], ["ManifestError"])
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
            process.join(5)
            process.close()
            parent.close()

    def test_legacy_maintenance_is_unavailable_without_queuing(self):
        fixture = self.fixture
        fixture.save(tuple(range(256)))
        for command in ('status', 'request'):
            with self.subTest(command=command), self.assertRaises(SystemExit) as raised:
                main((command, '--root', str(fixture.root)))
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse((fixture.root / 'state/deep-scrub-request.json').exists())

    def test_cleanup_attempts_directory_sync_when_temporary_unlink_fails(self):
        fixture = self.fixture
        primary = OSError("publication failed")
        original_unlink = type(fixture.store.root).unlink
        synced_tmp = []

        def fail(stage):
            if stage == "token_before_publish":
                raise primary

        def unlink(path, *args, **kwargs):
            if path.parent == fixture.root / "tmp":
                raise OSError("temporary unlink failed")
            return original_unlink(path, *args, **kwargs)

        def sync(path):
            if path == fixture.root / "tmp":
                synced_tmp.append(path)

        fixture.store._fault_hook = fail
        with patch.object(type(fixture.store.root), "unlink", unlink), patch(
            "spoolcache.rank_store._fsync_directory", side_effect=sync
        ), patch("spoolcache.token_files._fsync_directory", side_effect=sync):
            with self.assertRaises(OSError) as captured:
                fixture.save(tuple(range(256)))
        self.assertIs(captured.exception, primary)
        self.assertEqual(synced_tmp, [fixture.root / "tmp"])
