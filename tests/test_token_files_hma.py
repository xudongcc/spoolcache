"""Mixed HMA needs both shared full KV chunks and exact boundary state."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from spoolcache.errors import LayoutError, ManifestError
from spoolcache.hma import HMALayout
from spoolcache.prefix import prefix_digests
from spoolcache.token_files import TokenFileStore, boundary_state_key
from spoolcache.token_mover import TokenPageMover
from tests.token_fixtures import cpu_mover, layout_for


def mixed_layout(tokens_per_page=64):
    full = layout_for(tokens_per_page=tokens_per_page, page_bytes=64).groups[0]
    window = replace(
        layout_for("sliding", tokens_per_page=tokens_per_page, page_bytes=64).groups[0],
        group_index=1,
        layers=(replace(full.layers[0], name="window"),),
    )
    return HMALayout(100, 1, (full, window))


class TokenFileHMATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.layout = mixed_layout()
        self.root = Path(self.temp.name) / "rank"
        self.store = self.open_store()
        self.addCleanup(self.store.close)
        self.mover = cpu_mover(self.layout)
        self.mover.__class__ = TokenPageMover
        capture = self.mover._capture_packed

        def captured(*args):
            yield from capture(*args)

        self.mover._capture_packed = captured

    def open_store(self):
        return TokenFileStore(
            self.root,
            layout=self.layout,
            slot_bytes=16384,
            expected_deployment_digest="b" * 64,
            expected_rank_digest="c" * 64,
            expected_rank=0,
            expected_topology_digest="d" * 64,
            expected_profile=self.layout.profile,
            expected_layout_digest=self.layout.digest,
        )

    def keys(self, span):
        return prefix_digests(range(span), deployment_digest="b" * 64, chunk_tokens=self.store.chunk_tokens)

    def save(self, span):
        pages = span // 64
        # Historical window table entries have already been released by vLLM.
        table = (0,) * max(0, pages - 8) + tuple(range(max(1, pages - 7), pages + 1))
        return self.mover.commit_keys(
            self.store,
            prefixes=self.keys(span),
            block_tables=(tuple(range(1, pages + 1)), table),
        )

    def test_shared_full_chunks_require_exact_boundary_state(self):
        snapshot = self.save(1024)
        keys = self.keys(1024)
        self.assertEqual(len(snapshot.objects), 5)
        self.assertEqual(len(list(self.store.iter_manifest_paths())), 5)
        self.assertEqual(snapshot.objects[-1].key, boundary_state_key(keys[-1].digest))
        self.assertTrue(self.store.lookup(keys[-1].digest, verify_payloads=True).is_hit)
        self.assertFalse(self.store.lookup(keys[1].digest).is_hit)
        self.layout.validate_manifest_coverage(snapshot)
        captured = []

        @contextmanager
        def receive(desc):
            yield lambda data: captured.append((desc.kind, bytes(data)))

        with self.store.restore_view(keys[-1].digest) as view:
            self.store.stream_objects(view.descriptors, receive, lease=view)
        self.assertEqual(
            b"".join(data for kind, data in captured if kind == "data"),
            b"".join(bytes((i,)) * 64 for i in range(1, 17)),
        )
        self.assertEqual(
            captured[-1], ("state", b"".join(bytes((i,)) * 64 for i in range(9, 17)))
        )
        # Saving a shorter boundary reuses existing full KV and captures only
        # that boundary's window; it cannot invent state from the longer one.
        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(512)
            self.assertEqual(capture.call_count, 1)
            self.save(1024)
            self.assertEqual(capture.call_count, 1)
        self.assertEqual(len(list(self.store.iter_manifest_paths())), 6)
        self.assertTrue(self.store.lookup(keys[1].digest, verify_payloads=True).is_hit)
        self.store.close()
        original = os.open

        def ordinary(path, flags, *args, **kwargs):
            self.assertFalse(flags & os.O_DIRECT)
            return original(path, flags, *args, **kwargs)

        with patch("os.open", side_effect=ordinary), self.open_store() as reopened:
            self.assertTrue(
                reopened.lookup(keys[-1].digest, verify_payloads=True).is_hit
            )
            self.assertEqual(len(reopened.scan_offers(100)), 6)

    def test_window_state_is_pinned_fenced_and_independently_recoverable(self):
        snapshot = self.save(1024)
        state = snapshot.objects[-1]
        with self.store.restore_view(snapshot.entry_id) as view:
            self.assertFalse(self.store.evict(state.key))
            with self.store._manifest_path(state.key).open("r+b") as f:
                f.seek(-1, 2)
                f.write(b"!")

            @contextmanager
            def receive(desc):
                yield lambda data: None

            with self.assertRaises(ManifestError):
                self.store.stream_objects(view.descriptors, receive, lease=view)
            self.assertTrue(self.store._inventory_is_withdrawn(state.key))
        self.assertFalse(self.store.lookup(snapshot.entry_id).is_hit)

        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(1024)
            self.assertEqual(capture.call_count, 1)
        self.assertTrue(
            self.store.lookup(snapshot.entry_id, verify_payloads=True).is_hit
        )
        self.assertTrue(self.store.evict(self.keys(1024)[0].digest))
        self.assertFalse(self.store.lookup(snapshot.entry_id).is_hit)

    def test_boundary_publication_failure_leaves_only_data_offers(self):
        def fail(stage):
            if (
                stage == "token_before_publish"
                and len(list(self.store.iter_manifest_paths())) == 4
            ):
                raise OSError("state publication failure")

        self.store._fault_hook = fail
        with self.assertRaisesRegex(OSError, "state publication failure"):
            self.save(1024)
        key = self.keys(1024)[-1].digest
        self.assertEqual(len(self.store.scan_offers(100)), 4)
        self.assertNotIn(
            boundary_state_key(key), {x.entry_id for x in self.store.scan_offers(100)}
        )
        self.assertFalse(self.store.lookup(key).is_hit)
        self.assertFalse(list((self.root / "tmp").iterdir()))
        self.store._fault_hook = None
        with patch.object(
            self.mover, "_capture_packed", wraps=self.mover._capture_packed
        ) as capture:
            self.save(1024)
            self.assertEqual(capture.call_count, 1)
        self.assertTrue(self.store.lookup(key, verify_payloads=True).is_hit)

    def test_state_key_is_not_a_prefix_but_query_does_not_withdraw_it(self):
        snapshot = self.save(1024)
        state = snapshot.objects[-1]
        self.assertFalse(self.store.lookup(state.key).is_hit)
        self.assertIn(state.key, {x.entry_id for x in self.store.scan_offers(100)})
        self.assertFalse(self.store._inventory_is_withdrawn(state.key))

    def test_state_bound_is_checked_at_startup(self):
        group = self.layout.groups[1]
        large = replace(
            group, layers=(replace(group.layers[0], page_size_bytes=513 * 64 * 1024 * 1024),)
        )
        with self.assertRaisesRegex(LayoutError, "boundary state"):
            TokenPageMover.validate_layout(
                replace(self.layout, groups=(self.layout.groups[0], large))
            )

    def test_recurrent_alignment_determines_data_chunk_width(self):
        self.store.close()
        group = self.layout.groups[1]
        state = replace(
            group,
            block_size=1024,
            storage_block_size=1024,
            logical_tokens_per_page=1024,
            reuse_policy="recurrent_align",
            reuse_window_tokens=None,
            running_state_tail_pages=1,
        )
        self.layout = replace(self.layout, groups=(self.layout.groups[0], state))
        self.root = Path(self.temp.name) / "recurrent"
        self.store = self.open_store()
        self.addCleanup(self.store.close)
        self.mover.layout = self.layout
        # Active running state is table index 2, before one speculative tail.
        snapshot = self.mover.commit_keys(
            self.store,
            prefixes=self.keys(1024),
            block_tables=(tuple(range(1, 17)), (0, 0, 37, 38)),
        )
        self.assertEqual(snapshot.objects[-1].segments[0].page_count, 1)
        self.assertTrue(
            self.store.lookup(snapshot.entry_id, verify_payloads=True).is_hit
        )
