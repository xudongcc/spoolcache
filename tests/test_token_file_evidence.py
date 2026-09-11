"""The external verifier must check runtime identity before payload reads."""

import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from benchmarks.token_file_evidence import audit, digest
from tests import test_token_files_hma as fixtures
from tests.token_fixtures import cpu_mover, open_store
from spoolcache.prefix import prefix_digests


class TokenEvidenceTests(unittest.TestCase):
    @staticmethod
    def identity_for(layout, transfer_bytes):
        return {
            "deployment": "b" * 64, "rank_identity": "c" * 64,
            "physical_rank": 0, "topology": "d" * 64,
            "hma_layout": layout.digest, "transfer_bytes": transfer_bytes,
            "groups": [
                dict(index=g.group_index, layers=len(g.layers), block=g.block_size,
                     page_bytes=g.manager_page_size_bytes, dcp_shards=g.dcp_shard_count,
                     eagle=g.is_eagle_group, policy=g.reuse_policy, window=g.reuse_window_tokens,
                     layer_names_digest=digest({"layers": [x.name for x in g.layers]})[:12])
                for g in layout.groups
            ],
        }

    def test_auditor_derives_sharded_alignment_and_exact_state_without_scratch_capacity(self):
        original = fixtures.mixed_layout()
        full = replace(original.groups[0], block_size=192, storage_block_size=192,
                       dcp_shard_count=2, logical_tokens_per_page=384)
        state = replace(original.groups[1], block_size=320, storage_block_size=320,
                        logical_tokens_per_page=320, reuse_policy="recurrent_align",
                        reuse_window_tokens=None)
        scratch = replace(state, group_index=2, block_size=137, storage_block_size=137,
                          logical_tokens_per_page=137, reuse_policy="circular_one",
                          layers=(replace(state.layers[0], name="scratch"),))
        layout = replace(original, dcp_degree=2, groups=(full, state, scratch))
        self.assertEqual(layout.alignment_tokens, 1920)
        keys = prefix_digests(range(1920), deployment_digest="b" * 64, chunk_tokens=1920)
        with tempfile.TemporaryDirectory() as directory, open_store(
            Path(directory) / "rank", layout=layout, slot_bytes=4096
        ) as store:
            cpu_mover(layout, slot_bytes=4096).commit_keys(
                store, prefixes=keys, block_tables=((1, 2, 3, 4, 5), (0,) * 6 + (7,), (8,)))
            result = audit(store.root, keys[-1].digest, 1920, self.identity_for(layout, 4096))
            self.assertEqual(result["files"], 2)
            self.assertEqual(result["payload_bytes"], 7 * 64)

    def setUp(self):
        self.fixture = fixtures.TokenFileHMATests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.snapshot = self.fixture.save(1024)
        self.identity = {
            "deployment": "b" * 64,
            "rank_identity": "c" * 64,
            "physical_rank": 0,
            "topology": "d" * 64,
            "hma_layout": self.fixture.layout.digest,
            "transfer_bytes": self.fixture.store._pool.slot_bytes,
            "groups": [
                {
                    "index": g.group_index,
                    "layers": len(g.layers),
                    "block": g.block_size,
                    "page_bytes": g.manager_page_size_bytes,
                    "dcp_shards": g.dcp_shard_count,
                    "eagle": g.is_eagle_group,
                    "policy": g.reuse_policy,
                    "window": g.reuse_window_tokens,
                    "layer_names_digest": digest(
                        {"layers": [x.name for x in g.layers]}
                    )[:12],
                }
                for g in self.fixture.layout.groups
            ],
        }

    def test_independent_full_payload_and_state_corruption(self):
        result = audit(self.fixture.root, self.snapshot.entry_id, 1024, self.identity)
        self.assertEqual(result["files"], 5)
        self.assertEqual(result["payload_bytes"], 1536)
        state = self.snapshot.objects[-1]
        with self.fixture.store._manifest_path(state.key).open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        with self.assertRaisesRegex(ValueError, "checksum"):
            audit(self.fixture.root, self.snapshot.entry_id, 1024, self.identity)

    def test_wrong_identity_and_incomplete_chain_fail_before_payload_io(self):
        with patch(
            "benchmarks.token_file_evidence.os.read",
            side_effect=AssertionError("payload read"),
        ):
            with self.assertRaisesRegex(ValueError, "binding"):
                audit(
                    self.fixture.root,
                    self.snapshot.entry_id,
                    1024,
                    dict(self.identity, physical_rank=1),
                )
            self.fixture.store.evict(self.fixture.keys(1024)[1].digest)
            with self.assertRaises(FileNotFoundError):
                audit(self.fixture.root, self.snapshot.entry_id, 1024, self.identity)

    def test_packed_heterogeneous_pages_and_draft_state_use_group_totals(self):
        fixture = fixtures.TokenFileHMATests()
        original = fixtures.mixed_layout()
        groups = tuple(
            replace(
                g,
                manager_page_size_bytes=96,
                is_eagle_group=g.group_index == 1,
                layers=(
                    g.layers[0],
                    replace(
                        g.layers[0],
                        name=g.layers[0].name + "_small",
                        page_size_bytes=32,
                    ),
                ),
            )
            for g in original.groups
        )
        layout = replace(original, groups=groups)
        with patch.object(fixtures, "mixed_layout", return_value=layout):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        snapshot = fixture.save(1024)
        identity = dict(self.identity, hma_layout=layout.digest)
        identity["groups"] = [
            dict(
                self.identity["groups"][index],
                layers=2,
                page_bytes=96,
                eagle=index == 1,
                layer_names_digest=digest({"layers": [x.name for x in g.layers]})[:12],
            )
            for index, g in enumerate(groups)
        ]
        result = audit(fixture.root, snapshot.entry_id, 1024, identity)
        self.assertEqual(result["payload_bytes"], 2304)
        self.assertEqual(
            set(result["group_page_geometry"].values()), {"packed-group-pages"}
        )
        identity["groups"][0]["page_bytes"] += 1
        with (
            patch(
                "benchmarks.token_file_evidence.os.read",
                side_effect=AssertionError("payload read"),
            ),
            self.assertRaisesRegex(ValueError, "group byte total"),
        ):
            audit(fixture.root, snapshot.entry_id, 1024, identity)
