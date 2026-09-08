from __future__ import annotations

import unittest

from spoolcache.identity import sha256_json
from spoolcache.topology import (
    WorkerCoordinate,
    expected_worker_coordinates,
    rank_ownership_sha256,
)


class PipelineTopologyTests(unittest.TestCase):
    def test_pp2_coordinates_use_vllm_pp_then_tp_rank_order(self) -> None:
        coordinates = expected_worker_coordinates(
            tp_degree=2,
            pp_degree=2,
            dcp_degree=1,
        )
        self.assertEqual(
            coordinates,
            (
                WorkerCoordinate(0, 0, 0, 0),
                WorkerCoordinate(1, 0, 1, 0),
                WorkerCoordinate(2, 1, 0, 0),
                WorkerCoordinate(3, 1, 1, 0),
            ),
        )

    def test_dcp_rank_is_the_runtime_tp_subdivision(self) -> None:
        coordinate = WorkerCoordinate.from_runtime(
            global_rank=6,
            pp_rank=1,
            tp_rank=2,
            dcp_rank=0,
            tp_degree=4,
            pp_degree=2,
            dcp_degree=2,
        )
        self.assertEqual(coordinate, WorkerCoordinate(6, 1, 2, 0))

        invalid = (
            {"global_rank": 5},
            {"pp_rank": 0},
            {"tp_rank": 4},
            {"dcp_rank": 1},
        )
        for replacement in invalid:
            values = {
                "global_rank": 6,
                "pp_rank": 1,
                "tp_rank": 2,
                "dcp_rank": 0,
            }
            values.update(replacement)
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    WorkerCoordinate.from_runtime(
                        **values,
                        tp_degree=4,
                        pp_degree=2,
                        dcp_degree=2,
                    )

    def test_pp1_rank_ownership_digest_is_byte_stable(self) -> None:
        coordinate = WorkerCoordinate.from_runtime(
            global_rank=1,
            pp_rank=0,
            tp_rank=1,
            dcp_rank=0,
            tp_degree=2,
            pp_degree=1,
            dcp_degree=1,
        )
        expected = sha256_json(
            {
                "layers": ("layer.0", "layer.1"),
                "pp": 1,
                "dp_rank": 0,
            }
        )
        self.assertEqual(
            rank_ownership_sha256(
                layer_names=("layer.0", "layer.1"),
                shared_aliases=(),
                coordinate=coordinate,
                pp_degree=1,
                dp_rank=0,
            ),
            expected,
        )

    def test_pp2_ownership_binds_stage_local_facts_and_aliases(self) -> None:
        base = dict(
            layer_names=("layer.0", "layer.1"),
            shared_aliases=(("alias.0", ("layer.0",)),),
            coordinate=WorkerCoordinate(2, 1, 0, 0),
            pp_degree=2,
            dp_rank=0,
        )
        digest = rank_ownership_sha256(**base)
        variants = (
            {**base, "layer_names": ("layer.0",)},
            {**base, "shared_aliases": ()},
            {**base, "coordinate": WorkerCoordinate(3, 1, 1, 0)},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(digest, rank_ownership_sha256(**variant))


if __name__ == "__main__":
    unittest.main()
