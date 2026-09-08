#!/usr/bin/env python3
"""Probe vLLM's layer-hook order for a possible future G5 experiment.

This is qualification evidence only. SpoolCache does not currently implement
layerwise restore or publication, and the runtime package contains no planner
for either feature.
"""

from __future__ import annotations

import importlib
import inspect
import unittest
from unittest import mock

try:
    from vllm.model_executor.layers.attention import kv_transfer_utils
    from vllm.model_executor.layers.attention import mla_attention
except ImportError:
    kv_transfer_utils = None
    mla_attention = None


@unittest.skipIf(kv_transfer_utils is None, "an installed vLLM runtime is required")
class VLLMLayerHookContractProbe(unittest.TestCase):
    def test_wait_runs_before_attention_and_save_runs_after(self) -> None:
        runtime_attention = importlib.import_module(
            "vllm.model_executor.layers.attention.attention"
        )
        events: list[tuple[str, object]] = []
        metadata = object()
        kv_cache = object()

        class Connector:
            def has_connector_metadata(self) -> bool:
                return True

            def wait_for_layer_load(self, layer_name: str) -> None:
                events.append(("wait", layer_name))

            def save_kv_layer(self, layer_name, cache, attn_metadata) -> None:
                self_outer.assertIs(cache, kv_cache)
                self_outer.assertIs(attn_metadata, metadata)
                events.append(("save", layer_name))

        self_outer = self
        connector = Connector()

        def attention_operation(payload: str, layer_name: str) -> str:
            events.append(("attention", layer_name))
            return payload.upper()

        with (
            mock.patch.object(
                kv_transfer_utils,
                "has_kv_transfer_group",
                return_value=True,
            ),
            mock.patch.object(
                kv_transfer_utils,
                "is_v1_kv_transfer_group",
                return_value=True,
            ),
            mock.patch.object(
                kv_transfer_utils,
                "get_kv_transfer_group",
                return_value=connector,
            ),
            mock.patch.object(
                runtime_attention,
                "get_attention_context",
                return_value=(metadata, object(), kv_cache, object()),
            ),
        ):
            decorated = kv_transfer_utils.maybe_transfer_kv_layer(
                attention_operation
            )
            result = decorated("payload", "model.layers.2.attn")

        self.assertEqual(result, "PAYLOAD")
        self.assertEqual(
            events,
            [
                ("wait", "model.layers.2.attn"),
                ("attention", "model.layers.2.attn"),
                ("save", "model.layers.2.attn"),
            ],
        )

    def test_mla_attention_keeps_transfer_inside_cudagraph_break(self) -> None:
        source = inspect.getsource(mla_attention.unified_mla_attention_with_output)
        eager = source.index("@eager_break_during_capture")
        transfer = source.index("@maybe_transfer_kv_layer")
        self.assertLess(eager, transfer)


if __name__ == "__main__":
    unittest.main()
