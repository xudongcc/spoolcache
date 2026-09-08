from __future__ import annotations

import types
import unittest

from spoolcache.hma import build_hma_layout

try:
    import torch
    from vllm.v1 import kv_cache_interface as runtime_kv
except ImportError:
    torch = None
    runtime_kv = None


@unittest.skipIf(runtime_kv is None or torch is None, "a real vLLM runtime is required")
class VLLMCacheSemanticsRuntimeTests(unittest.TestCase):
    def _runtime_config(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            max_in_flight_tokens=32,
            model_config=types.SimpleNamespace(max_model_len=1024),
            parallel_config=types.SimpleNamespace(
                decode_context_parallel_size=1,
            ),
            cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
        )

    def _layout(self, specs: tuple[object, ...]):
        groups = tuple(
            types.SimpleNamespace(
                kv_cache_spec=spec,
                is_eagle_group=False,
                layer_names=(f"layer.{index}",),
            )
            for index, spec in enumerate(specs)
        )
        return build_hma_layout(
            types.SimpleNamespace(num_blocks=128, kv_cache_groups=groups),
            vllm_config=self._runtime_config(),
            spec_kind_resolver=runtime_kv.get_kv_cache_spec_kind,
        )

    def test_real_public_semantic_kinds_drive_reuse_policy(self) -> None:
        common = {
            "block_size": 16,
            "num_kv_heads": 1,
            "head_size": 8,
            "dtype": torch.float16,
        }
        full = runtime_kv.FullAttentionSpec(**common)
        sliding = runtime_kv.SlidingWindowSpec(
            **common,
            sliding_window=64,
        )
        recurrent = runtime_kv.MambaSpec(
            block_size=16,
            shapes=((1, 8),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        )
        layout = self._layout((full, sliding, recurrent))
        self.assertEqual(
            tuple(group.reuse_policy for group in layout.groups),
            ("full", "sliding", "recurrent_align"),
        )
        self.assertEqual(layout.groups[2].running_state_tail_pages, 0)

    def test_real_resolver_accepts_arbitrarily_named_subclass(self) -> None:
        renamed_type = type(
            "ArbitraryRuntimeCacheSpec",
            (runtime_kv.FullAttentionSpec,),
            {},
        )
        spec = renamed_type(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
        )
        layout = self._layout((spec,))
        self.assertEqual(layout.groups[0].reuse_policy, "full")

    def test_real_recurrent_spec_declares_running_state_tail(self) -> None:
        recurrent = runtime_kv.MambaSpec(
            block_size=16,
            shapes=((1, 8),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        layout = self._layout((recurrent,))
        group = layout.groups[0]
        self.assertEqual(group.running_state_tail_pages, 2)
        table = tuple(range(10, 18)) + (0, 0, 91, 92, 93)
        self.assertEqual(group.select_physical_pages(table, 128), (91,))

    def test_real_non_prefix_specs_use_page_ownership_contract(self) -> None:
        common = {
            "block_size": 16,
            "num_kv_heads": 1,
            "head_size": 8,
            "dtype": torch.float16,
        }
        scratch_specs: list[object] = []
        circular_type = getattr(runtime_kv, "CircularBufferSpec", None)
        if circular_type is not None:
            scratch_specs.append(circular_type(**common))
        kpool_type = getattr(runtime_kv, "KpoolTailSpec", None)
        if kpool_type is not None:
            scratch_specs.append(kpool_type(**common, sliding_window=16))
        if not scratch_specs:
            self.skipTest("this vLLM build exposes no non-prefix scratch spec")

        layout = self._layout(tuple(scratch_specs))
        self.assertTrue(
            all(group.reuse_policy == "circular_one" for group in layout.groups)
        )

        wrapped_specs = tuple(
            runtime_kv.UniformTypeKVCacheSpecs(
                block_size=spec.block_size,
                kv_cache_specs={f"layer.{index}": spec},
            )
            for index, spec in enumerate(scratch_specs)
        )
        wrapped_layout = self._layout(wrapped_specs)
        self.assertTrue(
            all(
                group.reuse_policy == "circular_one"
                for group in wrapped_layout.groups
            )
        )


if __name__ == "__main__":
    unittest.main()
