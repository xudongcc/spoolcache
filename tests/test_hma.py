from __future__ import annotations

import hashlib
import types
import unittest

from spoolcache.errors import LayoutError
from spoolcache.hma import (
    VLLM_RUNTIME_KV_PROFILE,
    build_hma_layout as _build_hma_layout,
)
from spoolcache.manifest import ObjectDescriptor, RankManifest, ordered_descriptors


class FullAttentionSpec:
    public_kind = "full_attention"

    def __init__(self, block_size: int, page_size_bytes: int = 4) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = page_size_bytes


class SlidingWindowSpec:
    public_kind = "sliding_window"

    def __init__(self, block_size: int, window: int, page_size_bytes: int = 4):
        self.block_size = block_size
        self.storage_block_size = block_size
        self.sliding_window = window
        self.page_size_bytes = page_size_bytes


class UniformTypeKVCacheSpecs:
    def __init__(self, specs: dict[str, SlidingWindowSpec], block_size: int) -> None:
        self.kv_cache_specs = specs
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = sum(spec.page_size_bytes for spec in specs.values())


class MambaSpec:
    public_kind = "mamba"

    def __init__(
        self,
        block_size: int,
        page_size_bytes: int = 12,
        num_speculative_blocks: int = 0,
    ) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = page_size_bytes
        self.mamba_cache_mode = "align"
        self.num_speculative_blocks = num_speculative_blocks


class CircularBufferSpec:
    def __init__(self, block_size: int, page_size_bytes: int = 10) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = page_size_bytes
        self.prefix_cacheable = False

    def max_memory_usage_bytes(self, _vllm_config: object) -> int:
        return self.page_size_bytes

    def max_num_blocks_per_req(
        self, _vllm_config: object, _max_len: int
    ) -> int:
        return 1


class NonPrefixScratchSpec(SlidingWindowSpec):
    def __init__(self, block_size: int, page_size_bytes: int = 14) -> None:
        super().__init__(block_size, block_size, page_size_bytes)
        self.participates_in_prefix_caching = False
        self.admission_blocks = 1
        self.block_table_blocks = 1
        self.calls: list[tuple[str, int, int]] = []

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        self.calls.append(("admission", max_in_flight_tokens, max_model_len))
        return self.admission_blocks

    def max_num_blocks_per_req(self, vllm_config: object, max_len: int) -> int:
        self.calls.append(
            ("block-table", vllm_config.max_in_flight_tokens, max_len)  # type: ignore[attr-defined]
        )
        return self.block_table_blocks

    def max_memory_usage_bytes(self, _vllm_config: object) -> int:
        return self.admission_blocks * self.page_size_bytes


class UnknownCacheSpec:
    block_size = 16
    storage_block_size = 16
    page_size_bytes = 4


def resolve_public_spec_kind(spec: object) -> str:
    return getattr(spec, "public_kind", "unknown")


def build_hma_layout(*args, **kwargs):
    kwargs.setdefault("spec_kind_resolver", resolve_public_spec_kind)
    return _build_hma_layout(*args, **kwargs)


def representative_cache_config() -> types.SimpleNamespace:
    # A large mixed-layout fixture ensures group distribution, byte counts,
    # and page planning are discovered rather than inferred from total layers.
    signatures = (
        (62, 256, None),
        (23, 64, 128),
        (23, 64, 128),
        (42, 4, 8),
        (20, 8, 128),
    )
    groups = []
    for group_index, (count, block_size, window) in enumerate(signatures):
        names = tuple(
            f"group{group_index:02d}.layer{layer_index:03d}"
            for layer_index in range(count)
        )
        if window is None:
            spec = FullAttentionSpec(block_size)
        else:
            spec = UniformTypeKVCacheSpecs(
                {
                    name: SlidingWindowSpec(block_size, window)
                    for name in names
                },
                block_size,
            )
        groups.append(
            types.SimpleNamespace(
                kv_cache_spec=spec,
                is_eagle_group=False,
                layer_names=names,
            )
        )
    return types.SimpleNamespace(num_blocks=1024, kv_cache_groups=tuple(groups))


def runtime_config(
    *, max_in_flight_tokens: int = 1537, max_model_len: int = 131_072
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        max_in_flight_tokens=max_in_flight_tokens,
        model_config=types.SimpleNamespace(max_model_len=max_model_len),
    )


class HMALayoutTests(unittest.TestCase):
    def test_public_semantic_kind_accepts_arbitrary_concrete_names(self) -> None:
        class RenamedFullState:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64
            public_kind = "full_attention"

        class RenamedWindowState:
            block_size = 8
            storage_block_size = 8
            page_size_bytes = 32
            sliding_window = 64
            public_kind = "sliding_window"

        def resolve_kind(spec: object) -> object:
            return types.SimpleNamespace(
                value=spec.public_kind  # type: ignore[attr-defined]
            )

        groups = (
            types.SimpleNamespace(
                kv_cache_spec=RenamedFullState(),
                is_eagle_group=False,
                layer_names=("layer.full",),
            ),
            types.SimpleNamespace(
                kv_cache_spec=RenamedWindowState(),
                is_eagle_group=False,
                layer_names=("layer.window",),
            ),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=128, kv_cache_groups=groups),
            spec_kind_resolver=resolve_kind,
        )
        self.assertEqual(
            tuple(group.reuse_policy for group in layout.groups),
            ("full", "sliding"),
        )

    def test_public_group_semantics_can_classify_opaque_members(self) -> None:
        class OpaqueMember:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64

        members = {
            "layer.0": OpaqueMember(),
            "layer.1": OpaqueMember(),
        }

        class RuntimeGroup:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 128
            kv_cache_specs = members

        runtime_group = RuntimeGroup()

        def resolve_kind(spec: object) -> str:
            return "full_attention" if spec is runtime_group else "unknown"

        layout = build_hma_layout(
            types.SimpleNamespace(
                num_blocks=128,
                kv_cache_groups=(
                    types.SimpleNamespace(
                        kv_cache_spec=runtime_group,
                        is_eagle_group=False,
                        layer_names=tuple(members),
                    ),
                ),
            ),
            spec_kind_resolver=resolve_kind,
        )
        self.assertEqual(layout.groups[0].reuse_policy, "full")

    def test_concrete_class_name_cannot_spoof_public_semantics(self) -> None:
        spoofed = type(
            "FullAttentionSpec",
            (),
            {
                "block_size": 16,
                "storage_block_size": 16,
                "page_size_bytes": 64,
            },
        )()
        group = types.SimpleNamespace(
            kv_cache_spec=spoofed,
            is_eagle_group=False,
            layer_names=("layer.spoofed",),
        )
        with self.assertRaisesRegex(LayoutError, "public KV cache semantic kind"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,)),
                spec_kind_resolver=lambda _spec: "unknown",
            )

    def test_invalid_or_failing_public_semantic_resolver_is_rejected(self) -> None:
        group = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(16),
            is_eagle_group=False,
            layer_names=("layer.full",),
        )
        config = types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,))
        for invalid in (None, True, "", types.SimpleNamespace(value=1)):
            with self.subTest(result=invalid):
                with self.assertRaisesRegex(LayoutError, "invalid public"):
                    build_hma_layout(
                        config,
                        spec_kind_resolver=lambda _spec, value=invalid: value,
                    )

        def raising_resolver(_spec: object) -> object:
            raise RuntimeError("resolver drift")

        with self.assertRaisesRegex(LayoutError, "cannot resolve public"):
            build_hma_layout(
                config,
                spec_kind_resolver=raising_resolver,
            )

    def test_unqualified_public_semantic_kinds_fail_closed(self) -> None:
        group = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(16),
            is_eagle_group=False,
            layer_names=("layer.unqualified",),
        )
        config = types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,))
        for kind in (
            "sink_full_attention",
            "chunked_local_attention",
            "encoder_only_attention",
            "cross_attention",
            "unknown",
        ):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(
                    LayoutError,
                    "unsupported public KV cache semantic kind",
                ):
                    build_hma_layout(
                        config,
                        spec_kind_resolver=lambda _spec, value=kind: value,
                    )

    def test_prefix_participation_marker_requires_exact_boolean(self) -> None:
        spec = FullAttentionSpec(16)
        spec.participates_in_prefix_caching = "false"
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            is_eagle_group=False,
            layer_names=("layer.full",),
        )
        with self.assertRaisesRegex(LayoutError, "must be boolean"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,))
            )

    def test_non_prefix_single_page_contract_needs_no_known_type(self) -> None:
        class RenamedScratchState:
            block_size = 13
            storage_block_size = 13
            page_size_bytes = 52
            prefix_cacheable = False

            def max_memory_usage_bytes(self, _vllm_config: object) -> int:
                return self.page_size_bytes

            def max_num_blocks_per_req(
                self, _vllm_config: object, _max_len: int
            ) -> int:
                return 1

        group = types.SimpleNamespace(
            kv_cache_spec=RenamedScratchState(),
            is_eagle_group=False,
            layer_names=("layer.scratch",),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
            vllm_config=runtime_config(),
            spec_kind_resolver=lambda _spec: "unknown",
        )
        self.assertEqual(layout.groups[0].reuse_policy, "circular_one")

    def test_discovers_layout_without_model_identity(self) -> None:
        config = representative_cache_config()
        discovered = build_hma_layout(config)
        self.assertEqual(discovered.profile, VLLM_RUNTIME_KV_PROFILE)

        config.kv_cache_groups[0].kv_cache_spec = UnknownCacheSpec()
        with self.assertRaisesRegex(LayoutError, "public KV cache semantic kind"):
            build_hma_layout(config)

    def test_runtime_layout_and_page_selection(self) -> None:
        layout = build_hma_layout(representative_cache_config())
        self.assertEqual(layout.alignment_tokens, 256)
        self.assertEqual(layout.selected_page_counts(1024), (4, 2, 2, 2, 16))
        boundary_counts = (4, 16, 16, 256, 128)
        tables = tuple(
            tuple(range(1000 * (index + 1), 1000 * (index + 1) + count))
            for index, count in enumerate(boundary_counts)
        )
        selected = layout.select_physical_pages(tables, 1024)
        self.assertEqual(selected[0], tables[0])
        self.assertEqual(selected[1], tables[1][-2:])
        self.assertEqual(selected[3], tables[3][-2:])
        self.assertEqual(selected[4], tables[4][-16:])

    def test_legacy_profile_argument_is_removed(self) -> None:
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            build_hma_layout(
                representative_cache_config(),
                profile="removed-static-profile",  # type: ignore[call-arg]
            )

    def test_runtime_profile_discovers_single_full_attention_group(self) -> None:
        """A conventional Transformer must not need a model-size table."""

        group = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(16, page_size_bytes=8192),
            is_eagle_group=False,
            layer_names=("model.layers.0.attn", "model.layers.1.attn"),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=4096, kv_cache_groups=(group,))
        )
        self.assertEqual(layout.profile, VLLM_RUNTIME_KV_PROFILE)
        self.assertEqual(layout.alignment_tokens, 16)
        self.assertEqual(len(layout.groups), 1)
        self.assertEqual(layout.groups[0].reuse_policy, "full")
        self.assertEqual(layout.groups[0].manager_page_size_bytes, 8192)
        self.assertEqual(
            tuple(layer.name for layer in layout.groups[0].layers),
            ("model.layers.0.attn", "model.layers.1.attn"),
        )

    def test_runtime_profile_discovers_mixed_known_cache_semantics(self) -> None:
        full = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(32, page_size_bytes=8),
            is_eagle_group=False,
            layer_names=("model.layers.0.attn",),
        )
        sliding_spec = SlidingWindowSpec(8, 64, page_size_bytes=6)
        sliding = types.SimpleNamespace(
            kv_cache_spec=UniformTypeKVCacheSpecs(
                {"model.layers.1.attn": sliding_spec}, 8
            ),
            is_eagle_group=False,
            layer_names=("model.layers.1.attn",),
        )
        recurrent = types.SimpleNamespace(
            kv_cache_spec=MambaSpec(16),
            is_eagle_group=False,
            layer_names=("model.layers.2.attn",),
        )
        circular = types.SimpleNamespace(
            kv_cache_spec=CircularBufferSpec(13),
            is_eagle_group=False,
            layer_names=("model.layers.3.compressor",),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(
                num_blocks=2048,
                kv_cache_groups=(full, sliding, recurrent, circular),
            ),
            vllm_config=runtime_config(),
        )
        self.assertEqual(
            tuple(group.reuse_policy for group in layout.groups),
            ("full", "sliding", "recurrent_align", "circular_one"),
        )
        self.assertEqual(layout.alignment_tokens, 32)
        self.assertEqual(layout.selected_page_counts(128), (4, 8, 1, 1))
        # At connector execution time align-mode Mamba has copied the state at
        # the eight-page boundary into the next active page.
        self.assertEqual(
            layout.groups[2].select_physical_pages(tuple(range(10, 19)), 128),
            (18,),
        )
        # A large scheduled chunk can leave null block-table positions between
        # the restored boundary and the current running-state page.
        self.assertEqual(
            layout.groups[2].select_physical_pages(
                tuple(range(10, 18)) + (0, 0, 91),
                128,
            ),
            (91,),
        )
        with self.assertRaisesRegex(LayoutError, "active running-state page"):
            layout.groups[2].select_physical_pages(tuple(range(10, 18)), 128)
        self.assertEqual(
            layout.groups[-1].select_physical_pages((99,), 128), (99,)
        )
        with self.assertRaisesRegex(LayoutError, "exactly one page"):
            layout.groups[-1].select_physical_pages((99, 100), 128)

    def test_recurrent_page_selection_honors_runtime_state_tail(self) -> None:
        recurrent = types.SimpleNamespace(
            kv_cache_spec=MambaSpec(16, num_speculative_blocks=2),
            is_eagle_group=False,
            layer_names=("model.layers.0.mixer",),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=128, kv_cache_groups=(recurrent,)),
            vllm_config=runtime_config(),
        )
        group = layout.groups[0]
        self.assertEqual(group.running_state_tail_pages, 2)
        # Eight pages cover the restored boundary. Two null gaps may follow,
        # then the running page and two runtime-declared speculative pages.
        table = tuple(range(10, 18)) + (0, 0, 91, 92, 93)
        self.assertEqual(group.select_physical_pages(table, 128), (91,))
        with self.assertRaisesRegex(LayoutError, "null/invalid"):
            group.select_physical_pages(
                tuple(range(10, 18)) + (0, 0, 0, 92, 93),
                128,
            )

        other = types.SimpleNamespace(
            kv_cache_spec=MambaSpec(16, num_speculative_blocks=1),
            is_eagle_group=False,
            layer_names=("model.layers.0.mixer",),
        )
        changed_tail = build_hma_layout(
            types.SimpleNamespace(num_blocks=128, kv_cache_groups=(other,)),
            vllm_config=runtime_config(),
        )
        self.assertNotEqual(layout.logical_digest, changed_tail.logical_digest)

        invalid = types.SimpleNamespace(
            kv_cache_spec=MambaSpec(16, num_speculative_blocks=-1),
            is_eagle_group=False,
            layer_names=("model.layers.0.mixer",),
        )
        with self.assertRaisesRegex(LayoutError, "non-negative integer"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=128, kv_cache_groups=(invalid,)),
                vllm_config=runtime_config(),
            )

    def test_recurrent_group_rejects_inconsistent_runtime_state_tails(self) -> None:
        first = MambaSpec(16, num_speculative_blocks=0)
        second = MambaSpec(16, num_speculative_blocks=2)
        group_spec = UniformTypeKVCacheSpecs(
            {"layer.a": first, "layer.b": second},
            16,
        )
        group_spec.public_kind = "mamba"
        group = types.SimpleNamespace(
            kv_cache_spec=group_spec,
            is_eagle_group=False,
            layer_names=("layer.a", "layer.b"),
        )
        with self.assertRaisesRegex(LayoutError, "different state tails"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

    def test_non_prefix_capabilities_must_not_conflict(self) -> None:
        spec = CircularBufferSpec(13)
        spec.prefix_cacheable = True
        spec.participates_in_prefix_caching = False
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            is_eagle_group=False,
            layer_names=("model.layers.0.compressor",),
        )
        with self.assertRaisesRegex(LayoutError, "capabilities conflict"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

    def test_group_and_member_prefix_capabilities_must_agree(self) -> None:
        member = FullAttentionSpec(16, page_size_bytes=64)

        class ContradictoryGroup:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64
            prefix_cacheable = False
            kv_cache_specs = {"layer.full": member}

            def max_memory_usage_bytes(self, _vllm_config: object) -> int:
                return self.page_size_bytes

            def max_num_blocks_per_req(
                self, _vllm_config: object, _max_len: int
            ) -> int:
                return 1

        group = types.SimpleNamespace(
            kv_cache_spec=ContradictoryGroup(),
            is_eagle_group=False,
            layer_names=("layer.full",),
        )
        with self.assertRaisesRegex(LayoutError, "group/member.*conflict"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        scratch = CircularBufferSpec(16, page_size_bytes=64)

        class ReusableGroup:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64
            prefix_cacheable = True
            kv_cache_specs = {"layer.scratch": scratch}

        group.kv_cache_spec = ReusableGroup()
        group.layer_names = ("layer.scratch",)
        with self.assertRaisesRegex(LayoutError, "group/member.*conflict"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

    def test_non_prefix_group_must_prove_shared_page_ownership(self) -> None:
        scratch = CircularBufferSpec(16, page_size_bytes=64)

        class UnprovenScratchGroup:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64
            kv_cache_specs = {"layer.scratch": scratch}

        group = types.SimpleNamespace(
            kv_cache_spec=UnprovenScratchGroup(),
            is_eagle_group=False,
            layer_names=("layer.scratch",),
        )
        with self.assertRaisesRegex(LayoutError, "block-table ownership"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        class InvalidScratchGroup:
            block_size = 16
            storage_block_size = 16
            page_size_bytes = 64
            prefix_cacheable = False
            kv_cache_specs = {"layer.scratch": scratch}

            def max_memory_usage_bytes(self, _vllm_config: object) -> int:
                return self.page_size_bytes

            def max_num_blocks_per_req(
                self, _vllm_config: object, _max_len: int
            ) -> int:
                return 2

        group.kv_cache_spec = InvalidScratchGroup()
        with self.assertRaisesRegex(LayoutError, "exactly one block-table page"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        class ProvenScratchGroup(UnprovenScratchGroup):
            def max_memory_usage_bytes(self, _vllm_config: object) -> int:
                return self.page_size_bytes

            def max_num_blocks_per_req(
                self, _vllm_config: object, _max_len: int
            ) -> int:
                return 1

        group.kv_cache_spec = ProvenScratchGroup()
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
            vllm_config=runtime_config(),
        )
        self.assertEqual(layout.groups[0].reuse_policy, "circular_one")

    def test_non_prefix_single_page_contract_uses_circular_semantics(self) -> None:
        """A SW-derived scratch spec must not be mistaken for a SW prefix."""

        spec = NonPrefixScratchSpec(20)
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            is_eagle_group=False,
            layer_names=("model.layers.0.indexer.kpool_tail",),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
            vllm_config=runtime_config(),
        )
        self.assertEqual(layout.groups[0].reuse_policy, "circular_one")
        self.assertEqual(layout.alignment_tokens, 1)
        self.assertEqual(layout.groups[0].selected_page_count(12345), 1)
        self.assertEqual(
            layout.groups[0].select_physical_pages((77,), 12345), (77,)
        )

        spec.admission_blocks = 2
        with self.assertRaisesRegex(LayoutError, "exactly one admission"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

    def test_non_prefix_contract_uses_real_deployment_bounds(self) -> None:
        spec = NonPrefixScratchSpec(20)
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            is_eagle_group=False,
            layer_names=("model.layers.0.scratch",),
        )
        config = runtime_config(max_in_flight_tokens=17, max_model_len=65_537)
        build_hma_layout(
            types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
            vllm_config=config,
        )
        self.assertEqual(
            spec.calls,
            [
                ("admission", 17, 65_537),
                ("block-table", 17, 65_537),
            ],
        )

    def test_non_prefix_contract_rejects_unprovable_or_non_integer_results(self) -> None:
        group = types.SimpleNamespace(
            is_eagle_group=False,
            layer_names=("model.layers.0.scratch",),
        )
        cases: tuple[tuple[str, object, str], ...] = (
            ("boolean admission", True, "positive integer"),
            ("multi-page admission", 2, "exactly one admission"),
        )
        for label, result, message in cases:
            with self.subTest(label=label):
                spec = NonPrefixScratchSpec(20)
                spec.admission_blocks = result  # type: ignore[assignment]
                group.kv_cache_spec = spec
                with self.assertRaisesRegex(LayoutError, message):
                    build_hma_layout(
                        types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                        vllm_config=runtime_config(),
                    )

        spec = NonPrefixScratchSpec(20)
        spec.block_table_blocks = 2
        group.kv_cache_spec = spec
        with self.assertRaisesRegex(LayoutError, "exactly one block-table"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        spec = NonPrefixScratchSpec(20)
        spec.block_table_blocks = True  # type: ignore[assignment]
        group.kv_cache_spec = spec
        with self.assertRaisesRegex(LayoutError, "positive integer"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        spec = NonPrefixScratchSpec(20)
        spec.max_memory_usage_bytes = lambda _config: 2 * spec.page_size_bytes  # type: ignore[method-assign]
        group.kv_cache_spec = spec
        with self.assertRaisesRegex(LayoutError, "exactly one physical page"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        spec = NonPrefixScratchSpec(20)
        spec.max_memory_usage_bytes = lambda _config: True  # type: ignore[method-assign]
        group.kv_cache_spec = spec
        with self.assertRaisesRegex(LayoutError, "positive integer"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        class MissingBlockTableContract(SlidingWindowSpec):
            participates_in_prefix_caching = False

            def max_admission_blocks_per_request(
                self, max_in_flight_tokens: int, max_model_len: int
            ) -> int:
                return 1

        group.kv_cache_spec = MissingBlockTableContract(20, 20)
        with self.assertRaisesRegex(LayoutError, "block-table ownership"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        class RaisingContract(NonPrefixScratchSpec):
            def max_num_blocks_per_req(
                self, vllm_config: object, max_len: int
            ) -> int:
                raise RuntimeError("runtime contract failed")

        group.kv_cache_spec = RaisingContract(20)
        with self.assertRaisesRegex(LayoutError, "cannot evaluate"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

        spec = NonPrefixScratchSpec(20)
        group.kv_cache_spec = spec
        with self.assertRaisesRegex(LayoutError, "requires the current vLLM config"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,))
            )

    def test_non_prefix_contract_getter_failure_is_normalized(self) -> None:
        class RaisingBlockTableCapability(NonPrefixScratchSpec):
            @property
            def max_num_blocks_per_req(self):
                raise RuntimeError("runtime property failed")

        group = types.SimpleNamespace(
            kv_cache_spec=RaisingBlockTableCapability(20),
            is_eagle_group=False,
            layer_names=("model.layers.0.scratch",),
        )
        with self.assertRaisesRegex(LayoutError, "cannot read.*block-table"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(),
            )

    def test_non_prefix_contract_cannot_evade_with_unsampled_lengths(self) -> None:
        class DeploymentBoundScratchSpec(NonPrefixScratchSpec):
            def max_admission_blocks_per_request(
                self, max_in_flight_tokens: int, max_model_len: int
            ) -> int:
                return 2 if max_in_flight_tokens == 17 else 1

        spec = DeploymentBoundScratchSpec(20)
        group = types.SimpleNamespace(
            kv_cache_spec=spec,
            is_eagle_group=False,
            layer_names=("model.layers.0.scratch",),
        )
        with self.assertRaisesRegex(LayoutError, "exactly one admission"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=8, kv_cache_groups=(group,)),
                vllm_config=runtime_config(max_in_flight_tokens=17),
            )

    def test_runtime_profile_fails_closed_for_unknown_cache_semantics(self) -> None:
        group = types.SimpleNamespace(
            kv_cache_spec=UnknownCacheSpec(),
            is_eagle_group=False,
            layer_names=("model.layers.0.attn",),
        )
        with self.assertRaisesRegex(LayoutError, "public KV cache semantic kind"):
            build_hma_layout(
                types.SimpleNamespace(num_blocks=128, kv_cache_groups=(group,))
            )

    def test_runtime_layout_digest_tracks_geometry_not_pool_capacity(self) -> None:
        config = representative_cache_config()
        automatic = build_hma_layout(config)
        config.num_blocks += 100
        resized = build_hma_layout(config)
        self.assertEqual(automatic.digest, resized.digest)
        config.kv_cache_groups[1].kv_cache_spec.kv_cache_specs[
            "group01.layer000"
        ].page_size_bytes += 1
        config.kv_cache_groups[1].kv_cache_spec.page_size_bytes += 1
        changed = build_hma_layout(config)
        self.assertNotEqual(resized.digest, changed.digest)
        self.assertEqual(resized.logical_digest, changed.logical_digest)

    def test_logical_layout_digest_tracks_cross_role_semantics(self) -> None:
        """Scheduler and worker byte views share one logical prefix identity."""

        scheduler_config = representative_cache_config()
        worker_config = representative_cache_config()
        for group in worker_config.kv_cache_groups:
            # Physical compressed storage width is also a worker concern. The
            # logical block table still advances in ``block_size`` units.
            group.kv_cache_spec.storage_block_size = max(
                1, group.kv_cache_spec.block_size // 2
            )
            per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            if per_layer is None:
                group.kv_cache_spec.page_size_bytes *= 128
            else:
                for spec in per_layer.values():
                    spec.storage_block_size = max(1, spec.block_size // 2)
                    spec.page_size_bytes *= 128
                group.kv_cache_spec.page_size_bytes = sum(
                    spec.page_size_bytes for spec in per_layer.values()
                )
        scheduler = build_hma_layout(scheduler_config)
        worker = build_hma_layout(worker_config)
        self.assertEqual(scheduler.logical_digest, worker.logical_digest)
        self.assertNotEqual(scheduler.digest, worker.digest)

        worker_config.kv_cache_groups[0].is_eagle_group = True
        changed_semantics = build_hma_layout(worker_config)
        self.assertNotEqual(worker.logical_digest, changed_semantics.logical_digest)

    def test_pp_coordination_digest_excludes_stage_local_layer_ownership(self) -> None:
        stage_zero = representative_cache_config()
        stage_one = representative_cache_config()
        for group_index, group in enumerate(stage_one.kv_cache_groups):
            names = tuple(
                f"stage1.group{group_index}.layer{index}"
                for index in range(max(1, len(group.layer_names) - 1))
            )
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                original = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
                group.kv_cache_spec = UniformTypeKVCacheSpecs(
                    {
                        name: SlidingWindowSpec(
                            original.block_size,
                            original.sliding_window,
                            original.page_size_bytes,
                        )
                        for name in names
                    },
                    original.block_size,
                )
            group.layer_names = names

        first = build_hma_layout(stage_zero)
        second = build_hma_layout(stage_one)
        self.assertNotEqual(first.logical_digest, second.logical_digest)
        self.assertEqual(first.coordination_digest, second.coordination_digest)

        for spec in stage_one.kv_cache_groups[1].kv_cache_spec.kv_cache_specs.values():
            spec.sliding_window += 64
        changed = build_hma_layout(stage_one)
        self.assertNotEqual(second.coordination_digest, changed.coordination_digest)

    def test_pp_stage_may_own_no_layers_in_one_global_group(self) -> None:
        global_names = ("other-stage.0", "other-stage.1")
        group = types.SimpleNamespace(
            kv_cache_spec=UniformTypeKVCacheSpecs(
                {
                    name: SlidingWindowSpec(16, 64)
                    for name in global_names
                },
                16,
            ),
            is_eagle_group=False,
            layer_names=(),
        )
        owned_group = types.SimpleNamespace(
            kv_cache_spec=FullAttentionSpec(16),
            is_eagle_group=False,
            layer_names=("this-stage.0",),
        )
        layout = build_hma_layout(
            types.SimpleNamespace(
                num_blocks=8,
                kv_cache_groups=(group, owned_group),
            )
        )
        self.assertEqual(layout.groups[0].layers, ())
        self.assertEqual(layout.groups[0].reuse_policy, "sliding")

    def test_eagle_group_is_identity_bound_not_rejected(self) -> None:
        config = representative_cache_config()
        base = build_hma_layout(config)
        config.kv_cache_groups[0].is_eagle_group = True
        eagle = build_hma_layout(config)
        self.assertTrue(eagle.groups[0].is_eagle_group)
        self.assertNotEqual(base.digest, eagle.digest)

    def test_runtime_manager_capacity_is_not_identity_bound(self) -> None:
        config = representative_cache_config()
        base = build_hma_layout(config)
        config.num_blocks += 137
        resized = build_hma_layout(config)
        self.assertNotEqual(base.num_manager_blocks, resized.num_manager_blocks)
        self.assertEqual(base.digest, resized.digest)

    def test_manifest_must_cover_every_layer_and_page(self) -> None:
        layout = build_hma_layout(representative_cache_config())
        counts = layout.selected_page_counts(256)
        descriptors = []
        for group, count in zip(layout.groups, counts, strict=True):
            for layer in group.layers:
                digest = hashlib.sha256(
                    f"{group.group_index}:{layer.name}".encode()
                ).hexdigest()
                descriptors.append(
                    ObjectDescriptor(
                        group_index=group.group_index,
                        layer_name=layer.name,
                        page_start=0,
                        page_count=count,
                        byte_length=count * layer.page_size_bytes,
                        stored_length=count * layer.page_size_bytes,
                        sha256=digest,
                        relative_path=f"objects/{digest[:2]}/{digest}.spool",
                    )
                )
        manifest = RankManifest(
            entry_id="a" * 64,
            deployment_identity_digest="b" * 64,
            rank_identity_digest="c" * 64,
            span_tokens=256,
            physical_rank=0,
            topology_digest="d" * 64,
            profile=VLLM_RUNTIME_KV_PROFILE,
            layout_digest=layout.digest,
            objects=ordered_descriptors(descriptors),
            created_at_unix_ns=1,
        )
        layout.validate_manifest_coverage(manifest)
        incomplete = RankManifest(
            **{
                **manifest.__dict__,
                "objects": manifest.objects[:-1],
            }
        )
        with self.assertRaisesRegex(LayoutError, "exactly every"):
            layout.validate_manifest_coverage(incomplete)


if __name__ == "__main__":
    unittest.main()
