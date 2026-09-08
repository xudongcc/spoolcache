from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from spoolcache.config import AccessMode, SpoolCacheConfig
from spoolcache.errors import ConfigurationError, IdentityError
from spoolcache.hma import VLLM_RUNTIME_KV_PROFILE
from spoolcache.identity import (
    DeploymentIdentity,
    RankIdentity,
    model_namespace_sha256,
)
from spoolcache.prefix import (
    MultimodalFeatureIdentity,
    aligned_prefix_span,
    prefix_digests,
)


class ConfigIdentityPrefixTests(unittest.TestCase):
    def _config(self, root: Path, **extra: object) -> SpoolCacheConfig:
        raw: dict[str, object] = {
            "spoolcache_root": root,
            "spoolcache_direct_io": "disabled",
        }
        raw.update(extra)
        return SpoolCacheConfig.from_mapping(raw)

    def test_strict_config_and_enabled_data_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self.assertEqual(config.access_mode, AccessMode.DISABLED)
            self.assertEqual(config.low_watermark_bytes, 180 * 1024**3)
            with self.assertRaisesRegex(ConfigurationError, "unknown"):
                self._config(Path(directory), surprise=True)
            for removed in (
                "spoolcache_profile",
                "spoolcache_expected_vllm_version",
                "spoolcache_qualified_gpu_mover",
                "spoolcache_runtime_compatibility",
                "spoolcache_checkpoint_sha256",
                "spoolcache_chunk_tokens",
                "spoolcache_min_span_tokens",
                "spoolcache_max_span_tokens",
                "spoolcache_slot_bytes",
                "spoolcache_slot_count",
                "spoolcache_max_pending_restores",
                "spoolcache_max_pending_stores",
                "spoolcache_startup_max_digests",
                "spoolcache_report_batch_size",
                "spoolcache_catalog_max_entries",
                "spoolcache_low_watermark_bytes",
                "spoolcache_require_cache_salt",
            ):
                with self.subTest(removed=removed):
                    with self.assertRaisesRegex(ConfigurationError, "unknown"):
                        self._config(Path(directory), **{removed: "removed"})
            enabled = self._config(
                Path(directory),
                spoolcache_access_mode="read-write",
            )
            self.assertEqual(enabled.access_mode, AccessMode.READ_WRITE)

    def test_low_watermark_is_a_bounded_internal_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory),
                spoolcache_max_bytes=10_001,
            )
            self.assertEqual(config.low_watermark_bytes, 9_000)
            with self.assertRaisesRegex(ConfigurationError, "at least 2"):
                self._config(Path(directory), spoolcache_max_bytes=1)

    def test_deployment_digest_is_mapping_order_independent(self) -> None:
        common = dict(
            schema="spoolcache-deployment/v2",
            profile=VLLM_RUNTIME_KV_PROFILE,
            model_namespace_sha256="1" * 64,
            model_config_sha256="2" * 64,
            execution_config_sha256="4" * 64,
            vllm_version="0.25.2.dev0+g752a3a504.d20260714",
            vllm_build_sha256="5" * 64,
            kv_cache_dtype="fp8_ds_mla",
            layout_sha256="3" * 64,
            chunk_tokens=256,
            spoolcache_version="0.1.0a0",
        )
        left = DeploymentIdentity(
            topology={
                "tp": 2,
                "pp": 1,
                "dcp": 1,
                "dp": 1,
                "dp_rank": 0,
                "world_size": 2,
                "world_size_across_dp": 2,
            },
            **common,
        )
        right = DeploymentIdentity(
            topology={
                "world_size_across_dp": 2,
                "world_size": 2,
                "dp_rank": 0,
                "dp": 1,
                "dcp": 1,
                "pp": 1,
                "tp": 2,
            },
            **common,
        )
        self.assertEqual(left.digest, right.digest)
        with self.assertRaisesRegex(IdentityError, "data-parallel"):
            DeploymentIdentity(
                topology={**dict(left.topology), "dp_rank": 1},
                **common,
            )
        with self.assertRaisesRegex(IdentityError, "malformed SHA-256"):
            DeploymentIdentity(
                topology=left.topology,
                **{**common, "vllm_build_sha256": "not-a-build-proof"},
            )
        with self.assertRaisesRegex(IdentityError, "schema"):
            DeploymentIdentity(
                topology=left.topology,
                **{**common, "schema": "spoolcache-deployment/v1"},
            )

        isolated = (
            replace(left, model_namespace_sha256="a" * 64),
            replace(left, model_config_sha256="b" * 64),
            replace(left, execution_config_sha256="c" * 64),
            replace(left, vllm_version="different-runtime"),
            replace(left, vllm_build_sha256="d" * 64),
            replace(left, kv_cache_dtype="different-dtype"),
            replace(left, layout_sha256="e" * 64),
            replace(
                left,
                topology={
                    "tp": 1,
                    "pp": 1,
                    "dcp": 1,
                    "dp": 1,
                    "dp_rank": 0,
                    "world_size": 1,
                    "world_size_across_dp": 1,
                },
            ),
        )
        for identity in isolated:
            with self.subTest(identity=identity):
                self.assertNotEqual(left.digest, identity.digest)

    def test_model_namespace_uses_public_locator_and_revision(self) -> None:
        base = SimpleNamespace(
            model="registry.example/org/model",
            model_weights="",
            revision="revision-a",
            served_model_name="public-alias-a",
        )

        class NewlyNamedModelConfig(SimpleNamespace):
            pass

        materialized = NewlyNamedModelConfig(
            model="/tmp/download-123/model",
            model_weights="registry.example/org/model",
            revision="revision-a",
            served_model_name="public-alias-b",
        )
        self.assertEqual(
            model_namespace_sha256(base),
            model_namespace_sha256(materialized),
        )
        self.assertNotEqual(
            model_namespace_sha256(base),
            model_namespace_sha256(
                SimpleNamespace(
                    model="registry.example/org/other-model",
                    model_weights="",
                    revision="revision-a",
                )
            ),
        )
        self.assertNotEqual(
            model_namespace_sha256(base),
            model_namespace_sha256(
                SimpleNamespace(
                    model="registry.example/org/model",
                    model_weights="",
                    revision="revision-b",
                )
            ),
        )

    def test_model_namespace_rejects_invalid_public_fields(self) -> None:
        for model_config in (
            SimpleNamespace(model="", model_weights="", revision=None),
            SimpleNamespace(model=True, model_weights="", revision=None),
            SimpleNamespace(model="model", model_weights=True, revision=None),
            SimpleNamespace(model="model", model_weights="", revision=True),
        ):
            with self.subTest(model_config=model_config):
                with self.assertRaises(IdentityError):
                    model_namespace_sha256(model_config)

    def test_rank_identity_rejects_non_integer_rank(self) -> None:
        with self.assertRaises(IdentityError):
            RankIdentity("a" * 64, "0", "b" * 64, "c" * 64)  # type: ignore[arg-type]

    def test_prefix_digest_is_exact_namespaced_and_incremental(self) -> None:
        tokens = list(range(768))
        first = prefix_digests(
            tokens,
            deployment_digest="d" * 64,
            namespace="tenant-a",
            cache_salt="salt",
            chunk_tokens=256,
        )
        selected = prefix_digests(
            tokens,
            deployment_digest="d" * 64,
            namespace="tenant-a",
            cache_salt="salt",
            chunk_tokens=256,
            boundaries=(256, 768),
        )
        self.assertEqual(selected, (first[0], first[2]))
        other = prefix_digests(
            tokens,
            deployment_digest="d" * 64,
            namespace="tenant-a",
            cache_salt="other",
            chunk_tokens=256,
        )
        self.assertNotEqual(first[0].digest, other[0].digest)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            prefix_digests(
                [*range(256), "bad"],  # type: ignore[list-item]
                deployment_digest="d" * 64,
                namespace="tenant-a",
            )

    def test_alignment_leaves_one_token_for_forward(self) -> None:
        self.assertEqual(
            aligned_prefix_span(
                1025,
                alignment=256,
                chunk_tokens=256,
                min_span_tokens=256,
            ),
            1024,
        )
        self.assertEqual(
            aligned_prefix_span(
                1024,
                alignment=256,
                chunk_tokens=256,
                min_span_tokens=256,
            ),
            768,
        )

    def test_store_boundary_can_cover_an_entire_aligned_producer(self) -> None:
        # Restore evaluates prompt_tokens and leaves one token. A producer
        # store evaluates prompt_tokens + 1, allowing its exact current state
        # to serve a longer consumer prompt.
        self.assertEqual(
            aligned_prefix_span(
                12800 + 1,
                alignment=1600,
                chunk_tokens=256,
                min_span_tokens=1024,
            ),
            12800,
        )
        self.assertEqual(
            aligned_prefix_span(
                12800,
                alignment=1600,
                chunk_tokens=256,
                min_span_tokens=1024,
            ),
            6400,
        )

    def test_multimodal_prefix_binds_content_and_placeholder_geometry(self) -> None:
        tokens = list(range(768))
        common = {
            "deployment_digest": "e" * 64,
            "namespace": "vision",
            "chunk_tokens": 256,
            "boundaries": (256, 512, 768),
        }
        red = MultimodalFeatureIdentity("image", "sha256:red", 300, 128)
        blue = MultimodalFeatureIdentity("image", "sha256:blue", 300, 128)
        shifted = MultimodalFeatureIdentity("image", "sha256:red", 301, 128)
        video = MultimodalFeatureIdentity("video", "sha256:red", 300, 128)
        text = prefix_digests(tokens, **common)
        red_keys = prefix_digests(tokens, multimodal_features=(red,), **common)
        blue_keys = prefix_digests(tokens, multimodal_features=(blue,), **common)
        shifted_keys = prefix_digests(tokens, multimodal_features=(shifted,), **common)
        video_keys = prefix_digests(tokens, multimodal_features=(video,), **common)
        red_again = prefix_digests(tokens, multimodal_features=(red,), **common)
        self.assertEqual(red_keys[0], text[0])
        self.assertNotEqual(red_keys[1].digest, text[1].digest)
        self.assertNotEqual(red_keys[1].digest, blue_keys[1].digest)
        self.assertNotEqual(red_keys[1].digest, shifted_keys[1].digest)
        self.assertNotEqual(red_keys[1].digest, video_keys[1].digest)
        self.assertEqual(red_keys, red_again)

    def test_multimodal_prefix_rejects_unprovable_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prefix_digests(
                list(range(256)),
                deployment_digest="f" * 64,
                namespace="vision",
                multimodal_features=(
                    MultimodalFeatureIdentity("image", "sha256:image", 200, 80),
                ),
            )


if __name__ == "__main__":
    unittest.main()
