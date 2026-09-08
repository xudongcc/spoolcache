from __future__ import annotations

import unittest

from spoolcache.vllm.config_json import build_config


class ConfigJsonTests(unittest.TestCase):
    def test_builds_valid_external_connector_config(self) -> None:
        rendered = build_config(
            {
                "SPOOLCACHE_MAX_BYTES": str(32 * 1024**3),
            }
        )
        self.assertEqual(rendered["kv_role"], "kv_both")
        self.assertEqual(
            rendered["kv_connector_module_path"], "spoolcache.vllm.connector"
        )
        extra = rendered["kv_connector_extra_config"]
        self.assertEqual(extra["spoolcache_max_bytes"], 32 * 1024**3)
        self.assertEqual(extra["spoolcache_deployment_namespace"], "default")
        self.assertEqual(
            set(extra),
            {
                "spoolcache_root",
                "spoolcache_deployment_namespace",
                "spoolcache_access_mode",
                "spoolcache_max_bytes",
                "spoolcache_direct_io",
            },
        )
        self.assertNotIn("spoolcache_profile", extra)
        self.assertNotIn("spoolcache_expected_vllm_version", extra)
        self.assertNotIn("spoolcache_qualified_gpu_mover", extra)

    def test_model_identity_needs_no_launcher_supplied_digest(self) -> None:
        rendered = build_config({})
        self.assertNotIn(
            "spoolcache_checkpoint_sha256",
            rendered["kv_connector_extra_config"],
        )

    def test_removed_environment_options_do_not_reach_connector_config(self) -> None:
        rendered = build_config(
            {
                "SPOOLCACHE_CHECKPOINT_SHA256": "a" * 64,
                "SPOOLCACHE_PROFILE": "legacy-value",
                "SPOOLCACHE_EXPECTED_VLLM_VERSION": "0.99.0",
                "SPOOLCACHE_RUNTIME_COMPATIBILITY": "strict",
                "SPOOLCACHE_CHUNK_TOKENS": "512",
                "SPOOLCACHE_MIN_SPAN_TOKENS": "2048",
                "SPOOLCACHE_MAX_SPAN_TOKENS": "4096",
                "SPOOLCACHE_SLOT_BYTES": "8388608",
                "SPOOLCACHE_SLOT_COUNT": "1",
                "SPOOLCACHE_MAX_PENDING_RESTORES": "8",
                "SPOOLCACHE_MAX_PENDING_STORES": "0",
                "SPOOLCACHE_STARTUP_MAX_DIGESTS": "64",
                "SPOOLCACHE_REPORT_BATCH_SIZE": "8",
                "SPOOLCACHE_CATALOG_MAX_ENTRIES": "1024",
                "SPOOLCACHE_LOW_WATERMARK_BYTES": "1024",
            }
        )
        extra = rendered["kv_connector_extra_config"]
        self.assertNotIn("spoolcache_checkpoint_sha256", extra)
        for removed in (
            "profile",
            "expected_vllm_version",
            "runtime_compatibility",
            "chunk_tokens",
            "min_span_tokens",
            "max_span_tokens",
            "slot_bytes",
            "slot_count",
            "max_pending_restores",
            "max_pending_stores",
            "startup_max_digests",
            "report_batch_size",
            "catalog_max_entries",
            "low_watermark_bytes",
        ):
            self.assertNotIn(f"spoolcache_{removed}", extra)


if __name__ == "__main__":
    unittest.main()
