from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from spoolcache.errors import ConfigurationError
from spoolcache.vllm.config_json import build_config


class ConfigJsonTests(unittest.TestCase):
    def test_default_path_uses_the_process_users_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"HOME": directory}):
                rendered = build_config({})
            expected = Path(directory) / ".cache" / "spoolcache"
            self.assertEqual(
                rendered["kv_connector_extra_config"]["spoolcache_path"],
                str(expected),
            )
            self.assertFalse(expected.exists())

    def test_path_override_expands_tilde_and_preserves_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"HOME": directory}):
                for raw, expected in (
                    ("~/custom cache", str(Path(directory) / "custom cache")),
                    ("/mnt/nvme/spoolcache", "/mnt/nvme/spoolcache"),
                ):
                    with self.subTest(raw=raw):
                        rendered = build_config({"SPOOLCACHE_PATH": raw})
                        self.assertEqual(
                            rendered["kv_connector_extra_config"]["spoolcache_path"],
                            expected,
                        )

    def test_invalid_path_override_is_rejected(self) -> None:
        for path in ("", "relative/cache", "/"):
            with self.subTest(path=path):
                with self.assertRaises(ConfigurationError):
                    build_config({"SPOOLCACHE_PATH": path})

    def test_builds_valid_external_connector_config(self) -> None:
        rendered = build_config(
            {
                "SPOOLCACHE_MAX_SIZE": "32.5",
            }
        )
        self.assertEqual(rendered["kv_role"], "kv_both")
        self.assertEqual(
            rendered["kv_connector_module_path"], "spoolcache.vllm.connector"
        )
        extra = rendered["kv_connector_extra_config"]
        self.assertEqual(extra["spoolcache_max_size"], 32.5)
        self.assertEqual(
            set(extra),
            {
                "spoolcache_path",
                "spoolcache_max_size",
            },
        )
        self.assertNotIn("spoolcache_profile", extra)
        self.assertNotIn("spoolcache_expected_vllm_version", extra)
        self.assertNotIn("spoolcache_qualified_gpu_mover", extra)

    def test_capacity_defaults_and_invalid_environment_values(self) -> None:
        extra = build_config({})["kv_connector_extra_config"]
        self.assertEqual(extra["spoolcache_max_size"], 200)
        for value in ("", "garbage", "true", "nan", "inf", "-inf", "0", "-1", "1e-9"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_config({"SPOOLCACHE_MAX_SIZE": value})

    def test_removed_access_mode_cannot_change_rendered_config(self) -> None:
        expected = build_config({})
        for value in ("disabled", "store-only", "restore-only", "read-write"):
            with self.subTest(value=value):
                self.assertEqual(build_config({"SPOOLCACHE_ACCESS_MODE": value}), expected)
        self.assertNotIn("spoolcache_access_mode", expected["kv_connector_extra_config"])

    def test_removed_namespace_cannot_change_rendered_config(self) -> None:
        expected = build_config({})
        for value in ("tenant-a", "tenant-b", ""):
            with self.subTest(value=value):
                self.assertEqual(build_config({"SPOOLCACHE_NAMESPACE": value}), expected)

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
                "SPOOLCACHE_DIRECT_IO": "disabled",
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
            "direct_io",
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
