from __future__ import annotations

import os
import unittest
from unittest import mock

from benchmarks.reset_gpu_prefix_cache import (
    MULTIMODAL_RESET_PATHS,
    RESET_QUERY,
    reset_gpu_prefix_cache,
    reset_multimodal_caches,
)


class _Response:
    status = 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class ResetGpuPrefixCacheTests(unittest.TestCase):
    def test_reset_explicitly_retains_external_cache(self) -> None:
        """The convenience tool must never depend on risky endpoint defaults."""

        with (
            mock.patch.dict(os.environ, {"VLLM_API_KEY": "test-secret"}, clear=True),
            mock.patch(
                "benchmarks.reset_gpu_prefix_cache.urllib.request.urlopen",
                return_value=_Response(),
            ) as open_url,
        ):
            self.assertEqual(reset_gpu_prefix_cache("http://127.0.0.1:8888/"), 200)

        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            f"http://127.0.0.1:8888/reset_prefix_cache?{RESET_QUERY}",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertIn("reset_running_requests=false", request.full_url)
        self.assertIn("reset_external=false", request.full_url)
        self.assertNotIn("reset_external=true", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")

    def test_multimodal_reset_clears_processor_and_encoder_caches(self) -> None:
        with mock.patch(
            "benchmarks.reset_gpu_prefix_cache.urllib.request.urlopen",
            return_value=_Response(),
        ) as open_url:
            self.assertEqual(
                reset_multimodal_caches("http://127.0.0.1:8888/"),
                (200, 200),
            )

        self.assertEqual(
            [call.args[0].full_url for call in open_url.call_args_list],
            [f"http://127.0.0.1:8888{path}" for path in MULTIMODAL_RESET_PATHS],
        )
        self.assertTrue(
            all(
                call.args[0].get_method() == "POST"
                for call in open_url.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
