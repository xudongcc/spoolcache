from __future__ import annotations

import unittest
from unittest import mock

from benchmarks.bench_prefix_e2e import (
    build_prompt_from_source,
    validate_complete_response,
)


class PrefixBenchmarkTests(unittest.TestCase):
    def test_shared_longer_source_produces_an_exact_shorter_prefix(self) -> None:
        source = [1, 2, 3, 4, 5]
        with mock.patch(
            "benchmarks.bench_prefix_e2e.build_prompt",
            return_value=source,
        ) as build:
            result = build_prompt_from_source("api", "model", 3, 5, "nonce")
        self.assertEqual(result, [1, 2, 3])
        build.assert_called_once_with("api", "model", 5, "nonce")

    def test_shared_source_rejects_invalid_ranges_before_tokenizing(self) -> None:
        for target, source_target in ((0, 5), (4, 3)):
            with self.subTest(target=target, source_target=source_target):
                with mock.patch(
                    "benchmarks.bench_prefix_e2e.build_prompt"
                ) as build:
                    with self.assertRaisesRegex(ValueError, "must cover"):
                        build_prompt_from_source(
                            "api", "model", target, source_target, "nonce"
                        )
                build.assert_not_called()

    def test_complete_stream_returns_authenticated_counts(self) -> None:
        self.assertEqual(
            validate_complete_response(
                usage={
                    "prompt_tokens": 257,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 256},
                },
                request_id="cmpl-test",
                output="answer",
                first_token_at=1.0,
                expected_prompt_tokens=257,
                expected_completion_tokens=8,
            ),
            (257, 256, 8),
        )

    def test_http_200_stream_cut_off_by_engine_death_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "ended without expected prompt usage",
        ):
            validate_complete_response(
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                request_id="",
                output="",
                first_token_at=None,
                expected_prompt_tokens=257,
                expected_completion_tokens=8,
            )

    def test_missing_prompt_token_details_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
            validate_complete_response(
                usage={"prompt_tokens": 257, "completion_tokens": 8},
                request_id="cmpl-test",
                output="answer",
                first_token_at=1.0,
                expected_prompt_tokens=257,
                expected_completion_tokens=8,
            )

    def test_boolean_and_string_usage_counts_are_rejected(self) -> None:
        for field, value in (
            ("prompt_tokens", True),
            ("completion_tokens", "8"),
            ("cached_tokens", False),
        ):
            with self.subTest(field=field, value=value):
                usage: dict[str, object] = {
                    "prompt_tokens": 257,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 256},
                }
                if field == "cached_tokens":
                    usage["prompt_tokens_details"] = {field: value}
                else:
                    usage[field] = value
                with self.assertRaisesRegex(RuntimeError, "invalid .* count"):
                    validate_complete_response(
                        usage=usage,
                        request_id="cmpl-test",
                        output="answer",
                        first_token_at=1.0,
                        expected_prompt_tokens=257,
                        expected_completion_tokens=8,
                    )

    def test_cached_usage_cannot_exceed_prompt_usage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "impossible cached token usage"):
            validate_complete_response(
                usage={
                    "prompt_tokens": 257,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 258},
                },
                request_id="cmpl-test",
                output="answer",
                first_token_at=1.0,
                expected_prompt_tokens=257,
                expected_completion_tokens=8,
            )


if __name__ == "__main__":
    unittest.main()
