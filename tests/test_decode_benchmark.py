from __future__ import annotations

import unittest

from benchmarks.bench_decode import _validated_decode_usage


class DecodeBenchmarkTests(unittest.TestCase):
    def test_decode_usage_requires_exact_non_coercive_counts(self) -> None:
        usage = {"prompt_tokens": 256, "completion_tokens": 128}
        self.assertEqual(
            _validated_decode_usage(
                usage,
                expected_completion_tokens=128,
            ),
            (256, 128),
        )
        cases = (
            ({}, "invalid prompt token count"),
            ({**usage, "prompt_tokens": True}, "invalid prompt token count"),
            (
                {**usage, "completion_tokens": "128"},
                "invalid completion token count",
            ),
            ({**usage, "completion_tokens": 127}, "expected generated tokens"),
        )
        for candidate, message in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(RuntimeError, message):
                    _validated_decode_usage(
                        candidate,
                        expected_completion_tokens=128,
                    )


if __name__ == "__main__":
    unittest.main()
