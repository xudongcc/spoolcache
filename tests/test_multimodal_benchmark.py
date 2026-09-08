import tempfile
import types
import unittest
from importlib import util
from pathlib import Path
from unittest import mock

from benchmarks.bench_multimodal_prefix_e2e import (
    _calibrate_exact_prompt,
    _data_url,
    _media_part,
    _normalized_output,
    _request_body,
    _reset_local_caches,
    _validate_content_oracle,
    _validated_usage,
)


class MultimodalBenchmarkTests(unittest.TestCase):
    def test_exact_calibration_can_use_padding_without_a_full_unit(self) -> None:
        def body(repetitions: int, padding: str) -> dict[str, int]:
            return {"tokens": 8 + repetitions * 3 + len(padding)}

        with mock.patch(
            "benchmarks.bench_multimodal_prefix_e2e._token_ids",
            side_effect=lambda _api, value: list(range(value["tokens"])),
        ):
            repetitions, padding, tokens = _calibrate_exact_prompt(
                "http://vllm", body, 10
            )
        self.assertEqual(repetitions, 0)
        self.assertEqual(padding, " x")
        self.assertEqual(len(tokens), 10)

    def test_local_cache_reset_works_when_imported_as_package(self) -> None:
        with (
            mock.patch(
                "benchmarks.reset_gpu_prefix_cache.reset_gpu_prefix_cache"
            ) as reset_prefix,
            mock.patch(
                "benchmarks.reset_gpu_prefix_cache.reset_multimodal_caches"
            ) as reset_multimodal,
        ):
            _reset_local_caches("http://vllm")
        reset_prefix.assert_called_once_with("http://vllm")
        reset_multimodal.assert_called_once_with("http://vllm")

    def test_local_cache_reset_works_when_executed_as_script(self) -> None:
        benchmark_path = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "bench_multimodal_prefix_e2e.py"
        )
        spec = util.spec_from_file_location("bench_multimodal_direct", benchmark_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__package__ = ""
        reset_prefix = mock.Mock()
        reset_multimodal = mock.Mock()
        reset_module = types.ModuleType("reset_gpu_prefix_cache")
        reset_module.reset_gpu_prefix_cache = reset_prefix
        reset_module.reset_multimodal_caches = reset_multimodal
        with mock.patch.dict(
            "sys.modules", {"reset_gpu_prefix_cache": reset_module}
        ):
            module._reset_local_caches("http://vllm")
        reset_prefix.assert_called_once_with("http://vllm")
        reset_multimodal.assert_called_once_with("http://vllm")

    def test_local_fixture_is_embedded_without_path_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            path.write_bytes(b"not-a-real-video-but-nonempty")
            url, mime, raw = _data_url("video", path)
        self.assertEqual(mime, "video/mp4")
        self.assertEqual(raw, b"not-a-real-video-but-nonempty")
        self.assertTrue(url.startswith("data:video/mp4;base64,"))
        part = _media_part("video", url, mime)
        self.assertEqual(part["type"], "video_url")
        self.assertNotIn(str(path), str(part))

    def test_skip_write_is_exact_boolean_and_opt_in(self) -> None:
        arguments = dict(
            model="model",
            media_part={"type": "image_url", "image_url": {"url": "data:x"}},
            nonce="nonce",
            repetitions=1,
            padding="",
            labels="A, B",
            cache_salt="salt",
            phase="consumer",
        )
        normal = _request_body(**arguments, skip_write=False)
        skip_write = _request_body(**arguments, skip_write=True)
        self.assertNotIn("kv_transfer_params", normal)
        for skip_read_flag, skip_write_flag in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(skip_read=skip_read_flag, skip_write=skip_write_flag):
                body = _request_body(**arguments, skip_read=skip_read_flag, skip_write=skip_write_flag)
                expected = {}
                if skip_read_flag:
                    expected["spoolcache.skip_read"] = True
                if skip_write_flag:
                    expected["spoolcache.skip_write"] = True
                self.assertEqual(body.get("kv_transfer_params", {}), expected)
        self.assertIs(skip_write["kv_transfer_params"]["spoolcache.skip_write"], True)
        self.assertEqual(normal["structured_outputs"], {"choice": ["A", "B"]})
        self.assertEqual(
            normal["chat_template_kwargs"],
            {"thinking": False, "enable_thinking": False},
        )

    def test_producer_turn_is_a_structural_prefix_of_consumer(self) -> None:
        common = dict(
            model="model",
            media_part={"type": "image_url", "image_url": {"url": "data:x"}},
            nonce="nonce",
            repetitions=1,
            padding=" x",
            labels="A, B",
            cache_salt="salt",
            skip_write=False,
        )
        producer = _request_body(**common, phase="producer")
        consumer = _request_body(**common, phase="consumer")
        self.assertEqual(producer["messages"], consumer["messages"][:1])
        self.assertFalse(producer["add_generation_prompt"])
        self.assertTrue(consumer["add_generation_prompt"])

    def test_shared_prefix_accepts_multiple_media_parts_in_order(self) -> None:
        media_parts = [
            {"type": "image_url", "image_url": {"url": "data:image"}},
            {"type": "input_audio", "input_audio": {"data": "audio"}},
            {"type": "video_url", "video_url": {"url": "data:video"}},
        ]
        common = dict(
            model="model",
            media_part=media_parts,
            nonce="nonce",
            repetitions=1,
            padding="",
            labels="ALL, OTHER",
            cache_salt="salt",
            skip_write=False,
        )
        producer = _request_body(**common, phase="producer")
        consumer = _request_body(**common, phase="consumer")
        self.assertEqual(
            producer["messages"][0]["content"][:-1],
            media_parts,
        )
        self.assertEqual(producer["messages"], consumer["messages"][:1])

    def test_consumer_extension_does_not_change_producer_turn(self) -> None:
        common = dict(
            model="model",
            media_part={"type": "image_url", "image_url": {"url": "data:x"}},
            nonce="nonce",
            repetitions=1,
            padding=" x",
            labels="A, B",
            cache_salt="salt",
            skip_write=False,
        )
        producer = _request_body(**common, phase="producer")
        consumer = _request_body(
            **common,
            phase="consumer",
            consumer_extension="extended synthetic context",
        )
        self.assertEqual(producer["messages"], consumer["messages"][:1])
        self.assertTrue(
            consumer["messages"][-1]["content"].startswith(
                "extended synthetic context\n"
            )
        )

    def test_output_digest_input_includes_reasoning_and_finish_state(self) -> None:
        normalized = _normalized_output(
            {
                "choices": [
                    {
                        "message": {"content": "CATS", "reasoning_content": "x"},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        self.assertEqual(
            normalized,
            {
                "content": "CATS",
                "reasoning": None,
                "reasoning_content": "x",
                "tool_calls": None,
                "finish_reason": "length",
            },
        )

    def test_content_oracle_rejects_empty_reasoning_envelope(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "constrained labels"):
            _validate_content_oracle(
                {
                    "content": None,
                    "reasoning": "still thinking",
                    "reasoning_content": None,
                    "tool_calls": None,
                    "finish_reason": "length",
                },
                "CATS, OTHER",
            )

    def test_content_oracle_accepts_exact_choice(self) -> None:
        self.assertEqual(
            _validate_content_oracle({"content": "CATS"}, "CATS, OTHER"),
            "CATS",
        )

    def test_usage_receipt_requires_exact_authenticated_counts(self) -> None:
        result = {
            "id": "chatcmpl-test",
            "usage": {
                "prompt_tokens": 6400,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 6400},
            },
        }
        self.assertEqual(
            _validated_usage(result, expected_prompt_tokens=6400),
            (6400, 6400, 1),
        )

    def test_usage_receipt_rejects_missing_or_ambiguous_evidence(self) -> None:
        cases = (
            ({"id": "x"}, "missing or invalid usage"),
            (
                {
                    "id": "x",
                    "usage": {"prompt_tokens": 6400, "completion_tokens": 1},
                },
                "prompt token details",
            ),
            (
                {
                    "id": "x",
                    "usage": {
                        "prompt_tokens": True,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 6400},
                    },
                },
                "invalid prompt token count",
            ),
            (
                {
                    "id": "x",
                    "usage": {
                        "prompt_tokens": 6400,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": "6400"},
                    },
                },
                "invalid cached token count",
            ),
            (
                {
                    "id": "x",
                    "usage": {
                        "prompt_tokens": 6401,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 6400},
                    },
                },
                "prompt usage differs",
            ),
            (
                {
                    "id": "x",
                    "usage": {
                        "prompt_tokens": 6400,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 6401},
                    },
                },
                "impossible cached token usage",
            ),
            (
                {
                    "id": "",
                    "usage": {
                        "prompt_tokens": 6400,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 6400},
                    },
                },
                "invalid request ID",
            ),
        )
        for result, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    _validated_usage(result, expected_prompt_tokens=6400)


if __name__ == "__main__":
    unittest.main()
