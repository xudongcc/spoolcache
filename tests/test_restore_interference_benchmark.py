from __future__ import annotations

import math

from benchmarks.bench_restore_interference import (
    gap_summary,
    kv_capacity_tokens,
    metric_value,
    overlapping_gaps,
    parse_labels,
    percentile,
    prometheus_samples,
    validated_stream_usage,
)


def test_prometheus_parser_handles_vllm_names_and_labels() -> None:
    text = """
# HELP vllm:external_prefix_cache_hits_total external hits
vllm:external_prefix_cache_hits_total{engine="0",model_name="model-a"} 31744.0
vllm:cache_config_info{engine="0",kv_cache_size_tokens="1467720"} 1.0
"""
    assert metric_value(text, "vllm:external_prefix_cache_hits_total") == 31744
    assert kv_capacity_tokens(text) == 1_467_720
    assert prometheus_samples(text, "does_not_exist") == []


def test_parse_labels_accepts_escaped_values() -> None:
    assert parse_labels(r'engine="0",model_name="a\"b"') == {
        "engine": "0",
        "model_name": 'a"b',
    }


def test_percentile_and_gap_summary_are_deterministic() -> None:
    assert percentile([], 99) == 0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    summary = gap_summary([10.0, 10.1, 10.3, 10.7])
    assert summary["event_count"] == 4
    assert summary["gap_count"] == 3
    assert math.isclose(float(summary["gap_max_seconds"]), 0.4)


def test_overlap_includes_a_gap_that_straddles_injection_boundary() -> None:
    # [1.0, 2.5] starts before the load but ends during it and must be counted:
    # this is exactly what a client-visible restore stall can look like.
    gaps = overlapping_gaps([0.0, 1.0, 2.5, 3.0, 5.0], 2.0, 4.0)
    assert gaps == [1.5, 0.5, 2.0]


def test_stream_usage_requires_complete_non_coercive_evidence() -> None:
    usage = {
        "prompt_tokens": 1025,
        "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 1024},
    }
    assert validated_stream_usage(
        usage,
        expected_completion_tokens=8,
    ) == (1025, 1024, 8)
    cases = (
        ({}, "prompt token details"),
        (
            {**usage, "prompt_tokens": True},
            "invalid prompt token count",
        ),
        (
            {
                **usage,
                "prompt_tokens_details": {"cached_tokens": "1024"},
            },
            "invalid cached token count",
        ),
        (
            {
                **usage,
                "prompt_tokens_details": {"cached_tokens": 1026},
            },
            "impossible cached token usage",
        ),
        (
            {**usage, "completion_tokens": 7},
            "expected generated tokens",
        ),
    )
    for candidate, message in cases:
        try:
            validated_stream_usage(candidate, expected_completion_tokens=8)
        except RuntimeError as error:
            assert message in str(error)
        else:  # pragma: no cover - assertion failure path
            raise AssertionError(f"accepted invalid usage: {candidate!r}")
