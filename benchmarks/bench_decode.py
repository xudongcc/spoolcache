#!/usr/bin/env python3
"""Matched short-prompt decode benchmark for SpoolCache enabled/disabled A/B."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
import urllib.request


def headers() -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        keys = os.environ.get("DSPARK_API_KEYS", "").split()
        key = keys[0] if keys else ""
    if key:
        result["Authorization"] = f"Bearer {key}"
    return result


def post_json(url: str, body: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("endpoint did not return a JSON object")
    return result


def token_count(api: str, model: str, prompt: str) -> int:
    value = post_json(f"{api}/tokenize", {"model": model, "prompt": prompt})
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenize endpoint returned an invalid token count")
    return count


def _validated_decode_usage(
    usage: object,
    *,
    expected_completion_tokens: int,
) -> tuple[int, int]:
    if not isinstance(usage, dict):
        raise RuntimeError("decode stream returned missing or invalid usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    for label, value in (
        ("prompt token count", prompt_tokens),
        ("completion token count", completion_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"decode stream returned invalid {label}")
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            "decode stream ended without expected generated tokens: "
            f"expected={expected_completion_tokens} actual={completion_tokens}"
        )
    return prompt_tokens, completion_tokens


def build_prompt(api: str, model: str, target: int, nonce: str) -> str:
    header = f"Decode benchmark {nonce}.\n"
    unit = "alpha beta gamma delta epsilon zeta eta theta. "
    low, high = 1, max(2, target // 4)
    while token_count(api, model, header + unit * high) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if token_count(api, model, header + unit * middle) < target:
            low = middle + 1
        else:
            high = middle
    return header + unit * low + "\nReturn numbered lowercase English words."


def stream_one(
    api: str,
    model: str,
    prompt: str,
    salt: str,
    seed: int,
    output_tokens: int,
    barrier: threading.Barrier,
) -> dict[str, float | int]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "seed": seed,
        "cache_salt": salt,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        f"{api}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers(),
        method="POST",
    )
    barrier.wait()
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, object] = {}
    request_id = ""
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            event_id = event.get("id", request_id)
            if not isinstance(event_id, str):
                raise RuntimeError("decode stream returned an invalid request ID")
            request_id = event_id
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if not isinstance(delta, dict):
                raise RuntimeError("decode stream returned an invalid output delta")
            observable = (
                delta.get("content"),
                delta.get("reasoning"),
                delta.get("reasoning_content"),
            )
            if any(value is not None and not isinstance(value, str) for value in observable):
                raise RuntimeError("decode stream returned non-text output")
            if first_token_at is None and any(observable):
                first_token_at = time.perf_counter()
            if event.get("usage") is not None:
                event_usage = event["usage"]
                if not isinstance(event_usage, dict):
                    raise RuntimeError("decode stream returned invalid usage")
                usage = event_usage
    finished = time.perf_counter()
    prompt_tokens, completion = _validated_decode_usage(
        usage,
        expected_completion_tokens=output_tokens,
    )
    if not request_id or first_token_at is None:
        raise RuntimeError("decode stream ended without a non-empty output")
    first = first_token_at
    return {
        "ttft_seconds": first - started,
        "elapsed_seconds": finished - started,
        "decode_seconds": finished - first,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "decode_tokens_s": completion / max(0.001, finished - first),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8888")
    model_default = os.environ.get("SERVED_MODEL_NAME")
    parser.add_argument(
        "--model",
        default=model_default,
        required=model_default is None,
    )
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    if (
        args.prompt_tokens <= 0
        or args.output_tokens <= 0
        or args.concurrency <= 0
        or args.repetitions <= 0
    ):
        parser.error("token counts, concurrency, and repetitions must be positive")

    trials: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        prompts = [
            build_prompt(
                args.api,
                args.model,
                args.prompt_tokens,
                f"{args.nonce}-trial{repetition}-lane{lane}",
            )
            for lane in range(args.concurrency)
        ]
        barrier = threading.Barrier(args.concurrency)
        wave_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    stream_one,
                    args.api,
                    args.model,
                    prompt,
                    f"{args.nonce}-trial{repetition}-lane{lane}",
                    20_000 + repetition * 100 + lane,
                    args.output_tokens,
                    barrier,
                )
                for lane, prompt in enumerate(prompts)
            ]
            requests = [future.result() for future in futures]
        wave_seconds = time.perf_counter() - wave_started
        total_output = sum(item["completion_tokens"] for item in requests)
        trial = {
            "repetition": repetition,
            "wave_seconds": wave_seconds,
            "aggregate_tokens_s": total_output / wave_seconds,
            "median_decode_tokens_s": statistics.median(
                float(item["decode_tokens_s"]) for item in requests
            ),
            "median_ttft_seconds": statistics.median(
                float(item["ttft_seconds"]) for item in requests
            ),
            "requests": requests,
        }
        trials.append(trial)
        print(json.dumps({"trial": trial}, sort_keys=True), flush=True)

    summary = {
        "prompt_tokens_target": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "repetitions": args.repetitions,
        "median_of_trial_decode_tokens_s": statistics.median(
            float(item["median_decode_tokens_s"]) for item in trials
        ),
        "median_aggregate_tokens_s": statistics.median(
            float(item["aggregate_tokens_s"]) for item in trials
        ),
        "median_ttft_seconds": statistics.median(
            float(item["median_ttft_seconds"]) for item in trials
        ),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
