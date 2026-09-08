#!/usr/bin/env python3
"""Run one deterministic streaming prefix request and report TTFT/cache use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        return json.load(response)


def tokenize(api: str, model: str, prompt: str) -> list[int]:
    result = post_json(f"{api}/tokenize", {"model": model, "prompt": prompt})
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in tokens
    ):
        raise RuntimeError("tokenize endpoint did not return integer token IDs")
    return tokens


def _usage_integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"completion stream returned invalid {label}")
    return value


def build_prompt(api: str, model: str, target: int, nonce: str) -> list[int]:
    if target <= 0:
        raise ValueError("target token count must be positive")
    header = f"SpoolCache benchmark {nonce}. Preserve this synthetic context.\n"
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa. "
    low, high = 1, max(2, target // 4)
    while len(tokenize(api, model, header + unit * high)) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(tokenize(api, model, header + unit * middle)) < target:
            low = middle + 1
        else:
            high = middle
    # The completions API accepts token IDs directly.  Truncating the synthetic
    # payload gives the benchmark an exact boundary instead of relying on a
    # tokenizer-dependent text overshoot.  That is essential when evaluating
    # stateful HMA groups whose safe snapshots exist only at exact boundaries.
    tokens = tokenize(api, model, header + unit * low)
    if len(tokens) < target:
        raise RuntimeError("failed to construct the requested token count")
    return tokens[:target]


def build_prompt_from_source(
    api: str,
    model: str,
    target: int,
    source_target: int,
    nonce: str,
) -> list[int]:
    """Take an exact prefix from one longer tokenizer-stable token stream."""

    if target <= 0 or source_target < target:
        raise ValueError("prompt source length must cover the positive target")
    source = build_prompt(api, model, source_target, nonce)
    if len(source) != source_target:
        raise RuntimeError("prompt source did not reach its requested token count")
    return source[:target]


def validate_complete_response(
    *,
    usage: dict[str, object],
    request_id: str,
    output: str,
    first_token_at: float | None,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> tuple[int, int, int]:
    """Reject a transport-successful stream truncated by engine failure."""

    if not isinstance(usage, dict):
        raise RuntimeError("completion stream returned invalid usage")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise RuntimeError(
            "completion stream returned missing or invalid prompt token details"
        )
    prompt_tokens = _usage_integer(
        usage.get("prompt_tokens"), label="prompt token count"
    )
    cached_tokens = _usage_integer(
        details.get("cached_tokens"), label="cached token count"
    )
    completion_tokens = _usage_integer(
        usage.get("completion_tokens"), label="completion token count"
    )
    if cached_tokens > prompt_tokens:
        raise RuntimeError("completion stream returned impossible cached token usage")
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            "completion stream ended without expected prompt usage: "
            f"expected={expected_prompt_tokens} actual={prompt_tokens}"
        )
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            "completion stream ended without expected generated tokens: "
            f"expected={expected_completion_tokens} actual={completion_tokens}"
        )
    if not request_id or first_token_at is None or not output:
        raise RuntimeError("completion stream ended without a non-empty output")
    return prompt_tokens, cached_tokens, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8888")
    model_default = os.environ.get("SERVED_MODEL_NAME")
    parser.add_argument(
        "--model",
        default=model_default,
        required=model_default is None,
    )
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument(
        "--prompt-source-tokens",
        type=int,
        help=(
            "build one longer deterministic token stream before slicing "
            "--target-tokens, so producer and consumer can prove an exact prefix"
        ),
    )
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed. The default makes correctness comparisons repeatable.",
    )
    parser.add_argument(
        "--bypass-spoolcache",
        action="store_true",
        help="Bypass external persistence for a correctness/control request.",
    )
    args = parser.parse_args()
    if args.target_tokens <= 0 or args.max_tokens <= 0:
        parser.error("--target-tokens and --max-tokens must be positive")

    source_tokens = (
        args.target_tokens
        if args.prompt_source_tokens is None
        else args.prompt_source_tokens
    )
    if source_tokens < args.target_tokens:
        parser.error("--prompt-source-tokens cannot be shorter than --target-tokens")
    prompt = build_prompt_from_source(
        args.api,
        args.model,
        args.target_tokens,
        source_tokens,
        args.nonce,
    )
    body = {
        "model": args.model,
        "prompt": prompt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "ignore_eos": True,
        "cache_salt": args.cache_salt,
    }
    if args.bypass_spoolcache:
        body["kv_transfer_params"] = {"spoolcache_bypass": True}
    request = urllib.request.Request(
        f"{args.api}/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers(),
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, object] = {}
    request_id = ""
    output: list[str] = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            event_id = event.get("id", request_id)
            if not isinstance(event_id, str):
                raise RuntimeError("completion stream returned an invalid request ID")
            request_id = event_id
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if not isinstance(text, str):
                raise RuntimeError("completion stream returned non-text output")
            if text and first_token_at is None:
                first_token_at = time.perf_counter()
            output.append(text)
            if event.get("usage") is not None:
                event_usage = event["usage"]
                if not isinstance(event_usage, dict):
                    raise RuntimeError("completion stream returned invalid usage")
                usage = event_usage
    finished = time.perf_counter()
    text = "".join(output)
    prompt_tokens, cached_tokens, completion_tokens = validate_complete_response(
        usage=usage,
        request_id=request_id,
        output=text,
        first_token_at=first_token_at,
        expected_prompt_tokens=args.target_tokens,
        expected_completion_tokens=args.max_tokens,
    )
    result = {
        "request_id": request_id,
        "target_tokens": args.target_tokens,
        "prompt_source_tokens": source_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": (first_token_at or finished) - started,
        "elapsed_seconds": finished - started,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_preview": text[:80],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
