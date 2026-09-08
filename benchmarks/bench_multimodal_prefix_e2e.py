#!/usr/bin/env python3
"""Run one deterministic multimodal prefix request and emit a safe receipt.

Each fixture is sent as a local data URL, so repeated qualification phases use
identical bytes and do not depend on a mutable remote URL.  Repeat
``--media-kind`` and ``--media-file`` in matching order to qualify a mixed
prompt.  ``--target-span`` uses vLLM's public ``/tokenize`` endpoint to choose
the smallest synthetic prompt whose token count reaches the requested HMA
boundary.  The same command can then be used for producer, bypass-control,
restore, and post-restart runs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


PROMPT_UNIT = "alpha beta gamma delta epsilon zeta eta theta iota kappa. "


def _headers() -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        raw_keys = os.environ.get("DSPARK_API_KEYS", "")
        fields = raw_keys.replace(",", " ").split()
        key = fields[0] if fields else ""
    if key:
        result["Authorization"] = f"Bearer {key}"
    return result


def _post_json(api: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api.rstrip('/')}{path}",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{path} did not return a JSON object")
    return result


def _data_url(kind: str, path: Path) -> tuple[str, str, bytes]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"media fixture is empty: {path}")
    guessed, _ = mimetypes.guess_type(path.name)
    fallbacks = {
        "image": "image/jpeg",
        "video": "video/mp4",
        "audio": "audio/wav",
    }
    mime = guessed or fallbacks[kind]
    if not mime.startswith(f"{kind}/"):
        raise ValueError(f"{path} has MIME type {mime}, not {kind}")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}", mime, raw


def _media_part(kind: str, data_url: str, mime: str) -> dict[str, Any]:
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": data_url}}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": data_url}}
    # OpenAI-compatible audio content carries raw base64 and the container
    # format separately rather than nesting a data URL under ``audio_url``.
    encoded = data_url.split(",", 1)[1]
    subtype = mime.split("/", 1)[1].split("+", 1)[0]
    return {
        "type": "input_audio",
        "input_audio": {"data": encoded, "format": subtype},
    }


def _prompt(nonce: str, repetitions: int, padding: str) -> str:
    return (
        f"SpoolCache multimodal persistence qualification {nonce}. "
        + PROMPT_UNIT * repetitions
        + padding
        + "\nEnd of shared persistent context."
    )


def _request_body(
    *,
    model: str,
    media_part: dict[str, Any] | Sequence[dict[str, Any]],
    nonce: str,
    repetitions: int,
    padding: str,
    labels: str,
    cache_salt: str,
    bypass: bool,
    phase: str,
    consumer_extension: str = "",
    max_tokens: int = 128,
) -> dict[str, Any]:
    choices = [label.strip() for label in labels.split(",") if label.strip()]
    if not choices or len(choices) != len(set(choices)):
        raise ValueError("--labels must contain distinct, non-empty comma-separated choices")
    media_parts = (
        [media_part]
        if isinstance(media_part, dict)
        else list(media_part)
    )
    if not media_parts or any(not isinstance(part, dict) for part in media_parts):
        raise ValueError("at least one valid media part is required")
    shared_message = {
        "role": "user",
        "content": [
            *media_parts,
            {
                "type": "text",
                "text": _prompt(nonce, repetitions, padding),
            },
        ],
    }
    if phase == "producer":
        messages = [shared_message]
    elif phase == "consumer":
        extension = f"{consumer_extension}\n" if consumer_extension else ""
        messages = [
            shared_message,
            {"role": "assistant", "content": "Shared context acknowledged."},
            {
                "role": "user",
                "content": (
                    extension
                    + "Inspect every media item from the shared context. Respond with "
                    f"exactly one of these labels: {', '.join(choices)}."
                ),
            },
        ]
    else:
        raise ValueError(f"unsupported qualification phase: {phase}")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        # The producer stops exactly after the first complete user turn.  The
        # consumer appends two turns, so the producer token sequence is an
        # exact prefix rather than a similar string with a different suffix.
        "add_generation_prompt": phase == "consumer",
        # Reasoning parsers may emit an internal reasoning field before the
        # constrained answer.  Eight forced tokens can therefore produce a
        # transport-successful response with no observable answer at all.
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "cache_salt": cache_salt,
        # Public chat templates in current runtimes use either spelling.  Both
        # are harmless template kwargs and avoid model-name branching in this
        # generic qualification client.
        "chat_template_kwargs": {
            "thinking": False,
            "enable_thinking": False,
        },
        # A byte-for-byte digest is only a content oracle when the response
        # contains observable model output.  Constraining generation to one
        # explicit label avoids accepting an empty reasoning-model envelope.
        "structured_outputs": {"choice": choices},
    }
    if bypass:
        body["kv_transfer_params"] = {"spoolcache_bypass": True}
    return body


def _token_ids(api: str, body: dict[str, Any]) -> list[int]:
    tokenize_body = {
        key: body[key]
        for key in (
            "model",
            "messages",
            "chat_template_kwargs",
            "add_generation_prompt",
        )
    }
    result = _post_json(api, "/tokenize", tokenize_body)
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in tokens
    ):
        raise RuntimeError("/tokenize did not return integer token IDs")
    return tokens


def _calibrate_exact_prompt(
    api: str,
    body_for_repetitions: Any,
    target_span: int,
) -> tuple[int, str, list[int]]:
    low, high = 0, 1
    while len(_token_ids(api, body_for_repetitions(high, ""))) < target_span:
        low = high + 1
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(_token_ids(api, body_for_repetitions(middle, ""))) < target_span:
            low = middle + 1
        else:
            high = middle
    exact = _token_ids(api, body_for_repetitions(low, ""))
    if len(exact) == target_span:
        return low, "", exact

    # One repetition below the crossing is close to the boundary.  Try several
    # tokenizer-neutral padding alphabets; the selected bytes are reported and
    # reused verbatim by producer and consumer.  Failure is closed rather than
    # silently publishing a shorter stateful snapshot.
    repetitions = max(0, low - 1)
    for unit in (" x", " a", " 1", ".", "\n"):
        for count in range(129):
            padding = unit * count
            token_ids = _token_ids(
                api, body_for_repetitions(repetitions, padding)
            )
            if len(token_ids) == target_span:
                return repetitions, padding, token_ids
            if len(token_ids) > target_span + 8:
                break
    raise RuntimeError(
        f"cannot calibrate producer prompt to exact {target_span}-token boundary"
    )


def _reset_local_caches(api: str) -> None:
    # Keep both supported invocation forms working: package imports used by
    # tests and direct ``python benchmarks/<script>.py`` runs used in the lab.
    if __package__:
        from .reset_gpu_prefix_cache import (  # noqa: PLC0415
            reset_gpu_prefix_cache,
            reset_multimodal_caches,
        )
    else:
        from reset_gpu_prefix_cache import (  # noqa: PLC0415
            reset_gpu_prefix_cache,
            reset_multimodal_caches,
        )

    reset_gpu_prefix_cache(api)
    reset_multimodal_caches(api)


def _normalized_output(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices") or []
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        message = {}
    return {
        "content": message.get("content"),
        "reasoning": message.get("reasoning"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": message.get("tool_calls"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }


def _validate_content_oracle(
    normalized: dict[str, Any], labels: str
) -> str:
    choices = [label.strip() for label in labels.split(",") if label.strip()]
    content = normalized.get("content")
    if not isinstance(content, str) or content not in choices:
        raise RuntimeError(
            "multimodal response did not contain one of the constrained labels: "
            f"content={content!r} choices={choices!r}"
        )
    return content


def _usage_integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"multimodal response returned invalid {label}")
    return value


def _validated_usage(
    result: dict[str, Any],
    *,
    expected_prompt_tokens: int,
) -> tuple[int, int, int]:
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("multimodal response returned missing or invalid usage")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise RuntimeError(
            "multimodal response returned missing or invalid prompt token details"
        )
    prompt_tokens = _usage_integer(
        usage.get("prompt_tokens"), label="prompt token count"
    )
    cached_tokens = _usage_integer(
        details.get("cached_tokens"), label="cached token count"
    )
    completion_tokens = _usage_integer(
        usage.get("completion_tokens"),
        label="completion token count",
        minimum=1,
    )
    if cached_tokens > prompt_tokens:
        raise RuntimeError("multimodal response returned impossible cached token usage")
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            "multimodal response prompt usage differs: "
            f"expected={expected_prompt_tokens} actual={prompt_tokens}"
        )
    request_id = result.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("multimodal response returned an invalid request ID")
    return prompt_tokens, cached_tokens, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8888")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--media-kind",
        choices=("image", "video", "audio"),
        action="append",
        required=True,
        help="media modality; repeat with --media-file for a mixed prompt",
    )
    parser.add_argument(
        "--media-file",
        type=Path,
        action="append",
        required=True,
        help="local media fixture; repeat in the same order as --media-kind",
    )
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--phase", choices=("producer", "consumer"), required=True)
    parser.add_argument("--labels", default="CATS, ARCHERY, STREET, KITCHEN, OTHER")
    parser.add_argument("--target-span", type=int, required=True)
    parser.add_argument(
        "--consumer-target-tokens",
        type=int,
        help=(
            "calibrate an extended consumer to this exact total prompt length; "
            "the producer remains --target-span tokens"
        ),
    )
    parser.add_argument("--alignment", type=int, required=True)
    parser.add_argument("--bypass-spoolcache", action="store_true")
    parser.add_argument("--reset-local", action="store_true")
    parser.add_argument("--dry-tokenize", action="store_true")
    parser.add_argument("--expect-cached-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.target_span <= 0 or args.alignment <= 0 or args.max_tokens <= 0:
        parser.error("--target-span, --alignment and --max-tokens must be positive")
    if len(args.media_kind) != len(args.media_file):
        parser.error("--media-kind and --media-file counts must match")
    if args.target_span % args.alignment:
        parser.error("--target-span must be aligned to --alignment")
    if args.consumer_target_tokens is not None:
        if args.phase != "consumer":
            parser.error("--consumer-target-tokens is valid only for consumers")
        if args.consumer_target_tokens <= args.target_span:
            parser.error("--consumer-target-tokens must exceed --target-span")

    parts: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    for kind, path in zip(args.media_kind, args.media_file, strict=True):
        data_url, mime, media = _data_url(kind, path)
        parts.append(_media_part(kind, data_url, mime))
        fixtures.append(
            {
                "kind": kind,
                "mime": mime,
                "bytes": len(media),
                "sha256": hashlib.sha256(media).hexdigest(),
            }
        )

    def body_for_repetitions(
        repetitions: int,
        padding: str,
        *,
        phase: str = "producer",
        consumer_extension: str = "",
    ) -> dict[str, Any]:
        return _request_body(
            model=args.model,
            media_part=parts,
            nonce=args.nonce,
            repetitions=repetitions,
            padding=padding,
            labels=args.labels,
            cache_salt=args.cache_salt,
            bypass=args.bypass_spoolcache,
            phase=phase,
            consumer_extension=consumer_extension,
            max_tokens=args.max_tokens,
        )

    repetitions, padding, producer_token_ids = _calibrate_exact_prompt(
        args.api, body_for_repetitions, args.target_span
    )
    consumer_extension_repetitions = 0
    consumer_extension_padding = ""
    if args.consumer_target_tokens is None:
        consumer_body = body_for_repetitions(
            repetitions, padding, phase="consumer"
        )
        consumer_token_ids = _token_ids(args.api, consumer_body)
    else:
        def body_for_consumer_extension(
            extension_repetitions: int,
            extension_padding: str,
        ) -> dict[str, Any]:
            return body_for_repetitions(
                repetitions,
                padding,
                phase="consumer",
                consumer_extension=(
                    PROMPT_UNIT * extension_repetitions + extension_padding
                ),
            )

        (
            consumer_extension_repetitions,
            consumer_extension_padding,
            consumer_token_ids,
        ) = _calibrate_exact_prompt(
            args.api,
            body_for_consumer_extension,
            args.consumer_target_tokens,
        )
        consumer_body = body_for_consumer_extension(
            consumer_extension_repetitions,
            consumer_extension_padding,
        )
    if consumer_token_ids[: len(producer_token_ids)] != producer_token_ids:
        raise RuntimeError("producer tokens are not an exact prefix of consumer tokens")
    if len(consumer_token_ids) <= len(producer_token_ids):
        raise RuntimeError("consumer did not extend the producer token prefix")
    prompt_tokens = (
        len(producer_token_ids)
        if args.phase == "producer"
        else len(consumer_token_ids)
    )
    fixture: dict[str, Any] | list[dict[str, Any]] = (
        fixtures[0] if len(fixtures) == 1 else fixtures
    )
    if args.dry_tokenize:
        print(
            json.dumps(
                {
                    "fixture": fixture,
                    "target_span": args.target_span,
                    "alignment": args.alignment,
                    "prompt_tokens": prompt_tokens,
                    "producer_prompt_tokens": len(producer_token_ids),
                    "consumer_prompt_tokens": len(consumer_token_ids),
                    "consumer_extension_repetitions": (
                        consumer_extension_repetitions
                    ),
                    "consumer_extension_padding_bytes": len(
                        consumer_extension_padding.encode()
                    ),
                    "prompt_unit_repetitions": repetitions,
                    "prompt_padding_bytes": len(padding.encode()),
                },
                sort_keys=True,
            )
        )
        return

    if args.reset_local:
        _reset_local_caches(args.api)
    started = time.perf_counter()
    result = _post_json(
        args.api,
        "/v1/chat/completions",
        (
            body_for_repetitions(repetitions, padding, phase="producer")
            if args.phase == "producer"
            else consumer_body
        ),
    )
    elapsed = time.perf_counter() - started
    prompt_usage, cached_tokens, completion_tokens = _validated_usage(
        result,
        expected_prompt_tokens=prompt_tokens,
    )
    if (
        args.expect_cached_tokens is not None
        and cached_tokens != args.expect_cached_tokens
    ):
        raise SystemExit(
            "cached token assertion failed: "
            f"expected {args.expect_cached_tokens}, got {cached_tokens}"
        )
    normalized = _normalized_output(result)
    oracle_label = (
        _validate_content_oracle(normalized, args.labels)
        if args.phase == "consumer"
        else None
    )
    normalized_json = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    print(
        json.dumps(
            {
                "request_id": result["id"],
                "fixture": fixture,
                "target_span": args.target_span,
                "alignment": args.alignment,
                "prompt_tokens": prompt_usage,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "prompt_unit_repetitions": repetitions,
                "prompt_padding_bytes": len(padding.encode()),
                "consumer_extension_repetitions": (
                    consumer_extension_repetitions
                ),
                "consumer_extension_padding_bytes": len(
                    consumer_extension_padding.encode()
                ),
                "phase": args.phase,
                "bypass_spoolcache": args.bypass_spoolcache,
                "wall_seconds": round(elapsed, 6),
                "output": normalized,
                "oracle_label": oracle_label,
                "output_sha256": hashlib.sha256(normalized_json.encode()).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
