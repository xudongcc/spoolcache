#!/usr/bin/env python3
"""Send one deterministic visual-prefix request and emit a JSON receipt.

This is intentionally a *single-request* probe.  To prove that SpoolCache, rather
than vLLM's in-process prefix cache, supplied the hit:

1. run it once and wait for both ranks to publish the entry;
2. restart every tensor-parallel rank;
3. run the exact same command again;
4. require ``cached_tokens > 0`` and matching restore logs on every rank.

The tiny embedded white PNG is not intended to evaluate model quality.  It
keeps the media bytes stable and makes the visual cache-identity path easy to
reproduce without downloading an external asset.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request


# A decoded 1 x 1 PNG is used instead of a file path.  vLLM therefore receives
# identical media bytes on both sides of a process restart, independent of the
# caller's working directory or filesystem metadata.
WHITE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAusB9Y9Z4aQAAAAASUVORK5CYII="
)
PROMPT_UNIT = "alpha beta gamma delta epsilon zeta eta theta. "


def _first_api_key() -> str:
    """Return the first configured key without ever logging it."""

    key = os.environ.get("VLLM_API_KEY", "")
    raw_keys = os.environ.get("DSPARK_API_KEYS", "")
    if raw_keys:
        # Deployments have used both whitespace- and comma-separated key lists.
        fields = raw_keys.replace(",", " ").split()
        key = fields[0] if fields else ""
    return key


def _visual_prompt(nonce: str, repetitions: int) -> str:
    # nonce must remain stable across the cold/store and post-restart hit runs.
    return (
        f"SpoolCache visual persistence {nonce}. "
        + PROMPT_UNIT * repetitions
        + "Describe the image briefly."
    )


def _payload(model: str, nonce: str, repetitions: int) -> dict[str, object]:
    # Decode once locally as a guard against accidentally checking in a broken
    # fixture.  The original base64 is sent because OpenAI-compatible APIs use a
    # data URL at this boundary.
    decoded = base64.b64decode(WHITE_PNG_BASE64, validate=True)
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("embedded visual fixture is not a PNG")

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{WHITE_PNG_BASE64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": _visual_prompt(nonce, repetitions),
                    },
                ],
            }
        ],
        "max_tokens": 12,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
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
    parser.add_argument(
        "--nonce",
        required=True,
        help="keep this byte-for-byte identical before and after restart",
    )
    parser.add_argument(
        "--prompt-unit-repetitions",
        type=int,
        default=500,
        help="500 yields roughly 5K prompt tokens on the qualified profile",
    )
    args = parser.parse_args()
    if args.prompt_unit_repetitions < 1:
        parser.error("--prompt-unit-repetitions must be positive")

    headers = {"Content-Type": "application/json"}
    api_key = _first_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{args.api.rstrip('/')}/v1/chat/completions",
        data=json.dumps(
            _payload(args.model, args.nonce, args.prompt_unit_repetitions)
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started

    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    # Only return the generated text and non-sensitive measurements.  Prompt
    # bytes, media bytes, salts, and credentials do not enter benchmark logs.
    receipt = {
        "object": body.get("object"),
        "request_id": body.get("id"),
        "text": message.get("content"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "wall_seconds": round(elapsed, 6),
    }
    print(json.dumps({"visual_prefix_receipt": receipt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
