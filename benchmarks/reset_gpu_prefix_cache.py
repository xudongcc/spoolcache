#!/usr/bin/env python3
"""Reset vLLM's in-process caches without deleting SpoolCache entries.

The endpoint exists only when vLLM starts with ``VLLM_SERVER_DEV_MODE=1``.
That mode also exposes unrelated development APIs, so this helper is intended
only for an isolated qualification environment.  Query parameters are written
explicitly to make an accidental external-cache reset visible in code review.
"""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request

RESET_QUERY = "reset_running_requests=false&reset_external=false"
MULTIMODAL_RESET_PATHS = ("/reset_mm_cache", "/reset_encoder_cache")


def _headers() -> dict[str, str]:
    result: dict[str, str] = {}
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        keys = os.environ.get("DSPARK_API_KEYS", "").split()
        key = keys[0] if keys else ""
    if key:
        result["Authorization"] = f"Bearer {key}"
    return result


def _post_reset(api: str, path: str) -> int:
    url = f"{api.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=b"",
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(
                "vLLM returned 404; restart it once with VLLM_SERVER_DEV_MODE=1"
            ) from error
        raise SystemExit(
            f"vLLM cache reset failed for {path} with HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"cannot reach vLLM cache reset endpoint {path}: {error.reason}"
        ) from error

    if status != 200:
        raise SystemExit(f"unexpected vLLM cache reset status for {path}: {status}")
    return status


def reset_gpu_prefix_cache(api: str) -> int:
    """Submit the deliberately local-only prefix reset and return its status."""

    return _post_reset(api, f"/reset_prefix_cache?{RESET_QUERY}")


def reset_multimodal_caches(api: str) -> tuple[int, ...]:
    """Clear vLLM media processor/encoder caches after the prefix reset."""

    return tuple(_post_reset(api, path) for path in MULTIMODAL_RESET_PATHS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Invalidate vLLM's GPU prefix cache while retaining the external "
            "SpoolCache NVMe store."
        )
    )
    parser.add_argument("--api", default="http://127.0.0.1:8888")
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="also clear vLLM's multimodal processor and encoder caches",
    )
    args = parser.parse_args()

    reset_gpu_prefix_cache(args.api)
    if args.multimodal:
        reset_multimodal_caches(args.api)
    scope = (
        "GPU prefix/encoder/multimodal caches"
        if args.multimodal
        else "GPU prefix cache"
    )
    print(f"{scope} reset accepted; external SpoolCache was retained.")


if __name__ == "__main__":
    main()
