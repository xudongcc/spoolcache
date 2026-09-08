"""Render a validated vLLM ``--kv-transfer-config`` from environment."""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..config import SpoolCacheConfig


def _size_gb(environment: Mapping[str, str], name: str, default: float) -> float:
    raw = environment.get(name, str(default))
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number in GB") from error


def build_config(environment: Mapping[str, str]) -> dict[str, object]:
    extra: dict[str, object] = {
        "spoolcache_max_size": _size_gb(
            environment, "SPOOLCACHE_MAX_SIZE", 200
        ),
    }
    if "SPOOLCACHE_PATH" in environment:
        extra["spoolcache_path"] = environment["SPOOLCACHE_PATH"]
    # Validation here makes container startup fail before vLLM imports the
    # model or allocates its 100+ GiB runtime.
    validated = SpoolCacheConfig.from_mapping(extra)
    extra["spoolcache_path"] = str(validated.path)
    return {
        "kv_connector": "SpoolCacheConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "spoolcache.vllm.connector",
        "kv_connector_extra_config": extra,
        "kv_load_failure_policy": "fail",
    }


def main() -> None:
    print(json.dumps(build_config(os.environ), separators=(",", ":")))


if __name__ == "__main__":
    main()


__all__ = ["build_config", "main"]
