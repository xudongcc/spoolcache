"""Render a validated vLLM ``--kv-transfer-config`` from environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from ..config import SpoolCacheConfig


def _integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def build_config(environment: Mapping[str, str]) -> dict[str, object]:
    extra: dict[str, object] = {
        "spoolcache_root": environment.get(
            "SPOOLCACHE_CONTAINER_ROOT", "/var/lib/spoolcache"
        ),
        "spoolcache_deployment_namespace": environment.get(
            "SPOOLCACHE_NAMESPACE", "default"
        ),
        "spoolcache_access_mode": environment.get(
            "SPOOLCACHE_ACCESS_MODE", "read-write"
        ),
        "spoolcache_max_bytes": _integer(
            environment, "SPOOLCACHE_MAX_BYTES", 200 * 1024**3
        ),
        "spoolcache_direct_io": environment.get(
            "SPOOLCACHE_DIRECT_IO", "required"
        ),
    }
    # Validation here makes container startup fail before vLLM imports the
    # model or allocates its 100+ GiB runtime.
    validated = SpoolCacheConfig.from_mapping(extra)
    extra["spoolcache_root"] = str(Path(validated.root))
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
