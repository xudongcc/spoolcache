"""Validated configuration for the SpoolCache 0.1 connector."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError

_MIB = 1024 * 1024
_GIB = 1024 * _MIB

# These are correctness/memory bounds, not deployment tuning knobs. Keeping
# them in one module makes the fixed resource budget auditable without asking
# every launcher to reproduce a matrix of low-level environment variables.
CACHE_CHUNK_TOKENS = 256
MIN_SPAN_TOKENS = 1024
MAX_SPAN_TOKENS = 1_048_576
STAGING_SLOT_BYTES = 64 * _MIB
STAGING_SLOT_COUNT = 2
MAX_PENDING_RESTORES = 2
MAX_PENDING_STORES = 1
STARTUP_MAX_DIGESTS = 512
REPORT_BATCH_SIZE = 64
CATALOG_MAX_ENTRIES = 100_000


class AccessMode(str, enum.Enum):
    READ_WRITE = "read-write"
    RESTORE_ONLY = "restore-only"
    STORE_ONLY = "store-only"
    DISABLED = "disabled"

    @property
    def restore_enabled(self) -> bool:
        return self in {self.READ_WRITE, self.RESTORE_ONLY}

    @property
    def store_enabled(self) -> bool:
        return self in {self.READ_WRITE, self.STORE_ONLY}


class DirectIOMode(str, enum.Enum):
    REQUIRED = "required"
    BEST_EFFORT = "best-effort"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SpoolCacheConfig:
    root: Path
    deployment_namespace: str = "default"
    access_mode: AccessMode = AccessMode.DISABLED
    max_bytes: int = 200 * _GIB
    direct_io: DirectIOMode = DirectIOMode.REQUIRED

    def __post_init__(self) -> None:
        root = self.root
        if not root.is_absolute():
            raise ConfigurationError("spoolcache_root must be absolute")
        if str(root) == "/":
            raise ConfigurationError("spoolcache_root cannot be filesystem root")
        if not self.deployment_namespace or len(self.deployment_namespace) > 256:
            raise ConfigurationError("deployment namespace is empty or too long")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int):
            raise ConfigurationError("max_bytes must be an integer")
        if self.max_bytes < 2:
            raise ConfigurationError("max_bytes must be at least 2 bytes")

    @property
    def restore_enabled(self) -> bool:
        return self.access_mode.restore_enabled

    @property
    def store_enabled(self) -> bool:
        return self.access_mode.store_enabled

    @property
    def low_watermark_bytes(self) -> int:
        """Fixed 90% GC target derived from the operator's capacity limit."""

        return self.max_bytes * 9 // 10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SpoolCacheConfig":
        aliases = {
            "spoolcache_root": "root",
            "spoolcache_deployment_namespace": "deployment_namespace",
            "spoolcache_access_mode": "access_mode",
            "spoolcache_max_bytes": "max_bytes",
            "spoolcache_direct_io": "direct_io",
        }
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            target = aliases.get(key, key)
            if target not in cls.__dataclass_fields__:
                raise ConfigurationError(f"unknown SpoolCache setting: {key}")
            if target in normalized:
                raise ConfigurationError(f"duplicate SpoolCache setting: {target}")
            normalized[target] = value
        missing = [key for key in ("root",) if key not in normalized]
        if missing:
            raise ConfigurationError(
                "missing required SpoolCache settings: " + ", ".join(missing)
            )
        normalized["root"] = Path(os.fspath(normalized["root"]))
        try:
            normalized["access_mode"] = AccessMode(
                normalized.get("access_mode", AccessMode.DISABLED)
            )
            normalized["direct_io"] = DirectIOMode(
                normalized.get("direct_io", DirectIOMode.REQUIRED)
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        for name in ("max_bytes",):
            if name in normalized and (
                isinstance(normalized[name], bool)
                or not isinstance(normalized[name], int)
            ):
                raise ConfigurationError(f"{name} must be an integer")
        return cls(**normalized)
