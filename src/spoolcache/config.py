"""Validated configuration for the SpoolCache 0.1 connector."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class SpoolCacheConfig:
    path: Path = field(default_factory=lambda: Path.home() / ".cache" / "spoolcache")
    max_size: float = 200

    def __post_init__(self) -> None:
        path = self.path.expanduser()
        if not path.is_absolute():
            raise ConfigurationError("spoolcache_path must be absolute")
        if str(path) == "/":
            raise ConfigurationError("spoolcache_path cannot be filesystem root")
        object.__setattr__(self, "path", path)
        if (
            isinstance(self.max_size, bool)
            or not isinstance(self.max_size, (int, float))
            or (isinstance(self.max_size, float) and not math.isfinite(self.max_size))
        ):
            raise ConfigurationError("max_size must be a finite number in GB")
        if self.max_bytes < 2:
            raise ConfigurationError("max_size must represent at least 2 bytes")

    @property
    def max_bytes(self) -> int:
        """Translate public GB (1024**3 bytes) to whole storage bytes."""

        # Integer arithmetic preserves byte boundaries and avoids overflowing a
        # finite float during conversion. Discard fractional bytes.
        numerator, denominator = self.max_size.as_integer_ratio()
        return numerator * _GIB // denominator

    @property
    def low_watermark_bytes(self) -> int:
        """Fixed 90% GC target derived from the operator's capacity limit."""

        return self.max_bytes * 9 // 10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SpoolCacheConfig":
        aliases = {
            "spoolcache_path": "path",
            "spoolcache_max_size": "max_size",
        }
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            target = aliases.get(key, key)
            if target not in cls.__dataclass_fields__:
                raise ConfigurationError(f"unknown SpoolCache setting: {key}")
            if target in normalized:
                raise ConfigurationError(f"duplicate SpoolCache setting: {target}")
            normalized[target] = value
        if "path" in normalized:
            normalized["path"] = Path(os.fspath(normalized["path"]))
        return cls(**normalized)
