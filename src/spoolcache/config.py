"""Validated configuration and fixed bounds for token-file caching."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError

_MIB = 1024 * 1024
_GIB = 1024 * _MIB

# These are correctness/memory bounds, not deployment tuning knobs. Keeping
# them in one module makes the fixed resource budget auditable without asking
# every launcher to reproduce a matrix of low-level environment variables.
STAGING_SLOT_BYTES = 64 * _MIB
STAGING_SLOT_COUNT = 2
MAX_PENDING_RESTORES = 2
MAX_PENDING_STORES = 1
REPORT_BATCH_SIZE = 64
INVENTORY_MEMORY_BYTES = 512 * _MIB

# Conservative ownership accounting for one fixed-width inventory record. This
# covers reporter/scheduler maps, checkpoint tuples, rank sets and temporary
# transport copies per rank. Dictionary/set slack is charged at singleton
# allocation cost, rather than assuming a full table's best-case load factor.
_KEY_EXAMPLE = "0" * 64
INVENTORY_ENTRY_BYTES = (
    4 * (sys.getsizeof(_KEY_EXAMPLE) + sys.getsizeof((1 << 63) - 1))
    + 6 * sys.getsizeof((_KEY_EXAMPLE, 1))
    + 4 * (sys.getsizeof({_KEY_EXAMPLE: 1}) - sys.getsizeof({}))
    + 2 * sys.getsizeof({0})
    + 16 * (sys.getsizeof((None,)) - sys.getsizeof(()))
)


def inventory_capacity(memory_bytes: int) -> int:
    """Record capacity derived from retained-memory allowance, not a key cap."""
    if type(memory_bytes) is not int or memory_bytes < INVENTORY_ENTRY_BYTES:
        raise ConfigurationError("inventory memory budget cannot hold one record")
    return memory_bytes // INVENTORY_ENTRY_BYTES


def aligned_chunk_tokens(alignment: int) -> int:
    """Smallest whole runtime alignment covering at least 256 tokens."""
    if type(alignment) is not int or alignment <= 0:
        raise ConfigurationError("chunk alignment must be a positive integer")
    # A floor avoids excess small files; it is not another page alignment.
    # In particular, alignment 192 gives 384, not lcm(192, 256) == 768.
    return ((256 + alignment - 1) // alignment) * alignment


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
    def trigger_watermark_bytes(self) -> int:
        """First whole byte at/above the 80% capacity trigger."""
        return (self.max_bytes * 4 + 4) // 5

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
