"""Deterministic deployment and physical-rank identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .errors import IdentityError


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode a mapping with one stable, version-independent JSON form."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IdentityError("identity contains a non-canonical value") from error


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def model_namespace_sha256(model_config: Any) -> str:
    """Hash vLLM's public model locator and revision without model dispatch.

    ``model_weights`` preserves an original object-storage URI when vLLM
    rewrites ``model`` to a role-local download directory.  Older runtimes may
    not expose it, so an absent or empty value falls back to the documented
    ``model`` locator.  API-facing served aliases are deliberately irrelevant.
    """

    model = getattr(model_config, "model", None)
    if not isinstance(model, str) or not model:
        raise IdentityError("vLLM model locator must be a non-empty string")
    original = getattr(model_config, "model_weights", "")
    if original is None:
        original = ""
    if not isinstance(original, str):
        raise IdentityError("vLLM original model locator must be a string")
    revision = getattr(model_config, "revision", None)
    if revision is not None and not isinstance(revision, str):
        raise IdentityError("vLLM model revision must be a string or null")
    return sha256_json(
        {
            "schema": "spoolcache-model-namespace/v1",
            "locator": original or model,
            "revision": revision,
        }
    )


@dataclass(frozen=True)
class DeploymentIdentity:
    schema: str
    profile: str
    model_namespace_sha256: str
    model_config_sha256: str
    execution_config_sha256: str
    vllm_version: str
    vllm_build_sha256: str
    kv_cache_dtype: str
    topology: Mapping[str, int]
    layout_sha256: str
    chunk_tokens: int
    spoolcache_version: str
    publication_policy: str = "prompt-snapshot-v1"

    def __post_init__(self) -> None:
        digests = (
            self.model_namespace_sha256,
            self.model_config_sha256,
            self.execution_config_sha256,
            self.vllm_build_sha256,
            self.layout_sha256,
        )
        if any(
            len(item) != 64
            or any(ch not in "0123456789abcdef" for ch in item)
            for item in digests
        ):
            raise IdentityError("deployment identity contains a malformed SHA-256")
        if self.schema != "spoolcache-deployment/v2":
            raise IdentityError("deployment identity schema is unsupported")
        if (
            not self.profile
            or not self.vllm_version
            or not self.spoolcache_version
            or not self.kv_cache_dtype
            or self.publication_policy != "prompt-snapshot-v1"
        ):
            raise IdentityError("deployment identity is incomplete")
        required_topology = {
            "tp",
            "pp",
            "dcp",
            "dp",
            "dp_rank",
            "world_size",
            "world_size_across_dp",
        }
        if set(self.topology) != required_topology:
            raise IdentityError("deployment topology fields are incomplete")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for name, value in self.topology.items()
            if name != "dp_rank"
        ):
            raise IdentityError("topology values must be positive integers")
        dp_rank = self.topology["dp_rank"]
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
            raise IdentityError("dp_rank must be a non-negative integer")
        if dp_rank >= self.topology["dp"]:
            raise IdentityError("dp_rank must be below the data-parallel size")
        if self.topology["world_size"] != self.topology["tp"] * self.topology["pp"]:
            raise IdentityError("world_size must equal tp multiplied by pp")
        if self.topology["world_size_across_dp"] != (
            self.topology["world_size"] * self.topology["dp"]
        ):
            raise IdentityError(
                "world_size_across_dp must include the data-parallel size"
            )
        if self.topology["tp"] % self.topology["dcp"]:
            raise IdentityError("dcp must divide tp")
        if (
            isinstance(self.chunk_tokens, bool)
            or not isinstance(self.chunk_tokens, int)
            or self.chunk_tokens <= 0
        ):
            raise IdentityError("chunk_tokens must be positive")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class RankIdentity:
    deployment_digest: str
    physical_rank: int
    shard_layout_sha256: str
    layer_ownership_sha256: str

    def __post_init__(self) -> None:
        for digest in (
            self.deployment_digest,
            self.shard_layout_sha256,
            self.layer_ownership_sha256,
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise IdentityError("rank identity contains a malformed SHA-256")
        if (
            isinstance(self.physical_rank, bool)
            or not isinstance(self.physical_rank, int)
            or self.physical_rank < 0
        ):
            raise IdentityError("physical_rank must be a non-negative integer")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
