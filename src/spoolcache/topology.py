"""Model-independent vLLM worker topology and rank ownership identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .identity import sha256_json


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _rank_integer(value: object, label: str, degree: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= degree
    ):
        raise ValueError(f"{label} is outside the runtime topology")
    return value


@dataclass(frozen=True)
class WorkerCoordinate:
    """One KV-owning worker in vLLM's public ``PP x TP`` rank layout."""

    global_rank: int
    pp_rank: int
    tp_rank: int
    dcp_rank: int

    @classmethod
    def from_runtime(
        cls,
        *,
        global_rank: object,
        pp_rank: object,
        tp_rank: object,
        dcp_rank: object,
        tp_degree: object,
        pp_degree: object,
        dcp_degree: object,
    ) -> "WorkerCoordinate":
        tp_size = _positive_integer(tp_degree, "tensor-parallel degree")
        pp_size = _positive_integer(pp_degree, "pipeline-parallel degree")
        dcp_size = _positive_integer(dcp_degree, "decode-context degree")
        if tp_size % dcp_size:
            raise ValueError("decode-context degree must divide tensor parallelism")
        world_size = tp_size * pp_size
        resolved_global = _rank_integer(global_rank, "global rank", world_size)
        resolved_pp = _rank_integer(pp_rank, "pipeline-parallel rank", pp_size)
        resolved_tp = _rank_integer(tp_rank, "tensor-parallel rank", tp_size)
        resolved_dcp = _rank_integer(
            dcp_rank,
            "decode-context rank",
            dcp_size,
        )
        if resolved_global != resolved_pp * tp_size + resolved_tp:
            raise ValueError("global rank disagrees with PP/TP coordinates")
        # vLLM DCP is a TP subdivision and does not add worker processes.
        if resolved_dcp != resolved_tp % dcp_size:
            raise ValueError("decode-context rank disagrees with TP subdivision")
        return cls(
            global_rank=resolved_global,
            pp_rank=resolved_pp,
            tp_rank=resolved_tp,
            dcp_rank=resolved_dcp,
        )


def expected_worker_coordinates(
    *,
    tp_degree: object,
    pp_degree: object,
    dcp_degree: object,
) -> tuple[WorkerCoordinate, ...]:
    """Return every required KV participant in vLLM global-rank order."""

    tp_size = _positive_integer(tp_degree, "tensor-parallel degree")
    pp_size = _positive_integer(pp_degree, "pipeline-parallel degree")
    dcp_size = _positive_integer(dcp_degree, "decode-context degree")
    if tp_size % dcp_size:
        raise ValueError("decode-context degree must divide tensor parallelism")
    return tuple(
        WorkerCoordinate.from_runtime(
            global_rank=pp_rank * tp_size + tp_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            dcp_rank=tp_rank % dcp_size,
            tp_degree=tp_size,
            pp_degree=pp_size,
            dcp_degree=dcp_size,
        )
        for pp_rank in range(pp_size)
        for tp_rank in range(tp_size)
    )


def rank_ownership_sha256(
    *,
    layer_names: Sequence[str],
    shared_aliases: Sequence[tuple[str, Sequence[str]]],
    coordinate: WorkerCoordinate,
    pp_degree: int,
    dp_rank: int,
) -> str:
    """Bind runtime-proven stage ownership while preserving PP=1 digests."""

    layers = tuple(layer_names)
    if any(not isinstance(name, str) or not name for name in layers):
        raise ValueError("owned layer names must be non-empty strings")
    if len(set(layers)) != len(layers):
        raise ValueError("owned layer names must be unique")
    if isinstance(pp_degree, bool) or not isinstance(pp_degree, int) or pp_degree <= 0:
        raise ValueError("pipeline-parallel degree must be positive")
    if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
        raise ValueError("data-parallel rank must be non-negative")

    aliases = tuple((name, tuple(targets)) for name, targets in shared_aliases)
    alias_names: set[str] = set()
    owned = set(layers)
    for name, targets in aliases:
        if (
            not isinstance(name, str)
            or not name
            or name in alias_names
            or not targets
            or any(target not in owned for target in targets)
        ):
            raise ValueError("shared layer aliases are invalid")
        alias_names.add(name)

    payload: dict[str, object] = {
        "layers": layers,
        "pp": pp_degree,
        "dp_rank": dp_rank,
    }
    if aliases:
        payload["shared_layer_aliases"] = aliases
    if pp_degree > 1:
        payload["worker_coordinate"] = {
            "global_rank": coordinate.global_rank,
            "pp_rank": coordinate.pp_rank,
            "tp_rank": coordinate.tp_rank,
            "dcp_rank": coordinate.dcp_rank,
        }
    return sha256_json(payload)


__all__ = [
    "WorkerCoordinate",
    "expected_worker_coordinates",
    "rank_ownership_sha256",
]
