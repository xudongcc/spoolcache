"""Small, deterministic policies for optional cache publication.

Restore admission is a correctness decision: once vLLM allocates blocks for an
external hit, failure is fatal in the current connector.  Store
admission is different.  A store is optional and may be skipped to protect the
request's latency or NVMe endurance.  Keeping that policy pure makes it easy to
test without importing vLLM or allocating model memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar


class StorePlanLike(Protocol):
    """The minimal scheduler plan surface needed by admission."""

    request_id: str
    entry_id: str
    span_tokens: int


PlanT = TypeVar("PlanT", bound=StorePlanLike)


@dataclass(frozen=True)
class StoreAdmissionResult(Generic[PlanT]):
    """Accepted plans plus bounded, non-sensitive skip counters."""

    admitted: tuple[PlanT, ...]
    skipped_duplicate: int
    skipped_budget: int

    @property
    def skipped(self) -> int:
        return self.skipped_duplicate + self.skipped_budget


def admit_store_plans(
    plans: Sequence[PlanT],
    *,
    max_plans: int,
) -> StoreAdmissionResult[PlanT]:
    """Deduplicate and bound synchronous Store work for one scheduler step.

    Longer exact prefixes are admitted first because they normally avoid more
    recompute and amortize the fixed model-runner interruption better.  Input
    order breaks ties, making the result stable and replayable in tests.

    A zero budget admits no plans and counts every unique plan as skipped.
    Production supplies its fixed internal per-step budget.
    """

    if isinstance(max_plans, bool) or not isinstance(max_plans, int):
        raise ValueError("store admission budget must be an integer")
    if max_plans < 0:
        raise ValueError("store admission budget cannot be negative")

    unique: list[tuple[int, PlanT]] = []
    spans_by_entry: dict[str, int] = {}
    skipped_duplicate = 0
    for position, plan in enumerate(plans):
        if not plan.request_id or not plan.entry_id or plan.span_tokens <= 0:
            raise ValueError("store admission received a malformed plan")
        previous_span = spans_by_entry.get(plan.entry_id)
        if previous_span is not None:
            # One cryptographic entry ID must identify one exact prefix span.
            # A disagreement is a caller/identity bug, not a harmless duplicate.
            if previous_span != plan.span_tokens:
                raise ValueError("one store entry ID has conflicting token spans")
            skipped_duplicate += 1
            continue
        spans_by_entry[plan.entry_id] = plan.span_tokens
        unique.append((position, plan))

    prioritized = sorted(
        unique,
        key=lambda item: (-item[1].span_tokens, item[0]),
    )
    admitted = tuple(plan for _, plan in prioritized[:max_plans])
    return StoreAdmissionResult(
        admitted=admitted,
        skipped_duplicate=skipped_duplicate,
        skipped_budget=max(0, len(prioritized) - len(admitted)),
    )


__all__ = [
    "StoreAdmissionResult",
    "StorePlanLike",
    "admit_store_plans",
]
