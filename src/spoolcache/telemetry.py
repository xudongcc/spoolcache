"""Bounded, serialization-safe telemetry for the vLLM connector channel.

The data carried here is deliberately independent of Prometheus and vLLM.
Workers and the scheduler emit only finite metric names/label tuples and plain
JSON-compatible values.  The API process can then bind those records to
vLLM's public connector Prometheus hook without transporting request data.
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


METRIC_PAYLOAD_SCHEMA = "spoolcache-metrics/v1"
MAX_HISTOGRAM_OBSERVATIONS = 4096
MAX_METRIC_RECORDS = 4096
MAX_METRIC_VALUE = (1 << 63) - 1
_SOURCE_RE = re.compile(r"^(?:scheduler|rank:(?:0|[1-9][0-9]*))$")


class MetricKind(str, enum.Enum):
    COUNTER = "counters"
    GAUGE = "gauges"
    HISTOGRAM = "histograms"


@dataclass(frozen=True)
class MetricDefinition:
    kind: MetricKind
    documentation: str
    label_names: tuple[str, ...] = ()
    allowed_labels: tuple[tuple[str, ...], ...] = ((),)
    gauge_aggregation: str = "max"
    buckets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.documentation:
            raise ValueError("metric documentation cannot be empty")
        if not self.allowed_labels:
            raise ValueError("metric label vocabulary cannot be empty")
        if any(
            len(labels) != len(self.label_names)
            for labels in self.allowed_labels
        ):
            raise ValueError("metric label tuples do not match label names")
        if len(set(self.allowed_labels)) != len(self.allowed_labels):
            raise ValueError("metric label vocabulary contains duplicates")
        if self.kind is MetricKind.GAUGE:
            if self.gauge_aggregation not in {"max", "sum"}:
                raise ValueError("gauge aggregation is unsupported")
        elif self.gauge_aggregation != "max":
            raise ValueError("only gauges can select an aggregation")
        if self.kind is MetricKind.HISTOGRAM:
            if (
                not self.buckets
                or tuple(sorted(set(self.buckets))) != self.buckets
                or any(
                    not math.isfinite(value) or value <= 0
                    for value in self.buckets
                )
            ):
                raise ValueError("histogram buckets must be finite and increasing")
        elif self.buckets:
            raise ValueError("only histograms can declare buckets")


LOOKUP_LABELS = (
    ("hit", "ready_entry"),
    ("bypass", "request_skip_read"),
    ("miss", "catalog_unavailable"),
    ("miss", "cache_salt"),
    ("miss", "request_shape"),
    ("miss", "multimodal_identity"),
    ("miss", "safe_span"),
    ("miss", "rank_quorum"),
    ("miss", "restore_budget"),
)

_SECONDS_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = {
    "spoolcache_lookup_total": MetricDefinition(
        MetricKind.COUNTER,
        "SpoolCache scheduler lookup attempts by bounded outcome and reason.",
        ("result", "reason"),
        LOOKUP_LABELS,
    ),
    "spoolcache_hit_tokens_total": MetricDefinition(
        MetricKind.COUNTER,
        "Prompt tokens admitted for a SpoolCache external restore.",
    ),
    "spoolcache_restore_bytes_total": MetricDefinition(
        MetricKind.COUNTER,
        "Logical KV bytes restored from rank-local storage.",
    ),
    "spoolcache_restore_seconds": MetricDefinition(
        MetricKind.HISTOGRAM,
        "Synchronous authenticated rank-local restore duration.",
        buckets=_SECONDS_BUCKETS,
    ),
    "spoolcache_store_bytes_total": MetricDefinition(
        MetricKind.COUNTER,
        "Logical KV bytes durably stored in rank-local storage.",
    ),
    "spoolcache_store_seconds": MetricDefinition(
        MetricKind.HISTOGRAM,
        "Synchronous durable rank-local store duration.",
        buckets=_SECONDS_BUCKETS,
    ),
    "spoolcache_store_skipped_total": MetricDefinition(
        MetricKind.COUNTER,
        "Optional stores skipped by a bounded reason.",
        ("reason",),
        tuple(
            (reason,)
            for reason in (
                "duplicate",
                "budget",
                "busy",
                "unsafe_boundary",
                "error",
            )
        ),
    ),
    "spoolcache_post_admission_failure_total": MetricDefinition(
        MetricKind.COUNTER,
        "Fatal restore failures after scheduler admission.",
        ("phase",),
        tuple(
            (phase,)
            for phase in ("lookup", "span", "manifest", "payload", "unknown")
        ),
    ),
    "spoolcache_rank_quorum_entries": MetricDefinition(
        MetricKind.GAUGE,
        "Entries currently offered by every required physical rank.",
    ),
    "spoolcache_rank_generation_changes_total": MetricDefinition(
        MetricKind.COUNTER,
        "Observed worker generation replacements after initial handshake.",
    ),
    "spoolcache_pinned_pool_bytes": MetricDefinition(
        MetricKind.GAUGE,
        "Configured pinned staging bytes across reporting physical ranks.",
        gauge_aggregation="sum",
    ),
    "spoolcache_delayed_store_requests": MetricDefinition(
        MetricKind.GAUGE,
        "Requests waiting to reach a safe synchronous store boundary.",
    ),
    "spoolcache_disk_bytes": MetricDefinition(
        MetricKind.GAUGE,
        "Manifest and immutable-object bytes across reporting rank roots.",
        gauge_aggregation="sum",
    ),
    "spoolcache_managed_disk_bytes": MetricDefinition(
        MetricKind.GAUGE,
        "All regular-file bytes owned by reporting rank-local cache roots.",
        gauge_aggregation="sum",
    ),
    "spoolcache_quarantine_bytes": MetricDefinition(
        MetricKind.GAUGE,
        "Quarantined evidence bytes retained across reporting rank roots.",
        gauge_aggregation="sum",
    ),
    "spoolcache_quarantined_entries_total": MetricDefinition(
        MetricKind.COUNTER,
        "Corrupt manifests or immutable objects quarantined by bounded reason.",
        ("reason",),
        tuple(
            (reason,)
            for reason in (
                "manifest_io",
                "manifest_validation",
                "object_collision",
                "payload_checksum",
                "payload_size",
                "unknown",
            )
        ),
    ),
    "spoolcache_scrub_payload_bytes_total": MetricDefinition(
        MetricKind.COUNTER,
        "Stored payload bytes authenticated by rank-local deep scrub.",
    ),
    "spoolcache_scrub_objects_total": MetricDefinition(
        MetricKind.COUNTER,
        "Immutable objects authenticated by rank-local deep scrub.",
    ),
    "spoolcache_scrub_manifests_total": MetricDefinition(
        MetricKind.COUNTER,
        "Rank manifests fully authenticated by rank-local deep scrub.",
    ),
    "spoolcache_scrub_cycles_total": MetricDefinition(
        MetricKind.COUNTER,
        "Complete resumable rank-local deep scrub cycles.",
    ),
    "spoolcache_scrub_objects_quarantined_total": MetricDefinition(
        MetricKind.COUNTER,
        "Corrupt or unrecognized managed objects moved to quarantine by scrub.",
    ),
    "spoolcache_scrub_failures_total": MetricDefinition(
        MetricKind.COUNTER,
        "Unexpected scheduled deep scrub step failures.",
    ),
    "spoolcache_scrub_namespace_items_total": MetricDefinition(
        MetricKind.COUNTER,
        "Managed namespace entries examined by incremental deep scrub snapshots.",
    ),
    "spoolcache_scrub_shutdown_failures_total": MetricDefinition(
        MetricKind.COUNTER,
        "Scheduled deep scrub shutdowns that exceeded the fixed join timeout.",
    ),
    "spoolcache_orphan_objects_removed_total": MetricDefinition(
        MetricKind.COUNTER,
        "Unreferenced immutable objects removed after a complete manifest pass.",
    ),
    "spoolcache_orphan_bytes_removed_total": MetricDefinition(
        MetricKind.COUNTER,
        "Unreferenced immutable-object bytes reclaimed by deep scrub.",
    ),
    "spoolcache_temporary_files_removed_total": MetricDefinition(
        MetricKind.COUNTER,
        "Abandoned publication temporary files removed by deep scrub.",
    ),
    "spoolcache_required_ranks": MetricDefinition(
        MetricKind.GAUGE,
        "Physical ranks required by the scheduler topology.",
    ),
    "spoolcache_ready_ranks": MetricDefinition(
        MetricKind.GAUGE,
        "Physical ranks with a current synchronized inventory generation.",
    ),
    "spoolcache_readiness": MetricDefinition(
        MetricKind.GAUGE,
        "Boolean SpoolCache scheduler readiness conditions.",
        ("condition",),
        (("rank_identity",), ("inventory_quorum",), ("fatal_clear",)),
    ),
    "spoolcache_telemetry_dropped_total": MetricDefinition(
        MetricKind.COUNTER,
        "Telemetry observations dropped to preserve a fixed memory bound.",
        ("kind",),
        (("histogram",),),
    ),
}


@dataclass(frozen=True)
class MetricRecord:
    name: str
    labels: tuple[str, ...]
    value: int | float | None = None
    values: tuple[int | float, ...] = ()
    source: str | None = None


def empty_metric_payload() -> dict[str, Any]:
    return {
        "schema": METRIC_PAYLOAD_SCHEMA,
        MetricKind.COUNTER.value: [],
        MetricKind.GAUGE.value: [],
        MetricKind.HISTOGRAM.value: [],
    }


def _validate_source(source: object) -> str:
    if (
        not isinstance(source, str)
        or len(source) > 32
        or _SOURCE_RE.fullmatch(source) is None
    ):
        raise ValueError("metric source must be scheduler or a physical rank")
    return source


def _validate_number(value: object, *, label: str) -> int | float:
    invalid = isinstance(value, bool) or not isinstance(value, (int, float))
    if isinstance(value, int) and not isinstance(value, bool):
        invalid = value < 0 or value > MAX_METRIC_VALUE
    elif isinstance(value, float):
        invalid = (
            not math.isfinite(value)
            or value < 0
            or value > MAX_METRIC_VALUE
        )
    if invalid:
        raise ValueError(f"{label} must be a finite non-negative number")
    return value


def _validate_metric(
    name: object,
    labels: Iterable[object],
    *,
    kind: MetricKind,
) -> tuple[str, tuple[str, ...], MetricDefinition]:
    if not isinstance(name, str) or name not in METRIC_DEFINITIONS:
        raise ValueError("unknown SpoolCache metric")
    definition = METRIC_DEFINITIONS[name]
    if definition.kind is not kind:
        raise TypeError(f"{name} is not a {kind.value[:-1]}")
    materialized = tuple(labels)
    if any(not isinstance(value, str) for value in materialized):
        raise TypeError("metric labels must be strings")
    if materialized not in definition.allowed_labels:
        raise ValueError(f"{name} labels are outside the bounded vocabulary")
    return name, materialized, definition


class TelemetryBuffer:
    """One role/rank's bounded delta buffer plus current gauges."""

    def __init__(self, *, source: str) -> None:
        self.source = _validate_source(source)
        self._counters: dict[tuple[str, tuple[str, ...]], int | float] = {}
        self._gauges: dict[tuple[str, tuple[str, ...]], int | float] = {}
        self._histograms: dict[
            tuple[str, tuple[str, ...]], list[int | float]
        ] = {}

    def increment(
        self,
        name: str,
        *,
        value: int | float = 1,
        labels: Iterable[str] = (),
    ) -> None:
        name, normalized, _ = _validate_metric(
            name, labels, kind=MetricKind.COUNTER
        )
        amount = _validate_number(value, label="counter increment")
        key = (name, normalized)
        self._counters[key] = _validate_number(
            self._counters.get(key, 0) + amount,
            label="counter total",
        )

    def set_gauge(
        self,
        name: str,
        value: int | float,
        *,
        labels: Iterable[str] = (),
    ) -> None:
        name, normalized, _ = _validate_metric(
            name, labels, kind=MetricKind.GAUGE
        )
        self._gauges[(name, normalized)] = _validate_number(
            value, label="gauge value"
        )

    def observe(
        self,
        name: str,
        value: int | float,
        *,
        labels: Iterable[str] = (),
    ) -> None:
        name, normalized, _ = _validate_metric(
            name, labels, kind=MetricKind.HISTOGRAM
        )
        observation = _validate_number(value, label="histogram observation")
        key = (name, normalized)
        values = self._histograms.setdefault(key, [])
        if len(values) >= MAX_HISTOGRAM_OBSERVATIONS:
            self.increment(
                "spoolcache_telemetry_dropped_total",
                labels=("histogram",),
            )
            return
        values.append(observation)

    def drain(self) -> dict[str, Any]:
        payload = empty_metric_payload()
        payload[MetricKind.COUNTER.value] = [
            {"name": name, "labels": list(labels), "value": value}
            for (name, labels), value in sorted(self._counters.items())
        ]
        payload[MetricKind.GAUGE.value] = [
            {
                "name": name,
                "labels": list(labels),
                "source": self.source,
                "value": value,
            }
            for (name, labels), value in sorted(self._gauges.items())
        ]
        payload[MetricKind.HISTOGRAM.value] = [
            {"name": name, "labels": list(labels), "values": list(values)}
            for (name, labels), values in sorted(self._histograms.items())
        ]
        self._counters.clear()
        self._histograms.clear()
        return payload


def _decode_records(
    payload: Mapping[str, Any], kind: MetricKind
) -> tuple[MetricRecord, ...]:
    raw_records = payload.get(kind.value)
    if not isinstance(raw_records, list):
        raise ValueError(f"metric payload {kind.value} must be a list")
    if len(raw_records) > MAX_METRIC_RECORDS:
        raise ValueError("metric payload record count exceeds the fixed bound")
    records: list[MetricRecord] = []
    seen: set[tuple[str, tuple[str, ...], str | None]] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("metric record must be a mapping")
        expected_keys = {"name", "labels"}
        if kind is MetricKind.HISTOGRAM:
            expected_keys.add("values")
        else:
            expected_keys.add("value")
        if kind is MetricKind.GAUGE:
            expected_keys.add("source")
        if set(raw) != expected_keys:
            raise ValueError("metric record fields differ from the schema")
        labels = raw["labels"]
        if not isinstance(labels, list):
            raise ValueError("metric record labels must be a list")
        name, normalized, _ = _validate_metric(
            raw["name"], labels, kind=kind
        )
        source = None
        if kind is MetricKind.GAUGE:
            source = _validate_source(raw["source"])
        key = (name, normalized, source)
        if key in seen:
            raise ValueError("metric payload contains duplicate records")
        seen.add(key)
        if kind is MetricKind.HISTOGRAM:
            values = raw["values"]
            if not isinstance(values, list):
                raise ValueError("histogram values must be a list")
            if len(values) > MAX_HISTOGRAM_OBSERVATIONS:
                raise ValueError("histogram observations exceed the fixed bound")
            records.append(
                MetricRecord(
                    name=name,
                    labels=normalized,
                    values=tuple(
                        _validate_number(value, label="histogram observation")
                        for value in values
                    ),
                )
            )
        else:
            records.append(
                MetricRecord(
                    name=name,
                    labels=normalized,
                    source=source,
                    value=_validate_number(
                        raw["value"],
                        label=("gauge value" if kind is MetricKind.GAUGE else "counter value"),
                    ),
                )
            )
    return tuple(records)


def validate_metric_payload(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("metric payload must be a mapping")
    if set(payload) != {
        "schema",
        MetricKind.COUNTER.value,
        MetricKind.GAUGE.value,
        MetricKind.HISTOGRAM.value,
    }:
        raise ValueError("metric payload fields differ from the schema")
    if payload.get("schema") != METRIC_PAYLOAD_SCHEMA:
        raise ValueError("metric payload schema is unsupported")
    for kind in MetricKind:
        _decode_records(payload, kind)


def iter_metric_records(
    payload: Mapping[str, Any], kind: MetricKind
) -> tuple[MetricRecord, ...]:
    validate_metric_payload(payload)
    return _decode_records(payload, kind)


def metric_payload_is_empty(payload: Mapping[str, Any]) -> bool:
    validate_metric_payload(payload)
    return not any(payload[kind.value] for kind in MetricKind)


def merge_metric_payloads(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    counters: dict[tuple[str, tuple[str, ...]], int | float] = {}
    gauges: dict[tuple[str, tuple[str, ...], str], int | float] = {}
    histograms: dict[tuple[str, tuple[str, ...]], list[int | float]] = {}
    overflow = 0
    for payload in payloads:
        validate_metric_payload(payload)
        for record in _decode_records(payload, MetricKind.COUNTER):
            assert record.value is not None
            key = (record.name, record.labels)
            counters[key] = _validate_number(
                counters.get(key, 0) + record.value,
                label="merged counter total",
            )
        for record in _decode_records(payload, MetricKind.GAUGE):
            assert record.value is not None and record.source is not None
            gauges[(record.name, record.labels, record.source)] = record.value
        for record in _decode_records(payload, MetricKind.HISTOGRAM):
            key = (record.name, record.labels)
            values = histograms.setdefault(key, [])
            remaining = MAX_HISTOGRAM_OBSERVATIONS - len(values)
            values.extend(record.values[:remaining])
            overflow += max(0, len(record.values) - remaining)
    if overflow:
        key = ("spoolcache_telemetry_dropped_total", ("histogram",))
        counters[key] = _validate_number(
            counters.get(key, 0) + overflow,
            label="merged dropped-observation total",
        )

    merged = empty_metric_payload()
    merged[MetricKind.COUNTER.value] = [
        {"name": name, "labels": list(labels), "value": value}
        for (name, labels), value in sorted(counters.items())
    ]
    merged[MetricKind.GAUGE.value] = [
        {
            "name": name,
            "labels": list(labels),
            "source": source,
            "value": value,
        }
        for (name, labels, source), value in sorted(gauges.items())
    ]
    merged[MetricKind.HISTOGRAM.value] = [
        {"name": name, "labels": list(labels), "values": list(values)}
        for (name, labels), values in sorted(histograms.items())
    ]
    # Each input is bounded independently, but aggregation can combine disjoint
    # sources or overflow a counter. Revalidate the result before it becomes
    # the next accumulator carried by vLLM.
    validate_metric_payload(merged)
    return merged


def aggregate_gauges(
    payload: Mapping[str, Any],
) -> dict[tuple[str, tuple[str, ...]], int | float]:
    grouped: dict[
        tuple[str, tuple[str, ...]], list[int | float]
    ] = {}
    for record in iter_metric_records(payload, MetricKind.GAUGE):
        assert record.value is not None
        grouped.setdefault((record.name, record.labels), []).append(record.value)
    result: dict[tuple[str, tuple[str, ...]], int | float] = {}
    for key, values in grouped.items():
        definition = METRIC_DEFINITIONS[key[0]]
        result[key] = _validate_number(
            sum(values)
            if definition.gauge_aggregation == "sum"
            else max(values),
            label="aggregated gauge value",
        )
    return result


def _summary_key(name: str, labels: tuple[str, ...]) -> str:
    return f"{name}{{{','.join(labels)}}}" if labels else name


def reduce_metric_payload(
    payload: Mapping[str, Any],
) -> dict[str, int | float]:
    summary: dict[str, int | float] = {}
    for record in iter_metric_records(payload, MetricKind.COUNTER):
        assert record.value is not None
        summary[_summary_key(record.name, record.labels)] = record.value
    for (name, labels), value in aggregate_gauges(payload).items():
        summary[_summary_key(name, labels)] = value
    for record in iter_metric_records(payload, MetricKind.HISTOGRAM):
        key = _summary_key(record.name, record.labels)
        summary[f"{key}_count"] = len(record.values)
        summary[f"{key}_sum"] = _validate_number(
            sum(record.values), label="histogram summary sum"
        )
    return summary


__all__ = [
    "LOOKUP_LABELS",
    "MAX_HISTOGRAM_OBSERVATIONS",
    "MAX_METRIC_RECORDS",
    "MAX_METRIC_VALUE",
    "METRIC_DEFINITIONS",
    "METRIC_PAYLOAD_SCHEMA",
    "MetricDefinition",
    "MetricKind",
    "MetricRecord",
    "TelemetryBuffer",
    "aggregate_gauges",
    "empty_metric_payload",
    "iter_metric_records",
    "merge_metric_payloads",
    "metric_payload_is_empty",
    "reduce_metric_payload",
    "validate_metric_payload",
]
