"""vLLM-native Prometheus exporter for bounded SpoolCache stats."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
)

from ..config import AccessMode, SpoolCacheConfig
from ..telemetry import (
    METRIC_DEFINITIONS,
    MetricKind,
    aggregate_gauges,
    iter_metric_records,
    merge_metric_payloads,
    validate_metric_payload,
)


class SpoolCachePromMetrics(KVConnectorPromMetrics):
    """Register once, then observe only pre-bound finite label tuples."""

    def __init__(
        self,
        vllm_config: Any,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        self._bound: dict[tuple[str, int, tuple[str, ...]], Any] = {}
        for name, definition in METRIC_DEFINITIONS.items():
            metric_type = {
                MetricKind.COUNTER: self._counter_cls,
                MetricKind.GAUGE: self._gauge_cls,
                MetricKind.HISTOGRAM: self._histogram_cls,
            }[definition.kind]
            kwargs: dict[str, Any] = {
                "name": name,
                "documentation": definition.documentation,
                "labelnames": labelnames + list(definition.label_names),
            }
            if definition.kind is MetricKind.HISTOGRAM:
                kwargs["buckets"] = definition.buckets
            metric = metric_type(**kwargs)
            for engine_idx, engine_labels in per_engine_labelvalues.items():
                for labels in definition.allowed_labels:
                    self._bound[(name, engine_idx, labels)] = metric.labels(
                        *(engine_labels + list(labels))
                    )
        self._initialize_startup_readiness(vllm_config)

    def _initialize_startup_readiness(self, vllm_config: Any) -> None:
        """Publish the startup handshake invariant before the first request.

        vLLM transports connector stats only after a scheduler iteration.  Its
        API does not become live until EngineCore construction (including the
        all-worker handshake) has completed, and SpoolCache rejects an
        incomplete handshake.  These initial gauges therefore describe a
        proven startup state; the first stats packet replaces them with the
        live scheduler values.
        """

        parallel = getattr(vllm_config, "parallel_config", None)
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        # Minimal metric test doubles intentionally lack runtime config. They
        # exercise registration/observation and do not claim startup readiness.
        if parallel is None and transfer is None:
            return
        tp_degree = getattr(parallel, "tensor_parallel_size", None)
        pp_degree = getattr(parallel, "pipeline_parallel_size", 1)
        if (
            isinstance(tp_degree, bool)
            or not isinstance(tp_degree, int)
            or tp_degree <= 0
            or isinstance(pp_degree, bool)
            or not isinstance(pp_degree, int)
            or pp_degree <= 0
        ):
            raise ValueError("SpoolCache Prometheus topology is invalid")
        required_workers = tp_degree * pp_degree
        raw = getattr(transfer, "kv_connector_extra_config", None)
        if not isinstance(raw, Mapping):
            raise ValueError("SpoolCache Prometheus connector config is invalid")
        config = SpoolCacheConfig.from_mapping(dict(raw))
        active = config.access_mode is not AccessMode.DISABLED
        for engine_idx in self.per_engine_labelvalues:
            self._bound[("spoolcache_required_ranks", engine_idx, ())].set(
                required_workers
            )
            self._bound[("spoolcache_ready_ranks", engine_idx, ())].set(
                required_workers if active else 0
            )
            for condition in ("rank_identity", "inventory_quorum"):
                self._bound[
                    ("spoolcache_readiness", engine_idx, (condition,))
                ].set(int(active))
            self._bound[
                ("spoolcache_readiness", engine_idx, ("fatal_clear",))
            ].set(1)

    def observe(
        self, transfer_stats_data: dict[str, Any], engine_idx: int = 0
    ) -> None:
        if engine_idx not in self.per_engine_labelvalues:
            raise ValueError("SpoolCache stats reference an unknown engine")
        validate_metric_payload(transfer_stats_data)
        for record in iter_metric_records(
            transfer_stats_data, MetricKind.COUNTER
        ):
            assert record.value is not None
            self._bound[(record.name, engine_idx, record.labels)].inc(record.value)
        for record in iter_metric_records(
            transfer_stats_data, MetricKind.HISTOGRAM
        ):
            metric = self._bound[(record.name, engine_idx, record.labels)]
            for value in record.values:
                metric.observe(value)
        for (name, labels), value in aggregate_gauges(
            transfer_stats_data
        ).items():
            self._bound[(name, engine_idx, labels)].set(value)


__all__ = ["SpoolCachePromMetrics", "merge_metric_payloads"]
