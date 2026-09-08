from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from spoolcache.telemetry import METRIC_DEFINITIONS, TelemetryBuffer


class _BoundMetric:
    def __init__(self) -> None:
        self.increments: list[float] = []
        self.values: list[float] = []
        self.observations: list[float] = []

    def inc(self, value: float = 1) -> None:
        self.increments.append(value)

    def set(self, value: float) -> None:
        self.values.append(value)

    def observe(self, value: float) -> None:
        self.observations.append(value)


class _Metric:
    instances: list["_Metric"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.bound: dict[tuple[object, ...], _BoundMetric] = {}
        type(self).instances.append(self)

    def labels(self, *values: object) -> _BoundMetric:
        return self.bound.setdefault(values, _BoundMetric())


class _Counter(_Metric):
    instances: list[_Metric] = []


class _Gauge(_Metric):
    instances: list[_Metric] = []


class _Histogram(_Metric):
    instances: list[_Metric] = []


class PrometheusExporterTests(unittest.TestCase):
    def _load(self):
        module_name = "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
        metrics_module = types.ModuleType(module_name)

        class KVConnectorPromMetrics:
            def __init__(
                self,
                _vllm_config,
                metric_types,
                labelnames,
                per_engine_labelvalues,
            ) -> None:
                self._counter_cls = metric_types[_Counter]
                self._gauge_cls = metric_types[_Gauge]
                self._histogram_cls = metric_types[_Histogram]
                self._labelnames = labelnames
                self.per_engine_labelvalues = per_engine_labelvalues

        metrics_module.KVConnectorPromMetrics = KVConnectorPromMetrics
        modules = {
            "vllm": types.ModuleType("vllm"),
            "vllm.distributed": types.ModuleType("vllm.distributed"),
            "vllm.distributed.kv_transfer": types.ModuleType(
                "vllm.distributed.kv_transfer"
            ),
            "vllm.distributed.kv_transfer.kv_connector": types.ModuleType(
                "vllm.distributed.kv_transfer.kv_connector"
            ),
            "vllm.distributed.kv_transfer.kv_connector.v1": types.ModuleType(
                "vllm.distributed.kv_transfer.kv_connector.v1"
            ),
            module_name: metrics_module,
        }
        with patch.dict(sys.modules, modules):
            sys.modules.pop("spoolcache.vllm.prometheus", None)
            return importlib.import_module("spoolcache.vllm.prometheus")

    def setUp(self) -> None:
        for metric_type in (_Counter, _Gauge, _Histogram):
            metric_type.instances.clear()

    def test_registers_exact_bounded_metric_surface(self) -> None:
        module = self._load()
        exporter = module.SpoolCachePromMetrics(
            object(),
            {_Counter: _Counter, _Gauge: _Gauge, _Histogram: _Histogram},
            ["model_name", "engine"],
            {0: ["served", "0"]},
        )
        registered = {
            metric.kwargs["name"]
            for metric in (*_Counter.instances, *_Gauge.instances, *_Histogram.instances)
        }
        self.assertEqual(registered, set(METRIC_DEFINITIONS))
        self.assertEqual(set(exporter._bound), {
            (name, 0, labels)
            for name, definition in METRIC_DEFINITIONS.items()
            for labels in definition.allowed_labels
        })
        for metric in (*_Counter.instances, *_Gauge.instances, *_Histogram.instances):
            name = metric.kwargs["name"]
            self.assertEqual(
                metric.kwargs["labelnames"],
                ["model_name", "engine", *METRIC_DEFINITIONS[name].label_names],
            )

    def test_initial_readiness_is_derived_from_runtime_topology_and_config(self) -> None:
        module = self._load()
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
            kv_transfer_config=types.SimpleNamespace(
                kv_connector_extra_config={
                    "spoolcache_root": "/var/lib/spoolcache",
                    "spoolcache_access_mode": "read-write",
                    "spoolcache_direct_io": "required",
                }
            ),
        )
        exporter = module.SpoolCachePromMetrics(
            config,
            {_Counter: _Counter, _Gauge: _Gauge, _Histogram: _Histogram},
            ["engine"],
            {0: ["0"]},
        )
        self.assertEqual(
            exporter._bound[("spoolcache_required_ranks", 0, ())].values,
            [4],
        )
        self.assertEqual(
            exporter._bound[("spoolcache_ready_ranks", 0, ())].values,
            [4],
        )
        for condition in ("rank_identity", "inventory_quorum", "fatal_clear"):
            self.assertEqual(
                exporter._bound[
                    ("spoolcache_readiness", 0, (condition,))
                ].values,
                [1],
            )

    def test_observe_maps_deltas_histograms_and_aggregated_gauges(self) -> None:
        module = self._load()
        exporter = module.SpoolCachePromMetrics(
            object(),
            {_Counter: _Counter, _Gauge: _Gauge, _Histogram: _Histogram},
            ["engine"],
            {3: ["3"]},
        )
        rank0 = TelemetryBuffer(source="rank:0")
        rank1 = TelemetryBuffer(source="rank:1")
        rank0.increment("spoolcache_restore_bytes_total", value=1024)
        rank0.observe("spoolcache_restore_seconds", 0.25)
        rank0.set_gauge("spoolcache_disk_bytes", 10)
        rank1.set_gauge("spoolcache_disk_bytes", 20)
        payload = module.merge_metric_payloads(rank0.drain(), rank1.drain())
        exporter.observe(payload, engine_idx=3)

        self.assertEqual(
            exporter._bound[("spoolcache_restore_bytes_total", 3, ())].increments,
            [1024],
        )
        self.assertEqual(
            exporter._bound[("spoolcache_restore_seconds", 3, ())].observations,
            [0.25],
        )
        self.assertEqual(
            exporter._bound[("spoolcache_disk_bytes", 3, ())].values,
            [30],
        )

    def test_observe_rejects_unregistered_engine_and_malformed_labels(self) -> None:
        module = self._load()
        exporter = module.SpoolCachePromMetrics(
            object(),
            {_Counter: _Counter, _Gauge: _Gauge, _Histogram: _Histogram},
            [],
            {0: []},
        )
        with self.assertRaises(ValueError):
            exporter.observe(
                {
                    "schema": "spoolcache-metrics/v1",
                    "counters": [
                        {
                            "name": "spoolcache_lookup_total",
                            "labels": ["hit", "secret-entry"],
                            "value": 1,
                        }
                    ],
                    "gauges": [],
                    "histograms": [],
                }
            )
        with self.assertRaises(ValueError):
            exporter.observe(TelemetryBuffer(source="scheduler").drain(), 9)


if __name__ == "__main__":
    unittest.main()
