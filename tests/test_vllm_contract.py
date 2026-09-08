from __future__ import annotations

import abc
import contextlib
import enum
import importlib
import json
import pickle
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from spoolcache.errors import ConfigurationError, UnsupportedRuntimeError
from spoolcache.maintenance import ScrubShutdownReport
from spoolcache.quorum import (
    InventoryCheckpoint,
    InventoryReporter,
    WorkerInventoryReport,
)
from spoolcache.telemetry import (
    MetricKind,
    TelemetryBuffer,
    aggregate_gauges,
    iter_metric_records,
)
from spoolcache.vllm.compat import (
    expandable_segments_enabled,
    require_qualified_allocator,
    verify_vllm_runtime,
)


class DummyBase:
    def __init__(self, vllm_config, role, kv_cache_config):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config
        self._kv_transfer_config = vllm_config.kv_transfer_config

    def register_kv_caches(self, kv_caches):
        pass

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        pass

    def on_new_request(self, request):
        pass

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        pass

    def get_finished(self, finished_req_ids):
        pass

    def get_block_ids_with_load_errors(self):
        pass

    def get_kv_connector_stats(self):
        pass

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        pass

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ):
        pass

    def update_connector_output(self, connector_output):
        pass

    def get_handshake_metadata(self):
        pass

    def set_xfer_handshake_metadata(self, metadata):
        pass

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        pass

    def shutdown(self):
        pass


class DummySupportsHMA:
    def request_finished_all_groups(self, request, block_ids):
        pass


class DummyConnector(DummyBase, DummySupportsHMA):
    pass


def resolve_spec_kind(spec: object) -> str:
    return getattr(spec, "public_kind", "unknown")


class FullAttentionSpec:
    public_kind = "full_attention"
    block_size = 256
    storage_block_size = 256
    page_size_bytes = 4


class SlidingWindowSpec:
    public_kind = "sliding_window"

    def __init__(self, block_size: int, window: int) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.sliding_window = window
        self.page_size_bytes = 4


class UniformTypeKVCacheSpecs:
    def __init__(self, specs: dict[str, SlidingWindowSpec], block_size: int) -> None:
        self.block_size = block_size
        self.storage_block_size = block_size
        self.kv_cache_specs = specs
        self.page_size_bytes = sum(spec.page_size_bytes for spec in specs.values())


def cache_config() -> types.SimpleNamespace:
    # The contract test is CPU-only, but its group distribution mirrors the
    # runtime receipt so connector logs and future byte assertions stay useful.
    signatures = (
        (62, 256, None),
        (23, 64, 128),
        (23, 64, 128),
        (42, 4, 8),
        (20, 8, 128),
    )
    groups = []
    for index, (count, block_size, window) in enumerate(signatures):
        names = tuple(f"g{index}.l{layer}" for layer in range(count))
        if window is None:
            spec = FullAttentionSpec()
        else:
            spec = UniformTypeKVCacheSpecs(
                {name: SlidingWindowSpec(block_size, window) for name in names},
                block_size,
            )
        groups.append(
            types.SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=names,
                is_eagle_group=False,
            )
        )
    return types.SimpleNamespace(num_blocks=1024, kv_cache_groups=tuple(groups))


class VLLMContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._runtime_directory = tempfile.TemporaryDirectory()
        self._runtime_root = Path(self._runtime_directory.name)
        (self._runtime_root / "runtime.py").write_text(
            "RUNTIME_CONTRACT = 1\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._runtime_directory.cleanup()

    def _runtime(self, version: str = "runtime-test") -> types.ModuleType:
        runtime = types.ModuleType("vllm")
        runtime.__version__ = version
        runtime.__path__ = [str(self._runtime_root)]  # type: ignore[attr-defined]
        return runtime

    def _verify(
        self,
        *,
        connector_base: type = DummyBase,
        connector_type: type = DummyConnector,
        supports_hma: type = DummySupportsHMA,
        spec_kind_resolver: object = resolve_spec_kind,
        version: str = "runtime-test",
        require_pp_aware: bool = False,
    ):
        return verify_vllm_runtime(
            vllm_module=self._runtime(version),
            connector_base=connector_base,
            connector_type=connector_type,
            supports_hma=supports_hma,
            spec_kind_resolver=spec_kind_resolver,
            package_roots=(self._runtime_root,),
            require_pp_aware=require_pp_aware,
        )

    def test_runtime_is_accepted_by_contract_not_version(self) -> None:
        receipt = self._verify(version="arbitrary-runtime-version")
        self.assertEqual(receipt.vllm_version, "arbitrary-runtime-version")
        self.assertEqual(receipt.compatibility_mode, "automatic-contract")
        self.assertEqual(len(receipt.vllm_build_sha256), 64)
        self.assertEqual(
            receipt.build_fingerprint_kind,
            "installed-package-content-sha256",
        )
        self.assertIn("start_load_kv", receipt.capabilities)
        self.assertIn("get_kv_cache_spec_kind", receipt.capabilities)
        self.assertEqual(
            receipt.connector_constructor,
            ("self", "vllm_config", "role", "kv_cache_config"),
        )

    def test_runtime_fingerprint_hashes_actual_installed_content(self) -> None:
        first = self._verify().vllm_build_sha256
        (self._runtime_root / "runtime.py").write_text(
            "RUNTIME_CONTRACT = 2\n", encoding="utf-8"
        )
        second = self._verify().vllm_build_sha256
        self.assertNotEqual(first, second)

        runtime = types.ModuleType("vllm")
        runtime.__version__ = "no-package-content"
        with self.assertRaisesRegex(UnsupportedRuntimeError, "package root"):
            verify_vllm_runtime(
                vllm_module=runtime,
                connector_base=DummyBase,
                connector_type=DummyConnector,
                supports_hma=DummySupportsHMA,
                spec_kind_resolver=resolve_spec_kind,
            )

    def test_cache_spec_semantic_resolver_call_shape_is_enforced(self) -> None:
        self._verify(
            spec_kind_resolver=lambda spec, future_option=None: "unknown"
        )
        for resolver in (
            True,
            lambda: "unknown",
            lambda spec, mandatory_future_argument: "unknown",
            lambda *, spec: "unknown",
        ):
            with self.subTest(resolver=resolver):
                with self.assertRaisesRegex(
                    UnsupportedRuntimeError,
                    "KV cache semantic-kind resolver",
                ):
                    self._verify(spec_kind_resolver=resolver)

    def test_required_contract_drift_is_rejected(self) -> None:
        class IncompatibleBase(DummyBase):
            def __init__(
                self,
                vllm_config,
                role,
                kv_cache_config,
                mandatory_future_argument,
            ):
                super().__init__(vllm_config, role, kv_cache_config)

        with self.assertRaisesRegex(UnsupportedRuntimeError, "required parameter"):
            self._verify(connector_base=IncompatibleBase)

        class FutureAbstractConnector(DummyConnector, abc.ABC):
            @abc.abstractmethod
            def future_vllm_hook(self):
                raise NotImplementedError

        with self.assertRaisesRegex(UnsupportedRuntimeError, "future_vllm_hook"):
            self._verify(connector_type=FutureAbstractConnector)

    def test_all_hooks_validate_shape_and_override_substitutability(self) -> None:
        class CompatibleKeywordExtension(DummyBase):
            def start_load_kv(
                self, forward_context, *, future_option=None, **kwargs
            ):
                pass

        class CompatibleConnector(DummyConnector):
            def start_load_kv(self, forward_context, **kwargs):
                pass

        receipt = self._verify(
            connector_base=CompatibleKeywordExtension,
            connector_type=CompatibleConnector,
        )
        self.assertIn("start_load_kv", receipt.capabilities)

        class RemovedParameter(DummyBase):
            def save_kv_layer(self, layer_name, kv_layer, **kwargs):
                pass

        class ReorderedParameters(DummyBase):
            def update_state_after_alloc(
                self, blocks, request, num_external_tokens
            ):
                pass

        class MandatoryExtension(DummyBase):
            def get_finished(self, finished_req_ids, mandatory_future_filter):
                pass

        class KeywordOnlyDrift(DummyBase):
            def build_connector_meta(self, *, scheduler_output):
                pass

        class RemovedVariadicKeyword(DummyBase):
            def start_load_kv(self, forward_context):
                pass

        for drifted_base, expected_message in (
            (RemovedParameter, "save_kv_layer contract differs"),
            (ReorderedParameters, "update_state_after_alloc contract differs"),
            (MandatoryExtension, "get_finished adds a required parameter"),
            (KeywordOnlyDrift, "build_connector_meta contract differs"),
            (RemovedVariadicKeyword, "start_load_kv contract differs"),
        ):
            with self.subTest(drifted_base=drifted_base.__name__):
                with self.assertRaisesRegex(
                    UnsupportedRuntimeError, expected_message
                ):
                    self._verify(connector_base=drifted_base)

    def test_pp_aware_handshake_hook_is_required_only_for_pp(self) -> None:
        receipt = self._verify(require_pp_aware=True)
        self.assertIn(
            "set_xfer_handshake_metadata_pp_aware",
            receipt.capabilities,
        )

        class LegacyBase(DummyBase):
            set_xfer_handshake_metadata_pp_aware = None

        class LegacyConnector(DummyConnector):
            set_xfer_handshake_metadata_pp_aware = None

        self._verify(
            connector_base=LegacyBase,
            connector_type=LegacyConnector,
        )
        with self.assertRaisesRegex(UnsupportedRuntimeError, "PP-aware"):
            self._verify(
                connector_base=LegacyBase,
                connector_type=LegacyConnector,
                require_pp_aware=True,
            )

    def test_each_of_the_eighteen_hook_signatures_is_enforced(self) -> None:
        hooks_with_only_self = {
            "wait_for_save",
            "get_block_ids_with_load_errors",
            "get_kv_connector_stats",
            "get_handshake_metadata",
            "shutdown",
        }
        hooks = (
            "register_kv_caches",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "get_num_new_matched_tokens",
            "on_new_request",
            "update_state_after_alloc",
            "build_connector_meta",
            "get_finished",
            "get_block_ids_with_load_errors",
            "get_kv_connector_stats",
            "build_kv_connector_stats",
            "build_prom_metrics",
            "update_connector_output",
            "get_handshake_metadata",
            "set_xfer_handshake_metadata",
            "shutdown",
        )
        self.assertEqual(len(hooks), 18)

        def missing_arguments(self):
            pass

        def adds_required(self, mandatory_future_argument):
            pass

        def drifted_classmethod(cls, mandatory_future_argument):
            pass

        for name in hooks:
            if name in {"build_kv_connector_stats", "build_prom_metrics"}:
                replacement = classmethod(drifted_classmethod)
            elif name in hooks_with_only_self:
                replacement = adds_required
            else:
                replacement = missing_arguments
            drifted_base = type(
                f"Drifted_{name}",
                (DummyBase,),
                {name: replacement},
            )
            with self.subTest(hook=name):
                with self.assertRaisesRegex(UnsupportedRuntimeError, name):
                    self._verify(connector_base=drifted_base)

    def test_optional_and_variadic_base_extensions_require_override_support(self) -> None:
        class OptionalPositional(DummyBase):
            def get_finished(self, finished_req_ids, future_filter=None):
                pass

        class OptionalConstructor(DummyBase):
            def __init__(
                self, vllm_config, role, kv_cache_config, future_option=None
            ):
                super().__init__(vllm_config, role, kv_cache_config)

        class VariadicPositional(DummyBase):
            def get_finished(self, finished_req_ids, *filters):
                pass

        class VariadicKeyword(DummyBase):
            def get_finished(self, finished_req_ids, **filters):
                pass

        for drifted_base in (
            OptionalPositional,
            OptionalConstructor,
            VariadicPositional,
            VariadicKeyword,
        ):
            with self.subTest(drifted_base=drifted_base.__name__):
                with self.assertRaisesRegex(
                    UnsupportedRuntimeError, "cannot accept vLLM call shape"
                ):
                    self._verify(connector_base=drifted_base)

        class VariadicBase(DummyBase):
            def get_finished(self, finished_req_ids, *filters, **options):
                pass

        class VariadicConnector(DummyConnector):
            def get_finished(self, finished_req_ids, *filters, **options):
                pass

        self._verify(
            connector_base=VariadicBase,
            connector_type=VariadicConnector,
        )

        class OrderedOptionalBase(DummyBase):
            def get_finished(self, finished_req_ids, first=None, second=None):
                pass

        class ReorderedOptionalConnector(DummyConnector):
            def get_finished(self, finished_req_ids, second=None, first=None):
                pass

        with self.assertRaisesRegex(
            UnsupportedRuntimeError, "reorders vLLM positional parameters"
        ):
            self._verify(
                connector_base=OrderedOptionalBase,
                connector_type=ReorderedOptionalConnector,
            )

    def test_hma_override_must_accept_the_public_mixin_contract(self) -> None:
        class ExtendedHMA(DummySupportsHMA):
            def request_finished_all_groups(
                self, request, block_ids, *, completion_epoch=None
            ):
                pass

        with self.assertRaisesRegex(
            UnsupportedRuntimeError, "cannot accept vLLM call shape"
        ):
            self._verify(supports_hma=ExtendedHMA)

        class ExtendedConnector(DummyConnector):
            def request_finished_all_groups(
                self, request, block_ids, *, completion_epoch=None
            ):
                pass

        self._verify(
            supports_hma=ExtendedHMA,
            connector_type=ExtendedConnector,
        )

    def test_keyword_only_required_hook_parameter_is_rejected(self) -> None:
        class KeywordOnlyRequired(DummyBase):
            def get_finished(self, finished_req_ids, *, required_filter):
                pass

        with self.assertRaisesRegex(UnsupportedRuntimeError, "required parameter"):
            self._verify(connector_base=KeywordOnlyRequired)

    def test_classmethod_call_shape_is_checked(self) -> None:
        class ExtendedClassmethod(DummyBase):
            @classmethod
            def build_kv_connector_stats(cls, data=None, *, epoch=None):
                pass

        with self.assertRaisesRegex(
            UnsupportedRuntimeError, "cannot accept vLLM call shape"
        ):
            self._verify(connector_base=ExtendedClassmethod)

        class ExtendedConnector(DummyConnector):
            @classmethod
            def build_kv_connector_stats(cls, data=None, *, epoch=None):
                pass

        self._verify(
            connector_base=ExtendedClassmethod,
            connector_type=ExtendedConnector,
        )

    def test_legacy_version_selectors_are_not_part_of_runtime_verification(self) -> None:
        receipt = verify_vllm_runtime(
            vllm_module=self._runtime("0.26.0"),
            connector_base=DummyBase,
            connector_type=DummyConnector,
            supports_hma=DummySupportsHMA,
            spec_kind_resolver=resolve_spec_kind,
            package_roots=(self._runtime_root,),
        )
        self.assertEqual(receipt.vllm_version, "0.26.0")
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            verify_vllm_runtime(
                vllm_module=self._runtime("0.26.0"),
                connector_base=DummyBase,
                connector_type=DummyConnector,
                supports_hma=DummySupportsHMA,
                spec_kind_resolver=resolve_spec_kind,
                package_roots=(self._runtime_root,),
                expected_version=None,  # type: ignore[call-arg]
            )

    def test_allocator_configuration_is_detected_without_mutation(self) -> None:
        environment = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True"
        }
        self.assertTrue(expandable_segments_enabled(environment))
        with self.assertRaisesRegex(UnsupportedRuntimeError, "expandable_segments"):
            require_qualified_allocator(environment)
        self.assertFalse(
            expandable_segments_enabled(
                {"PYTORCH_ALLOC_CONF": "expandable_segments:False"}
            )
        )

    def test_external_connector_enables_data_path_and_preserves_contracts(self) -> None:
        base_module_name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
        stub_vllm = types.ModuleType("vllm")
        stub_vllm.__version__ = "runtime-test"
        stub_vllm.__path__ = [str(self._runtime_root)]  # type: ignore[attr-defined]
        base_module = types.ModuleType(base_module_name)

        class KVConnectorRole(enum.Enum):
            SCHEDULER = 0
            WORKER = 1

        class KVConnectorMetadata:
            pass

        class KVConnectorHandshakeMetadata:
            pass

        @dataclass
        class KVConnectorStats:
            data: dict = field(default_factory=dict)

        class KVConnectorBaseV1(DummyBase, abc.ABC):
            pass

        class SupportsHMA(DummySupportsHMA, abc.ABC):
            pass

        base_module.KVConnectorBase_V1 = KVConnectorBaseV1
        base_module.KVConnectorHandshakeMetadata = KVConnectorHandshakeMetadata
        base_module.KVConnectorMetadata = KVConnectorMetadata
        base_module.KVConnectorRole = KVConnectorRole
        base_module.SupportsHMA = SupportsHMA
        cache_interface_module_name = "vllm.v1.kv_cache_interface"
        cache_interface_module = types.ModuleType(cache_interface_module_name)
        cache_interface_module.get_kv_cache_spec_kind = (
            lambda spec: getattr(spec, "public_kind", "unknown")
        )
        modules = {
            "vllm": stub_vllm,
            "vllm.v1": types.ModuleType("vllm.v1"),
            cache_interface_module_name: cache_interface_module,
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
            base_module_name: base_module,
        }
        metrics_module_name = (
            "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
        )
        metrics_module = types.ModuleType(metrics_module_name)
        metrics_module.KVConnectorStats = KVConnectorStats
        modules[metrics_module_name] = metrics_module
        parallel_state_module_name = "vllm.distributed.parallel_state"
        parallel_groups = {
            "pp": types.SimpleNamespace(world_size=2, rank_in_group=1),
            "tp": types.SimpleNamespace(world_size=2, rank_in_group=1),
            "dcp": types.SimpleNamespace(world_size=1, rank_in_group=0),
        }
        parallel_state_module = types.ModuleType(parallel_state_module_name)
        parallel_state_module.get_pp_group = lambda: parallel_groups["pp"]
        parallel_state_module.get_tp_group = lambda: parallel_groups["tp"]
        parallel_state_module.get_dcp_group = lambda: parallel_groups["dcp"]
        modules[parallel_state_module_name] = parallel_state_module

        class TransferConfig:
            kv_connector_extra_config = {
                "spoolcache_path": "/tmp/spoolcache-contract-test",
            }

        vllm_config = types.SimpleNamespace(
            compute_hash=lambda: "vllm-config-test",
            kv_transfer_config=TransferConfig(),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_size=1,
                data_parallel_rank=0,
                world_size=2,
                world_size_across_dp=2,
            ),
            model_config=types.SimpleNamespace(
                model="model-test",
                model_weights="",
                revision="revision",
                served_model_name="model-test",
                is_multimodal_model=False,
                hf_config=types.SimpleNamespace(to_dict=lambda: {"layers": 170}),
            ),
            cache_config=types.SimpleNamespace(cache_dtype="nvfp4_ds_mla"),
        )
        with patch.dict(sys.modules, modules):
            sys.modules.pop("spoolcache.vllm.connector", None)
            connector_module = importlib.import_module("spoolcache.vllm.connector")
            # Per-step metadata transports plans through vLLM. It has no
            # independent schema negotiation or persisted representation.
            plan = connector_module.SpoolCachePlan(
                request_id="metadata-roundtrip", entry_id="e" * 64,
                span_tokens=1024, block_ids_by_group=((0, 2), (3,)),
            )
            metadata = connector_module.SpoolCacheMetadata(loads=(plan,), stores=(plan,))
            self.assertEqual(pickle.loads(pickle.dumps(metadata)), metadata)
            with self.assertRaises(TypeError):
                connector_module.SpoolCacheMetadata(schema="unused")
            get_spec_kind_resolver = connector_module._get_public_spec_kind_resolver
            self.assertIs(
                get_spec_kind_resolver(),
                cache_interface_module.get_kv_cache_spec_kind,
            )
            del cache_interface_module.get_kv_cache_spec_kind
            with self.assertRaisesRegex(
                UnsupportedRuntimeError,
                "lacks the public KV cache semantic-kind resolver",
            ):
                get_spec_kind_resolver()
            cache_interface_module.get_kv_cache_spec_kind = None
            with self.assertRaisesRegex(
                UnsupportedRuntimeError,
                "semantic-kind resolver is not callable",
            ):
                get_spec_kind_resolver()
            cache_interface_module.get_kv_cache_spec_kind = (
                lambda spec: getattr(spec, "public_kind", "unknown")
            )
            choose_store_span = connector_module._pre_forward_store_span
            skips_write = connector_module._request_skips_write
            skips_read = connector_module._request_skips_read
            normalize_block_ids = connector_module._normalize_block_ids
            nonnegative_runtime_int = connector_module._nonnegative_runtime_int
            request_cache_salt = connector_module._request_cache_salt
            runtime_worker_coordinate = connector_module._runtime_worker_coordinate
            self.assertEqual(
                normalize_block_ids(((0, 2), [3])),
                ((0, 2), (3,)),
            )
            for invalid in (((True,),), (("1",),), ((-1,),), (None,)):
                with self.subTest(invalid_block_table=invalid):
                    with self.assertRaisesRegex(ValueError, "block"):
                        normalize_block_ids(invalid)
            self.assertEqual(
                nonnegative_runtime_int(0, label="test count"),
                0,
            )
            for invalid_count in (True, -1, "1"):
                with self.subTest(invalid_runtime_count=invalid_count):
                    with self.assertRaisesRegex(RuntimeError, "test count"):
                        nonnegative_runtime_int(
                            invalid_count,
                            label="test count",
                        )
            self.assertEqual(
                request_cache_salt(types.SimpleNamespace(cache_salt="salt")),
                "salt",
            )
            self.assertEqual(request_cache_salt(types.SimpleNamespace()), "")
            for invalid_salt in (True, 1, b"salt", "\ud800"):
                with self.subTest(invalid_cache_salt=repr(invalid_salt)):
                    self.assertIsNone(
                        request_cache_salt(
                            types.SimpleNamespace(cache_salt=invalid_salt)
                        )
                    )
            self.assertTrue(
                skips_write(
                    types.SimpleNamespace(
                        kv_transfer_params={"spoolcache.skip_write": True}
                    )
                )
            )
            self.assertFalse(
                skips_write(
                    types.SimpleNamespace(
                        kv_transfer_params={"spoolcache.skip_write": "true"}
                    )
                )
            )
            self.assertFalse(skips_write(types.SimpleNamespace()))
            self.assertTrue(skips_read(types.SimpleNamespace(
                kv_transfer_params={"spoolcache.skip_read": True}
            )))
            for value in (False, 0, 1, "true", None):
                self.assertFalse(skips_read(types.SimpleNamespace(
                    kv_transfer_params={"spoolcache.skip_read": value}
                )))
            self.assertFalse(skips_read(types.SimpleNamespace()))
            for value in (False, 0, 1, "true", None):
                self.assertFalse(skips_write(types.SimpleNamespace(
                    kv_transfer_params={"spoolcache.skip_write": value}
                )))
            self.assertFalse(skips_write(types.SimpleNamespace(
                kv_transfer_params={"spoolcache_bypass": True}
            )))
            self.assertEqual(
                runtime_worker_coordinate(
                    vllm_config=types.SimpleNamespace(
                        parallel_config=types.SimpleNamespace(rank=3)
                    ),
                    tp_degree=2,
                    pp_degree=2,
                    dcp_degree=1,
                ),
                connector_module.WorkerCoordinate(3, 1, 1, 0),
            )
            parallel_groups["tp"].world_size = 3
            with self.assertRaisesRegex(RuntimeError, "group size"):
                runtime_worker_coordinate(
                    vllm_config=types.SimpleNamespace(
                        parallel_config=types.SimpleNamespace(rank=3)
                    ),
                    tp_degree=2,
                    pp_degree=2,
                    dcp_degree=1,
                )
            parallel_groups["tp"].world_size = 2
            self.assertIsNone(
                choose_store_span(
                    before_tokens=2048,
                    scheduled_tokens=1024,
                    target_span_tokens=4096,
                    quantum_tokens=256,
                    min_span_tokens=1024,
                )
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=30720,
                    scheduled_tokens=919,
                    target_span_tokens=31488,
                    quantum_tokens=256,
                    min_span_tokens=1024,
                ),
                30720,
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=31488,
                    scheduled_tokens=152,
                    target_span_tokens=31488,
                    quantum_tokens=256,
                    min_span_tokens=1024,
                ),
                31488,
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=0,
                    scheduled_tokens=4096,
                    target_span_tokens=3840,
                    quantum_tokens=256,
                    min_span_tokens=1024,
                ),
                0,
            )
            self.assertIsNone(
                choose_store_span(
                    before_tokens=0,
                    scheduled_tokens=6400,
                    target_span_tokens=6400,
                    quantum_tokens=6400,
                    min_span_tokens=1024,
                    require_exact_boundary=True,
                )
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=6400,
                    scheduled_tokens=1,
                    target_span_tokens=6400,
                    quantum_tokens=6400,
                    min_span_tokens=1024,
                    require_exact_boundary=True,
                ),
                6400,
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=8192,
                    scheduled_tokens=8192,
                    target_span_tokens=12800,
                    quantum_tokens=6400,
                    min_span_tokens=1024,
                    require_exact_boundary=True,
                ),
                None,
            )
            self.assertEqual(
                choose_store_span(
                    before_tokens=16384,
                    scheduled_tokens=1,
                    target_span_tokens=12800,
                    quantum_tokens=6400,
                    min_span_tokens=1024,
                    require_exact_boundary=True,
                ),
                0,
            )
            media = types.SimpleNamespace(
                modality="image",
                identifier="sha256:image",
                mm_position=types.SimpleNamespace(offset=32, length=64),
            )
            identities = connector_module._multimodal_feature_identities(
                types.SimpleNamespace(mm_features=[media]),
                256,
                enabled_modalities=frozenset({"image", "video", "audio"}),
            )
            self.assertEqual(
                identities,
                (
                    connector_module.MultimodalFeatureIdentity(
                        "image", "sha256:image", 32, 64
                    ),
                ),
            )
            self.assertIsNone(
                connector_module._multimodal_feature_identities(
                    types.SimpleNamespace(mm_features=[object()]),
                    256,
                    enabled_modalities=frozenset({"image", "video", "audio"}),
                )
            )
            audio = types.SimpleNamespace(
                modality="audio",
                identifier="sha256:audio",
                mm_position=types.SimpleNamespace(offset=32, length=64),
            )
            self.assertEqual(
                connector_module._multimodal_feature_identities(
                    types.SimpleNamespace(mm_features=[audio]),
                    256,
                    enabled_modalities=frozenset({"image", "video", "audio"}),
                ),
                (
                    connector_module.MultimodalFeatureIdentity(
                        "audio", "sha256:audio", 32, 64
                    ),
                ),
            )
            video = types.SimpleNamespace(
                modality="video",
                identifier="sha256:video",
                mm_position=types.SimpleNamespace(offset=96, length=96),
            )
            self.assertEqual(
                connector_module._multimodal_feature_identities(
                    types.SimpleNamespace(mm_features=[video]),
                    256,
                    enabled_modalities=frozenset({"image", "video"}),
                ),
                (
                    connector_module.MultimodalFeatureIdentity(
                        "video", "sha256:video", 96, 96
                    ),
                ),
            )
            self.assertIsNone(
                connector_module._multimodal_feature_identities(
                    types.SimpleNamespace(mm_features=[audio]),
                    256,
                    enabled_modalities=frozenset({"image", "video"}),
                )
            )

            class MultiModalConfig:
                limits = {
                    "image": 4,
                    "video": 1,
                    "audio": 0,
                    "depth": 2,
                }

                def get_limit_per_prompt(self, modality):
                    return self.limits[modality]

            multimodal_model_config = types.SimpleNamespace(
                is_multimodal_model=True,
                get_multimodal_config=lambda: MultiModalConfig(),
            )

            class Registry:
                def supports_multimodal_inputs(self, model_config):
                    return model_config is multimodal_model_config

                def get_processing_info(self, model_config):
                    self.requested_model_config = model_config
                    return types.SimpleNamespace(
                        supported_mm_limits={
                            "image": None,
                            "video": 1,
                            "audio": None,
                            "depth": None,
                            "point_cloud": 0,
                        }
                    )

            registry = Registry()
            self.assertEqual(
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(model_config=multimodal_model_config),
                    registry=registry,
                ),
                frozenset({"image", "video", "depth"}),
            )
            self.assertIs(registry.requested_model_config, multimodal_model_config)
            with self.assertRaisesRegex(
                connector_module.UnsupportedRuntimeError,
                "invalid multimodal input name",
            ):
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(model_config=multimodal_model_config),
                    registry=types.SimpleNamespace(
                        supports_multimodal_inputs=lambda _config: True,
                        get_processing_info=lambda _config: types.SimpleNamespace(
                            supported_mm_limits={"": 1}
                        ),
                    ),
                )
            with self.assertRaisesRegex(
                connector_module.UnsupportedRuntimeError,
                "capability discovery contract differs",
            ):
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(model_config=multimodal_model_config),
                    registry=types.SimpleNamespace(
                        supports_multimodal_inputs=lambda _config: True,
                    ),
                )
            with self.assertRaisesRegex(
                connector_module.UnsupportedRuntimeError,
                "multimodal model flag is not boolean",
            ):
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(
                        model_config=types.SimpleNamespace(
                            is_multimodal_model="false"
                        )
                    ),
                    registry=registry,
                )
            with self.assertRaisesRegex(
                connector_module.UnsupportedRuntimeError,
                "registry support result is not boolean",
            ):
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(model_config=multimodal_model_config),
                    registry=types.SimpleNamespace(
                        supports_multimodal_inputs=lambda _config: 1,
                        get_processing_info=lambda _config: types.SimpleNamespace(
                            supported_mm_limits={"image": 1}
                        ),
                    ),
                )
            self.assertEqual(
                connector_module._discover_multimodal_modalities(
                    types.SimpleNamespace(
                        model_config=types.SimpleNamespace(
                            is_multimodal_model=False
                        )
                    ),
                    registry=object(),
                ),
                frozenset(),
            )
            with patch.object(connector_module, "require_qualified_allocator") as allocator_gate:
                connector = connector_module.SpoolCacheConnector(
                    vllm_config, KVConnectorRole.SCHEDULER, cache_config()
                )
                allocator_gate.assert_called_once_with()
            store_plan = connector_module.SpoolCachePlan(
                request_id="default-store", entry_id="a" * 64,
                span_tokens=1024, block_ids_by_group=(),
            )
            with patch.object(connector, "_track_new_requests", return_value=[store_plan]), patch.object(
                connector, "_track_cached_requests", return_value=[]
            ):
                metadata = connector.build_connector_meta(
                    types.SimpleNamespace(finished_req_ids=())
                )
                self.assertEqual(metadata.stores, (store_plan,))
            self.assertIsInstance(connector, SupportsHMA)
            self.assertEqual(connector._multimodal_modalities, frozenset())
            initial_stats = connector.get_kv_connector_stats()
            self.assertIsNotNone(initial_stats)
            initial_gauges = aggregate_gauges(initial_stats.data)
            self.assertEqual(
                initial_gauges[("spoolcache_required_ranks", ())], 2
            )
            self.assertEqual(initial_gauges[("spoolcache_ready_ranks", ())], 0)
            self.assertEqual(
                initial_gauges[
                    ("spoolcache_readiness", ("rank_identity",))
                ],
                0,
            )

            pp_config = types.SimpleNamespace(**vars(vllm_config))
            pp_config.parallel_config = types.SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                decode_context_parallel_size=1,
                data_parallel_size=1,
                data_parallel_rank=0,
                world_size=4,
                world_size_across_dp=4,
            )
            pp_connector = connector_module.SpoolCacheConnector(
                pp_config,
                KVConnectorRole.SCHEDULER,
                cache_config(),
            )
            self.assertEqual(
                pp_connector._catalog.expected_ranks,
                frozenset(range(4)),
            )
            pp_entry = "9" * 64
            pp_handshake = {
                (pp_rank, tp_rank): connector_module.SpoolCacheHandshakeMetadata(
                    inventories=(
                        connector_module.SpoolCacheStartupInventory(
                            rank=pp_rank * 2 + tp_rank,
                            coordination_digest=pp_connector._coordination_digest,
                            generation=f"pp{pp_rank}-tp{tp_rank}",
                            generation_epoch=pp_rank * 2 + tp_rank + 1,
                            entries=((pp_entry, 1024),),
                        ),
                    )
                )
                for pp_rank in range(2)
                for tp_rank in range(2)
            }
            with self.assertRaisesRegex(RuntimeError, "missing.*required"):
                pp_connector.set_xfer_handshake_metadata_pp_aware(
                    {key: value for key, value in pp_handshake.items() if key[0] == 0}
                )
            wrong_stage_rank = dict(pp_handshake)
            wrong_stage_rank[(1, 0)] = connector_module.SpoolCacheHandshakeMetadata(
                inventories=(
                    connector_module.SpoolCacheStartupInventory(
                        rank=0,
                        coordination_digest=pp_connector._coordination_digest,
                        generation="duplicate-global-rank",
                        generation_epoch=10,
                        entries=((pp_entry, 1024),),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "transport.*rank"):
                pp_connector.set_xfer_handshake_metadata_pp_aware(wrong_stage_rank)
            with self.assertRaisesRegex(RuntimeError, "PP-aware"):
                pp_connector.set_xfer_handshake_metadata(
                    {rank: value for rank, value in enumerate(pp_handshake.values())}
                )
            pp_connector.set_xfer_handshake_metadata_pp_aware(pp_handshake)
            self.assertTrue(pp_connector._catalog.has_quorum(pp_entry, 1024))
            pp_connector.update_connector_output(
                types.SimpleNamespace(
                    kv_connector_stats=connector_module.SpoolCacheStats(
                        reports=(
                            WorkerInventoryReport(
                                rank=2,
                                generation="pp1-tp0-restarted",
                                generation_epoch=20,
                                checkpoint=InventoryCheckpoint(
                                    sequence=0,
                                    cycle=1,
                                    index=0,
                                    count=1,
                                    held_count=0,
                                    entries=(),
                                ),
                            ),
                        )
                    )
                )
            )
            self.assertFalse(pp_connector._catalog.has_quorum(pp_entry, 1024))

            scanned_limits: list[int] = []

            class CatalogProbe:
                @contextlib.contextmanager
                def _exclusive(self):
                    yield

                def scan_offers(self, limit: int):
                    scanned_limits.append(limit)
                    return ()

            self.assertEqual(
                connector_module._scan_worker_catalog(CatalogProbe()),
                (),
            )
            self.assertEqual(
                scanned_limits,
                [connector_module.CATALOG_MAX_ENTRIES],
            )
            self.assertGreater(
                connector_module.CATALOG_MAX_ENTRIES,
                connector_module.STARTUP_MAX_DIGESTS,
            )

            held_marker_entry = "1" * 64
            absent_marker_entry = "2" * 64
            fenced_object = "3" * 64

            class MarkerReconciliationStore:
                def __init__(self) -> None:
                    self.events: list[object] = []

                @contextlib.contextmanager
                def _exclusive(self):
                    self.events.append("lock")
                    yield

                def pending_inventory_withdrawals(self, entry_ids):
                    self.events.append(("pending", tuple(entry_ids)))
                    return (held_marker_entry,)

                def inventory_withdrawal_marker_batch(self, limit, *, after=None):
                    self.events.append(("entry-batch", limit, after))
                    return (absent_marker_entry,), absent_marker_entry

                def object_withdrawal_marker_batch(self, limit, *, after=None):
                    self.events.append(("object-batch", limit, after))
                    return (fenced_object,), fenced_object

                def acknowledge_absent_inventory_withdrawals(self, entry_ids):
                    self.events.append(("ack-entry", tuple(entry_ids)))

                def acknowledge_unreferenced_object_withdrawals(self, digests):
                    self.events.append(("ack-object", tuple(digests)))

            marker_store = MarkerReconciliationStore()
            marker_reporter = InventoryReporter(
                rank=0,
                generation="marker-worker",
                generation_epoch=1,
                max_entries=connector_module.CATALOG_MAX_ENTRIES,
                max_report_entries=connector_module.REPORT_BATCH_SIZE,
            )
            marker_reporter.replace({held_marker_entry: 256})
            marker_reporter.startup(8)
            marker_view = types.SimpleNamespace(
                _role=KVConnectorRole.WORKER,
                _reporter=marker_reporter,
                _store=marker_store,
                _inventory_marker_cursor=None,
                _object_marker_cursor=None,
                _telemetry=TelemetryBuffer(source="rank:0"),
                _drain_scrub_maintenance=lambda: None,
                _refresh_worker_metrics=lambda: None,
            )
            marker_stats = connector_module.SpoolCacheConnector.get_kv_connector_stats(
                marker_view
            )
            self.assertEqual(
                marker_stats.reports[0].delta.removed,
                (held_marker_entry,),
            )
            self.assertIn(
                ("ack-entry", (absent_marker_entry,)),
                marker_store.events,
            )
            self.assertIn(
                ("ack-object", (fenced_object,)),
                marker_store.events,
            )
            self.assertEqual(
                marker_view._inventory_marker_cursor,
                absent_marker_entry,
            )
            self.assertEqual(marker_view._object_marker_cursor, fenced_object)

            class StartupCapacityProbe:
                def __init__(self, *, usage: int, fail: bool = False) -> None:
                    self.usage = usage
                    self.fail = fail
                    self.events: list[str] = []

                @contextlib.contextmanager
                def _exclusive(self):
                    self.events.append("lock")
                    yield

                def disk_usage_bytes(self):
                    self.events.append("usage")
                    return self.usage

                def maintain_capacity(self, *, max_bytes, low_watermark_bytes):
                    self.events.append(f"capacity:{max_bytes}:{low_watermark_bytes}")
                    if self.fail:
                        raise OSError("startup capacity failed")
                    self.usage = low_watermark_bytes

                def scan_offers(self, limit):
                    self.events.append(f"scan:{limit}")
                    return ()

            overfull = StartupCapacityProbe(usage=101)
            self.assertEqual(
                connector_module._prepare_worker_catalog(
                    overfull,
                    max_bytes=100,
                    low_watermark_bytes=90,
                ),
                (),
            )
            self.assertEqual(
                overfull.events,
                [
                    "lock",
                    "usage",
                    "capacity:100:90",
                    f"scan:{connector_module.CATALOG_MAX_ENTRIES}",
                ],
            )
            failed_startup = StartupCapacityProbe(usage=101, fail=True)
            with self.assertRaisesRegex(OSError, "startup capacity failed"):
                connector_module._prepare_worker_catalog(
                    failed_startup,
                    max_bytes=100,
                    low_watermark_bytes=90,
                )
            self.assertFalse(
                any(event.startswith("scan:") for event in failed_startup.events)
            )

            class FailedRescanScheduler:
                def __init__(self) -> None:
                    self.acknowledged: list[int] = []

                def pending_inventory_rescan_epoch(self):
                    return 1

                def inventory_rescan_requires_withdrawal(self, _epoch: int):
                    return False

                def acknowledge_inventory_rescan(self, epoch: int) -> None:
                    self.acknowledged.append(epoch)

                def drain_metrics(self):
                    return {}

            class RecordingReporter:
                def __init__(self) -> None:
                    self.replacements: list[dict[str, int]] = []

                def replace(self, entries):
                    self.replacements.append(dict(entries))

            class FailedRescanStore:
                @contextlib.contextmanager
                def _exclusive(self):
                    yield

                def scan_offers(self, _limit: int):
                    raise OSError("simulated inventory read failure")

            failed_scheduler = FailedRescanScheduler()
            recording_reporter = RecordingReporter()
            maintenance_view = types.SimpleNamespace(
                _scrub_scheduler=failed_scheduler,
                _telemetry=TelemetryBuffer(source="rank:0"),
                _reporter=recording_reporter,
                _store=FailedRescanStore(),
                _inventory_withdrawn_epoch=0,
            )
            with patch.object(connector_module.logger, "exception"):
                connector_module.SpoolCacheConnector._drain_scrub_maintenance(
                    maintenance_view
                )
            self.assertEqual(recording_reporter.replacements, [{}])
            self.assertEqual(failed_scheduler.acknowledged, [])

            class RescanScheduler:
                def __init__(self, *, force: bool = False) -> None:
                    self.force = force
                    self.acknowledged: list[int] = []

                def pending_inventory_rescan_epoch(self):
                    return 1

                def inventory_rescan_requires_withdrawal(self, _epoch: int):
                    return self.force

                def acknowledge_inventory_rescan(self, epoch: int) -> None:
                    self.acknowledged.append(epoch)

                def drain_metrics(self):
                    return {}

            entry = "f" * 64
            scan_started = threading.Event()
            mutation_attempted = threading.Event()
            mutation_finished = threading.Event()

            class CoordinatedStore:
                def __init__(self) -> None:
                    self.lock = threading.Lock()
                    self.owner: int | None = None

                @contextlib.contextmanager
                def _exclusive(self):
                    with self.lock:
                        self.owner = threading.get_ident()
                        try:
                            yield
                        finally:
                            self.owner = None

                def lock_is_owned(self) -> bool:
                    return self.owner == threading.get_ident()

                def scan_offers(self, _limit: int):
                    scan_started.set()
                    if not mutation_attempted.wait(5):
                        raise AssertionError("rescan mutation did not start")
                    return (types.SimpleNamespace(entry_id=entry, span_tokens=256),)

            coordinated_store = CoordinatedStore()

            class CoordinatedReporter(InventoryReporter):
                def replace(self, entries):
                    # On the buggy path replace happened outside the store
                    # lock. Waiting for the mutator then deterministically
                    # reproduced stale re-addition. The fixed path owns the
                    # lock, publishes first, and lets the later remove win.
                    if scan_started.is_set() and not coordinated_store.lock_is_owned():
                        if not mutation_finished.wait(5):
                            raise AssertionError("rescan mutation did not finish")
                    super().replace(entries)

            coordinated_reporter = CoordinatedReporter(
                rank=0,
                generation="boot",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            coordinated_reporter.replace({entry: 256})
            coordinated_reporter.startup(8)
            rescan_scheduler = RescanScheduler()
            coordinated_view = types.SimpleNamespace(
                _scrub_scheduler=rescan_scheduler,
                _telemetry=TelemetryBuffer(source="rank:0"),
                _reporter=coordinated_reporter,
                _store=coordinated_store,
                _inventory_withdrawn_epoch=0,
            )

            def mutate_after_scan() -> None:
                self.assertTrue(scan_started.wait(5))
                mutation_attempted.set()
                with coordinated_store._exclusive():
                    coordinated_reporter.remove(entry)
                mutation_finished.set()

            mutator = threading.Thread(target=mutate_after_scan)
            mutator.start()
            connector_module.SpoolCacheConnector._drain_scrub_maintenance(
                coordinated_view
            )
            mutator.join(5)
            self.assertFalse(mutator.is_alive())
            report = coordinated_reporter.next_report(8)
            self.assertIsNotNone(report.delta)
            self.assertIn(entry, report.delta.removed)
            self.assertEqual(rescan_scheduler.acknowledged, [1])

            published_entry = "e" * 64
            publication_complete = threading.Event()
            publication_mutation_attempted = threading.Event()
            publication_mutation_finished = threading.Event()
            publication_events: list[str] = []

            class PublicationStore:
                def __init__(self) -> None:
                    self.lock = threading.Lock()
                    self.owner: int | None = None

                @contextlib.contextmanager
                def _exclusive(self):
                    with self.lock:
                        self.owner = threading.get_ident()
                        try:
                            yield
                        finally:
                            self.owner = None

                def lock_is_owned(self) -> bool:
                    return self.owner == threading.get_ident()

            publication_store = PublicationStore()

            class PublicationMover:
                def commit(self, _store, **_kwargs):
                    publication_complete.set()
                    if not publication_mutation_attempted.wait(5):
                        raise AssertionError("publication mutation did not start")
                    return types.SimpleNamespace(
                        objects=(types.SimpleNamespace(byte_length=4096),)
                    )

            class PublicationReporter(InventoryReporter):
                def add(self, entry_id, span_tokens):
                    # Before the fix, add ran after the store lock was released.
                    # Let the quarantine/remove operation finish first to make
                    # the stale re-add deterministic on that buggy ordering.
                    if not publication_store.lock_is_owned():
                        if not publication_mutation_finished.wait(5):
                            raise AssertionError(
                                "publication mutation did not finish"
                            )
                    super().add(entry_id, span_tokens)
                    publication_events.append("add")

                def remove(self, entry_id):
                    super().remove(entry_id)
                    publication_events.append("remove")

            publication_reporter = PublicationReporter(
                rank=0,
                generation="publication-race",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            publication_reporter.startup(8)
            publication_view = types.SimpleNamespace(
                _worker_data_path=lambda: (PublicationMover(), publication_store),
                deployment_identity=types.SimpleNamespace(digest="a" * 64),
                _rank_identity=types.SimpleNamespace(digest="b" * 64),
                _physical_rank=0,
                _topology_digest="c" * 64,
                _reporter=publication_reporter,
                _telemetry=None,
                _maintain_capacity=lambda: None,
                _refresh_worker_metrics=lambda *, force_disk: None,
            )

            def quarantine_after_publication() -> None:
                self.assertTrue(publication_complete.wait(5))
                publication_mutation_attempted.set()
                with publication_store._exclusive():
                    publication_reporter.remove(published_entry)
                publication_mutation_finished.set()

            publication_mutator = threading.Thread(
                target=quarantine_after_publication
            )
            publication_mutator.start()
            connector_module.SpoolCacheConnector._commit_store_plans(
                publication_view,
                (
                    connector_module.SpoolCachePlan(
                        request_id="publication-race",
                        entry_id=published_entry,
                        span_tokens=256,
                        block_ids_by_group=((0,),),
                    ),
                ),
            )
            publication_mutator.join(5)
            self.assertFalse(publication_mutator.is_alive())
            self.assertEqual(publication_events, ["add", "remove"])
            publication_report = publication_reporter.next_report(8)
            self.assertEqual(publication_report.checkpoint.entries, ())
            self.assertIsNone(publication_report.delta)

            forced_scheduler = RescanScheduler(force=True)
            forced_reporter = InventoryReporter(
                rank=0,
                generation="boot",
                generation_epoch=1,
                max_entries=8,
                max_report_entries=8,
            )
            forced_reporter.replace({entry: 256})
            forced_reporter.startup(8)
            forced_view = types.SimpleNamespace(
                _scrub_scheduler=forced_scheduler,
                _telemetry=TelemetryBuffer(source="rank:0"),
                _reporter=forced_reporter,
                _store=CatalogProbe(),
                _inventory_withdrawn_epoch=0,
            )
            connector_module.SpoolCacheConnector._drain_scrub_maintenance(
                forced_view
            )
            forced_empty = forced_reporter.next_report(8)
            self.assertIsNotNone(forced_empty.delta)
            self.assertIn(entry, forced_empty.delta.removed)
            self.assertEqual(forced_scheduler.acknowledged, [])
            connector_module.SpoolCacheConnector._drain_scrub_maintenance(
                forced_view
            )
            self.assertEqual(forced_scheduler.acknowledged, [1])

            class FailedCapacityStore:
                @contextlib.contextmanager
                def _exclusive(self):
                    yield

                def disk_usage_bytes(self):
                    return 101

                def maintain_capacity(self, *, max_bytes, low_watermark_bytes):
                    self.capacity_args = (max_bytes, low_watermark_bytes)
                    raise OSError("capacity directory fsync failed")

            class CapacityRescanScheduler:
                def __init__(self) -> None:
                    self.force_withdrawal: list[bool] = []

                def require_inventory_rescan(self, *, force_withdrawal=False):
                    self.force_withdrawal.append(force_withdrawal)

            failed_capacity_store = FailedCapacityStore()
            capacity_reporter = RecordingReporter()
            capacity_scheduler = CapacityRescanScheduler()
            capacity_view = types.SimpleNamespace(
                _store=failed_capacity_store,
                _reporter=capacity_reporter,
                _scrub_scheduler=capacity_scheduler,
                config=types.SimpleNamespace(
                    max_bytes=100,
                    low_watermark_bytes=90,
                ),
            )
            with patch.object(connector_module.logger, "exception"):
                connector_module.SpoolCacheConnector._maintain_capacity(
                    capacity_view
                )
            self.assertEqual(failed_capacity_store.capacity_args, (100, 90))
            self.assertEqual(capacity_reporter.replacements, [{}])
            self.assertEqual(capacity_scheduler.force_withdrawal, [True])

            original_catalog = connector._catalog
            class AlwaysHitCatalog:
                def longest(self, _candidates):
                    return 1024, "e" * 64

            connector._catalog = AlwaysHitCatalog()
            connector._pending_loads = {
                f"active-{index}": connector_module.SpoolCachePlan(
                    request_id=f"active-{index}",
                    entry_id="d" * 64,
                    span_tokens=1024,
                    block_ids_by_group=(),
                )
                for index in range(connector_module.MAX_PENDING_RESTORES)
            }
            waiting_request = types.SimpleNamespace(
                request_id="waiting",
                prompt_token_ids=[1] * 1025,
                prompt_embeds=None,
                lora_request=None,
                cache_salt="",
                mm_features=None,
                mm_hashes=None,
                kv_transfer_params=None,
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(waiting_request, 0),
                (0, False),
            )
            self.assertNotIn("waiting", connector._need_load)
            connector._pending_loads.pop("active-0")
            self.assertEqual(
                connector.get_num_new_matched_tokens(waiting_request, 0),
                (1024, False),
            )
            self.assertIn("waiting", connector._need_load)
            lookup_payload = connector._telemetry.drain()
            lookup_counters = {
                (record.name, record.labels): record.value
                for record in iter_metric_records(
                    lookup_payload, MetricKind.COUNTER
                )
            }
            self.assertEqual(
                lookup_counters[
                    ("spoolcache_lookup_total", ("miss", "restore_budget"))
                ],
                1,
            )
            self.assertEqual(
                lookup_counters[
                    ("spoolcache_lookup_total", ("hit", "ready_entry"))
                ],
                1,
            )
            self.assertEqual(
                lookup_counters[("spoolcache_hit_tokens_total", ())], 1024
            )
            connector._need_load.clear()
            connector._pending_loads.clear()
            connector._catalog = original_catalog

            skip_request = types.SimpleNamespace(
                **{**vars(waiting_request), "request_id": "skip-write",
                   "kv_transfer_params": {"spoolcache.skip_write": True}},
            )
            connector.on_new_request(skip_request)
            with patch.object(connector, "_catalog", AlwaysHitCatalog()):
                self.assertEqual(connector.get_num_new_matched_tokens(skip_request, 0), (1024, False))
            self.assertIn("skip-write", connector._need_load)
            connector._need_load.clear()
            restore_plan = connector_module.SpoolCachePlan(
                request_id="skip-write", entry_id="e" * 64,
                span_tokens=1024, block_ids_by_group=(),
            )
            connector._pending_loads["skip-write"] = restore_plan
            # A skip-write request must retain its restore plan and be skipped
            # before store-side token/page access, even with default config.
            metadata = connector.build_connector_meta(types.SimpleNamespace(
                finished_req_ids=(),
                scheduled_new_reqs=[types.SimpleNamespace(req_id="skip-write")],
                scheduled_cached_reqs=types.SimpleNamespace(req_ids=()),
            ))
            self.assertEqual(metadata.loads, (restore_plan,))
            self.assertEqual(metadata.stores, ())
            connector.build_connector_meta(types.SimpleNamespace(
                finished_req_ids=("skip-write",), scheduled_new_reqs=(),
                scheduled_cached_reqs=types.SimpleNamespace(req_ids=()),
            ))
            self.assertNotIn("skip-write", connector._skip_write_requests)

            for skip_write in (False, True):
                with self.subTest(skip_read=True, skip_write=skip_write):
                    read_request = types.SimpleNamespace(
                        **{**vars(waiting_request), "request_id": "skip-read",
                           "kv_transfer_params": {
                               "spoolcache.skip_read": True,
                               "spoolcache.skip_write": skip_write,
                           }},
                    )
                    connector.on_new_request(read_request)
                    with patch.object(connector, "_catalog") as catalog:
                        catalog.longest.return_value = (1024, "e" * 64)
                        catalog.has_quorum.return_value = False
                        self.assertEqual(
                            connector.get_num_new_matched_tokens(read_request, 0),
                            (0, False),
                        )
                        catalog.longest.assert_not_called()
                        self.assertNotIn("skip-read", connector._need_load)
                        new_request = types.SimpleNamespace(
                            req_id="skip-read", prompt_token_ids=[1] * 1025,
                            block_ids=(), num_computed_tokens=1024,
                        )
                        with patch.object(connector, "_plan_from_progress", return_value=store_plan):
                            result = connector.build_connector_meta(types.SimpleNamespace(
                                finished_req_ids=(), scheduled_new_reqs=[new_request],
                                scheduled_cached_reqs=types.SimpleNamespace(req_ids=()),
                                num_scheduled_tokens={"skip-read": 1},
                            ))
                        self.assertEqual(result.loads, ())
                        self.assertEqual(result.stores, () if skip_write else (store_plan,))
                    connector.build_connector_meta(types.SimpleNamespace(
                        finished_req_ids=("skip-read",), scheduled_new_reqs=(),
                        scheduled_cached_reqs=types.SimpleNamespace(req_ids=()),
                    ))
                    self.assertNotIn("skip-read", connector._skip_write_requests)

            # The live vLLM runtime materializes ModelConfig differently in
            # scheduler and worker processes.  That role-local difference must
            # isolate their manifests, but must not make the startup quorum
            # handshake fail when all cross-role facts still agree.
            materialized_config = types.SimpleNamespace(
                compute_hash=lambda: "vllm-config-test-materialized",
                kv_transfer_config=vllm_config.kv_transfer_config,
                parallel_config=vllm_config.parallel_config,
                model_config=types.SimpleNamespace(
                    model="/tmp/materialized-model",
                    model_weights="model-test",
                    revision="revision",
                    served_model_name="model-test",
                    is_multimodal_model=False,
                    hf_config=types.SimpleNamespace(
                        to_dict=lambda: {"layers": 170, "materialized": True}
                    ),
                ),
                # Backends may specialize the worker's physical dtype label
                # (for example fp8 -> fp8_ds_mla) after the scheduler config is
                # frozen. This remains bound into each role-local deployment
                # and rank manifest, but cannot be a cross-role handshake fact.
                cache_config=types.SimpleNamespace(cache_dtype="fp8_ds_mla"),
            )
            materialized = connector_module.SpoolCacheConnector(
                materialized_config, KVConnectorRole.SCHEDULER, cache_config()
            )
            self.assertNotEqual(
                connector.deployment_identity.digest,
                materialized.deployment_identity.digest,
            )
            self.assertEqual(
                connector._coordination_digest,
                materialized._coordination_digest,
            )
            self.assertEqual(
                connector._model_namespace_sha256,
                materialized._model_namespace_sha256,
            )

            alias_values = dict(vars(vllm_config.model_config))
            alias_values["served_model_name"] = "another-public-alias"
            alias_config = types.SimpleNamespace(
                **{
                    **vars(vllm_config),
                    "model_config": types.SimpleNamespace(**alias_values),
                }
            )
            alias_connector = connector_module.SpoolCacheConnector(
                alias_config, KVConnectorRole.SCHEDULER, cache_config()
            )
            self.assertEqual(
                connector.deployment_identity.digest,
                alias_connector.deployment_identity.digest,
            )
            self.assertEqual(
                connector._coordination_digest,
                alias_connector._coordination_digest,
            )

            revision_values = dict(vars(vllm_config.model_config))
            revision_values["revision"] = "different-revision"
            revision_config = types.SimpleNamespace(
                **{
                    **vars(vllm_config),
                    "model_config": types.SimpleNamespace(**revision_values),
                }
            )
            revision_connector = connector_module.SpoolCacheConnector(
                revision_config, KVConnectorRole.SCHEDULER, cache_config()
            )
            self.assertNotEqual(
                connector._model_namespace_sha256,
                revision_connector._model_namespace_sha256,
            )
            self.assertNotEqual(
                connector._coordination_digest,
                revision_connector._coordination_digest,
            )

            locator_values = dict(vars(vllm_config.model_config))
            locator_values["model"] = "different-model-locator"
            locator_config = types.SimpleNamespace(
                **{
                    **vars(vllm_config),
                    "model_config": types.SimpleNamespace(**locator_values),
                }
            )
            locator_connector = connector_module.SpoolCacheConnector(
                locator_config, KVConnectorRole.SCHEDULER, cache_config()
            )
            self.assertNotEqual(
                connector._model_namespace_sha256,
                locator_connector._model_namespace_sha256,
            )
            self.assertNotEqual(
                connector._coordination_digest,
                locator_connector._coordination_digest,
            )

            namespaced_transfer = types.SimpleNamespace(
                kv_connector_extra_config={
                    **TransferConfig.kv_connector_extra_config,
                    "spoolcache_deployment_namespace": "different-tenant",
                }
            )
            namespaced_config = types.SimpleNamespace(
                **{
                    **vars(vllm_config),
                    "kv_transfer_config": namespaced_transfer,
                }
            )
            with self.assertRaisesRegex(ConfigurationError, "unknown SpoolCache setting"):
                connector_module.SpoolCacheConnector(
                    namespaced_config, KVConnectorRole.SCHEDULER, cache_config()
                )

            def identity_with(
                *, processor_size: int, attention_backend: str
            ):
                model_values = dict(vars(vllm_config.model_config))
                model_values["multimodal_config"] = types.SimpleNamespace(
                    mm_processor_kwargs={"image_size": processor_size},
                    media_io_kwargs={"num_frames": 8},
                )
                identity_config = types.SimpleNamespace(
                    **{
                        **vars(vllm_config),
                        "model_config": types.SimpleNamespace(**model_values),
                        "attention_config": types.SimpleNamespace(
                            backend=attention_backend,
                        ),
                    }
                )
                return connector_module._build_deployment_identity(
                    identity_config,
                    connector.layout,
                    model_namespace_sha256=connector._model_namespace_sha256,
                    chunk_tokens=256,
                    vllm_version="runtime-test",
                    vllm_build_sha256="b" * 64,
                )

            identity_a = identity_with(
                processor_size=448,
                attention_backend="backend-a",
            )
            identity_b = identity_with(
                processor_size=896,
                attention_backend="backend-a",
            )
            identity_c = identity_with(
                processor_size=448,
                attention_backend="backend-b",
            )
            self.assertNotEqual(
                identity_a.execution_config_sha256,
                identity_b.execution_config_sha256,
            )
            self.assertNotEqual(
                identity_a.execution_config_sha256,
                identity_c.execution_config_sha256,
            )
            self.assertEqual(connector.get_num_new_matched_tokens(object(), 0), (0, False))
            self.assertEqual(
                connector.request_finished_all_groups(object(), tuple()), (False, None)
            )
            self.assertTrue(
                connector_module.SpoolCacheStats().is_empty()
            )
            for malformed_stats in ({}, {"schema": "wrong"}):
                with self.subTest(malformed_stats=malformed_stats):
                    with self.assertRaises(ValueError):
                        connector_module.SpoolCacheStats(data=malformed_stats)
                    with self.assertRaises(ValueError):
                        connector_module.SpoolCacheConnector.build_kv_connector_stats(
                            malformed_stats
                        )
            with self.assertRaisesRegex(RuntimeError, "stats type"):
                connector.update_connector_output(
                    types.SimpleNamespace(kv_connector_stats=object())
                )
            with self.assertRaises(TypeError):
                connector_module.SpoolCacheStats(reports=[])
            oversized_report = WorkerInventoryReport(
                rank=0,
                generation="bounded",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(
                    sequence=0,
                    cycle=1,
                    index=0,
                    count=1,
                    held_count=connector_module.REPORT_BATCH_SIZE + 1,
                    entries=(("e" * 64, 1024),)
                    * (connector_module.REPORT_BATCH_SIZE + 1),
                ),
            )
            with self.assertRaisesRegex(ValueError, "checkpoint bounds"):
                connector_module.SpoolCacheStats(reports=(oversized_report,))
            valid_report = WorkerInventoryReport(
                rank=0,
                generation="bounded",
                generation_epoch=1,
                checkpoint=InventoryCheckpoint(
                    sequence=0,
                    cycle=1,
                    index=0,
                    count=1,
                    held_count=0,
                    entries=(),
                ),
            )
            with self.assertRaisesRegex(ValueError, "ranks are duplicated"):
                connector_module.SpoolCacheStats(
                    reports=(valid_report, valid_report)
                )
            duplicate_accumulator = connector_module.SpoolCacheStats(
                reports=(valid_report,)
            )
            with self.assertRaisesRegex(ValueError, "ranks are duplicated"):
                duplicate_accumulator.aggregate(
                    connector_module.SpoolCacheStats(reports=(valid_report,))
                )
            mismatched = connector_module.SpoolCacheHandshakeMetadata(
                inventories=(
                    connector_module.SpoolCacheStartupInventory(
                        rank=0,
                        coordination_digest="f" * 64,
                        generation="worker-generation",
                        generation_epoch=1,
                        entries=(),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "coordination identities differ"):
                connector.set_xfer_handshake_metadata({0: mismatched})
            with self.assertRaisesRegex(RuntimeError, "missing.*required ranks"):
                connector.set_xfer_handshake_metadata({})
            wrong_rank = connector_module.SpoolCacheHandshakeMetadata(
                inventories=(
                    connector_module.SpoolCacheStartupInventory(
                        rank=1,
                        coordination_digest=connector._coordination_digest,
                        generation="wrong-rank",
                        generation_epoch=1,
                        entries=(),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "transport.*rank.*differ"):
                connector.set_xfer_handshake_metadata({0: wrong_rank})
            boolean_rank = connector_module.SpoolCacheHandshakeMetadata(
                inventories=(
                    connector_module.SpoolCacheStartupInventory(
                        rank=False,
                        coordination_digest=connector._coordination_digest,
                        generation="boolean-rank",
                        generation_epoch=1,
                        entries=(),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "transport.*rank.*differ"):
                connector.set_xfer_handshake_metadata({0: boolean_rank})
            oversized = connector_module.SpoolCacheHandshakeMetadata(
                inventories=(
                    connector_module.SpoolCacheStartupInventory(
                        rank=0,
                        coordination_digest=connector._coordination_digest,
                        generation="bounded",
                        generation_epoch=1,
                        entries=(("e" * 64, 1024),)
                        * (connector_module.STARTUP_MAX_DIGESTS + 1),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "exceeds its bound"):
                connector.set_xfer_handshake_metadata({0: oversized})
            shutdown_release = threading.Event()
            shutdown_finalizer_entered = threading.Event()
            shutdown_journal_release = threading.Event()
            shutdown_journal_entered = threading.Event()

            class TimedOutScrubScheduler:
                def __init__(self) -> None:
                    self.calls = 0

                def close(self):
                    self.calls += 1
                    if self.calls == 1:
                        return ScrubShutdownReport(
                            schema="spoolcache-scrub-shutdown/v1",
                            status="timeout",
                            thread_alive=True,
                            waited_seconds=5.0,
                        )
                    shutdown_finalizer_entered.set()
                    shutdown_release.wait(5)
                    return ScrubShutdownReport(
                        schema="spoolcache-scrub-shutdown/v1",
                        status="stopped",
                        thread_alive=False,
                        waited_seconds=0.0,
                    )

            class ShutdownJournal:
                def __init__(self) -> None:
                    self.names: list[str] = []

                def increment(self, name: str) -> None:
                    self.names.append(name)
                    shutdown_journal_entered.set()
                    shutdown_journal_release.wait(5)

            class ShutdownResource:
                def __init__(self) -> None:
                    self.closed = threading.Event()

                def close(self) -> None:
                    self.closed.set()

            timed_out_scheduler = TimedOutScrubScheduler()
            shutdown_journal = ShutdownJournal()
            shutdown_mover = ShutdownResource()
            shutdown_store = ShutdownResource()
            shutdown_view = types.SimpleNamespace(
                _scrub_scheduler=timed_out_scheduler,
                _event_journal=shutdown_journal,
                _mover=shutdown_mover,
                _store=shutdown_store,
            )
            with patch.object(connector_module.logger, "error") as shutdown_log:
                started = time.monotonic()
                connector_module.SpoolCacheConnector.shutdown(shutdown_view)
                self.assertLess(time.monotonic() - started, 0.1)
            self.assertIsNone(shutdown_view._scrub_scheduler)
            self.assertIsNone(shutdown_view._store)
            self.assertIsNone(shutdown_view._mover)
            self.assertTrue(shutdown_journal_entered.wait(1))
            self.assertEqual(
                shutdown_journal.names,
                ["spoolcache_scrub_shutdown_failures_total"],
            )
            self.assertFalse(shutdown_store.closed.is_set())
            shutdown_journal_release.set()
            self.assertTrue(shutdown_finalizer_entered.wait(1))
            self.assertFalse(shutdown_store.closed.is_set())
            receipt = json.loads(shutdown_log.call_args.args[1])
            self.assertEqual(receipt["status"], "timeout")
            self.assertTrue(receipt["thread_alive"])
            shutdown_release.set()
            self.assertTrue(shutdown_store.closed.wait(1))
            self.assertTrue(shutdown_mover.closed.is_set())

            shared_entry = "e" * 64
            matching = {
                rank: connector_module.SpoolCacheHandshakeMetadata(
                    inventories=(
                        connector_module.SpoolCacheStartupInventory(
                            rank=rank,
                            coordination_digest=connector._coordination_digest,
                            generation=f"worker-{rank}",
                            generation_epoch=rank + 1,
                            entries=((shared_entry, 1024),),
                        ),
                    )
                )
                for rank in range(2)
            }
            connector.set_xfer_handshake_metadata(matching)
            self.assertTrue(connector._catalog.has_quorum(shared_entry, 1024))
            ready_stats = connector.get_kv_connector_stats()
            self.assertIsNotNone(ready_stats)
            ready_gauges = aggregate_gauges(ready_stats.data)
            self.assertEqual(ready_gauges[("spoolcache_ready_ranks", ())], 2)
            self.assertEqual(
                ready_gauges[
                    ("spoolcache_readiness", ("inventory_quorum",))
                ],
                1,
            )

            rank0_metrics = TelemetryBuffer(source="rank:0")
            rank1_metrics = TelemetryBuffer(source="rank:1")
            rank0_metrics.increment(
                "spoolcache_restore_bytes_total", value=1024
            )
            rank1_metrics.increment(
                "spoolcache_restore_bytes_total", value=2048
            )
            combined = connector_module.SpoolCacheStats(
                data=rank0_metrics.drain()
            )
            combined.aggregate(
                connector_module.SpoolCacheStats(data=rank1_metrics.drain())
            )
            self.assertEqual(
                combined.reduce()["spoolcache_restore_bytes_total"], 3072
            )
            self.assertIn(
                "build_prom_metrics", connector.runtime_receipt.capabilities
            )
            self.assertEqual(
                connector_module._fatal_restore_phase(
                    connector_module.FatalRestoreError(
                        "SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED"
                    )
                ),
                "payload",
            )


if __name__ == "__main__":
    unittest.main()
