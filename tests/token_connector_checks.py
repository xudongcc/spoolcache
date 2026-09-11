"""Run inside the existing public-vLLM stub scope, without another fake runtime."""

import importlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spoolcache.config import INVENTORY_ENTRY_BYTES, INVENTORY_MEMORY_BYTES, inventory_capacity
from spoolcache.config import SpoolCacheConfig
from spoolcache.prefix import PrefixDigest
from spoolcache.quorum import InventoryReporter, QuorumCatalog
from spoolcache.token_files import TokenFileStore, boundary_state_key
from spoolcache.token_mover import TokenPageMover
from tests.token_fixtures import cpu_mover, layout_for
from tests.test_token_files_hma import mixed_layout


def check_token_connector(test, module):
    token_module = importlib.import_module("spoolcache.vllm.connector")
    connector_type = token_module.SpoolCacheConnector
    check_worker_registration(test, module)
    scheduler = object.__new__(connector_type)
    scheduler.layout = layout_for(tokens_per_page=256, page_bytes=16384)
    scheduler.deployment_identity = SimpleNamespace(digest="b" * 64)
    scheduler._catalog = QuorumCatalog(
        expected_ranks=(0, 1), max_bytes=(100) * INVENTORY_ENTRY_BYTES, max_report_entries=100
    )
    progress = module._StoreProgress(
        tuple(range(1280)), (), "salt", 1280, (tuple(range(1, 21)),)
    )
    plan = scheduler._plan_from_progress("token-fixture", progress, 1280)
    test.assertEqual([p.span_tokens for p in plan.prefixes], [256, 512, 768, 1024])
    keys = (*plan.prefixes, PrefixDigest(plan.span_tokens, plan.entry_id))
    entries = tuple((p.digest, p.span_tokens) for p in keys)
    scheduler._catalog.apply_startup(
        rank=0, generation="a", generation_epoch=1, entries=entries
    )
    test.assertIsNone(scheduler._select_prefix(keys, 256))
    scheduler._catalog.apply_startup(
        rank=1, generation="b", generation_epoch=1, entries=entries[:2] + entries[3:]
    )
    test.assertEqual(scheduler._select_prefix(keys, 256), (512, keys[1].digest))
    test.assertIsNone(scheduler._select_prefix(keys, 768))
    test.assertFalse(scheduler._cached_plan(plan))
    scheduler._catalog.apply_startup(
        rank=1, generation="c", generation_epoch=2, entries=entries
    )
    test.assertTrue(scheduler._cached_plan(plan))
    test.assertEqual(scheduler._select_prefix(keys, 256), (1280, plan.entry_id))
    test.assertEqual(scheduler._safe_restore_span(513), 512)
    # Data-file quorum alone must never advertise an HMA restore. Every rank
    # must independently offer the state belonging to that exact boundary.
    scheduler.layout = mixed_layout(tokens_per_page=256)
    test.assertIsNone(scheduler._select_prefix(keys, 256))
    test.assertFalse(scheduler._cached_plan(plan))
    states = (
        (boundary_state_key(keys[1].digest), 512),
        (boundary_state_key(plan.entry_id), 1280),
    )
    scheduler._catalog.apply_startup(
        rank=0, generation="state-a", generation_epoch=2, entries=entries + states
    )
    test.assertIsNone(scheduler._select_prefix(keys, 256))
    scheduler._catalog.apply_startup(
        rank=1, generation="state-b", generation_epoch=3, entries=entries + states[:1]
    )
    test.assertEqual(scheduler._select_prefix(keys, 256), (512, keys[1].digest))
    test.assertFalse(scheduler._cached_plan(plan))
    scheduler._catalog.apply_startup(
        rank=1, generation="state-c", generation_epoch=4, entries=entries + states
    )
    test.assertEqual(scheduler._select_prefix(keys, 256), (1280, plan.entry_id))
    test.assertTrue(scheduler._cached_plan(plan))
    # Evicting an intermediate data file cuts the prefix even if its later
    # boundary-state offer is still present.
    scheduler._catalog.apply_startup(
        rank=1,
        generation="state-d",
        generation_epoch=5,
        entries=entries[:2] + entries[3:] + states,
    )
    test.assertEqual(scheduler._select_prefix(keys, 256), (512, keys[1].digest))
    scheduler.layout = layout_for(tokens_per_page=256, page_bytes=16384)
    # Store planning after a load must still retain the request's uncaptured
    # extension until its complete pages are ready at a pre-forward boundary.
    scheduler._skip_write_requests = set()
    scheduler._pending_loads = {}
    scheduler._request_salts = {"extension": "salt"}
    scheduler._multimodal_modalities = frozenset()
    scheduler._store_progress = {}
    scheduler._telemetry = None
    scheduler._catalog = None
    request = SimpleNamespace(
        req_id="extension",
        prompt_token_ids=list(range(1280)),
        mm_features=(),
        block_ids=(list(range(1, 21)),),
        num_computed_tokens=512,
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[request], num_scheduled_tokens={"extension": 256}
    )
    test.assertEqual(scheduler._track_new_requests(output), [])
    test.assertIn("extension", scheduler._store_progress)
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["extension"],
        new_block_ids=[None],
        resumed_req_ids=set(),
        num_computed_tokens=[1280],
    )
    output.num_scheduled_tokens = {"extension": 1}
    test.assertEqual(len(scheduler._track_cached_requests(output)), 1)

    with tempfile.TemporaryDirectory() as directory:
        store = TokenFileStore(
            Path(directory) / "rank",
            layout=scheduler.layout,
            slot_bytes=16384,
            expected_deployment_digest="b" * 64,
            expected_rank_digest="c" * 64,
            expected_rank=0,
            expected_topology_digest="d" * 64,
            expected_profile=scheduler.layout.profile,
            expected_layout_digest=scheduler.layout.digest,
        )
        with store:
            mover = cpu_mover(scheduler.layout)
            mover.__class__ = TokenPageMover
            capture = mover._capture_packed

            def captured(*args):
                yield from capture(*args)

            mover._capture_packed = captured
            worker = object.__new__(connector_type)
            worker._worker_data_path = lambda: (mover, store)
            worker._reporter = InventoryReporter(
                rank=0,
                generation="worker",
                generation_epoch=1,
                max_bytes=(100) * INVENTORY_ENTRY_BYTES,
                max_report_entries=100,
            )
            worker._telemetry = None
            worker._physical_rank = 0
            worker._maintain_capacity = lambda: None
            worker._refresh_worker_metrics = lambda **kw: None
            worker._commit_store_plans((plan,))
            test.assertEqual(
                set(worker._reporter.held_entry_ids()), {p.digest for p in keys}
            )
            test.assertTrue(store.lookup(plan.entry_id, verify_payloads=True).is_hit)


def check_worker_registration(test, module):
    """Exercise startup through ready logging, metrics and resource shutdown.

    Only tensor binding/CUDA allocation and process-group discovery are faked;
    the default store, inventory owner, generation, scrub and metrics are real.
    """
    epochs = []
    with tempfile.TemporaryDirectory() as directory:
        for attempt in range(2):
            with test.subTest(worker_registration=attempt):
                worker = object.__new__(module.SpoolCacheConnector)
                worker._role = module.KVConnectorRole.WORKER
                worker._vllm_config = SimpleNamespace()
                worker._tp_degree = worker._pp_degree = worker._dcp_degree = 1
                worker.layout = layout_for()
                worker.config = SpoolCacheConfig(path=Path(directory))
                worker.deployment_identity = SimpleNamespace(
                    digest="b" * 64, topology={"dp_rank": 0}
                )
                worker._topology_digest = "d" * 64
                mover = SimpleNamespace(
                    geometry_digest="e" * 64, pinned_budget_bytes=32768, close=Mock()
                )
                worker._create_mover = lambda *args, **kwargs: mover
                coordinate = module.WorkerCoordinate(0, 0, 0, 0)
                # The second boot has more keys than the former 512-entry
                # startup slice. Every held key must be available immediately,
                # without unrelated inference to advance rolling reports.
                offers = tuple(SimpleNamespace(entry_id=f"{i:064x}", span_tokens=64)
                               for i in range(640 if attempt else 0))
                with (
                    patch.object(module, "STAGING_SLOT_BYTES", 16384),
                    patch.object(module, "bind_group_owned_kv_caches",
                                 return_value=({"layer": object()}, ())),
                    patch.object(module, "_runtime_worker_coordinate",
                                 return_value=coordinate),
                    patch.object(module, "_prepare_worker_catalog", return_value=offers),
                ):
                    try:
                        worker.register_kv_caches({})
                        test.assertIsInstance(worker._store, TokenFileStore)
                        test.assertEqual(worker._store.buffer_budget_bytes, 16384)
                        test.assertEqual(worker._startup_inventory,
                                         tuple((o.entry_id, o.span_tokens) for o in offers))
                        epochs.append(worker._reporter.generation_epoch)
                        store = worker._store
                    finally:
                        worker.shutdown()
                    test.assertTrue(store._closed)
                    mover.close.assert_called_once()
        test.assertEqual(len(epochs), 2)
        test.assertGreater(epochs[1], epochs[0])
