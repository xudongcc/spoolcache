"""Patch-free SpoolCache connector with startup contract discovery.

The first qualified implementation intentionally performs NVMe transfers
synchronously at the model-runner boundary.  That keeps vLLM block ownership
simple: restore finishes before the forward pass and store finishes before the
next scheduler step can reuse a page.  Both paths use fixed-size staging pools.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats

from ..admission import admit_store_plans
from .. import __version__ as SPOOLCACHE_VERSION
from ..config import (
    CACHE_CHUNK_TOKENS,
    CATALOG_MAX_ENTRIES,
    MAX_PENDING_RESTORES,
    MAX_PENDING_STORES,
    MAX_SPAN_TOKENS,
    MIN_SPAN_TOKENS,
    REPORT_BATCH_SIZE,
    STAGING_SLOT_BYTES,
    STAGING_SLOT_COUNT,
    STARTUP_MAX_DIGESTS,
    SpoolCacheConfig,
)
from ..errors import (
    FatalRestoreError,
    IdentityError,
    StoreBusyError,
    UnsupportedRuntimeError,
)
from ..event_journal import PersistentEventJournal
from ..fail_stop import terminate_worker_after_fatal_restore
from ..gpu import TorchPageMover, bind_group_owned_kv_caches
from ..hma import HMALayout, build_hma_layout
from ..identity import (
    DeploymentIdentity,
    RankIdentity,
    model_namespace_sha256,
    sha256_json,
)
from ..maintenance import DeepScrubber, ScheduledDeepScrubber
from ..prefix import (
    MultimodalFeatureIdentity,
    aligned_prefix_span,
    prefix_digests,
    validate_multimodal_features,
)
from ..quorum import (
    MAX_INVENTORY_REPORTS,
    InventoryReporter,
    QuorumCatalog,
    WorkerInventoryReport,
    validate_inventory_report,
)
from ..store import ManifestOffer, ManifestStore
from ..telemetry import (
    TelemetryBuffer,
    empty_metric_payload,
    merge_metric_payloads,
    metric_payload_is_empty,
    reduce_metric_payload,
    validate_metric_payload,
)
from ..topology import (
    WorkerCoordinate,
    expected_worker_coordinates,
    rank_ownership_sha256,
)
from .compat import require_qualified_allocator, verify_vllm_runtime

if TYPE_CHECKING:
    import torch

    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


logger = logging.getLogger(__name__)
_DISK_METRIC_INTERVAL_SECONDS = 300.0


def _finish_deferred_scrub_shutdown(
    scheduler: ScheduledDeepScrubber,
    journal: PersistentEventJournal | None,
    mover: TorchPageMover | None,
    store: ManifestStore | None,
) -> None:
    """Release data-path resources only after a timed-out scrub actually exits.

    The finalizer is daemonized so a stuck filesystem operation cannot make
    process exit unbounded. Keeping the store alive until the scrub thread has
    stopped prevents the timeout path from turning into a use-after-close.
    """

    if journal is not None:
        try:
            journal.increment("spoolcache_scrub_shutdown_failures_total")
        except Exception:
            logger.exception(
                "spoolcache: could not persist scrub shutdown failure"
            )
    while True:
        try:
            report = scheduler.close()
        except Exception:
            logger.exception(
                "spoolcache: deferred scrub shutdown could not be finalized"
            )
            return
        if not report.thread_alive:
            break
    try:
        if mover is not None:
            mover.close()
    finally:
        if store is not None:
            store.close()


@dataclass(frozen=True)
class SpoolCachePlan:
    request_id: str
    entry_id: str
    span_tokens: int
    block_ids_by_group: tuple[tuple[int, ...], ...]


@dataclass
class SpoolCacheMetadata(KVConnectorMetadata):
    loads: tuple[SpoolCachePlan, ...] = ()
    stores: tuple[SpoolCachePlan, ...] = ()


@dataclass(frozen=True)
class SpoolCacheStartupInventory:
    rank: int
    coordination_digest: str
    generation: str
    generation_epoch: int
    entries: tuple[tuple[str, int], ...]


@dataclass
class SpoolCacheHandshakeMetadata(KVConnectorHandshakeMetadata):
    inventories: tuple[SpoolCacheStartupInventory, ...] = ()


@dataclass
class SpoolCacheStats(KVConnectorStats):
    """Bounded per-rank inventory reports carried by vLLM's native channel."""

    data: dict[str, Any] = field(default_factory=empty_metric_payload)
    reports: tuple[WorkerInventoryReport, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_metric_payload(self.data)
        if not isinstance(self.reports, tuple) or any(
            not isinstance(report, WorkerInventoryReport)
            for report in self.reports
        ):
            raise TypeError("SpoolCache inventory reports must be a tuple")
        if len(self.reports) > MAX_INVENTORY_REPORTS:
            raise ValueError("SpoolCache inventory report count exceeds its bound")
        ranks: set[int] = set()
        for report in self.reports:
            validate_inventory_report(
                report,
                max_entries=CATALOG_MAX_ENTRIES,
                max_report_entries=REPORT_BATCH_SIZE,
            )
            if report.rank in ranks:
                raise ValueError("SpoolCache inventory report ranks are duplicated")
            ranks.add(report.rank)

    def reset(self) -> None:
        self.data = empty_metric_payload()
        self.reports = ()

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not isinstance(other, SpoolCacheStats):
            raise TypeError("cannot aggregate another connector's statistics")
        self.__post_init__()
        other.__post_init__()
        current_ranks = {report.rank for report in self.reports}
        other_ranks = {report.rank for report in other.reports}
        if current_ranks.intersection(other_ranks):
            raise ValueError("SpoolCache inventory report ranks are duplicated")
        latest = {report.rank: report for report in (*self.reports, *other.reports)}
        if len(latest) > MAX_INVENTORY_REPORTS:
            raise ValueError("SpoolCache inventory report count exceeds its bound")
        merged_data = merge_metric_payloads(self.data, other.data)
        merged_reports = tuple(latest[rank] for rank in sorted(latest))
        # Do not partially mutate the accumulator if bounded telemetry
        # validation rejects the merged payload.
        self.data = merged_data
        self.reports = merged_reports
        return self

    def reduce(self) -> dict[str, int | float]:
        return {
            **reduce_metric_payload(self.data),
            "spoolcache_ranks": len(self.reports),
            "spoolcache_reported_entries": sum(
                report.checkpoint.held_count for report in self.reports
            ),
        }

    def is_empty(self) -> bool:
        return not self.reports and metric_payload_is_empty(self.data)


@dataclass
class _StoreProgress:
    token_ids: Sequence[int]
    multimodal_features: tuple[MultimodalFeatureIdentity, ...]
    cache_salt: str
    target_span_tokens: int
    block_ids_by_group: list[list[int]]


class SpoolCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """Exact-prefix, all-HMA-group, rank-local persistent cache."""

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        raw = dict(self._kv_transfer_config.kv_connector_extra_config)
        self.config = SpoolCacheConfig.from_mapping(raw)
        spec_kind_resolver = _get_public_spec_kind_resolver()
        parallel = vllm_config.parallel_config
        self._tp_degree = _runtime_integer(
            getattr(parallel, "tensor_parallel_size", None),
            label="tensor_parallel_size",
            minimum=1,
        )
        self._pp_degree = _runtime_integer(
            getattr(parallel, "pipeline_parallel_size", None),
            label="pipeline_parallel_size",
            minimum=1,
        )
        self._dcp_degree = _runtime_integer(
            getattr(parallel, "decode_context_parallel_size", 1),
            label="decode_context_parallel_size",
            minimum=1,
        )
        self._expected_coordinates = expected_worker_coordinates(
            tp_degree=self._tp_degree,
            pp_degree=self._pp_degree,
            dcp_degree=self._dcp_degree,
        )
        self.runtime_receipt = verify_vllm_runtime(
            connector_type=type(self),
            spec_kind_resolver=spec_kind_resolver,
            require_pp_aware=self._pp_degree > 1,
        )
        logger.warning(
            "spoolcache: vLLM compatibility accepted=%s version=%s "
            "build_sha256=%s attestation=%s",
            self.runtime_receipt.compatibility_mode,
            self.runtime_receipt.vllm_version,
            self.runtime_receipt.vllm_build_sha256[:12],
            self.runtime_receipt.build_fingerprint_kind,
        )
        require_qualified_allocator()

        self._multimodal_modalities = _discover_multimodal_modalities(vllm_config)
        logger.warning(
            "spoolcache: vLLM-declared enabled multimodal inputs=%s",
            sorted(self._multimodal_modalities),
        )

        self.layout = build_hma_layout(
            kv_cache_config,
            dcp_degree=self._dcp_degree,
            vllm_config=vllm_config,
            spec_kind_resolver=spec_kind_resolver,
        )
        logger.warning(
            "spoolcache: HMA runtime layout profile=%s groups=%s "
            "layers=%d alignment=%d "
            "logical_digest=%s physical_digest=%s",
            self.layout.profile,
            [
                {
                    "index": group.group_index,
                    "layers": len(group.layers),
                    "block": group.block_size,
                    "storage_block": group.storage_block_size,
                    "policy": group.reuse_policy,
                    "window": group.reuse_window_tokens,
                    "running_state_tail_pages": (
                        group.running_state_tail_pages
                    ),
                    "dcp_replicated": group.dcp_replicated,
                    "dcp_shards": group.dcp_shard_count,
                    "eagle": group.is_eagle_group,
                    "layer_names_digest": sha256_json(
                        {"layers": [layer.name for layer in group.layers]}
                    )[:12],
                    "page_bytes": group.manager_page_size_bytes,
                }
                for group in self.layout.groups
            ],
            sum(len(group.layers) for group in self.layout.groups),
            self.layout.alignment_tokens,
            self.layout.logical_digest[:12],
            self.layout.digest[:12],
        )
        try:
            self._model_namespace_sha256 = model_namespace_sha256(
                vllm_config.model_config
            )
        except IdentityError as error:
            raise UnsupportedRuntimeError(
                "cannot derive the public vLLM model namespace"
            ) from error
        self.deployment_identity = _build_deployment_identity(
            vllm_config,
            self.layout,
            model_namespace_sha256=self._model_namespace_sha256,
            chunk_tokens=CACHE_CHUNK_TOKENS,
            vllm_version=self.runtime_receipt.vllm_version,
            vllm_build_sha256=self.runtime_receipt.vllm_build_sha256,
        )
        self._topology_digest = sha256_json(
            dict(self.deployment_identity.topology)
        )
        logger.warning(
            "spoolcache: deployment identity role=%s deployment=%s "
            "model_namespace=%s model_config=%s execution_config=%s topology=%s "
            "logical_layout=%s",
            role,
            self.deployment_identity.digest,
            self._model_namespace_sha256,
            self.deployment_identity.model_config_sha256,
            self.deployment_identity.execution_config_sha256,
            self._topology_digest,
            self.layout.logical_digest,
        )
        # vLLM mutates some ModelConfig details differently in the scheduler
        # and workers while loading the model. Their full DeploymentIdentity
        # values are therefore intentionally role-local: the scheduler binds
        # prefix IDs to its complete view, and each worker binds manifests to
        # its complete physical view. This smaller receipt contains only
        # facts that must agree across roles and is checked before quorum.
        self._coordination_layout_digest = (
            self.layout.logical_digest
            if self._pp_degree == 1
            else self.layout.coordination_digest
        )
        self._coordination_digest = sha256_json(
            {
                "schema": "spoolcache-coordination/v2",
                "profile": self.layout.profile,
                "model_namespace_sha256": self._model_namespace_sha256,
                "chunk_tokens": CACHE_CHUNK_TOKENS,
                "spoolcache_version": SPOOLCACHE_VERSION,
                "vllm_version": self.runtime_receipt.vllm_version,
                "vllm_build_sha256": self.runtime_receipt.vllm_build_sha256,
                "topology": dict(self.deployment_identity.topology),
                "logical_layout_digest": self._coordination_layout_digest,
            }
        )

        self._catalog: QuorumCatalog | None = None
        self._need_load: dict[str, tuple[str, int, int]] = {}
        self._pending_loads: dict[str, SpoolCachePlan] = {}
        self._store_progress: dict[str, _StoreProgress] = {}
        self._restored_requests: set[str] = set()
        self._request_salts: dict[str, str | None] = {}
        # Remember skip_write and invalid-salt requests until completion so
        # store tracking cannot publish them. Persistent read control is
        # independent and checked at lookup through ``spoolcache.skip_read``.
        # Keep only request IDs here so arbitrary client data is never copied
        # into connector metadata or logs.
        self._skip_write_requests: set[str] = set()
        self._store: ManifestStore | None = None
        self._mover: TorchPageMover | None = None
        self._reporter: InventoryReporter | None = None
        self._telemetry: TelemetryBuffer | None = None
        self._event_journal: PersistentEventJournal | None = None
        self._scrub_scheduler: ScheduledDeepScrubber | None = None
        self._inventory_withdrawn_epoch = 0
        self._inventory_marker_cursor: str | None = None
        self._object_marker_cursor: str | None = None
        self._event_journal_cursor: dict[
            tuple[str, tuple[str, ...]], int
        ] = {}
        self._reported_generation_changes = 0
        self._last_disk_sample = 0.0
        self._physical_rank: int | None = None
        self._worker_coordinate: WorkerCoordinate | None = None
        self._rank_identity: RankIdentity | None = None
        self._startup_inventory: tuple[tuple[str, int], ...] = ()

        if role == KVConnectorRole.SCHEDULER:
            self._catalog = QuorumCatalog(
                expected_ranks=(
                    coordinate.global_rank
                    for coordinate in self._expected_coordinates
                ),
                max_entries=CATALOG_MAX_ENTRIES,
                max_report_entries=REPORT_BATCH_SIZE,
            )
            self._telemetry = TelemetryBuffer(source="scheduler")
            self._refresh_scheduler_gauges()

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self._role != KVConnectorRole.WORKER:
            return
        physical_kv_caches, shared_aliases = bind_group_owned_kv_caches(
            self.layout,
            kv_caches,
        )
        if shared_aliases:
            logger.info(
                "spoolcache: excluded runtime-proven shared KV aliases "
                "count=%d digest=%s",
                len(shared_aliases),
                sha256_json({"aliases": shared_aliases})[:12],
            )
        coordinate = _runtime_worker_coordinate(
            vllm_config=self._vllm_config,
            tp_degree=self._tp_degree,
            pp_degree=self._pp_degree,
            dcp_degree=self._dcp_degree,
        )
        rank = coordinate.global_rank
        layer_names = tuple(sorted(physical_kv_caches))
        mover = TorchPageMover(
            self.layout,
            physical_kv_caches,
            slot_bytes=STAGING_SLOT_BYTES,
            slot_count=STAGING_SLOT_COUNT,
        )
        rank_identity = RankIdentity(
            deployment_digest=self.deployment_identity.digest,
            physical_rank=rank,
            shard_layout_sha256=mover.geometry_digest,
            layer_ownership_sha256=rank_ownership_sha256(
                layer_names=layer_names,
                shared_aliases=shared_aliases,
                coordinate=coordinate,
                pp_degree=self._pp_degree,
                dp_rank=self.deployment_identity.topology["dp_rank"],
            ),
        )
        rank_root = (
            Path(self.config.path)
            / self.deployment_identity.digest
            / f"rank-{rank:04d}"
        )
        store: ManifestStore | None = None
        try:
            store = ManifestStore(
                rank_root,
                slot_bytes=STAGING_SLOT_BYTES,
                slot_count=STAGING_SLOT_COUNT,
                expected_deployment_digest=self.deployment_identity.digest,
                expected_rank_digest=rank_identity.digest,
                expected_rank=rank,
                expected_topology_digest=self._topology_digest,
                expected_profile=self.layout.profile,
                expected_layout_digest=self.layout.digest,
            )
            # A maintenance operation is short-lived, whereas a reporter owns
            # an in-memory scheduler image for the worker lifetime. Prevent two
            # generations for the same physical rank from advertising
            # independent snapshots of one root.
            store.acquire_inventory_owner()
            event_journal = PersistentEventJournal(store.root / "state")
            store.set_quarantine_hook(
                lambda reason: event_journal.increment(
                    "spoolcache_quarantined_entries_total",
                    (reason,),
                )
            )
            telemetry = TelemetryBuffer(source=f"rank:{rank}")
            # Materialize the complete bounded local catalog. Only the first
            # STARTUP_MAX_DIGESTS entries cross the synchronous handshake;
            # rolling checkpoints advertise the remainder after startup.
            offers = _prepare_worker_catalog(
                store,
                max_bytes=self.config.max_bytes,
                low_watermark_bytes=self.config.low_watermark_bytes,
            )
        except Exception:
            if store is not None:
                store.close()
            mover.close()
            raise
        scrub_scheduler: ScheduledDeepScrubber | None = None
        try:
            generation = str(uuid.uuid4())
            # The scheduler orders delayed reports by this value. Wall-clock
            # time alone can move backwards, so reserve a crash-consistent
            # rank-local epoch before exposing the generation UUID.
            generation_epoch = store.reserve_inventory_generation_epoch()
            reporter = InventoryReporter(
                rank=rank,
                generation=generation,
                generation_epoch=generation_epoch,
                max_entries=CATALOG_MAX_ENTRIES,
                max_report_entries=REPORT_BATCH_SIZE,
            )
            entries = {offer.entry_id: offer.span_tokens for offer in offers}
            with store._exclusive():
                reporter.replace(entries)
                pending_withdrawals = store.pending_inventory_withdrawals(
                    reporter.held_entry_ids()
                )
                marker_batch, inventory_marker_cursor = (
                    store.inventory_withdrawal_marker_batch(
                        REPORT_BATCH_SIZE,
                    )
                )
                object_batch, object_marker_cursor = (
                    store.object_withdrawal_marker_batch(1)
                )
                for entry_id in pending_withdrawals:
                    reporter.remove(entry_id)
                startup_inventory = reporter.startup(STARTUP_MAX_DIGESTS)
                store.acknowledge_absent_inventory_withdrawals(
                    pending_withdrawals
                )
                store.acknowledge_absent_inventory_withdrawals(marker_batch)
                store.acknowledge_unreferenced_object_withdrawals(object_batch)
            # A deep scrub removes the durable manifest before this callback
            # returns. InventoryReporter is synchronized because the scrub
            # driver runs independently from vLLM's stats callback.
            store.set_withdraw_hook(reporter.remove)
            scrub_scheduler = ScheduledDeepScrubber(DeepScrubber(store))
            self._physical_rank = rank
            self._worker_coordinate = coordinate
            self._rank_identity = rank_identity
            self._store = store
            self._mover = mover
            self._telemetry = telemetry
            self._event_journal = event_journal
            self._event_journal_cursor = {}
            self._reporter = reporter
            self._startup_inventory = startup_inventory
            self._scrub_scheduler = scrub_scheduler
            self._inventory_marker_cursor = inventory_marker_cursor
            self._object_marker_cursor = object_marker_cursor
            scrub_scheduler.start()
        except Exception:
            if scrub_scheduler is not None:
                scrub_scheduler.close()
            self._scrub_scheduler = None
            self._store = None
            self._mover = None
            self._reporter = None
            self._telemetry = None
            self._event_journal = None
            self._event_journal_cursor = {}
            self._physical_rank = None
            self._worker_coordinate = None
            self._rank_identity = None
            self._startup_inventory = ()
            store.close()
            mover.close()
            raise
        logger.warning(
            "spoolcache: rank identity rank=%d pp_rank=%d tp_rank=%d "
            "dcp_rank=%d deployment=%s "
            "rank_identity=%s topology=%s physical_layout=%s hma_layout=%s",
            rank,
            coordinate.pp_rank,
            coordinate.tp_rank,
            coordinate.dcp_rank,
            self.deployment_identity.digest,
            rank_identity.digest,
            self._topology_digest,
            mover.geometry_digest,
            self.layout.digest,
        )
        logger.warning(
            "spoolcache: worker ready rank=%d pp_rank=%d tp_rank=%d "
            "dcp_rank=%d root=%s entries=%d "
            "pinned_bytes=%d direct_io=True",
            rank,
            coordinate.pp_rank,
            coordinate.tp_rank,
            coordinate.dcp_rank,
            rank_root,
            len(entries),
            self._mover.pinned_budget_bytes,
        )
        self._refresh_worker_metrics(force_disk=True)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._worker_metadata()
        if metadata.loads:
            mover, store = self._worker_data_path()
            for plan in metadata.loads:
                started = time.perf_counter()
                try:
                    manifest = mover.restore_entry(
                        store,
                        entry_id=plan.entry_id,
                        span_tokens=plan.span_tokens,
                        block_tables=plan.block_ids_by_group,
                    )
                except FatalRestoreError as error:
                    phase = _fatal_restore_phase(error)
                    if self._event_journal is not None:
                        try:
                            self._event_journal.increment(
                                "spoolcache_post_admission_failure_total",
                                (phase,),
                            )
                        except Exception:
                            logger.exception(
                                "spoolcache: failed to persist fatal telemetry"
                            )
                    if self._reporter is not None:
                        self._reporter.remove(plan.entry_id)
                    terminate_worker_after_fatal_restore(
                        error,
                        rank=self._physical_rank,
                        entry_id=plan.entry_id,
                    )
                    raise AssertionError("fatal restore terminator returned")
                elapsed = time.perf_counter() - started
                if self._telemetry is not None:
                    self._telemetry.increment(
                        "spoolcache_restore_bytes_total",
                        value=_manifest_logical_bytes(manifest),
                    )
                    self._telemetry.observe(
                        "spoolcache_restore_seconds", elapsed
                    )
                logger.warning(
                    "spoolcache: restore rank=%d request=%s tokens=%d entry=%s",
                    self._physical_rank,
                    plan.request_id,
                    plan.span_tokens,
                    plan.entry_id[:12],
                )
        # HMA sliding-window managers recycle physical pages during forward.
        # Capture a boundary while it is still the live pre-forward state;
        # wait_for_save() is too late once this step advances beyond it.
        if metadata.stores:
            self._commit_store_plans(metadata.stores)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        # Stores are deliberately completed in start_load_kv(), before forward
        # can recycle a sliding-window page selected for the snapshot.
        return

    def _commit_store_plans(self, plans: Sequence[SpoolCachePlan]) -> None:
        mover, store = self._worker_data_path()
        assert self._rank_identity is not None
        assert self._physical_rank is not None
        for plan in plans:
            started = time.perf_counter()
            try:
                # Keep manifest visibility and reporter admission in one
                # rank-maintenance critical section. A scrub can therefore
                # happen either before publication or after reporter.add(),
                # where its remove callback necessarily wins.
                with store._exclusive():
                    manifest = mover.commit(
                        store,
                        entry_id=plan.entry_id,
                        deployment_identity_digest=self.deployment_identity.digest,
                        rank_identity_digest=self._rank_identity.digest,
                        span_tokens=plan.span_tokens,
                        physical_rank=self._physical_rank,
                        topology_digest=self._topology_digest,
                        block_tables=plan.block_ids_by_group,
                    )
                    if self._reporter is not None:
                        self._reporter.add(plan.entry_id, plan.span_tokens)
            except StoreBusyError:
                if self._telemetry is not None:
                    self._telemetry.increment(
                        "spoolcache_store_skipped_total",
                        labels=("busy",),
                    )
                logger.warning(
                    "spoolcache: store busy rank=%d request=%s entry=%s",
                    self._physical_rank,
                    plan.request_id,
                    plan.entry_id[:12],
                )
                continue
            except Exception:
                if self._telemetry is not None:
                    self._telemetry.increment(
                        "spoolcache_store_skipped_total",
                        labels=("error",),
                    )
                logger.exception(
                    "spoolcache: store skipped rank=%d request=%s entry=%s",
                    self._physical_rank,
                    plan.request_id,
                    plan.entry_id[:12],
                )
                continue
            elapsed = time.perf_counter() - started
            if self._telemetry is not None:
                self._telemetry.increment(
                    "spoolcache_store_bytes_total",
                    value=_manifest_logical_bytes(manifest),
                )
                self._telemetry.observe("spoolcache_store_seconds", elapsed)
            logger.warning(
                "spoolcache: store rank=%d request=%s tokens=%d entry=%s",
                self._physical_rank,
                plan.request_id,
                plan.span_tokens,
                plan.entry_id[:12],
            )
        self._maintain_capacity()
        self._refresh_worker_metrics(force_disk=True)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if _request_skips_read(request):
            self._record_lookup("bypass", "request_skip_read")
            return 0, False
        if self._catalog is None:
            self._record_lookup("miss", "catalog_unavailable")
            return 0, False
        cache_salt = _request_cache_salt(request)
        if cache_salt is None:
            self._record_lookup("miss", "cache_salt")
            return 0, False
        tokens = _eligible_token_ids(request)
        if tokens is None:
            self._record_lookup("miss", "request_shape")
            return 0, False
        if (
            isinstance(num_computed_tokens, bool)
            or not isinstance(num_computed_tokens, int)
            or not 0 <= num_computed_tokens <= len(tokens)
        ):
            raise RuntimeError("vLLM reported an invalid computed-token count")
        multimodal_features = _multimodal_feature_identities(
            request,
            len(tokens),
            enabled_modalities=self._multimodal_modalities,
        )
        if multimodal_features is None:
            self._record_lookup("miss", "multimodal_identity")
            return 0, False
        ceiling = self._safe_restore_span(len(tokens))
        first = max(
            MIN_SPAN_TOKENS,
            ((num_computed_tokens // CACHE_CHUNK_TOKENS) + 1)
            * CACHE_CHUNK_TOKENS,
        )
        if ceiling < first:
            self._record_lookup("miss", "safe_span")
            return 0, False
        candidates = prefix_digests(
            tokens,
            deployment_digest=self.deployment_identity.digest,
            cache_salt=cache_salt,
            chunk_tokens=CACHE_CHUNK_TOKENS,
            boundaries=range(first, ceiling + 1, CACHE_CHUNK_TOKENS),
            multimodal_features=multimodal_features,
        )
        selected = self._catalog.longest(
            (candidate.span_tokens, candidate.digest) for candidate in candidates
        )
        if selected is None:
            self._record_lookup("miss", "rank_quorum")
            return 0, False
        active_restores = set(self._need_load).union(self._pending_loads)
        if (
            request.request_id not in active_restores
            and len(active_restores) >= MAX_PENDING_RESTORES
        ):
            self._record_lookup("miss", "restore_budget")
            return 0, False
        span_tokens, entry_id = selected
        expected_external_tokens = span_tokens - num_computed_tokens
        if expected_external_tokens <= 0:
            raise RuntimeError("SpoolCache selected a non-positive external span")
        self._need_load[request.request_id] = (
            entry_id,
            span_tokens,
            expected_external_tokens,
        )
        self._record_lookup("hit", "ready_entry")
        if self._telemetry is not None:
            self._telemetry.increment(
                "spoolcache_hit_tokens_total",
                value=expected_external_tokens,
            )
        logger.warning(
            "spoolcache: hit request=%s tokens=%d entry=%s",
            request.request_id,
            span_tokens,
            entry_id[:12],
        )
        # Restore is synchronous in start_load_kv, before this step's forward.
        return span_tokens - num_computed_tokens, False

    def on_new_request(self, request: "Request") -> None:
        salt = _request_cache_salt(request)
        self._request_salts[request.request_id] = salt or None
        if salt is None or _request_skips_write(request):
            self._skip_write_requests.add(request.request_id)
        else:
            self._skip_write_requests.discard(request.request_id)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        external_tokens = _nonnegative_runtime_int(
            num_external_tokens,
            label="external-token count",
        )
        pending = self._need_load.pop(request.request_id, None)
        if external_tokens == 0:
            return
        if pending is None:
            raise RuntimeError("SpoolCache load allocation has no admitted entry")
        entry_id, span_tokens, expected_external_tokens = pending
        if external_tokens != expected_external_tokens:
            raise RuntimeError(
                "vLLM external-token allocation differs from the admitted span"
            )
        block_ids = _normalize_block_ids(blocks.get_block_ids())
        self.layout.select_physical_pages(block_ids, span_tokens)
        self._pending_loads[request.request_id] = SpoolCachePlan(
            request_id=request.request_id,
            entry_id=entry_id,
            span_tokens=span_tokens,
            block_ids_by_group=block_ids,
        )
        self._restored_requests.add(request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> SpoolCacheMetadata:
        loads = tuple(self._pending_loads.values())
        self._pending_loads.clear()
        candidates: list[SpoolCachePlan] = []

        finished = set(scheduler_output.finished_req_ids or ())
        for request_id in finished:
            self._store_progress.pop(request_id, None)
            self._restored_requests.discard(request_id)
            self._need_load.pop(request_id, None)
            self._request_salts.pop(request_id, None)
            self._skip_write_requests.discard(request_id)

        candidates.extend(self._track_new_requests(scheduler_output))
        candidates.extend(self._track_cached_requests(scheduler_output))

        # Store is synchronous in the current patch-free connector.  Bounding
        # plans here prevents a busy scheduler step from serializing an
        # unbounded number of full HMA snapshots before forward can continue.
        # Duplicate exact prefixes in the same batch are also written once.
        admission = admit_store_plans(
            candidates,
            max_plans=MAX_PENDING_STORES,
        )
        if self._telemetry is not None:
            if admission.skipped_duplicate:
                self._telemetry.increment(
                    "spoolcache_store_skipped_total",
                    value=admission.skipped_duplicate,
                    labels=("duplicate",),
                )
            if admission.skipped_budget:
                self._telemetry.increment(
                    "spoolcache_store_skipped_total",
                    value=admission.skipped_budget,
                    labels=("budget",),
                )
        if admission.skipped:
            logger.warning(
                "spoolcache: store admission admitted=%d "
                "skipped_duplicate=%d skipped_budget=%d budget=%d",
                len(admission.admitted),
                admission.skipped_duplicate,
                admission.skipped_budget,
                MAX_PENDING_STORES,
            )
        return SpoolCacheMetadata(loads=loads, stores=admission.admitted)

    def _track_new_requests(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[SpoolCachePlan]:
        result: list[SpoolCachePlan] = []
        load_ids = {plan.request_id for plan in self._pending_loads.values()}
        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            if (
                request_id in self._skip_write_requests
                or request_id in self._restored_requests
                or request_id in load_ids
            ):
                continue
            tokens = _eligible_new_request_tokens(request)
            if tokens is None:
                continue
            multimodal_features = _multimodal_feature_identities(
                request,
                len(tokens),
                enabled_modalities=self._multimodal_modalities,
            )
            if multimodal_features is None:
                continue
            target_span_tokens = self._safe_store_span(len(tokens))
            if target_span_tokens <= 0:
                continue
            cache_salt = self._request_salts.get(request_id)
            target_entry_id = _entry_id(
                tokens,
                span_tokens=target_span_tokens,
                deployment_digest=self.deployment_identity.digest,
                cache_salt=cache_salt or "",
                chunk_tokens=CACHE_CHUNK_TOKENS,
                multimodal_features=multimodal_features,
            )
            if self._catalog is not None and self._catalog.has_quorum(
                target_entry_id, target_span_tokens
            ):
                continue
            block_ids = [list(group) for group in request.block_ids]
            progress = _StoreProgress(
                token_ids=tokens,
                multimodal_features=multimodal_features,
                cache_salt=cache_salt or "",
                target_span_tokens=target_span_tokens,
                block_ids_by_group=block_ids,
            )
            before = request.num_computed_tokens
            scheduled = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            candidate = _pre_forward_store_span(
                before_tokens=before,
                scheduled_tokens=scheduled,
                target_span_tokens=target_span_tokens,
                quantum_tokens=self._store_quantum,
                min_span_tokens=MIN_SPAN_TOKENS,
                require_exact_boundary=self._store_requires_exact_boundary,
            )
            if candidate is None:
                self._store_progress[request_id] = progress
            elif candidate > 0:
                plan = self._plan_from_progress(request_id, progress, candidate)
                if self._catalog is None or not self._catalog.has_quorum(
                    plan.entry_id, plan.span_tokens
                ):
                    result.append(plan)
            elif self._telemetry is not None:
                self._telemetry.increment(
                    "spoolcache_store_skipped_total",
                    labels=("unsafe_boundary",),
                )
        return result

    def _track_cached_requests(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[SpoolCachePlan]:
        result: list[SpoolCachePlan] = []
        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            progress = self._store_progress.get(request_id)
            if progress is None:
                continue
            new_groups = cached.new_block_ids[index]
            if new_groups is not None:
                normalized = [list(group) for group in new_groups]
                if len(normalized) != len(progress.block_ids_by_group):
                    self._store_progress.pop(request_id, None)
                    continue
                if request_id in cached.resumed_req_ids:
                    progress.block_ids_by_group = normalized
                else:
                    for existing, added in zip(
                        progress.block_ids_by_group, normalized, strict=True
                    ):
                        existing.extend(added)
            before = cached.num_computed_tokens[index]
            scheduled = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            candidate = _pre_forward_store_span(
                before_tokens=before,
                scheduled_tokens=scheduled,
                target_span_tokens=progress.target_span_tokens,
                quantum_tokens=self._store_quantum,
                min_span_tokens=MIN_SPAN_TOKENS,
                require_exact_boundary=self._store_requires_exact_boundary,
            )
            if candidate is not None:
                self._store_progress.pop(request_id, None)
                if candidate > 0:
                    plan = self._plan_from_progress(request_id, progress, candidate)
                    if self._catalog is None or not self._catalog.has_quorum(
                        plan.entry_id, plan.span_tokens
                    ):
                        result.append(plan)
                elif self._telemetry is not None:
                    self._telemetry.increment(
                        "spoolcache_store_skipped_total",
                        labels=("unsafe_boundary",),
                    )
        return result

    @property
    def _store_quantum(self) -> int:
        return math.lcm(self.layout.alignment_tokens, CACHE_CHUNK_TOKENS)

    @property
    def _store_requires_exact_boundary(self) -> bool:
        # Recurrent and circular groups retain only the state at the request's
        # current boundary.  Unlike full-attention pages, they cannot prove an
        # older prefix merely because the corresponding table slot exists:
        # vLLM may already have replaced it with the null block or advanced the
        # ring.  Defer publication until a scheduler step starts at the exact
        # target boundary.  This is derived solely from discovered cache-spec
        # semantics, never from a model identity.
        return any(
            group.reuse_policy in {"recurrent_align", "circular_one"}
            for group in self.layout.groups
        )

    def _plan_from_progress(
        self,
        request_id: str,
        progress: _StoreProgress,
        span_tokens: int,
    ) -> SpoolCachePlan:
        block_ids = _normalize_block_ids(progress.block_ids_by_group)
        self.layout.select_physical_pages(block_ids, span_tokens)
        return SpoolCachePlan(
            request_id=request_id,
            entry_id=_entry_id(
                progress.token_ids,
                span_tokens=span_tokens,
                deployment_digest=self.deployment_identity.digest,
                cache_salt=progress.cache_salt,
                chunk_tokens=CACHE_CHUNK_TOKENS,
                multimodal_features=progress.multimodal_features,
            ),
            span_tokens=span_tokens,
            block_ids_by_group=block_ids,
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del request, block_ids
        # Stores complete pre-forward before vLLM may reuse these blocks.
        return False, None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        return None, None

    def get_kv_connector_stats(self) -> SpoolCacheStats | None:
        reports: tuple[WorkerInventoryReport, ...] = ()
        if self._role == KVConnectorRole.WORKER and self._reporter is not None:
            self._drain_scrub_maintenance()
            self._refresh_worker_metrics()
            store = self._store
            if store is None:
                raise RuntimeError("SpoolCache worker inventory store is unavailable")
            # Standalone maintenance and a previous failed process communicate
            # withdrawals through durable markers. Consume them under the same
            # lock as report construction so no scan/report operation can
            # overwrite the fail-closed image.
            with store._exclusive():
                pending_withdrawals = store.pending_inventory_withdrawals(
                    self._reporter.held_entry_ids()
                )
                marker_batch, next_inventory_marker_cursor = (
                    store.inventory_withdrawal_marker_batch(
                        REPORT_BATCH_SIZE,
                        after=self._inventory_marker_cursor,
                    )
                )
                object_batch, next_object_marker_cursor = (
                    store.object_withdrawal_marker_batch(
                        1,
                        after=self._object_marker_cursor,
                    )
                )
                for entry_id in pending_withdrawals:
                    self._reporter.remove(entry_id)
                reports = (self._reporter.next_report(REPORT_BATCH_SIZE),)
                store.acknowledge_absent_inventory_withdrawals(
                    pending_withdrawals
                )
                store.acknowledge_absent_inventory_withdrawals(marker_batch)
                store.acknowledge_unreferenced_object_withdrawals(object_batch)
                self._inventory_marker_cursor = next_inventory_marker_cursor
                self._object_marker_cursor = next_object_marker_cursor
        elif self._role == KVConnectorRole.SCHEDULER:
            self._refresh_scheduler_metrics()
        data = (
            self._telemetry.drain()
            if self._telemetry is not None
            else empty_metric_payload()
        )
        stats = SpoolCacheStats(data=data, reports=reports)
        return None if stats.is_empty() else stats

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> SpoolCacheStats:
        return SpoolCacheStats(
            data=empty_metric_payload() if data is None else data
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: Any,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Any:
        from .prometheus import SpoolCachePromMetrics

        return SpoolCachePromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        if self._catalog is None:
            return
        stats = getattr(connector_output, "kv_connector_stats", None)
        if stats is None:
            return
        if not isinstance(stats, SpoolCacheStats):
            raise RuntimeError("SpoolCache connector stats type is incompatible")
        report_ranks = tuple(report.rank for report in stats.reports)
        if (
            len(report_ranks) > len(self._catalog.expected_ranks)
            or len(set(report_ranks)) != len(report_ranks)
            or any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank not in self._catalog.expected_ranks
                for rank in report_ranks
            )
        ):
            raise RuntimeError("SpoolCache connector report ranks are incompatible")
        for report in stats.reports:
            self._catalog.apply_report(report)
        self._refresh_scheduler_gauges()

    def get_handshake_metadata(self) -> SpoolCacheHandshakeMetadata | None:
        if self._role != KVConnectorRole.WORKER or self._reporter is None:
            return None
        return SpoolCacheHandshakeMetadata(
            inventories=(
                SpoolCacheStartupInventory(
                    rank=self._reporter.rank,
                    coordination_digest=self._coordination_digest,
                    generation=self._reporter.generation,
                    generation_epoch=self._reporter.generation_epoch,
                    entries=self._startup_inventory,
                ),
            )
        )

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        if self._catalog is None:
            return
        if self._pp_degree != 1:
            raise RuntimeError(
                "SpoolCache PP>1 requires vLLM's PP-aware handshake"
            )
        if not isinstance(metadata, dict):
            raise RuntimeError("SpoolCache worker handshake mapping is incompatible")
        pp_metadata: dict[
            tuple[int, int], KVConnectorHandshakeMetadata
        ] = {}
        for rank, handshake in metadata.items():
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise RuntimeError(
                    "SpoolCache received handshake metadata from an unexpected rank"
                )
            pp_metadata[(0, rank)] = handshake
        self._set_pp_aware_handshake_metadata(pp_metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self,
        metadata: dict[
            tuple[int, int], KVConnectorHandshakeMetadata
        ],
    ) -> None:
        if self._catalog is None:
            return
        self._set_pp_aware_handshake_metadata(metadata)

    def _set_pp_aware_handshake_metadata(
        self,
        metadata: dict[
            tuple[int, int], KVConnectorHandshakeMetadata
        ],
    ) -> None:
        assert self._catalog is not None
        if not isinstance(metadata, dict):
            raise RuntimeError("SpoolCache worker handshake mapping is incompatible")
        expected = {
            (coordinate.pp_rank, coordinate.tp_rank): coordinate
            for coordinate in self._expected_coordinates
        }
        if len(metadata) > len(expected):
            raise RuntimeError(
                "SpoolCache received handshake metadata from too many ranks"
            )
        received_coordinates = set(metadata)
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in key)
            for key in received_coordinates
        ) or not received_coordinates.issubset(expected):
            raise RuntimeError(
                "SpoolCache received handshake metadata from an unexpected PP/TP rank"
            )
        validated: list[SpoolCacheStartupInventory] = []
        inventory_ranks: set[int] = set()
        for transport_coordinate, handshake in sorted(metadata.items()):
            if not isinstance(handshake, SpoolCacheHandshakeMetadata):
                raise RuntimeError("SpoolCache worker handshake type is incompatible")
            inventories = handshake.inventories
            if not isinstance(inventories, tuple) or len(inventories) != 1:
                raise RuntimeError(
                    "SpoolCache requires exactly one startup inventory per rank"
                )
            inventory = inventories[0]
            if not isinstance(inventory, SpoolCacheStartupInventory):
                raise RuntimeError("SpoolCache startup inventory type is incompatible")
            if (
                not isinstance(inventory.entries, tuple)
                or len(inventory.entries) > STARTUP_MAX_DIGESTS
            ):
                raise RuntimeError("SpoolCache startup inventory exceeds its bound")
            if (
                isinstance(inventory.rank, bool)
                or not isinstance(inventory.rank, int)
                or inventory.rank
                != expected[transport_coordinate].global_rank
                or inventory.rank in inventory_ranks
            ):
                raise RuntimeError(
                    "SpoolCache transport rank and startup inventory rank differ"
                )
            if inventory.coordination_digest != self._coordination_digest:
                raise RuntimeError(
                    "SpoolCache scheduler/worker coordination identities differ; "
                    "refusing to build a cache quorum"
                )
            validated.append(inventory)
            inventory_ranks.add(inventory.rank)
        if received_coordinates != set(expected):
            raise RuntimeError(
                "SpoolCache startup handshake is missing one or more required ranks"
            )
        for inventory in validated:
            self._catalog.apply_startup(
                rank=inventory.rank,
                generation=inventory.generation,
                generation_epoch=inventory.generation_epoch,
                entries=inventory.entries,
            )
        if not self._catalog.is_ready:
            raise RuntimeError("SpoolCache startup inventories did not form a quorum")
        self._refresh_scheduler_gauges()

    def _record_lookup(self, result: str, reason: str) -> None:
        if self._telemetry is not None:
            self._telemetry.increment(
                "spoolcache_lookup_total",
                labels=(result, reason),
            )

    def _refresh_scheduler_gauges(self) -> None:
        telemetry, catalog = self._telemetry, self._catalog
        if telemetry is None or catalog is None:
            return
        telemetry.set_gauge(
            "spoolcache_required_ranks", len(catalog.expected_ranks)
        )
        telemetry.set_gauge(
            "spoolcache_ready_ranks", catalog.ready_rank_count
        )
        telemetry.set_gauge(
            "spoolcache_rank_quorum_entries", catalog.quorum_count
        )
        telemetry.set_gauge(
            "spoolcache_readiness",
            int(catalog.has_all_rank_identities),
            labels=("rank_identity",),
        )
        telemetry.set_gauge(
            "spoolcache_readiness",
            int(catalog.is_ready),
            labels=("inventory_quorum",),
        )
        # A scheduler capable of exporting this sample has not entered the
        # worker fatal-exit path. Deployment-level readiness may additionally
        # combine this with API and rank liveness outside SpoolCache.
        telemetry.set_gauge(
            "spoolcache_readiness", 1, labels=("fatal_clear",)
        )

    def _refresh_scheduler_metrics(self) -> None:
        telemetry, catalog = self._telemetry, self._catalog
        if telemetry is None or catalog is None:
            return
        changes = catalog.generation_changes_total
        if changes < self._reported_generation_changes:
            raise RuntimeError("SpoolCache generation counter moved backwards")
        if changes > self._reported_generation_changes:
            telemetry.increment(
                "spoolcache_rank_generation_changes_total",
                value=changes - self._reported_generation_changes,
            )
            self._reported_generation_changes = changes
        telemetry.set_gauge(
            "spoolcache_delayed_store_requests", len(self._store_progress)
        )
        self._refresh_scheduler_gauges()

    def _refresh_worker_metrics(self, *, force_disk: bool = False) -> None:
        telemetry, mover, store = self._telemetry, self._mover, self._store
        if telemetry is None or mover is None or store is None:
            return
        telemetry.set_gauge(
            "spoolcache_pinned_pool_bytes", mover.pinned_budget_bytes
        )
        now = time.monotonic()
        if force_disk or now - self._last_disk_sample >= _DISK_METRIC_INTERVAL_SECONDS:
            usage = store.managed_disk_usage()
            telemetry.set_gauge("spoolcache_disk_bytes", usage.cache_bytes)
            telemetry.set_gauge(
                "spoolcache_managed_disk_bytes",
                usage.total_bytes,
            )
            telemetry.set_gauge(
                "spoolcache_quarantine_bytes",
                usage.quarantine_bytes,
            )
            self._last_disk_sample = now
        journal = self._event_journal
        if journal is None:
            return
        totals = journal.totals()
        for key, total in totals.items():
            previous = self._event_journal_cursor.get(key, 0)
            if total < previous:
                raise RuntimeError("SpoolCache persistent event counter moved backwards")
            if total > previous:
                name, labels = key
                telemetry.increment(name, value=total - previous, labels=labels)
        self._event_journal_cursor = totals

    def shutdown(self) -> None:
        scrub_scheduler = self._scrub_scheduler
        if scrub_scheduler is not None:
            shutdown_report = scrub_scheduler.close()
            if shutdown_report.thread_alive:
                journal = self._event_journal
                logger.error(
                    "spoolcache: scrub shutdown failed receipt=%s",
                    json.dumps(
                        asdict(shutdown_report),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                mover, store = self._mover, self._store
                self._scrub_scheduler = None
                self._mover = None
                self._store = None
                self._event_journal = None
                threading.Thread(
                    target=_finish_deferred_scrub_shutdown,
                    args=(scrub_scheduler, journal, mover, store),
                    name="spoolcache-scrub-shutdown-finalizer",
                    daemon=True,
                ).start()
                return
        self._scrub_scheduler = None
        mover, store = self._mover, self._store
        self._mover = None
        self._store = None
        self._event_journal = None
        if mover is not None:
            mover.close()
        if store is not None:
            store.close()

    def _drain_scrub_maintenance(self) -> None:
        scheduler = self._scrub_scheduler
        telemetry = self._telemetry
        reporter = self._reporter
        store = self._store
        if scheduler is None or telemetry is None or reporter is None or store is None:
            return
        rescan_epoch = scheduler.pending_inventory_rescan_epoch()
        if rescan_epoch is not None:
            force_withdrawal = scheduler.inventory_rescan_requires_withdrawal(
                rescan_epoch
            )
            if (
                force_withdrawal
                and self._inventory_withdrawn_epoch < rescan_epoch
            ):
                # Emit an empty image for at least one stats transport before
                # rebuilding after an unexpected maintenance failure. This
                # prevents a partially applied quarantine/fsync failure from
                # being hidden by an immediate successful metadata scan.
                reporter.replace({})
                self._inventory_withdrawn_epoch = rescan_epoch
            else:
                try:
                    # The scrub callback also takes this lock before removing
                    # an offer. Keeping scan, replace, and acknowledgement in
                    # one critical section prevents a post-scan quarantine
                    # from being overwritten by the stale snapshot.
                    with store._exclusive():
                        offers = _scan_worker_catalog(store)
                        reporter.replace(
                            {
                                offer.entry_id: offer.span_tokens
                                for offer in offers
                            }
                        )
                        scheduler.acknowledge_inventory_rescan(rescan_epoch)
                except Exception:
                    # A failed reconciliation cannot leave the previous
                    # catalog advertised. Leave the epoch pending; the empty
                    # image crosses the stats channel before the next retry.
                    reporter.replace({})
                    self._inventory_withdrawn_epoch = rescan_epoch
                    logger.exception("spoolcache: scrub inventory rescan failed")
        names = {
            "payload_bytes": "spoolcache_scrub_payload_bytes_total",
            "objects_authenticated": "spoolcache_scrub_objects_total",
            "manifests_authenticated": "spoolcache_scrub_manifests_total",
            "cycles": "spoolcache_scrub_cycles_total",
            "objects_quarantined": (
                "spoolcache_scrub_objects_quarantined_total"
            ),
            "failures": "spoolcache_scrub_failures_total",
            "namespace_items_scanned": (
                "spoolcache_scrub_namespace_items_total"
            ),
            "shutdown_failures": (
                "spoolcache_scrub_shutdown_failures_total"
            ),
            "orphan_objects_removed": "spoolcache_orphan_objects_removed_total",
            "orphan_bytes_removed": "spoolcache_orphan_bytes_removed_total",
            "temporary_files_removed": "spoolcache_temporary_files_removed_total",
        }
        for key, value in scheduler.drain_metrics().items():
            metric = names.get(key)
            if metric is None:
                raise RuntimeError("deep scrub scheduler emitted an unknown metric")
            telemetry.increment(metric, value=value)

    def _worker_metadata(self) -> SpoolCacheMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SpoolCacheMetadata):
            raise RuntimeError("SpoolCache worker received incompatible metadata")
        return metadata

    def _worker_data_path(self) -> tuple[TorchPageMover, ManifestStore]:
        if self._mover is None or self._store is None:
            raise RuntimeError("SpoolCache worker data path is not registered")
        return self._mover, self._store

    def _safe_restore_span(self, prompt_tokens: int) -> int:
        return aligned_prefix_span(
            prompt_tokens,
            alignment=self.layout.alignment_tokens,
            chunk_tokens=CACHE_CHUNK_TOKENS,
            min_span_tokens=MIN_SPAN_TOKENS,
            max_span_tokens=MAX_SPAN_TOKENS,
        )

    def _safe_store_span(self, prompt_tokens: int) -> int:
        # A producer may publish state after its entire prompt when that prompt
        # ends on a safe boundary.  A consumer still uses
        # ``_safe_restore_span`` and therefore retains at least one local token
        # to execute.  Separate ceilings let stateful HMA layouts persist exact
        # current state without changing the scheduler batch size or adding a
        # model-specific launch setting.
        return aligned_prefix_span(
            prompt_tokens + 1,
            alignment=self.layout.alignment_tokens,
            chunk_tokens=CACHE_CHUNK_TOKENS,
            min_span_tokens=MIN_SPAN_TOKENS,
            max_span_tokens=MAX_SPAN_TOKENS,
        )

    def _maintain_capacity(self) -> None:
        if self._store is None or self._reporter is None:
            return
        try:
            with self._store._exclusive():
                if self._store.disk_usage_bytes() <= self.config.max_bytes:
                    return
                self._store.maintain_capacity(
                    max_bytes=self.config.max_bytes,
                    low_watermark_bytes=self.config.low_watermark_bytes,
                )
                offers = _scan_worker_catalog(self._store)
                self._reporter.replace(
                    {offer.entry_id: offer.span_tokens for offer in offers}
                )
        except Exception:
            # Namespace mutation may already have happened even if a following
            # directory fsync or callback failed. Withdraw the whole local
            # image now and reconcile on later stats cycles.
            self._reporter.replace({})
            if self._scrub_scheduler is not None:
                self._scrub_scheduler.require_inventory_rescan(
                    force_withdrawal=True
                )
            logger.exception("spoolcache: capacity maintenance failed")


def _manifest_logical_bytes(manifest: object) -> int:
    objects = getattr(manifest, "objects", None)
    if not isinstance(objects, tuple):
        raise RuntimeError("SpoolCache manifest objects are malformed")
    total = 0
    for descriptor in objects:
        byte_length = getattr(descriptor, "byte_length", None)
        if (
            isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length <= 0
        ):
            raise RuntimeError("SpoolCache manifest byte length is malformed")
        total += byte_length
    return total


def _scan_worker_catalog(store: ManifestStore) -> tuple[ManifestOffer, ...]:
    """Load the full fixed local catalog, independently of handshake size."""

    return store.scan_offers(CATALOG_MAX_ENTRIES)


def _prepare_worker_catalog(
    store: ManifestStore,
    *,
    max_bytes: int,
    low_watermark_bytes: int,
) -> tuple[ManifestOffer, ...]:
    """Enforce startup capacity before forming the only handshake image."""

    with store._exclusive():
        if store.disk_usage_bytes() > max_bytes:
            store.maintain_capacity(
                max_bytes=max_bytes,
                low_watermark_bytes=low_watermark_bytes,
            )
        return _scan_worker_catalog(store)


def _fatal_restore_phase(error: FatalRestoreError) -> str:
    code = str(error).partition(":")[0]
    return {
        "SPOOLCACHE_POST_ADMISSION_LOOKUP_FAILED": "lookup",
        "SPOOLCACHE_POST_ADMISSION_SPAN_MISMATCH": "span",
        "SPOOLCACHE_POST_ADMISSION_MANIFEST_CHANGED": "manifest",
        "SPOOLCACHE_POST_ADMISSION_RESTORE_FAILED": "payload",
    }.get(code, "unknown")


def _normalize_block_ids(
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    try:
        for group in groups:
            block_ids = tuple(group)
            if any(
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                for block_id in block_ids
            ):
                raise ValueError(
                    "vLLM block IDs must be non-negative integers"
                )
            normalized.append(block_ids)
    except TypeError as error:
        raise ValueError("vLLM block tables must be sequences of block IDs") from error
    return tuple(normalized)


def _nonnegative_runtime_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"vLLM reported an invalid {label}")
    return value


def _runtime_worker_coordinate(
    *,
    vllm_config: object,
    tp_degree: int,
    pp_degree: int,
    dcp_degree: int,
) -> WorkerCoordinate:
    """Read and cross-check vLLM's public process-group coordinates."""

    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_pp_group,
        get_tp_group,
    )

    groups = (
        (get_pp_group(), pp_degree, "pipeline-parallel"),
        (get_tp_group(), tp_degree, "tensor-parallel"),
        (get_dcp_group(), dcp_degree, "decode-context"),
    )
    ranks: list[int] = []
    for group, expected_size, label in groups:
        group_size = _runtime_integer(
            getattr(group, "world_size", None),
            label=f"{label} group size",
            minimum=1,
        )
        if group_size != expected_size:
            raise RuntimeError(f"vLLM {label} group size disagrees with topology")
        ranks.append(
            _nonnegative_runtime_int(
                getattr(group, "rank_in_group", None),
                label=f"{label} rank",
            )
        )
    pp_rank, tp_rank, dcp_rank = ranks
    derived_global_rank = pp_rank * tp_degree + tp_rank
    parallel = getattr(vllm_config, "parallel_config", None)
    configured_global_rank = getattr(parallel, "rank", derived_global_rank)
    try:
        return WorkerCoordinate.from_runtime(
            global_rank=configured_global_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            dcp_rank=dcp_rank,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dcp_degree=dcp_degree,
        )
    except ValueError as error:
        raise RuntimeError("vLLM worker rank coordinates are inconsistent") from error


def _request_skips_read(request: object) -> bool:
    """Skip persistent restore admission without changing store or GPU reuse."""

    params = getattr(request, "kv_transfer_params", None)
    return isinstance(params, Mapping) and params.get("spoolcache.skip_read") is True


def _request_skips_write(request: object) -> bool:
    """Skip persistence writes only when the public request flag is JSON true.

    vLLM carries ``kv_transfer_params`` into the internal Request. The key
    ``spoolcache.skip_write`` leaves persistent reads and GPU prefix reuse active.
    Strings and integers are ignored; only the literal boolean enables it.
    """

    params = getattr(request, "kv_transfer_params", None)
    return isinstance(params, Mapping) and params.get("spoolcache.skip_write") is True


def _pre_forward_store_span(
    *,
    before_tokens: int,
    scheduled_tokens: int,
    target_span_tokens: int,
    quantum_tokens: int,
    min_span_tokens: int,
    require_exact_boundary: bool = False,
) -> int | None:
    """Choose a completed aligned boundary before this forward can recycle it.

    ``None`` means the target has not been reached yet. ``0`` means this step
    crosses the target without any safe boundary worth publishing, so tracking
    must stop. A positive result is safe to capture in ``start_load_kv``.
    """

    values = (
        before_tokens,
        scheduled_tokens,
        target_span_tokens,
        quantum_tokens,
        min_span_tokens,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("store boundary values must be integers")
    if before_tokens < 0 or scheduled_tokens < 0:
        raise ValueError("store progress cannot be negative")
    if target_span_tokens <= 0 or quantum_tokens <= 0 or min_span_tokens < 0:
        raise ValueError("store boundary configuration is invalid")
    if target_span_tokens % quantum_tokens:
        raise ValueError("store target is not quantum aligned")
    if not isinstance(require_exact_boundary, bool):
        raise ValueError("exact-boundary policy must be boolean")
    if before_tokens == target_span_tokens:
        return target_span_tokens
    if require_exact_boundary:
        # Stateful groups expose only the state at ``before_tokens``.  Keep
        # tracking while the target is ahead; if a step overshoots it, the next
        # call returns 0 and retires the unsafe candidate without touching GPU
        # pages.  An exactly aligned prompt reaches the target on its following
        # decode step and is published there.
        return None if before_tokens < target_span_tokens else 0
    if before_tokens > target_span_tokens:
        return 0
    if before_tokens + scheduled_tokens < target_span_tokens:
        return None
    candidate = before_tokens // quantum_tokens * quantum_tokens
    return candidate if candidate >= min_span_tokens else 0


def _entry_id(
    tokens: Sequence[int],
    *,
    span_tokens: int,
    deployment_digest: str,
    cache_salt: str,
    chunk_tokens: int,
    multimodal_features: Sequence[MultimodalFeatureIdentity] = (),
) -> str:
    digests = prefix_digests(
        tokens,
        deployment_digest=deployment_digest,
        cache_salt=cache_salt,
        chunk_tokens=chunk_tokens,
        boundaries=(span_tokens,),
        multimodal_features=multimodal_features,
    )
    if len(digests) != 1:
        raise RuntimeError("SpoolCache could not derive the store entry ID")
    return digests[0].digest


def _eligible_token_ids(request: Any) -> Sequence[int] | None:
    tokens = getattr(request, "prompt_token_ids", None)
    if (
        tokens is None
        or getattr(request, "prompt_embeds", None) is not None
        or getattr(request, "lora_request", None) is not None
    ):
        return None
    return tokens


def _request_cache_salt(request: object) -> str | None:
    """Return a canonical public salt, or ``None`` for an invalid identity."""

    salt = getattr(request, "cache_salt", None)
    if salt is None:
        return ""
    if not isinstance(salt, str):
        return None
    try:
        salt.encode("utf-8")
    except UnicodeError:
        return None
    return salt


def _eligible_new_request_tokens(request: Any) -> Sequence[int] | None:
    tokens = getattr(request, "prompt_token_ids", None)
    if (
        tokens is None
        or getattr(request, "prompt_embeds", None) is not None
        or getattr(request, "lora_request", None) is not None
    ):
        return None
    # The canonical salt was captured from the public Request in
    # ``on_new_request`` and is joined with these tokens by the caller.
    return tokens


def _multimodal_feature_identities(
    request: Any,
    token_count: int,
    *,
    enabled_modalities: frozenset[str],
) -> tuple[MultimodalFeatureIdentity, ...] | None:
    """Extract stable vLLM media identity without retaining the media bytes.

    The enabled set comes from vLLM's model registry and per-prompt limits at
    connector startup. SpoolCache intentionally has no model or modality
    allowlist. A request outside that declared set, or without a complete media
    identity, bypasses persistent reuse instead of falling back to token-only
    identity.
    """

    raw = getattr(request, "mm_features", None)
    if raw is None:
        return None if getattr(request, "mm_hashes", None) else ()
    try:
        raw = tuple(raw)
    except TypeError:
        return None
    if not raw:
        return ()
    try:
        if any(feature.modality not in enabled_modalities for feature in raw):
            return None
        features = tuple(
            MultimodalFeatureIdentity(
                modality=feature.modality,
                identifier=feature.identifier,
                offset=feature.mm_position.offset,
                length=feature.mm_position.length,
            )
            for feature in raw
        )
        return validate_multimodal_features(features, token_count)
    except (AttributeError, TypeError, ValueError):
        # An incomplete media identity can only be a cache miss. Token-only
        # aliasing here would allow two different media inputs to reuse KV.
        return None


def _discover_multimodal_modalities(
    vllm_config: Any,
    *,
    registry: Any | None = None,
) -> frozenset[str]:
    """Read enabled input modalities from vLLM's public multimodal registry.

    Model support and deployment limits are both authoritative. No model name,
    architecture, or modality name is embedded here, so newly registered vLLM
    modalities automatically participate in the same identity contract.
    """

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        raise UnsupportedRuntimeError("vLLM model configuration is unavailable")
    is_multimodal = getattr(model_config, "is_multimodal_model", None)
    if not isinstance(is_multimodal, bool):
        raise UnsupportedRuntimeError(
            "vLLM multimodal model flag is not boolean"
        )
    if not is_multimodal:
        return frozenset()

    if registry is None:
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY
        except (ImportError, AttributeError) as error:
            raise UnsupportedRuntimeError(
                "multimodal model requires vLLM's public MULTIMODAL_REGISTRY"
            ) from error
        registry = MULTIMODAL_REGISTRY

    supports_inputs = getattr(registry, "supports_multimodal_inputs", None)
    get_processing_info = getattr(registry, "get_processing_info", None)
    get_multimodal_config = getattr(model_config, "get_multimodal_config", None)
    if not all(
        callable(method)
        for method in (
            supports_inputs,
            get_processing_info,
            get_multimodal_config,
        )
    ):
        raise UnsupportedRuntimeError(
            "vLLM multimodal capability discovery contract differs"
        )

    try:
        supported = supports_inputs(model_config)
        if not isinstance(supported, bool):
            raise UnsupportedRuntimeError(
                "vLLM multimodal registry support result is not boolean"
            )
        if not supported:
            return frozenset()
        processing_info = get_processing_info(model_config)
        supported_limits = processing_info.supported_mm_limits
        multimodal_config = get_multimodal_config()
        get_limit_per_prompt = multimodal_config.get_limit_per_prompt
    except (AttributeError, TypeError, ValueError) as error:
        raise UnsupportedRuntimeError(
            "cannot discover vLLM multimodal input capabilities"
        ) from error

    if not isinstance(supported_limits, Mapping) or not callable(
        get_limit_per_prompt
    ):
        raise UnsupportedRuntimeError(
            "vLLM multimodal capability discovery contract differs"
        )

    enabled: set[str] = set()
    for modality, model_limit in supported_limits.items():
        if not isinstance(modality, str) or not modality:
            raise UnsupportedRuntimeError(
                "vLLM reported an invalid multimodal input name"
            )
        if model_limit is not None and (
            isinstance(model_limit, bool)
            or not isinstance(model_limit, int)
            or model_limit < 0
        ):
            raise UnsupportedRuntimeError(
                f"vLLM reported an invalid limit for modality {modality!r}"
            )
        if model_limit == 0:
            continue
        try:
            configured_limit = get_limit_per_prompt(modality)
        except (KeyError, TypeError, ValueError) as error:
            raise UnsupportedRuntimeError(
                f"cannot read vLLM's configured limit for modality {modality!r}"
            ) from error
        if (
            isinstance(configured_limit, bool)
            or not isinstance(configured_limit, int)
            or configured_limit < 0
        ):
            raise UnsupportedRuntimeError(
                f"vLLM reported an invalid configured limit for modality {modality!r}"
            )
        if configured_limit > 0:
            enabled.add(modality)
    return frozenset(enabled)


def _json_safe(value: Any) -> Any:
    return _json_safe_inner(value, active=set())


def _json_safe_inner(value: Any, *, active: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, enum.Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _json_safe_inner(value.value, active=active),
        }
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return {"callable": f"{module}.{qualname}"}

    identity = id(value)
    if identity in active:
        raise UnsupportedRuntimeError(
            "vLLM configuration contains a recursive public value"
        )
    active.add(identity)
    try:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return _json_safe_inner(to_dict(), active=active)
            except Exception as error:
                raise UnsupportedRuntimeError(
                    "cannot serialize a public vLLM configuration"
                ) from error
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key)
                if normalized in result:
                    raise UnsupportedRuntimeError(
                        "vLLM configuration keys are not canonically unique"
                    )
                result[normalized] = _json_safe_inner(item, active=active)
            return result
        if isinstance(value, (list, tuple)):
            return [_json_safe_inner(item, active=active) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [
                _json_safe_inner(item, active=active) for item in value
            ]
            return sorted(
                normalized,
                key=lambda item: sha256_json({"item": item}),
            )
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field_info.name: _json_safe_inner(
                    getattr(value, field_info.name), active=active
                )
                for field_info in fields(value)
                if not field_info.name.startswith("_")
                and field_info.name != "compute_hash"
            }
        try:
            raw_public = vars(value)
        except TypeError:
            raw_public = {}
        public = {
            name: item
            for name, item in raw_public.items()
            if not name.startswith("_") and name != "compute_hash"
        }
        if public:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    name: _json_safe_inner(item, active=active)
                    for name, item in public.items()
                },
            }
        rendered = str(value)
        if re.search(r"\bat 0x[0-9a-fA-F]+\b", rendered):
            raise UnsupportedRuntimeError(
                "vLLM configuration contains an opaque public value"
            )
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": rendered,
        }
    finally:
        active.remove(identity)


def _public_config_snapshot(
    value: Any,
    *,
    excluded: frozenset[str] = frozenset(),
) -> Any:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        names = tuple(field_info.name for field_info in fields(value))
    else:
        try:
            names = tuple(vars(value))
        except TypeError as error:
            raise UnsupportedRuntimeError(
                "cannot enumerate a public vLLM configuration"
            ) from error
    return {
        name: _json_safe(getattr(value, name))
        for name in names
        if not name.startswith("_")
        and name != "compute_hash"
        and name not in excluded
    }


def _compute_hash(value: Any, *, label: str) -> str:
    compute_hash = getattr(value, "compute_hash", None)
    if not callable(compute_hash):
        raise UnsupportedRuntimeError(f"{label} lacks public compute_hash()")
    try:
        result = compute_hash()
    except Exception as error:
        raise UnsupportedRuntimeError(f"cannot compute {label} identity") from error
    if not isinstance(result, str) or not result:
        raise UnsupportedRuntimeError(f"{label} compute_hash() is invalid")
    return result


def _component_hashes(vllm_config: Any) -> Mapping[str, str]:
    if is_dataclass(vllm_config) and not isinstance(vllm_config, type):
        names = tuple(field_info.name for field_info in fields(vllm_config))
    else:
        try:
            names = tuple(vars(vllm_config))
        except TypeError as error:
            raise UnsupportedRuntimeError(
                "cannot enumerate VllmConfig public components"
            ) from error
    result: dict[str, str] = {}
    for name in names:
        if name.startswith("_") or name == "compute_hash":
            continue
        component = getattr(vllm_config, name, None)
        if component is not None and callable(getattr(component, "compute_hash", None)):
            result[name] = _compute_hash(component, label=f"VllmConfig.{name}")
    return result


def _runtime_integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UnsupportedRuntimeError(f"{label} is not a valid runtime integer")
    return value


def _get_public_spec_kind_resolver() -> Callable[[object], object]:
    """Load vLLM's public cache-spec semantic classifier or fail closed."""

    try:
        from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind
    except (ImportError, AttributeError) as error:
        raise UnsupportedRuntimeError(
            "vLLM lacks the public KV cache semantic-kind resolver"
        ) from error
    if not callable(get_kv_cache_spec_kind):
        raise UnsupportedRuntimeError(
            "vLLM KV cache semantic-kind resolver is not callable"
        )
    return get_kv_cache_spec_kind


def _build_deployment_identity(
    vllm_config: Any,
    layout: HMALayout,
    *,
    model_namespace_sha256: str,
    chunk_tokens: int,
    vllm_version: str,
    vllm_build_sha256: str,
) -> DeploymentIdentity:
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    model_payload = _public_config_snapshot(
        model,
        # Never fold an authentication secret or API-facing served alias into
        # a persistent cache key. The public locator/revision namespace is
        # bound separately; the remaining model view captures execution facts.
        excluded=frozenset({"hf_token", "served_model_name"}),
    )

    vllm_config_hash = _compute_hash(vllm_config, label="VllmConfig")
    multimodal = getattr(model, "multimodal_config", None)
    if multimodal is None:
        get_multimodal_config = getattr(model, "get_multimodal_config", None)
        if callable(get_multimodal_config):
            try:
                multimodal = get_multimodal_config()
            except Exception as error:
                raise UnsupportedRuntimeError(
                    "cannot read the public multimodal configuration"
                ) from error
    execution_payload = {
        "schema": "spoolcache-vllm-execution-config/v1",
        "vllm_config_compute_hash": vllm_config_hash,
        # Automatically consume every public component hash offered by this
        # vLLM build, including components introduced by future releases.
        "component_compute_hashes": _component_hashes(vllm_config),
        # vLLM intentionally omits some non-graph settings from compute_hash.
        # These public supplements bind the complete opaque-KV interpretation
        # and every multimodal preprocessing option without model branches.
        "attention_config": _public_config_snapshot(
            getattr(vllm_config, "attention_config", None)
        ),
        "cache_config": _public_config_snapshot(
            cache,
            excluded=frozenset(
                {
                    "gpu_memory_utilization",
                    "num_gpu_blocks_override",
                    "num_gpu_blocks",
                    "num_cpu_blocks",
                    "kv_cache_size_tokens",
                    "kv_cache_max_concurrency",
                    "kv_cache_memory_bytes",
                }
            ),
        ),
        "mamba_config": _public_config_snapshot(
            getattr(vllm_config, "mamba_config", None)
        ),
        "quant_config": _public_config_snapshot(
            getattr(vllm_config, "quant_config", None)
        ),
        "speculative_config": _public_config_snapshot(
            getattr(vllm_config, "speculative_config", None)
        ),
        "multimodal_config": _public_config_snapshot(multimodal),
    }

    tp = _runtime_integer(
        getattr(parallel, "tensor_parallel_size", None),
        label="tensor_parallel_size",
        minimum=1,
    )
    pp = _runtime_integer(
        getattr(parallel, "pipeline_parallel_size", None),
        label="pipeline_parallel_size",
        minimum=1,
    )
    dcp = _runtime_integer(
        getattr(parallel, "decode_context_parallel_size", 1),
        label="decode_context_parallel_size",
        minimum=1,
    )
    dp = _runtime_integer(
        getattr(parallel, "data_parallel_size", 1),
        label="data_parallel_size",
        minimum=1,
    )
    raw_dp_rank = getattr(parallel, "data_parallel_rank", 0)
    if raw_dp_rank is None and dp == 1:
        raw_dp_rank = 0
    dp_rank = _runtime_integer(
        raw_dp_rank,
        label="data_parallel_rank",
        minimum=0,
    )
    world_size = _runtime_integer(
        getattr(parallel, "world_size", tp * pp),
        label="world_size",
        minimum=1,
    )
    world_size_across_dp = _runtime_integer(
        getattr(parallel, "world_size_across_dp", world_size * dp),
        label="world_size_across_dp",
        minimum=1,
    )
    topology = {
        "tp": tp,
        "pp": pp,
        "dcp": dcp,
        "dp": dp,
        "dp_rank": dp_rank,
        "world_size": world_size,
        "world_size_across_dp": world_size_across_dp,
    }
    return DeploymentIdentity(
        schema="spoolcache-deployment/v2",
        profile=layout.profile,
        model_namespace_sha256=model_namespace_sha256,
        model_config_sha256=sha256_json(_json_safe(model_payload)),
        execution_config_sha256=sha256_json(_json_safe(execution_payload)),
        vllm_version=vllm_version,
        vllm_build_sha256=vllm_build_sha256,
        kv_cache_dtype=str(getattr(cache, "cache_dtype", "unknown")),
        topology=topology,
        # This is a role-local deployment identity.  vLLM may expose a reduced
        # ModelConfig to the scheduler and the fully materialized config to a
        # worker, so their model_config_sha256 values need not match.  The
        # connector separately checks its smaller cross-role coordination
        # receipt before accepting worker inventory.  RankIdentity and each
        # manifest additionally bind layout.digest (physical geometry).
        layout_sha256=layout.logical_digest,
        chunk_tokens=chunk_tokens,
        spoolcache_version=SPOOLCACHE_VERSION,
    )


__all__ = [
    "SpoolCacheConnector",
    "SpoolCacheHandshakeMetadata",
    "SpoolCacheMetadata",
    "SpoolCachePlan",
    "SpoolCacheStartupInventory",
    "SpoolCacheStats",
]
