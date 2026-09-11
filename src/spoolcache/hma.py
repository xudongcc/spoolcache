"""Fail-closed HMA page geometry discovered from vLLM at startup.

The module intentionally uses duck typing instead of importing vLLM.  Both
the scheduler and worker can therefore derive the same identity from a
``KVCacheConfig``-shaped object, while CPU-only tests exercise the contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .errors import LayoutError
from .identity import sha256_json
from .manifest import TokenSnapshot

# This is an internal persisted protocol identity, not an operator-selectable
# profile or model compatibility gate. Every deployment uses runtime discovery.
VLLM_RUNTIME_KV_PROFILE = "vllm-runtime-kv-v1"
# v2 invalidated snapshots captured after a forward had already recycled the
# sliding-window pages for their advertised boundary. v3 recorded the vLLM V1
# connector timing for align-mode recurrent state, but assumed that the next
# running-state page was adjacent to the restored boundary. v4 follows the
# runtime block-table contract instead: chunked scheduling may leave null gaps
# before the current running-state page, followed by runtime-declared
# speculative state pages.
LAYOUT_SCHEMA = "spoolcache-hma-layout/v4"


@dataclass(frozen=True)
class LayerGeometry:
    name: str
    spec_name: str
    page_size_bytes: int

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 512:
            raise LayoutError("HMA layer name is invalid")
        if not self.spec_name or self.page_size_bytes <= 0:
            raise LayoutError("HMA layer page geometry is incomplete")


@dataclass(frozen=True)
class GroupGeometry:
    group_index: int
    spec_name: str
    block_size: int
    storage_block_size: int
    manager_page_size_bytes: int
    dcp_replicated: bool
    dcp_shard_count: int
    logical_tokens_per_page: int
    reuse_policy: str
    reuse_window_tokens: int | None
    running_state_tail_pages: int
    is_eagle_group: bool
    layers: tuple[LayerGeometry, ...]

    def __post_init__(self) -> None:
        integers = (
            self.group_index,
            self.block_size,
            self.storage_block_size,
            self.manager_page_size_bytes,
            self.dcp_shard_count,
            self.logical_tokens_per_page,
            self.running_state_tail_pages,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integers
        ):
            raise LayoutError("HMA group integer geometry is malformed")
        positive_integers = (
            self.block_size,
            self.storage_block_size,
            self.manager_page_size_bytes,
            self.dcp_shard_count,
            self.logical_tokens_per_page,
        )
        if self.group_index < 0 or any(value <= 0 for value in positive_integers):
            raise LayoutError("HMA group geometry must be positive")
        if self.storage_block_size > self.block_size:
            raise LayoutError("HMA storage block exceeds its logical block")
        if self.logical_tokens_per_page != self.block_size * self.dcp_shard_count:
            raise LayoutError("HMA logical page width is inconsistent")
        if self.reuse_policy not in {
            "full",
            "sliding",
            "recurrent_align",
            "circular_one",
        }:
            raise LayoutError("HMA group reuse policy is unsupported")
        if self.reuse_policy == "sliding":
            if self.reuse_window_tokens is None or self.reuse_window_tokens <= 1:
                raise LayoutError("sliding HMA group has an invalid window")
        elif self.reuse_window_tokens is not None:
            raise LayoutError("non-sliding HMA group declares a window")
        if self.running_state_tail_pages < 0:
            raise LayoutError("HMA running-state tail must not be negative")
        if (
            self.reuse_policy != "recurrent_align"
            and self.running_state_tail_pages != 0
        ):
            raise LayoutError("only recurrent HMA groups may declare a state tail")
        names = tuple(layer.name for layer in self.layers)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise LayoutError("HMA group layers must be sorted and unique")

    def selected_page_count(self, span_tokens: int) -> int:
        if isinstance(span_tokens, bool) or not isinstance(span_tokens, int):
            raise LayoutError("HMA span must be an integer")
        if span_tokens <= 0:
            raise LayoutError("HMA span must be positive")
        # A one-page non-prefix state is request-owned regardless of prefix
        # length. Its block_size is scratch capacity, not a durable prefix
        # boundary quantum, so it must not constrain cache alignment.
        if self.reuse_policy == "circular_one":
            return 1
        if span_tokens % self.logical_tokens_per_page:
            raise LayoutError("HMA span is not aligned to every manager group")
        required = span_tokens // self.logical_tokens_per_page
        if self.reuse_policy == "full":
            return required
        if self.reuse_policy == "sliding":
            assert self.reuse_window_tokens is not None
            window_pages = math.ceil(
                (self.reuse_window_tokens - 1) / self.logical_tokens_per_page
            )
            return min(required, window_pages)
        if self.reuse_policy == "recurrent_align":
            return 1
        raise AssertionError("unreachable reuse policy")

    def select_physical_pages(
        self,
        block_table: Sequence[int],
        span_tokens: int,
    ) -> tuple[int, ...]:
        selected = self.selected_page_count(span_tokens)
        boundary_pages = span_tokens // self.logical_tokens_per_page
        if self.reuse_policy == "circular_one":
            if len(block_table) != 1:
                raise LayoutError("circular HMA group must expose exactly one page")
            pages = (block_table[0],)
        elif self.reuse_policy == "recurrent_align":
            # vLLM's align-mode recurrent preprocessing runs before the
            # connector hook. It copies the completed boundary state into the
            # current step's running-state page. A large chunk may span several
            # recurrent blocks, leaving null placeholders between the boundary
            # and that current page; speculative state pages, when declared by
            # the runtime cache spec, trail it. Select from that public
            # block-table lifecycle instead of assuming adjacency. This rule
            # contains no model identity or fixed layer geometry.
            active_index = len(block_table) - 1 - self.running_state_tail_pages
            if active_index < boundary_pages:
                raise LayoutError(
                    "recurrent HMA block table lacks the active running-state page"
                )
            pages = (block_table[active_index],)
        else:
            if len(block_table) < boundary_pages:
                raise LayoutError("request block table is shorter than the cache span")
            pages = tuple(block_table[boundary_pages - selected : boundary_pages])
        if any(
            isinstance(page, bool) or not isinstance(page, int) or page <= 0
            for page in pages
        ):
            raise LayoutError("selected HMA page contains vLLM's null/invalid block")
        if len(set(pages)) != len(pages):
            raise LayoutError("selected HMA page range contains duplicates")
        return pages


@dataclass(frozen=True)
class HMALayout:
    num_manager_blocks: int
    dcp_degree: int
    groups: tuple[GroupGeometry, ...]
    schema: str = LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LAYOUT_SCHEMA:
            raise LayoutError("unknown HMA layout schema")
        if self.num_manager_blocks <= 0 or self.dcp_degree <= 0:
            raise LayoutError("HMA manager capacity and DCP degree must be positive")
        if tuple(group.group_index for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise LayoutError("HMA groups must be complete and ordered")
        names = [layer.name for group in self.groups for layer in group.layers]
        if not names:
            raise LayoutError("HMA layout has no worker-owned layers")
        if len(names) != len(set(names)):
            raise LayoutError("one HMA layer belongs to multiple groups")

    @property
    def profile(self) -> str:
        """Return the fixed persisted runtime-layout protocol identity."""

        return VLLM_RUNTIME_KV_PROFILE

    @property
    def digest(self) -> str:
        """Physical worker layout used to validate rank-local payload bytes."""

        # ``num_manager_blocks`` is allocation capacity, not persisted page
        # geometry.  vLLM derives it from the free GPU-memory measurement, so
        # it may legitimately vary between otherwise identical boots.  Keep
        # the value for runtime bounds checking, but never let that ephemeral
        # capacity invalidate durable cache entries.
        return sha256_json(
            {
                "schema": self.schema,
                "profile": self.profile,
                "dcp_degree": self.dcp_degree,
                "groups": [asdict(group) for group in self.groups],
            }
        )

    @property
    def logical_digest(self) -> str:
        """Cross-role layout identity used to build logical prefix keys.

        vLLM gives the scheduler a logical KV configuration and gives workers
        the packed physical page geometry after model allocation.  Page byte
        sizes can therefore differ legitimately between those roles.  Prefix
        IDs must bind the shared group/layer/reuse semantics, while rank-local
        manifests continue to bind :attr:`digest`, which includes every byte
        size.  Keeping the two identities separate prevents both false misses
        and unsafe cross-layout payload reuse.
        """

        return sha256_json(
            {
                "schema": self.schema,
                "profile": self.profile,
                "dcp_degree": self.dcp_degree,
                "groups": [
                    {
                        "group_index": group.group_index,
                        "block_size": group.block_size,
                        "dcp_replicated": group.dcp_replicated,
                        "dcp_shard_count": group.dcp_shard_count,
                        "logical_tokens_per_page": group.logical_tokens_per_page,
                        "reuse_policy": group.reuse_policy,
                        "reuse_window_tokens": group.reuse_window_tokens,
                        "running_state_tail_pages": (
                            group.running_state_tail_pages
                        ),
                        "is_eagle_group": group.is_eagle_group,
                        "layers": [layer.name for layer in group.layers],
                    }
                    for group in self.groups
                ],
            }
        )

    @property
    def coordination_digest(self) -> str:
        """Page-selection semantics shared by every pipeline stage.

        vLLM projects global cache groups onto each PP worker.  Layer names,
        layer counts, EAGLE ownership, and physical byte geometry are therefore
        stage-local facts.  The scheduler only needs the ordered manager-group
        semantics that make one block-table plan valid on every stage.
        """

        return sha256_json(
            {
                "schema": self.schema,
                "profile": self.profile,
                "dcp_degree": self.dcp_degree,
                "groups": [
                    {
                        "group_index": group.group_index,
                        "block_size": group.block_size,
                        "dcp_replicated": group.dcp_replicated,
                        "dcp_shard_count": group.dcp_shard_count,
                        "logical_tokens_per_page": group.logical_tokens_per_page,
                        "reuse_policy": group.reuse_policy,
                        "reuse_window_tokens": group.reuse_window_tokens,
                        "running_state_tail_pages": (
                            group.running_state_tail_pages
                        ),
                    }
                    for group in self.groups
                ],
            }
        )

    @property
    def alignment_tokens(self) -> int:
        widths = tuple(
            group.logical_tokens_per_page
            for group in self.groups
            if group.reuse_policy != "circular_one"
        )
        return math.lcm(*widths) if widths else 1

    def selected_page_counts(self, span_tokens: int) -> tuple[int, ...]:
        return tuple(group.selected_page_count(span_tokens) for group in self.groups)

    def select_physical_pages(
        self,
        block_tables: Sequence[Sequence[int]],
        span_tokens: int,
    ) -> tuple[tuple[int, ...], ...]:
        if len(block_tables) != len(self.groups):
            raise LayoutError("request block tables disagree with HMA groups")
        return tuple(
            group.select_physical_pages(table, span_tokens)
            for group, table in zip(self.groups, block_tables, strict=True)
        )

    def validate_manifest_coverage(self, manifest: TokenSnapshot) -> None:
        """Prove that a manifest covers every required opaque page exactly once."""

        if manifest.profile != self.profile or manifest.layout_digest != self.digest:
            raise LayoutError("manifest layout identity differs from this worker")
        expected_counts = self.selected_page_counts(manifest.span_tokens)
        # Files arrive in prefix order, with boundary state last. Keep one
        # cursor per layer instead of retaining every file/layer descriptor.
        expected = {
            (group.group_index, layer.name): (count, layer.page_size_bytes)
            for group, count in zip(self.groups, expected_counts, strict=True)
            for layer in group.layers
        }
        cursors: dict[tuple[int, str], int] = {}
        for descriptor in manifest.objects:
            for segment in descriptor.segments:
                key = (segment.group_index, segment.layer_name)
                if key not in expected:
                    raise LayoutError("manifest does not contain exactly every HMA layer")
                if segment.page_start != cursors.get(key, 0):
                    raise LayoutError("manifest HMA page ranges contain a gap")
                if segment.byte_length != segment.page_count * expected[key][1]:
                    raise LayoutError("manifest HMA object byte length differs")
                cursors[key] = segment.page_start + segment.page_count
        if set(cursors) != set(expected):
            raise LayoutError("manifest does not contain exactly every HMA layer")
        if any(cursors[key] != count for key, (count, _) in expected.items()):
            raise LayoutError("manifest HMA page coverage is incomplete")


_MISSING = object()


def _optional_bool_capability(spec: object, name: str) -> bool | None:
    """Read one public reuse capability without accepting truthy lookalikes."""

    try:
        value = getattr(spec, name, _MISSING)
    except Exception as error:
        raise LayoutError(f"cannot read HMA {name} capability") from error
    if value is _MISSING:
        return None
    if not isinstance(value, bool):
        raise LayoutError(f"HMA {name} capability must be boolean")
    return value


def _optional_contract(spec: object, name: str, *, label: str) -> object:
    """Read one public method contract and normalize descriptor failures."""

    try:
        return getattr(spec, name, _MISSING)
    except Exception as error:
        raise LayoutError(f"cannot read non-prefix HMA {label} contract") from error


def _prefix_participation_state(spec: object) -> bool | None:
    """Return an explicit prefix-sharing capability, if one is advertised."""

    markers = tuple(
        value
        for value in (
            _optional_bool_capability(spec, "participates_in_prefix_caching"),
            _optional_bool_capability(spec, "prefix_cacheable"),
        )
        if value is not None
    )
    if True in markers and False in markers:
        raise LayoutError("HMA prefix participation capabilities conflict")
    if not markers:
        return None
    return markers[0]


def _is_non_prefix_state(spec: object) -> bool:
    return _prefix_participation_state(spec) is False


def _single_page_non_prefix_policy(
    spec: object,
    *,
    vllm_config: object | None,
) -> tuple[str, None]:
    """Prove request ownership and physical size for opaque scratch state."""

    if vllm_config is None:
        raise LayoutError("non-prefix HMA state requires the current vLLM config")
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = _positive_int(
        getattr(model_config, "max_model_len", None),
        "vLLM maximum model length",
    )

    admission = _optional_contract(
        spec,
        "max_admission_blocks_per_request",
        label="admission-block",
    )
    if admission is not _MISSING:
        if not callable(admission):
            raise LayoutError(
                "non-prefix HMA state has an invalid admission-block contract"
            )
        max_in_flight_tokens = _positive_int(
            getattr(vllm_config, "max_in_flight_tokens", None),
            "vLLM maximum in-flight token count",
        )
        try:
            admission_blocks = _positive_int(
                admission(
                    max_in_flight_tokens=max_in_flight_tokens,
                    max_model_len=max_model_len,
                ),
                "non-prefix HMA admission-block result",
            )
        except LayoutError:
            raise
        except Exception as error:
            raise LayoutError(
                "cannot evaluate non-prefix HMA admission-block contract"
            ) from error
        if admission_blocks != 1:
            raise LayoutError(
                "non-prefix HMA state must admit exactly one admission block "
                "per request"
            )

    block_table = _optional_contract(
        spec,
        "max_num_blocks_per_req",
        label="block-table ownership",
    )
    if not callable(block_table):
        raise LayoutError(
            "non-prefix HMA state lacks a block-table ownership contract"
        )
    memory_usage = _optional_contract(
        spec,
        "max_memory_usage_bytes",
        label="physical-memory bound",
    )
    if not callable(memory_usage):
        raise LayoutError(
            "non-prefix HMA state lacks a physical-memory bound contract"
        )
    try:
        block_table_blocks = _positive_int(
            block_table(vllm_config, max_model_len),
            "non-prefix HMA block-table result",
        )
        memory_bytes = _positive_int(
            memory_usage(vllm_config),
            "non-prefix HMA memory-bound result",
        )
    except LayoutError:
        raise
    except Exception as error:
        raise LayoutError(
            "cannot evaluate non-prefix HMA page-ownership contracts"
        ) from error
    if block_table_blocks != 1:
        raise LayoutError(
            "non-prefix HMA state must own exactly one block-table page "
            "per request"
        )
    page_size_bytes = _positive_int(
        getattr(spec, "page_size_bytes", None),
        "non-prefix HMA page size",
    )
    if memory_bytes != page_size_bytes:
        raise LayoutError(
            "non-prefix HMA state must occupy exactly one physical page "
            "per request"
        )
    return "circular_one", None


def _public_semantic_kind(
    spec: object,
    resolver: Callable[[object], object],
) -> str:
    if not callable(resolver):
        raise LayoutError("vLLM KV cache semantic-kind resolver is not callable")
    try:
        result = resolver(spec)
    except Exception as error:
        raise LayoutError("cannot resolve public KV cache semantic kind") from error
    try:
        value = getattr(result, "value", result)
    except Exception as error:
        raise LayoutError("cannot read public KV cache semantic kind") from error
    if not isinstance(value, str) or not value:
        raise LayoutError("vLLM returned an invalid public KV cache semantic kind")
    return value


def _reuse_policy(
    spec: object,
    *,
    vllm_config: object | None,
    spec_kind_resolver: Callable[[object], object],
    declared_kind: str | None = None,
) -> tuple[str, int | None]:
    # Scratch managers are intentionally outside vLLM's reusable-prefix kind
    # taxonomy. Accept any concrete class only after its public capabilities
    # prove non-participation, one request-owned block-table page, and one
    # physical page under this deployment's real bounds.
    if _is_non_prefix_state(spec):
        return _single_page_non_prefix_policy(spec, vllm_config=vllm_config)

    kind = (
        declared_kind
        if declared_kind is not None
        else _public_semantic_kind(spec, spec_kind_resolver)
    )
    if kind in {"sliding_window", "sliding_window_mla"}:
        window = _positive_int(getattr(spec, "sliding_window", None), "sliding window")
        if window <= 1:
            raise LayoutError("sliding window must exceed one token")
        return "sliding", window
    if kind in {"full_attention", "mla_attention"}:
        return "full", None
    if kind == "mamba":
        if getattr(spec, "mamba_cache_mode", None) != "align":
            raise LayoutError("recurrent HMA state requires mamba_cache_mode='align'")
        return "recurrent_align", None
    raise LayoutError(f"unsupported public KV cache semantic kind: {kind!r}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LayoutError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LayoutError(f"{label} must be a non-negative integer")
    return value


def build_hma_layout(
    kv_cache_config: object,
    *,
    dcp_degree: int = 1,
    vllm_config: object | None = None,
    spec_kind_resolver: Callable[[object], object],
) -> HMALayout:
    """Validate and freeze the layout advertised by vLLM at engine startup.

    Any number of groups is accepted when its page-selection semantics are
    explicitly understood by :func:`_reuse_policy`. It is discovery, not
    guesswork: an unknown spec still fails closed before a cache file can be
    read or written.
    """

    dcp_degree = _positive_int(dcp_degree, "DCP degree")
    num_blocks = _positive_int(
        getattr(kv_cache_config, "num_blocks", None), "manager block count"
    )
    runtime_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if not runtime_groups:
        raise LayoutError("vLLM reported no KV cache groups")
    groups: list[GroupGeometry] = []
    assigned: set[str] = set()
    for index, runtime_group in enumerate(runtime_groups):
        names = tuple(sorted(tuple(getattr(runtime_group, "layer_names", ()) or ())))
        if len(names) != len(set(names)):
            raise LayoutError(f"HMA group {index} requires unique layers")
        if assigned.intersection(names):
            raise LayoutError("one runtime layer belongs to multiple HMA groups")
        assigned.update(names)

        group_spec = getattr(runtime_group, "kv_cache_spec", None)
        block_size = _positive_int(
            getattr(group_spec, "block_size", None), f"group {index} block size"
        )
        storage_block_size = _positive_int(
            getattr(group_spec, "storage_block_size", block_size),
            f"group {index} storage block size",
        )
        manager_page_size = _positive_int(
            getattr(group_spec, "page_size_bytes", None),
            f"group {index} manager page size",
        )
        per_layer = getattr(group_spec, "kv_cache_specs", None)
        if per_layer is not None and not isinstance(per_layer, Mapping):
            raise LayoutError("per-layer HMA specifications must be a mapping")
        if (
            isinstance(per_layer, Mapping)
            and names
            and set(per_layer) != set(names)
        ):
            raise LayoutError("per-layer HMA specifications disagree with group layers")
        group_prefix_state = _prefix_participation_state(group_spec)
        group_semantic_kind: str | None = None
        if isinstance(per_layer, Mapping):
            if group_prefix_state is not False:
                # vLLM's public resolver can classify a registered uniform
                # group even when its concrete member implementations have
                # different or opaque types. Prefer that aggregate
                # declaration; UNKNOWN still falls back to each member's own
                # public declaration.
                resolved_group_kind = _public_semantic_kind(
                    group_spec,
                    spec_kind_resolver,
                )
                if resolved_group_kind != "unknown":
                    group_semantic_kind = resolved_group_kind

        layers: list[LayerGeometry] = []
        policies: set[tuple[str, int | None]] = set()
        recurrent_tail_pages: set[int] = set()
        for name in names:
            layer_spec = per_layer[name] if isinstance(per_layer, Mapping) else group_spec
            layer_block = _positive_int(
                getattr(layer_spec, "block_size", block_size),
                f"layer {name} block size",
            )
            if layer_block != block_size:
                raise LayoutError("one HMA group mixes different layer block sizes")
            page_size = _positive_int(
                getattr(layer_spec, "page_size_bytes", None),
                f"layer {name} page size",
            )
            layer_policy = _reuse_policy(
                layer_spec,
                vllm_config=vllm_config,
                spec_kind_resolver=spec_kind_resolver,
                declared_kind=group_semantic_kind,
            )
            policies.add(layer_policy)
            if layer_policy[0] == "recurrent_align":
                recurrent_tail_pages.add(
                    _nonnegative_int(
                        getattr(layer_spec, "num_speculative_blocks", 0),
                        f"layer {name} recurrent state tail",
                    )
                )
            layers.append(
                LayerGeometry(
                    name=name,
                    spec_name=type(layer_spec).__name__,
                    page_size_bytes=page_size,
                )
            )
        if not names:
            semantic_specs = (
                tuple(per_layer.values())
                if isinstance(per_layer, Mapping) and per_layer
                else (group_spec,)
            )
            for semantic_spec in semantic_specs:
                layer_policy = _reuse_policy(
                    semantic_spec,
                    vllm_config=vllm_config,
                    spec_kind_resolver=spec_kind_resolver,
                    declared_kind=group_semantic_kind,
                )
                policies.add(layer_policy)
                if layer_policy[0] == "recurrent_align":
                    recurrent_tail_pages.add(
                        _nonnegative_int(
                            getattr(semantic_spec, "num_speculative_blocks", 0),
                            f"group {index} recurrent state tail",
                        )
                    )
        if len(policies) != 1:
            raise LayoutError("one HMA group mixes incompatible reuse policies")
        policy, window = policies.pop()
        if policy == "recurrent_align":
            if len(recurrent_tail_pages) != 1:
                raise LayoutError(
                    "one recurrent HMA group mixes different state tails"
                )
            running_state_tail_pages = recurrent_tail_pages.pop()
        else:
            running_state_tail_pages = 0
        if isinstance(per_layer, Mapping):
            if (
                group_prefix_state is False and policy != "circular_one"
            ) or (
                group_prefix_state is True and policy == "circular_one"
            ) or (
                group_semantic_kind is not None and policy == "circular_one"
            ):
                raise LayoutError(
                    "HMA group/member prefix participation conflict"
                )
            if policy == "circular_one":
                # Prove the contract used by the actual shared allocator and
                # block table, not only the contracts exposed by its members.
                _single_page_non_prefix_policy(
                    group_spec,
                    vllm_config=vllm_config,
                )
            if names and manager_page_size != sum(
                layer.page_size_bytes for layer in layers
            ):
                raise LayoutError("packed manager page size differs from its layers")
        elif names and any(
            layer.page_size_bytes != manager_page_size for layer in layers
        ):
            raise LayoutError("manager and layer page sizes disagree")

        dcp_replicated_raw = getattr(group_spec, "dcp_replicated", False)
        eagle_raw = getattr(runtime_group, "is_eagle_group", False)
        if not isinstance(dcp_replicated_raw, bool) or not isinstance(eagle_raw, bool):
            raise LayoutError("HMA DCP/EAGLE flags must be booleans")
        dcp_replicated = dcp_replicated_raw
        dcp_shards = 1 if dcp_replicated else dcp_degree
        groups.append(
            GroupGeometry(
                group_index=index,
                spec_name=type(group_spec).__name__,
                block_size=block_size,
                storage_block_size=storage_block_size,
                manager_page_size_bytes=manager_page_size,
                dcp_replicated=dcp_replicated,
                dcp_shard_count=dcp_shards,
                logical_tokens_per_page=block_size * dcp_shards,
                reuse_policy=policy,
                reuse_window_tokens=window,
                running_state_tail_pages=running_state_tail_pages,
                is_eagle_group=eagle_raw,
                layers=tuple(layers),
            )
        )
    return HMALayout(
        num_manager_blocks=num_blocks,
        dcp_degree=dcp_degree,
        groups=tuple(groups),
    )


__all__ = [
    "GroupGeometry",
    "HMALayout",
    "LAYOUT_SCHEMA",
    "LayerGeometry",
    "VLLM_RUNTIME_KV_PROFILE",
    "build_hma_layout",
]
