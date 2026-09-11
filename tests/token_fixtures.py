"""Model-independent token-file page geometry and deterministic CPU capture."""
import types
from spoolcache.hma import GroupGeometry, HMALayout, LayerGeometry
from spoolcache.token_files import TokenFileStore
from spoolcache.token_mover import TokenPageMover

def layout_for(policy="full", tokens_per_page=64, page_bytes=4096, dcp=1):
    return HMALayout(
        num_manager_blocks=100,
        dcp_degree=dcp,
        groups=(
            GroupGeometry(
                group_index=0,
                spec_name="fixture",
                block_size=tokens_per_page // dcp,
                storage_block_size=tokens_per_page // dcp,
                manager_page_size_bytes=page_bytes,
                dcp_replicated=False,
                dcp_shard_count=dcp,
                logical_tokens_per_page=tokens_per_page,
                reuse_policy=policy,
                reuse_window_tokens=513 if policy == "sliding" else None,
                running_state_tail_pages=0,
                is_eagle_group=False,
                layers=(LayerGeometry("layer", "fixture", page_bytes),),
            ),
        ),
    )


def cpu_mover(layout, slot_bytes=16384):
    """Use production capture selection with a deterministic page-byte provider."""
    mover = object.__new__(TokenPageMover)
    mover.layout = layout
    mover._closed = False
    mover._pool = types.SimpleNamespace(slot_bytes=slot_bytes)
    mover._bound = {(0, "layer"): None}
    def capture(segments, selected, *, byte_start=0, byte_length=None):
        data = b"".join(
            bytes((page % 251,)) * (segment.byte_length // segment.page_count)
            for segment in segments
            for page in selected[segment.group_index][
                segment.page_start : segment.page_start + segment.page_count
            ]
        )
        yield memoryview(data)[byte_start : None if byte_length is None else byte_start + byte_length]
    mover._capture_packed = capture
    return mover


def open_store(root, **kwargs):
    layout = kwargs.pop('layout', layout_for(tokens_per_page=256, page_bytes=16384))
    options = dict(slot_bytes=16384, slot_count=1,
        expected_deployment_digest='b' * 64, expected_rank_digest='c' * 64,
        expected_rank=0, expected_topology_digest='d' * 64,
        expected_profile=layout.profile, expected_layout_digest=layout.digest)
    options.update(kwargs)
    return TokenFileStore(root, layout=layout, **options)
