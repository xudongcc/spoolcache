#!/usr/bin/env python3
"""Offline, full-payload verification for one rank-local cache entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path

from spoolcache.hma import VLLM_RUNTIME_KV_PROFILE
from spoolcache.manifest import RankManifest, decode_manifest
from spoolcache.topology import WorkerCoordinate


class VerificationError(ValueError):
    """The authenticated manifest does not describe the expected deployment."""


_LOWER_HEX = frozenset("0123456789abcdef")


def validate_expected_inputs(
    *,
    entry_id: str,
    span_tokens: int,
    deployment_identity_digest: str,
    rank_identity_digest: str,
    physical_rank: int,
    topology_digest: str,
    layout_digest: str,
    tp_degree: int,
    pp_degree: int,
    dcp_degree: int,
    pp_rank: int,
    tp_rank: int,
    dcp_rank: int,
    groups: int,
    layers: int,
    group_layers: str,
    pages: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate operator-supplied identity before deriving any filesystem path."""

    digests = {
        "--entry-id": entry_id,
        "--expected-deployment-identity-digest": deployment_identity_digest,
        "--expected-rank-identity-digest": rank_identity_digest,
        "--expected-topology-digest": topology_digest,
        "--expected-layout-digest": layout_digest,
    }
    for option, value in digests.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or not set(value) <= _LOWER_HEX
        ):
            raise VerificationError(
                f"{option} must be one lowercase 64-character SHA-256 digest"
            )
    if (
        isinstance(span_tokens, bool)
        or not isinstance(span_tokens, int)
        or span_tokens <= 0
    ):
        raise VerificationError("--expected-span must be positive")
    if (
        isinstance(physical_rank, bool)
        or not isinstance(physical_rank, int)
        or physical_rank < 0
    ):
        raise VerificationError("--expected-physical-rank must be non-negative")
    try:
        WorkerCoordinate.from_runtime(
            global_rank=physical_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            dcp_rank=dcp_rank,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dcp_degree=dcp_degree,
        )
    except ValueError as error:
        raise VerificationError(
            "expected worker coordinate is inconsistent with PP/TP/DCP topology"
        ) from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (groups, layers)
    ):
        raise VerificationError(
            "--expected-groups and --expected-layers must be positive"
        )
    if not isinstance(group_layers, str):
        raise VerificationError(
            "--expected-group-layers must contain one non-negative decimal count per group"
        )
    raw_group_layers = group_layers.split(",")
    if any(
        not value or not value.isascii() or not value.isdecimal()
        for value in raw_group_layers
    ):
        raise VerificationError(
            "--expected-group-layers must contain one non-negative decimal count per group"
        )
    expected_group_layers = tuple(int(value) for value in raw_group_layers)
    if (
        len(expected_group_layers) != groups
        or sum(expected_group_layers) != layers
    ):
        raise VerificationError(
            "--expected-group-layers must match expected groups and layers"
        )
    if not isinstance(pages, str):
        raise VerificationError(
            "--expected-pages must contain one non-negative decimal count per group"
        )
    raw_pages = pages.split(",")
    if any(
        not value or not value.isascii() or not value.isdecimal()
        for value in raw_pages
    ):
        raise VerificationError(
            "--expected-pages must contain one non-negative decimal count per group"
        )
    expected_pages = tuple(int(value) for value in raw_pages)
    if len(expected_pages) != groups:
        raise VerificationError(
            "--expected-pages must contain one non-negative decimal count per group"
        )
    if any(
        (layer_count == 0) != (page_count == 0)
        for layer_count, page_count in zip(
            expected_group_layers, expected_pages, strict=True
        )
    ):
        raise VerificationError(
            "--expected-pages must be zero exactly for stage-local empty groups"
        )
    return expected_pages, expected_group_layers


def validate_manifest_identity(
    manifest: RankManifest,
    *,
    entry_id: str,
    span_tokens: int,
    deployment_identity_digest: str,
    rank_identity_digest: str,
    physical_rank: int,
    topology_digest: str,
    layout_digest: str,
    tp_degree: int,
    pp_degree: int,
    dcp_degree: int,
    pp_rank: int,
    tp_rank: int,
    dcp_rank: int,
) -> None:
    """Bind an offline content result to one exact rank-local deployment view."""

    expected = {
        "entry ID": (manifest.entry_id, entry_id),
        "span": (manifest.span_tokens, span_tokens),
        "deployment identity": (
            manifest.deployment_identity_digest,
            deployment_identity_digest,
        ),
        "rank identity": (manifest.rank_identity_digest, rank_identity_digest),
        "physical rank": (manifest.physical_rank, physical_rank),
        "topology": (manifest.topology_digest, topology_digest),
        "layout protocol": (manifest.profile, VLLM_RUNTIME_KV_PROFILE),
        "layout": (manifest.layout_digest, layout_digest),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise VerificationError(f"manifest {label} differs from expectation")
    try:
        WorkerCoordinate.from_runtime(
            global_rank=manifest.physical_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            dcp_rank=dcp_rank,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            dcp_degree=dcp_degree,
        )
    except ValueError as error:
        raise VerificationError(
            "manifest worker coordinate differs from expectation"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-root", type=Path, required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--expected-span", type=int, required=True)
    parser.add_argument("--expected-deployment-identity-digest", required=True)
    parser.add_argument("--expected-rank-identity-digest", required=True)
    parser.add_argument("--expected-physical-rank", type=int, required=True)
    parser.add_argument("--expected-tp-degree", type=int, required=True)
    parser.add_argument("--expected-pp-degree", type=int, required=True)
    parser.add_argument("--expected-dcp-degree", type=int, required=True)
    parser.add_argument("--expected-pp-rank", type=int, required=True)
    parser.add_argument("--expected-tp-rank", type=int, required=True)
    parser.add_argument("--expected-dcp-rank", type=int, required=True)
    parser.add_argument("--expected-topology-digest", required=True)
    parser.add_argument("--expected-layout-digest", required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-layers", type=int, required=True)
    parser.add_argument(
        "--expected-group-layers",
        required=True,
        help="comma-separated stage-local layer count for every runtime group",
    )
    parser.add_argument(
        "--expected-pages",
        required=True,
        help=(
            "comma-separated selected page count for each runtime group; "
            "use zero exactly for stage-local empty groups"
        ),
    )
    args = parser.parse_args()

    try:
        expected_pages, expected_group_layers = validate_expected_inputs(
            entry_id=args.entry_id,
            span_tokens=args.expected_span,
            deployment_identity_digest=args.expected_deployment_identity_digest,
            rank_identity_digest=args.expected_rank_identity_digest,
            physical_rank=args.expected_physical_rank,
            topology_digest=args.expected_topology_digest,
            layout_digest=args.expected_layout_digest,
            tp_degree=args.expected_tp_degree,
            pp_degree=args.expected_pp_degree,
            dcp_degree=args.expected_dcp_degree,
            pp_rank=args.expected_pp_rank,
            tp_rank=args.expected_tp_rank,
            dcp_rank=args.expected_dcp_rank,
            groups=args.expected_groups,
            layers=args.expected_layers,
            group_layers=args.expected_group_layers,
            pages=args.expected_pages,
        )
    except VerificationError as error:
        raise SystemExit(str(error)) from error
    manifest_path = (
        args.rank_root
        / "manifests"
        / args.entry_id[:2]
        / f"{args.entry_id}.json"
    )
    envelope = decode_manifest(manifest_path.read_bytes())
    manifest = envelope.manifest
    try:
        validate_manifest_identity(
            manifest,
            entry_id=args.entry_id,
            span_tokens=args.expected_span,
            deployment_identity_digest=args.expected_deployment_identity_digest,
            rank_identity_digest=args.expected_rank_identity_digest,
            physical_rank=args.expected_physical_rank,
            topology_digest=args.expected_topology_digest,
            layout_digest=args.expected_layout_digest,
            tp_degree=args.expected_tp_degree,
            pp_degree=args.expected_pp_degree,
            dcp_degree=args.expected_dcp_degree,
            pp_rank=args.expected_pp_rank,
            tp_rank=args.expected_tp_rank,
            dcp_rank=args.expected_dcp_rank,
        )
    except VerificationError as error:
        # Identity rejection deliberately precedes any payload path lookup.
        raise SystemExit(str(error)) from error

    coverage: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    aggregate = hashlib.sha256()
    for descriptor in manifest.objects:
        path = args.rank_root / descriptor.relative_path
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != descriptor.stored_length:
            raise SystemExit(f"object type/length differs: {descriptor.relative_path}")
        digest = hashlib.sha256()
        remaining = descriptor.byte_length
        padding_nonzero = False
        with path.open("rb", buffering=0) as source:
            while chunk := source.read(8 * 1024 * 1024):
                logical = min(remaining, len(chunk))
                digest.update(chunk[:logical])
                remaining -= logical
                padding_nonzero = padding_nonzero or any(chunk[logical:])
        if remaining or padding_nonzero or digest.hexdigest() != descriptor.sha256:
            raise SystemExit(f"object content differs: {descriptor.relative_path}")
        aggregate.update(descriptor.sha256.encode())
        aggregate.update(descriptor.relative_path.encode())
        coverage[(descriptor.group_index, descriptor.layer_name)].append(
            (descriptor.page_start, descriptor.page_count)
        )

    group_layers: dict[int, int] = {}
    group_pages: dict[int, int] = {}
    for group_index in range(args.expected_groups):
        layers = {
            layer_name: ranges
            for (index, layer_name), ranges in coverage.items()
            if index == group_index
        }
        expected_layer_count = expected_group_layers[group_index]
        if len(layers) != expected_layer_count:
            raise SystemExit(
                f"group {group_index} layer coverage differs: {len(layers)}"
            )
        if not layers:
            group_layers[group_index] = 0
            group_pages[group_index] = 0
            continue
        totals: set[int] = set()
        for layer_name, ranges in layers.items():
            cursor = 0
            for page_start, page_count in sorted(ranges):
                if page_start != cursor:
                    raise SystemExit(
                        f"page gap in group {group_index} layer {layer_name}"
                    )
                cursor += page_count
            totals.add(cursor)
        if totals != {expected_pages[group_index]}:
            raise SystemExit(
                f"group {group_index} page coverage differs: {sorted(totals)}"
            )
        group_layers[group_index] = len(layers)
        group_pages[group_index] = totals.pop()

    expected_nonempty_groups = {
        index
        for index, count in enumerate(expected_group_layers)
        if count
    }
    if set(index for index, _ in coverage) != expected_nonempty_groups:
        raise SystemExit("manifest contains an unexpected group index")
    if len(coverage) != args.expected_layers:
        raise SystemExit(
            f"manifest covers {len(coverage)} layers, expected {args.expected_layers}"
        )
    print(
        json.dumps(
            {
                "entry_id": manifest.entry_id,
                "deployment_identity_digest": manifest.deployment_identity_digest,
                "rank_identity_digest": manifest.rank_identity_digest,
                "physical_rank": manifest.physical_rank,
                "worker_coordinate": {
                    "pp_rank": args.expected_pp_rank,
                    "tp_rank": args.expected_tp_rank,
                    "dcp_rank": args.expected_dcp_rank,
                },
                "topology_digest": manifest.topology_digest,
                "profile": manifest.profile,
                "layout_digest": manifest.layout_digest,
                "span_tokens": manifest.span_tokens,
                "groups": args.expected_groups,
                "layers": len(coverage),
                "objects": len(manifest.objects),
                "group_layers": group_layers,
                "group_pages": group_pages,
                "logical_bytes": manifest.logical_bytes,
                "stored_bytes": manifest.stored_bytes,
                "manifest_payload_sha256": envelope.payload_sha256,
                "content_index_sha256": aggregate.hexdigest(),
                "status": "all-payloads-authenticated",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
