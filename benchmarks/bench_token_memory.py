"""CPU token-file and inventory memory qualification with an external RSS sampler.

The page bytes are synthetic. This does not run a million-token model prefill.
The allocation check proves that a burst below an earlier process high-water
is still detected; acceptance uses current VmRSS, never inherited ru_maxrss.
"""

import argparse
from contextlib import contextmanager
import json
import mmap
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

from spoolcache.buffers import AlignedBufferPool
from spoolcache.config import INVENTORY_MEMORY_BYTES, inventory_capacity
from spoolcache.hma import GroupGeometry, HMALayout, LayerGeometry
from spoolcache.prefix import prefix_digests
from spoolcache.quorum import InventoryReporter, QuorumCatalog
from spoolcache.token_files import TokenFileStore

from benchmarks.token_file_evidence import audit, digest

MIB = 1024**2


def touch(mapping):
    for offset in range(0, len(mapping), mmap.PAGESIZE):
        mapping[offset] = 1


def workload(args):
    if args.mode == "sampler":
        with mmap.mmap(-1, 256 * MIB) as earlier:
            touch(earlier)
    print("ready", flush=True)
    assert sys.stdin.readline().strip() == "start"
    if args.mode == "sampler":
        old_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with mmap.mmap(-1, 128 * MIB) as burst:
            touch(burst)
            time.sleep(0.2)
        return {"old_high_water_kib": old_peak, "burst_bytes": 128 * MIB}

    count = inventory_capacity(INVENTORY_MEMORY_BYTES)
    entries = {f"{i:064x}": (i + 1) * 256 for i in range(count)}
    reporter = InventoryReporter(rank=0, generation="probe", generation_epoch=1,
        max_bytes=INVENTORY_MEMORY_BYTES, max_report_entries=64)
    reporter.replace(entries)
    wire = pickle.dumps(reporter.startup(), protocol=5)
    received = pickle.loads(wire)
    catalog = QuorumCatalog(expected_ranks=(0,), max_bytes=INVENTORY_MEMORY_BYTES,
                            max_report_entries=64)
    catalog.apply_startup(rank=0, generation="probe", generation_epoch=1,
                          entries=received)
    assert catalog.quorum_count == count > 100_000
    reporter.add("f" * 64, 256)
    catalog.apply_report(reporter.next_report(64))
    assert catalog.is_ready
    assert not catalog.has_quorum(f"{count - 1:064x}")
    del entries, received, wire

    layers = tuple(LayerGeometry(f"layer{i:03d}", "synthetic", 1) for i in range(64))
    group = GroupGeometry(0, "synthetic", 256, 256, 1, False, 1, 256, "full",
                          None, 0, False, layers)
    layout = HMALayout(args.chunks + 1, 1, (group,))
    assert args.reuse_cache or not args.cache_root.exists(), "use a new probe cache root"
    identity = {"deployment": "b" * 64, "rank_identity": "c" * 64,
        "physical_rank": 0, "topology": "d" * 64, "hma_layout": layout.digest,
        "transfer_bytes": 64 * MIB,
        "groups": [dict(index=0, layers=64, block=256, page_bytes=1, dcp_shards=1,
                        eagle=False, policy="full", window=None,
                        layer_names_digest=digest({"layers": [x.name for x in layers]})[:12])]}
    with TokenFileStore(args.cache_root, layout=layout, slot_bytes=64 * MIB,
        expected_deployment_digest="b" * 64, expected_rank_digest="c" * 64,
        expected_rank=0, expected_topology_digest="d" * 64,
        expected_profile=layout.profile, expected_layout_digest=layout.digest) as store:
        # Two resident CPU credits stand in for the two CUDA host credits in
        # this CPU-only budget probe. Installed CUDA tests qualify real pinning.
        with AlignedBufferPool(slot_bytes=64 * MIB, slot_count=2) as credits:
            for pool in (store._pool, credits):
                for slot in pool._slots:
                    slot.zero()
            store.validate_prefix_headers(args.chunks * 256)
            prefixes = prefix_digests(range(args.chunks * 256),
                                      deployment_digest="b" * 64, chunk_tokens=256)
            parent = None
            for index, prefix in enumerate(prefixes, 1):
                with store._exclusive():
                    existing = (store.valid_chunk(prefix.digest, parent, prefix.span_tokens)
                                if args.reuse_cache else None)
                    if existing is None:
                        store.commit_chunk(prefix.digest, parent, prefix.span_tokens,
                                           bytes([index % 251]) * 64)
                parent = prefix.digest
            seen = 0

            @contextmanager
            def receiver(descriptor):
                def receive(data):
                    nonlocal seen
                    assert bytes(data) == bytes([(descriptor.span_tokens // 256) % 251]) * 64
                    seen += 1
                yield receive

            with store.restore_view(parent) as lease:
                assert lease.result.is_hit
                assert not store.evict(prefixes[0].digest)
                store.stream_objects(lease.descriptors, receiver, lease=lease)
            assert seen == args.chunks
            evidence = audit(store.root, parent, args.chunks * 256, identity)
            assert evidence["metadata_bytes"] > 4 * MIB
    return dict(inventory_keys=count, inventory_allowance_bytes=INVENTORY_MEMORY_BYTES,
                reused_cache=args.reuse_cache,
                explicit_resident_credits_bytes=192 * MIB,
                audit=evidence, model_prefill=False)


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("sampler", "storage"), default="storage")
    parser.add_argument("--chunks", type=int, default=8193)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(workload(args)), flush=True)
        return
    command = [sys.executable, "-m", "benchmarks.bench_token_memory", *sys.argv[1:], "--child"]
    with args.output.with_suffix(".stderr").open("w") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True)
        assert process.stdout.readline().strip() == "ready"
        baseline = rss(process.pid)
        assert baseline is not None
        process.stdin.write("start\n")
        process.stdin.flush()
        samples = []
        while process.poll() is None:
            value = rss(process.pid)
            if value is not None:
                samples.append(value)
            time.sleep(0.01)
        assert process.returncode == 0, "child failed; see retained stderr"
        result = json.loads(process.stdout.read())
        peak = max(samples)
        result.update(baseline_rss_kib=baseline, peak_rss_kib=peak,
                      incremental_peak_kib=peak - baseline, rss_samples=len(samples))
        if args.mode == "sampler":
            assert 120 * 1024 <= peak - baseline < 160 * 1024
            assert peak < result["old_high_water_kib"]
        else:
            assert peak - baseline < 1024 * 1024
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
