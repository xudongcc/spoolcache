#!/usr/bin/env python3
"""Measure whether an NVMe KV restore stalls requests that are already decoding.

This is deliberately a no-restart benchmark.  An immediate prompt replay is not
useful here because vLLM can serve it from its own GPU prefix cache without ever
calling SpoolCache.  Instead, the benchmark:

1. stores a small, deterministic set of long prefixes (the restore set);
2. computes a disjoint 128K working set slightly larger than the authoritative
   GPU KV capacity, naturally evicting the older restore set without restarting;
3. primes a distinct 128K prefix with SpoolCache's per-request bypass, then in
   that same full-cache state measures five long-lived foreground decodes alone
   and with that GPU-local prefix in the sixth slot; and
4. repeats five foreground decodes while injecting the evicted restore set from
   NVMe through the sixth request slot.

The foreground result records every non-empty SSE delivery timestamp.  OpenAI
stream chunks can contain more than one accepted speculative token, so these
are correctly named ``event gaps``, not per-token TPOT.  P95/P99/max event gaps
are nevertheless the most direct client-visible signal for a restore-induced
pause.  Aggregate decode throughput is recorded as the complementary measure.

The preparation phase can take many minutes: filling a roughly 2.42M-token KV
pool necessarily requires computing roughly that many new tokens once.  Its
time and synchronous Store cost are excluded from the interference comparison.
No prompt text or API key is written to the JSON report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Iterable
import urllib.request


EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
LOCAL_HITS = "vllm:prefix_cache_hits_total"
PREEMPTIONS = "vllm:num_preemptions_total"
KV_USAGE = "vllm:kv_cache_usage_perc"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"


def headers() -> dict[str, str]:
    """Return authenticated JSON headers without exposing the selected key."""

    result = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        # The deployment accepts comma- or whitespace-separated keys.  Only the
        # first is needed by this client, and it is never included in reports.
        fields = os.environ.get("DSPARK_API_KEYS", "").replace(",", " ").split()
        key = fields[0] if fields else ""
    if key:
        result["Authorization"] = f"Bearer {key}"
    return result


def post_json(url: str, body: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("endpoint did not return a JSON object")
    return result


def get_text(url: str) -> str:
    request = urllib.request.Request(url, headers=headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def token_count(api: str, model: str, prompt: str) -> int:
    response = post_json(f"{api}/tokenize", {"model": model, "prompt": prompt})
    count = response.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenize endpoint returned an invalid token count")
    return count


def validated_stream_usage(
    usage: object,
    *,
    expected_completion_tokens: int,
) -> tuple[int, int, int]:
    """Require complete non-coercive evidence from one completion stream."""

    if not isinstance(usage, dict):
        raise RuntimeError("completion stream returned missing or invalid usage")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise RuntimeError(
            "completion stream returned missing or invalid prompt token details"
        )

    def integer(value: object, *, label: str, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RuntimeError(f"completion stream returned invalid {label}")
        return value

    prompt_tokens = integer(
        usage.get("prompt_tokens"),
        label="prompt token count",
        minimum=1,
    )
    cached_tokens = integer(
        details.get("cached_tokens"),
        label="cached token count",
    )
    completion_tokens = integer(
        usage.get("completion_tokens"),
        label="completion token count",
        minimum=1,
    )
    if cached_tokens > prompt_tokens:
        raise RuntimeError("completion stream returned impossible cached token usage")
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            "completion stream ended without expected generated tokens: "
            f"expected={expected_completion_tokens} actual={completion_tokens}"
        )
    return prompt_tokens, cached_tokens, completion_tokens


def build_prompt(api: str, model: str, target: int, label: str) -> str:
    """Build a deterministic prompt with at least ``target`` tokenizer tokens.

    The unique label is placed at the start.  Consequently different working
    set members diverge in their first hash block instead of accidentally
    sharing almost the entire prefix through vLLM's local prefix cache.
    """

    header = f"SpoolCache restore-interference fixture {label}.\n"
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa. "
    low, high = 1, max(2, target // 4)
    while token_count(api, model, header + unit * high) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if token_count(api, model, header + unit * middle) < target:
            low = middle + 1
        else:
            high = middle
    return header + unit * low + "\nReturn numbered lowercase English words."


def parse_labels(raw: str) -> dict[str, str]:
    """Parse the simple quoted-label subset emitted by Prometheus."""

    labels: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"', raw):
        labels[match.group(1)] = bytes(
            match.group(2), "utf-8"
        ).decode("unicode_escape")
    return labels


def prometheus_samples(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    """Extract numeric samples for one exact Prometheus metric name."""

    pattern = re.compile(
        rf"^{re.escape(name)}(?:\{{(?P<labels>.*)\}})?\s+"
        r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )
    result: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            result.append(
                (parse_labels(match.group("labels") or ""), float(match.group("value")))
            )
    return result


def metric_value(text: str, name: str) -> float:
    samples = prometheus_samples(text, name)
    if len(samples) != 1:
        raise RuntimeError(f"expected one {name} sample, found {len(samples)}")
    return samples[0][1]


def kv_capacity_tokens(text: str) -> int:
    """Read vLLM's resolved token capacity from ``cache_config_info`` labels."""

    samples = prometheus_samples(text, "vllm:cache_config_info")
    capacities = {
        int(labels["kv_cache_size_tokens"])
        for labels, _value in samples
        if labels.get("kv_cache_size_tokens", "").isdigit()
    }
    if len(capacities) != 1:
        raise RuntimeError(
            "could not resolve one kv_cache_size_tokens value from cache_config_info"
        )
    return capacities.pop()


def percentile(values: Iterable[float], percent: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gap_summary(event_times: list[float]) -> dict[str, float | int]:
    gaps = [right - left for left, right in zip(event_times, event_times[1:])]
    return gap_distribution_summary(gaps, event_count=len(event_times))


def gap_distribution_summary(
    gaps: list[float], *, event_count: int | None = None
) -> dict[str, float | int]:
    """Summarize already-computed event gaps from one or many streams."""

    return {
        "event_count": event_count if event_count is not None else len(gaps) + 1,
        "gap_count": len(gaps),
        "gap_p50_seconds": percentile(gaps, 50),
        "gap_p95_seconds": percentile(gaps, 95),
        "gap_p99_seconds": percentile(gaps, 99),
        "gap_max_seconds": max(gaps, default=0.0),
    }


def overlapping_gaps(
    event_times: list[float], start: float, end: float
) -> list[float]:
    """Return gaps whose time interval overlaps a background-load window."""

    return [
        right - left
        for left, right in zip(event_times, event_times[1:])
        if left < end and right > start
    ]


def cache_salt(run_id: str, label: str) -> str:
    return hashlib.sha256(f"{run_id}\0{label}".encode()).hexdigest()


def stream_completion(
    api: str,
    model: str,
    prompt: str,
    salt: str,
    output_tokens: int,
    seed: int,
    ready: threading.Event | None = None,
    barrier: threading.Barrier | None = None,
    bypass_spoolcache: bool = False,
) -> dict[str, object]:
    """Run one completion and retain timing metadata but not generated text."""

    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "seed": seed,
        "cache_salt": salt,
    }
    if bypass_spoolcache:
        # This is handled by SpoolCache through vLLM's public passthrough field.
        # vLLM's own GPU prefix cache remains enabled, while persistent lookup
        # and Store are both skipped for this request only.
        body["kv_transfer_params"] = {"spoolcache_bypass": True}
    request = urllib.request.Request(
        f"{api}/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers(),
        method="POST",
    )
    if barrier is not None:
        barrier.wait()
    started = time.perf_counter()
    first_at: float | None = None
    event_times: list[float] = []
    usage: dict[str, object] = {}
    request_id = ""
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            event_id = event.get("id", request_id)
            if not isinstance(event_id, str):
                raise RuntimeError("completion stream returned an invalid request ID")
            request_id = event_id
            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if not isinstance(text, str):
                raise RuntimeError("completion stream returned non-text output")
            if text:
                now = time.perf_counter()
                event_times.append(now)
                digest.update(text.encode())
                if first_at is None:
                    first_at = now
                    if ready is not None:
                        ready.set()
            if event.get("usage") is not None:
                event_usage = event["usage"]
                if not isinstance(event_usage, dict):
                    raise RuntimeError("completion stream returned invalid usage")
                usage = event_usage
    finished = time.perf_counter()
    first = first_at or finished
    if ready is not None and not ready.is_set():
        ready.set()
    prompt_tokens, cached_tokens, completion = validated_stream_usage(
        usage,
        expected_completion_tokens=output_tokens,
    )
    if not request_id or first_at is None or not event_times:
        raise RuntimeError("completion stream ended without a non-empty output")
    return {
        "request_id": request_id,
        "started_at": started,
        "first_at": first,
        "finished_at": finished,
        "ttft_seconds": first - started,
        "elapsed_seconds": finished - started,
        "decode_seconds": finished - first,
        "decode_tokens_s": completion / max(0.001, finished - first),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion,
        "event_times": event_times,
        "event_gaps": gap_summary(event_times),
        "output_sha256": digest.hexdigest(),
    }


def run_wave(
    *,
    api: str,
    model: str,
    prompts: list[tuple[str, str]],
    output_tokens: int,
    concurrency: int,
    seed_base: int,
    bypass_spoolcache: bool = False,
) -> list[dict[str, object]]:
    """Run preparation requests in bounded waves."""

    results: list[dict[str, object]] = []
    for offset in range(0, len(prompts), concurrency):
        wave = prompts[offset : offset + concurrency]
        barrier = threading.Barrier(len(wave))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [
                executor.submit(
                    stream_completion,
                    api,
                    model,
                    prompt,
                    salt,
                    output_tokens,
                    seed_base + offset + lane,
                    None,
                    barrier,
                    bypass_spoolcache,
                )
                for lane, (prompt, salt) in enumerate(wave)
            ]
            results.extend(future.result() for future in futures)
        print(
            json.dumps(
                {
                    "preparation_progress": {
                        "completed": min(offset + len(wave), len(prompts)),
                        "total": len(prompts),
                    }
                }
            ),
            flush=True,
        )
    return results


def metric_snapshot(api: str) -> dict[str, float]:
    text = get_text(f"{api}/metrics")
    return {
        "external_hit_tokens": metric_value(text, EXTERNAL_HITS),
        "local_hit_tokens": metric_value(text, LOCAL_HITS),
        "preemptions": metric_value(text, PREEMPTIONS),
    }


def runtime_snapshot(api: str) -> dict[str, float]:
    """Read gauges used to prove that an interference phase was near full."""

    text = get_text(f"{api}/metrics")
    return {
        "kv_cache_usage_perc": metric_value(text, KV_USAGE),
        "requests_running": metric_value(text, RUNNING),
        "requests_waiting": metric_value(text, WAITING),
    }


def run_interference_phase(
    *,
    name: str,
    api: str,
    model: str,
    foreground_prompts: list[tuple[str, str]],
    injection_prompts: list[tuple[str, str]],
    foreground_output_tokens: int,
    injection_output_tokens: int,
    settle_seconds: float,
    seed_base: int,
    injection_bypass_spoolcache: bool = False,
) -> dict[str, object]:
    """Decode in the foreground and inject one background request at a time."""

    foreground_count = len(foreground_prompts)
    barrier = threading.Barrier(foreground_count)
    ready = [threading.Event() for _ in foreground_prompts]
    metrics_before = metric_snapshot(api)
    runtime_samples: list[dict[str, float]] = []
    monitor_stop = threading.Event()

    def monitor_runtime() -> None:
        # A 500 ms interval is frequent enough to catch a seconds-long restore,
        # while avoiding an aggressive /metrics polling client that itself
        # becomes part of the workload under test.
        while not monitor_stop.is_set():
            try:
                runtime_samples.append(runtime_snapshot(api))
            except Exception:
                # The request results and connector counters are authoritative;
                # one failed observability scrape must not abort inference.
                pass
            monitor_stop.wait(0.5)

    monitor = threading.Thread(target=monitor_runtime, daemon=True)
    monitor.start()
    phase_started = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=foreground_count + 1
        ) as executor:
            foreground_futures = [
                executor.submit(
                    stream_completion,
                    api,
                    model,
                    prompt,
                    salt,
                    foreground_output_tokens,
                    seed_base + lane,
                    ready[lane],
                    barrier,
                )
                for lane, (prompt, salt) in enumerate(foreground_prompts)
            ]
            deadline = time.monotonic() + 300
            for event in ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not event.wait(remaining):
                    raise RuntimeError("foreground streams did not all reach first output")
            time.sleep(settle_seconds)
            injection_started = time.perf_counter()
            injection_results = [
                stream_completion(
                    api,
                    model,
                    prompt,
                    salt,
                    injection_output_tokens,
                    seed_base + 10_000 + index,
                    bypass_spoolcache=injection_bypass_spoolcache,
                )
                for index, (prompt, salt) in enumerate(injection_prompts)
            ]
            injection_finished = time.perf_counter()
            foreground_results = [future.result() for future in foreground_futures]
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
    phase_finished = time.perf_counter()
    metrics_after = metric_snapshot(api)

    overlap: list[float] = []
    all_gaps: list[float] = []
    all_event_count = 0
    for result in foreground_results:
        event_times = list(result["event_times"])
        all_event_count += len(event_times)
        all_gaps.extend(
            right - left for left, right in zip(event_times, event_times[1:])
        )
        overlap.extend(
            overlapping_gaps(
                event_times, injection_started, injection_finished
            )
        )
        # Absolute monotonic timestamps aid interval analysis in memory but make
        # reports noisy and are not useful across processes.
        result.pop("event_times", None)

    total_completion = sum(item["completion_tokens"] for item in foreground_results)
    foreground_decode_start = min(
        float(item["first_at"]) for item in foreground_results
    )
    foreground_finish = max(
        float(item["finished_at"]) for item in foreground_results
    )
    for result in foreground_results:
        result.pop("started_at", None)
        result.pop("first_at", None)
        result.pop("finished_at", None)
    for result in injection_results:
        result.pop("event_times", None)
        result.pop("started_at", None)
        result.pop("first_at", None)
        result.pop("finished_at", None)

    return {
        "name": name,
        "phase_seconds": phase_finished - phase_started,
        "injection_seconds": injection_finished - injection_started,
        "foreground_aggregate_decode_tokens_s": total_completion
        / max(0.001, foreground_finish - foreground_decode_start),
        "foreground_median_decode_tokens_s": statistics.median(
            float(item["decode_tokens_s"]) for item in foreground_results
        ),
        "foreground_all_event_gaps": gap_distribution_summary(
            all_gaps, event_count=all_event_count
        ),
        "foreground_overlap_gap_count": len(overlap),
        "foreground_overlap_gap_p50_seconds": percentile(overlap, 50),
        "foreground_overlap_gap_p95_seconds": percentile(overlap, 95),
        "foreground_overlap_gap_p99_seconds": percentile(overlap, 99),
        "foreground_overlap_gap_max_seconds": max(overlap, default=0.0),
        "metrics_delta": {
            key: metrics_after[key] - metrics_before[key]
            for key in metrics_before
        },
        "runtime_observation": {
            "sample_count": len(runtime_samples),
            "max_kv_cache_usage_perc": max(
                (item["kv_cache_usage_perc"] for item in runtime_samples),
                default=0.0,
            ),
            "max_requests_running": max(
                (item["requests_running"] for item in runtime_samples),
                default=0.0,
            ),
            "max_requests_waiting": max(
                (item["requests_waiting"] for item in runtime_samples),
                default=0.0,
            ),
        },
        "foreground": foreground_results,
        "injections": injection_results,
    }


def sanitize_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise argparse.ArgumentTypeError(
            "run id must be 1-80 ASCII letters, digits, dot, underscore, or dash"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8888")
    model_default = os.environ.get("SERVED_MODEL_NAME")
    parser.add_argument(
        "--model",
        default=model_default,
        required=model_default is None,
    )
    parser.add_argument("--run-id", required=True, type=sanitize_run_id)
    parser.add_argument(
        "--reuse-persisted-fixtures-from",
        default=None,
        type=sanitize_run_id,
        metavar="RUN_ID",
        help=(
            "reuse the restore/eviction fixture identities already persisted "
            "by RUN_ID; skips cold restore-set priming"
        ),
    )
    parser.add_argument("--prefix-tokens", type=int, default=131072)
    parser.add_argument("--restore-entries", type=int, default=3)
    parser.add_argument(
        "--gpu-kv-capacity-tokens",
        type=int,
        default=0,
        help=(
            "authoritative 'GPU KV cache size' value from the boot log; required "
            "for HMA profiles whose cache_config_info label has another meaning"
        ),
    )
    parser.add_argument(
        "--eviction-factor",
        type=float,
        default=1.08,
        help="working-set tokens divided by authoritative GPU KV capacity",
    )
    parser.add_argument("--prepare-concurrency", type=int, default=6)
    parser.add_argument("--foreground-concurrency", type=int, default=5)
    parser.add_argument("--foreground-prompt-tokens", type=int, default=256)
    parser.add_argument("--foreground-output-tokens", type=int, default=2048)
    parser.add_argument("--injection-output-tokens", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output",
        default="",
        help="JSON report path (default: results/restore-interference-RUN_ID.json)",
    )
    args = parser.parse_args()
    if args.restore_entries < 1:
        parser.error("--restore-entries must be positive")
    if not 1 <= args.foreground_concurrency <= 5:
        parser.error("--foreground-concurrency must be 1-5, leaving one vLLM slot")
    if not 1 <= args.prepare_concurrency <= 6:
        parser.error("--prepare-concurrency must be 1-6")
    if args.prefix_tokens < 1024:
        parser.error("--prefix-tokens must reach the 1,024-token Store threshold")
    if args.foreground_prompt_tokens <= 0:
        parser.error("--foreground-prompt-tokens must be positive")
    if args.foreground_output_tokens <= 0 or args.injection_output_tokens <= 0:
        parser.error("foreground and injection output token counts must be positive")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    if args.gpu_kv_capacity_tokens < 0:
        parser.error("--gpu-kv-capacity-tokens must be positive or zero")
    if args.eviction_factor <= 1.0:
        parser.error("--eviction-factor must be greater than 1.0")

    report_path = Path(
        args.output
        or f"results/restore-interference-{args.run_id}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = get_text(f"{args.api}/metrics")
    metric_capacity = kv_capacity_tokens(metrics)
    capacity = args.gpu_kv_capacity_tokens or metric_capacity
    eviction_entries = math.ceil(
        capacity * args.eviction_factor / args.prefix_tokens
    )
    working_set_tokens = eviction_entries * args.prefix_tokens
    print(
        f"authoritative GPU KV capacity={capacity:,} tokens "
        f"(cache_config_info label={metric_capacity:,}); eviction working set="
        f"{eviction_entries} x ~{args.prefix_tokens:,} = {working_set_tokens:,} "
        f"tokens ({working_set_tokens / capacity:.1%})",
        flush=True,
    )

    fixture_run_id = args.reuse_persisted_fixtures_from or args.run_id

    def fixture(kind: str, index: int) -> tuple[str, str]:
        label = f"{fixture_run_id}-{kind}-{index:04d}"
        return (
            build_prompt(args.api, args.model, args.prefix_tokens, label),
            cache_salt(fixture_run_id, label),
        )

    restore_set = [fixture("restore", index) for index in range(args.restore_entries)]
    # Keep prompt text identical across phases.  Changing the actual words can
    # change speculative-decoding acceptance and make a connector comparison
    # look faster or slower for model-content reasons.  Distinct cache salts
    # prevent the earlier phase from donating a local GPU prefix hit while the
    # token sequence and greedy output remain matched.
    foreground_prompts = [
        build_prompt(
            args.api,
            args.model,
            args.foreground_prompt_tokens,
            f"{args.run_id}-foreground-{lane}",
        )
        for lane in range(args.foreground_concurrency)
    ]
    foreground_sets = {
        name: [
            (
                foreground_prompts[lane],
                cache_salt(args.run_id, f"foreground-{name}-{lane}"),
            )
            for lane in range(args.foreground_concurrency)
        ]
        for name in ("baseline", "local-hit", "disk-hit")
    }

    if args.reuse_persisted_fixtures_from:
        print(
            "reusing the persisted restore set; cold priming skipped",
            flush=True,
        )
        prime: list[dict[str, object]] = []
    else:
        print(
            "priming the deterministic restore set (preparation; excluded)",
            flush=True,
        )
        prime = run_wave(
            api=args.api,
            model=args.model,
            prompts=restore_set,
            output_tokens=args.injection_output_tokens,
            concurrency=1,
            seed_base=10_000,
        )

    print("building the oversized 128K working set (preparation; excluded)", flush=True)
    for offset in range(0, eviction_entries, args.prepare_concurrency):
        size = min(args.prepare_concurrency, eviction_entries - offset)
        wave = [fixture("evict", offset + lane) for lane in range(size)]
        run_wave(
            api=args.api,
            model=args.model,
            prompts=wave,
            output_tokens=args.injection_output_tokens,
            concurrency=size,
            seed_base=15_000 + offset,
        )
        print(
            json.dumps(
                {
                    "eviction_progress": {
                        "completed": offset + size,
                        "total": eviction_entries,
                    }
                }
            ),
            flush=True,
        )

    # Use a fixture that is absent from the persistent catalog and bypass both
    # SpoolCache read and write while priming it.  A later replay with the same
    # bypass can therefore only use vLLM's GPU prefix cache (plus any small tail
    # vLLM elects to recompute); connector logs and external-hit metrics must
    # stay completely quiet.
    local_label = f"{args.run_id}-local-control-0000"
    local_resident = (
        build_prompt(args.api, args.model, args.prefix_tokens, local_label),
        cache_salt(args.run_id, local_label),
    )
    print(
        "priming the SpoolCache-bypassed GPU-local control (excluded)",
        flush=True,
    )
    local_prime = run_wave(
        api=args.api,
        model=args.model,
        prompts=[local_resident],
        output_tokens=args.injection_output_tokens,
        concurrency=1,
        seed_base=18_000,
        bypass_spoolcache=True,
    )
    local_injection = [local_resident] * args.restore_entries

    baseline = run_interference_phase(
        name="foreground-only",
        api=args.api,
        model=args.model,
        foreground_prompts=foreground_sets["baseline"],
        injection_prompts=[],
        foreground_output_tokens=args.foreground_output_tokens,
        injection_output_tokens=args.injection_output_tokens,
        settle_seconds=0,
        seed_base=20_000,
    )
    local = run_interference_phase(
        name="gpu-local-hit-injection",
        api=args.api,
        model=args.model,
        foreground_prompts=foreground_sets["local-hit"],
        injection_prompts=local_injection,
        foreground_output_tokens=args.foreground_output_tokens,
        injection_output_tokens=args.injection_output_tokens,
        settle_seconds=args.settle_seconds,
        # Foreground seeds must match the baseline exactly.  Different seeds
        # can change DSpark speculative acceptance even for temperature=0.
        seed_base=20_000,
        injection_bypass_spoolcache=True,
    )

    disk = run_interference_phase(
        name="nvme-restore-injection",
        api=args.api,
        model=args.model,
        foreground_prompts=foreground_sets["disk-hit"],
        injection_prompts=restore_set,
        foreground_output_tokens=args.foreground_output_tokens,
        injection_output_tokens=args.injection_output_tokens,
        settle_seconds=args.settle_seconds,
        seed_base=20_000,
    )

    disk_cached = sum(
        1 for item in disk["injections"] if item["cached_tokens"] > 0
    )
    local_cached = sum(
        1 for item in local["injections"] if item["cached_tokens"] > 0
    )
    external_delta = float(disk["metrics_delta"]["external_hit_tokens"])
    local_external_delta = float(local["metrics_delta"]["external_hit_tokens"])
    verdict = (
        "PASS"
        if (
            disk_cached == args.restore_entries
            and external_delta > 0
            and local_cached == args.restore_entries
            and local_external_delta == 0
        )
        else "INVALID"
    )
    report = {
        "verdict": verdict,
        "run_id": args.run_id,
        "fixture_run_id": fixture_run_id,
        "reused_persisted_fixtures": bool(args.reuse_persisted_fixtures_from),
        "model": args.model,
        "no_service_restart": True,
        "gpu_kv_capacity_tokens": capacity,
        "cache_config_info_capacity_label": metric_capacity,
        "prefix_tokens_target": args.prefix_tokens,
        "eviction_factor": args.eviction_factor,
        "eviction_entries": eviction_entries,
        "working_set_tokens_target": working_set_tokens,
        "restore_entries": args.restore_entries,
        "foreground_concurrency": args.foreground_concurrency,
        "foreground_output_tokens": args.foreground_output_tokens,
        "prime_cached_tokens": [item["cached_tokens"] for item in prime],
        "local_prime_cached_tokens": [
            item["cached_tokens"] for item in local_prime
        ],
        "phases": [baseline, local, disk],
        "validation": {
            "local_injections_with_cached_tokens": local_cached,
            "local_external_hit_tokens_delta": local_external_delta,
            "disk_injections_with_cached_tokens": disk_cached,
            "external_hit_tokens_delta": external_delta,
            "note": (
                "PASS requires a SpoolCache-bypassed GPU-local control with "
                "zero external-hit tokens, plus an older restore set proven "
                "to load through the external connector."
            ),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report}, sort_keys=True), flush=True)
    print(f"report: {report_path}", flush=True)
    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
