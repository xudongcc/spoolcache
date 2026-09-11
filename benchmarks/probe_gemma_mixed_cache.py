#!/usr/bin/env python3
"""Compare cold, native-GPU and disk KV reuse at the same mixed-media boundary.

This is a diagnostic, not a replacement for run_gemma_e2e.py's cold-output gate.
It leaves the externally managed service running and preserves persistent state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import bench_multimodal_prefix_e2e as client
from benchmarks.token_file_evidence import TokenFileEvidence as Evidence
from benchmarks.reset_gpu_prefix_cache import reset_gpu_prefix_cache, reset_multimodal_caches
from benchmarks.run_gemma_e2e import MEDIA, MODEL, REVISION, check_sample, write_json

NONCE = "2640792bcfca21673e81c16d52364dda-mixed"
LABELS = "CATS_MARY_ARCHERY,DOGS_SPEECH_STREET,OTHER"
SPAN = 2560


def common_prefix_length(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def classify(cold, disk, native):
    if cold["cached_tokens"] != 0 or disk["cached_tokens"] != SPAN:
        raise ValueError("invalid cold/disk span evidence")
    if not native or any(row["cached_tokens"] != SPAN for row in native):
        raise ValueError("native comparison requires the same cached span")
    if any((row["output_sha256"], row["logprobs"]) !=
           (disk["output_sha256"], disk["logprobs"]) for row in native):
        raise ValueError("native output/probabilities differ from disk restore")
    if cold["output_sha256"] != disk["output_sha256"]:
        return "divergence-reproduced-by-native-cache"
    if cold["logprobs"] != disk["logprobs"]:
        return "same-output-with-native-matched-probability-drift"
    return "cold-native-disk-equivalent"


class Probe:
    def __init__(self, args):
        self.args = args
        self.salt = uuid.uuid4().hex
        self.evidence = Evidence(args, MODEL, REVISION)
        self.rows = {}

    def reset(self):
        reset_gpu_prefix_cache(self.args.api)
        reset_multimodal_caches(self.args.api)

    def sample(self, name, body, salt, cached, *, read=False, write=False, complete=True):
        body = copy.deepcopy(body)
        body["cache_salt"] = salt
        body["kv_transfer_params"] = {
            "spoolcache.skip_read": not read, "spoolcache.skip_write": not write,
        }
        body.update(logprobs=True, top_logprobs=5)
        response = client._post_json(self.args.api, "/v1/chat/completions", body)
        write_json(self.args.output / f"{name}-response.json", response)
        prompt, actual, _ = client._validated_usage(
            response, expected_prompt_tokens=len(client._token_ids(self.args.api, body)))
        output = client._normalized_output(response)
        if complete:
            client._validate_content_oracle(output, LABELS)
        probabilities = response["choices"][0].get("logprobs")
        if not isinstance(probabilities, dict) or not probabilities.get("content"):
            raise ValueError("missing generated-token probability evidence")
        row = {
            "request_id": response["id"], "prompt_tokens": prompt, "cached_tokens": actual,
            "output": output, "logprobs": probabilities,
            "output_sha256": hashlib.sha256(json.dumps(
                output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "cache_salt": salt, "skip_read": not read, "skip_write": not write,
        }
        self.rows[name] = row
        write_json(self.args.output / f"{name}.json", row)
        check_sample(row, cached)
        print(name, actual, output["content"], flush=True)
        return row

    def run(self):
        self.evidence.refresh()
        parts = []
        for kind, filename in (("video", "archery.mp4"), ("image", "cats.jpg"),
                               ("audio", "mary_had_lamb.ogg")):
            url, mime, data = client._data_url(kind, self.args.media_dir / filename)
            if (len(data), hashlib.sha256(data).hexdigest()) != MEDIA[filename]:
                raise ValueError(f"media fixture mismatch: {filename}")
            parts.append(client._media_part(kind, url, mime))

        def body(repetitions, padding, phase="producer"):
            return client._request_body(
                model=MODEL, media_part=parts, nonce=NONCE, repetitions=repetitions,
                padding=padding, labels=LABELS, cache_salt=self.salt,
                skip_write=False, phase=phase, max_tokens=32)
        repetitions, padding, producer_tokens = client._calibrate_exact_prompt(
            self.args.api, body, 4096)
        producer = body(repetitions, padding)
        consumer = body(repetitions, padding, "consumer")
        consumer_tokens = client._token_ids(self.args.api, consumer)
        if consumer_tokens[:len(producer_tokens)] != producer_tokens:
            raise ValueError("full producer is not an exact consumer prefix")

        # Keep the complete video and image, omit the following audio/text.
        # The resulting real request ends just after the image. Its common token
        # prefix rounds down to 2560 in native APC, without altering cache state.
        native_producer = copy.deepcopy(producer)
        native_producer["messages"][0]["content"] = parts[:2]
        native_producer["max_tokens"] = 1
        native_tokens = client._token_ids(self.args.api, native_producer)
        common = common_prefix_length(native_tokens, consumer_tokens)
        if not SPAN <= common < SPAN + 32:
            raise ValueError(f"native producer no longer establishes the fixture boundary: {common}")
        write_json(self.args.output / "inputs.json", {
            "model": MODEL, "revision": REVISION, "nonce": NONCE, "salt": self.salt,
            "repetitions": repetitions, "padding": padding, "media": MEDIA,
            "producer_tokens": producer_tokens, "consumer_tokens": consumer_tokens,
            "native_producer_tokens": native_tokens, "common_prefix_tokens": common,
            "image_id": self.args.image_id,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        self.reset()
        cold = self.sample("cold1", consumer, self.salt + "-cold1", 0)
        self.reset()
        cold2 = self.sample("cold2", consumer, self.salt + "-cold2", 0)
        self.require_equal(cold, cold2, "cold controls")
        self.reset()
        self.sample("producer", producer, self.salt, 0, read=True, write=True)
        reset_gpu_prefix_cache(self.args.api)
        retained = self.sample("restore-encoder-retained", consumer, self.salt, SPAN, read=True)
        self.reset()
        disk = self.sample("restore", consumer, self.salt, SPAN, read=True)
        self.evidence.verify("restore", {"mixed": disk})
        native = []
        for number in (1, 2, 3):
            salt = self.salt + f"-native{number}"
            self.reset()
            manifests = self.evidence.manifests()
            self.sample(f"native-producer{number}", native_producer, salt, 0, complete=False)
            reset_multimodal_caches(self.args.api)
            native.append(self.sample(f"native{number}", consumer, salt, SPAN))
            if self.evidence.manifests() != manifests:
                raise ValueError("native-only controls changed persistent manifests")
        self.reset()
        cold3 = self.sample("cold3", consumer, self.salt + "-cold3", 0)
        self.require_equal(cold, cold3, "final cold control")
        self.reset()
        disk2 = self.sample("restore-repeat", consumer, self.salt, SPAN, read=True)
        self.require_equal(disk, disk2, "disk repeats")
        result = classify(cold, disk, native)
        write_json(self.args.output / "summary.json", {
            "status": result, "native_matches_disk_probabilities": True,
            "cold_matches_disk_output": cold["output_sha256"] == disk["output_sha256"],
            "encoder_cache_retention_matches_disk":
                (retained["output_sha256"], retained["logprobs"]) ==
                (disk["output_sha256"], disk["logprobs"]),
            "cold": cold["output"]["content"], "disk": disk["output"]["content"],
            "native": [r["output"]["content"] for r in native],
            "native_and_disk_cached_tokens": SPAN, "request_count": len(self.rows),
            "scope": "diagnostic only; the e2e cold-output assertion remains unchanged",
        })
        print(result, flush=True)

    @staticmethod
    def require_equal(left, right, label):
        if (left["output_sha256"], left["logprobs"]) != (right["output_sha256"], right["logprobs"]):
            raise ValueError(f"unstable {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--worker-host", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("image-id must be an immutable Docker SHA-256")
    args.topology = "pp2"
    args.head_container = "spoolcache-gemma-pp2-head"
    args.worker_container = "spoolcache-gemma-pp2-worker"
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        Probe(args).run()
    except Exception as error:
        write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
