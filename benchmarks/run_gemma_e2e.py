#!/usr/bin/env python3
"""Fixed Gemma cache regression; run against an isolated installed-wheel service."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

# Also support direct invocation without putting the package source on PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.gemma_e2e_evidence import Evidence

MODEL = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
BENCHMARKS = Path(__file__).resolve().parent
MEDIA = {
    "cats.jpg": (173131, "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"),
    "archery.mp4": (549197, "8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562"),
    "mary_had_lamb.ogg": (65449, "c8f0a87f8d7e44f2d6e0f88ec63f6401b4f153f53fd14a9d730a5d1ba9927c4e"),
}


@dataclass(frozen=True)
class Case:
    name: str
    media: tuple[tuple[str, str], ...] = ()
    labels: str = ""
    hit: int = 2048


CASES = (
    Case("text"),
    Case("image", (("image", "cats.jpg"),), "CATS,DOGS,OTHER"),
    Case("audio", (("audio", "mary_had_lamb.ogg"),),
         "MARY_HAD_A_LITTLE_LAMB,TWINKLE_TWINKLE_LITTLE_STAR,OTHER"),
    Case("video", (("video", "archery.mp4"),), "ARCHERY,STREET,OTHER"),
    Case("mixed", (("video", "archery.mp4"), ("image", "cats.jpg"),
                   ("audio", "mary_had_lamb.ogg")),
         "CATS_MARY_ARCHERY,DOGS_SPEECH_STREET,OTHER", 2560),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_sample(row, expected, output=None):
    cached, prompt = row.get("cached_tokens"), row.get("prompt_tokens")
    if type(cached) is not int or type(prompt) is not int or not 0 <= cached <= prompt:
        raise ValueError("missing or invalid cached/prompt token evidence")
    if cached != expected:
        raise ValueError(f"cached tokens: expected {expected}, got {cached}")
    digest = row.get("output_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("missing output digest")
    if not isinstance(row.get("request_id"), str) or not row["request_id"]:
        raise ValueError("missing request identity")
    if output is not None and digest != output:
        raise ValueError("output differs from repeated cold control")


class Regression:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.nonce = uuid.uuid4().hex
        self.controls = {}
        self.restored = {}
        self.evidence = Evidence(args, MODEL, REVISION)
        write_json(self.output / "run.json", {"nonce": self.nonce, "model": MODEL, "revision": REVISION})

    def reset(self):
        subprocess.run([sys.executable, str(BENCHMARKS / "reset_gpu_prefix_cache.py"),
                        "--api", self.args.api, "--multimodal"],
                       check=True, stdout=subprocess.DEVNULL, timeout=60)

    def request(self, case, phase, salt, skip_read=False, skip_write=False):
        base = ["--api", self.args.api, "--model", MODEL,
                "--nonce", "spoolcache-gemma-e2e-v1-" + case.name, "--cache-salt", salt]
        if case.name == "text":
            command = [sys.executable, str(BENCHMARKS / "bench_prefix_e2e.py"), *base,
                       "--prompt-source-tokens", "4128", "--target-tokens",
                       "4096" if phase == "producer" else "4128", "--max-tokens", "8"]
        else:
            command = [sys.executable, str(BENCHMARKS / "bench_multimodal_prefix_e2e.py"),
                       *base, "--target-span", "4096", "--alignment", "32",
                       "--phase", phase, "--labels", case.labels, "--max-tokens", "32"]
            for kind, filename in case.media:
                command += ["--media-kind", kind, "--media-file", str(self.args.media_dir / filename)]
        if skip_read:
            command.append("--skip-read")
        if skip_write:
            command.append("--skip-write")
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode:
            (self.output / "client-failure.log").write_text(result.stdout + "\n" + result.stderr)
            raise RuntimeError("request failed; see client-failure.log")
        return {**json.loads(result.stdout), "nonce": "spoolcache-gemma-e2e-v1-" + case.name,
                "cache_salt": salt, "command": command}

    def sample(self, case, label, phase, salt, expected, oracle=None, **flags):
        row = self.request(case, phase, salt, **flags)
        write_json(self.output / f"{case.name}-{label}.json", row)
        print(case.name, label, row.get("cached_tokens"), row.get("output_sha256"), flush=True)
        check_sample(row, expected, oracle)
        return row

    def flags(self):
        case = CASES[0]
        prefix = self.nonce + "-flags-"
        before = self.evidence.manifests()
        rows = []
        def sample(label, phase, salt, expected, oracle=None, **flags):
            self.reset()
            row = self.sample(case, label, phase, prefix + salt, expected, oracle, **flags)
            rows.append({"case": label, "flags": flags, "request": row})
            write_json(self.output / "request-flags.json", rows)
            return row["output_sha256"]
        oracle = sample("flags-control1", "consumer", "control1", 0,
                        skip_read=True, skip_write=True)
        sample("flags-control2", "consumer", "control2", 0, oracle,
               skip_read=True, skip_write=True)
        sample("skip-write-producer", "producer", "no-write", 0, skip_write=True)
        sample("skip-write-consumer-misses", "consumer", "no-write", 0, oracle, skip_write=True)
        if self.evidence.manifests() != before:
            raise ValueError("skip_write changed persistent manifests")
        sample("skip-read-producer-stores", "producer", "save", 0, skip_read=True)
        sample("skip-write-consumer-restores", "consumer", "save", case.hit, oracle, skip_write=True)
        before = self.evidence.manifests()
        sample("skip-read-existing-entry", "consumer", "save", 0, oracle,
               skip_read=True, skip_write=True)
        if self.evidence.manifests() != before:
            raise ValueError("disabled read/write changed persistent manifests")

    def restart(self):
        # Lifecycle stays in the caller's Compose/PP helper; no shell evaluation.
        with (self.output / "restart.log").open("w") as log:
            subprocess.run(shlex.split(self.args.restart_command), check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        self.wait_ready()
        self.evidence.refresh(require_restarted=True)

    def wait_ready(self):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.args.api.rstrip("/") + "/health", timeout=3) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(5)
        raise RuntimeError("API readiness timed out")

    def execute(self):
        for case in CASES:
            salt = self.nonce + "-" + case.name
            oracle = None
            for number in (1, 2):
                self.reset()
                row = self.sample(case, f"control{number}", "consumer", salt + f"-control{number}",
                                  0, oracle, skip_read=True, skip_write=True)
                oracle = row["output_sha256"]
            self.controls[case.name] = oracle
            self.reset()
            self.sample(case, "producer", "producer", salt, 0)
            self.reset()
            self.restored[case.name] = self.sample(case, "restore", "consumer", salt, case.hit, oracle)
        self.evidence.verify("restore", self.restored)
        self.flags()
        self.restart()
        for case in CASES:
            self.restored[case.name] = self.sample(
                case, "post-restart", "consumer", self.nonce + "-" + case.name,
                case.hit, self.controls[case.name])
        self.evidence.verify("post-restart", self.restored)
        write_json(self.output / "summary.json", {
            "status": "passed", "oracle": "repeated cold output equivalence",
            "model": MODEL, "revision": REVISION, "nonce": self.nonce,
            "cases": [case.name for case in CASES],
            "expected_cached_tokens": {case.name: case.hit for case in CASES},
            "request_controls": "passed", "cross_restart": "passed",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new receipt directory")
    parser.add_argument("--topology", choices=("pp1", "pp2"), required=True)
    parser.add_argument("--head-container", required=True)
    parser.add_argument("--worker-container")
    parser.add_argument("--worker-host")
    parser.add_argument("--image-id", required=True, help="trusted immutable Docker image ID")
    parser.add_argument("--restart-command", required=True, help="whole-group restart argv, parsed without a shell")
    args = parser.parse_args()
    if args.topology == "pp2" and not (args.worker_container and args.worker_host):
        parser.error("PP=2 requires worker container and SSH host")
    if args.topology == "pp1" and (args.worker_container or args.worker_host):
        parser.error("PP=1 has no remote worker")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("image-id must be an immutable sha256 digest")
    if not shlex.split(args.restart_command):
        parser.error("restart-command must not be empty")
    for filename, (size, digest) in MEDIA.items():
        data = (args.media_dir / filename).read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            parser.error(f"media fixture mismatch: {filename}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "inputs.json", {
        **{key: str(value) for key, value in vars(args).items()},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "media": MEDIA,
    })
    try:
        runner = Regression(args)
        runner.wait_ready()
        runner.evidence.refresh()
        runner.execute()
    except Exception as error:
        write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
