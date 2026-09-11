#!/usr/bin/env python3
"""Build a wheel from one clean Git commit, never from the live source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def build(root: Path, output: Path, *, python: Path = Path(sys.executable)) -> Path:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("release requires a clean committed tree")
    commit = git("rev-parse", "HEAD")
    epoch = git("show", "-s", "--format=%ct", commit)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previously published artifact or receipt.
    if any(output.iterdir()):
        raise ValueError("release output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="spoolcache-release-") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(root), "archive", commit, "pyproject.toml", "README.md", "LICENSE", "src"]
        )
        subprocess.run(["tar", "-xf", "-", "-C", str(source)], input=archive, check=True)
        subprocess.run(
            ["uv", "build", "--python", str(python.resolve()), "--wheel", "--out-dir", str(output.resolve()), str(source)],
            env={**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONHASHSEED": "0"},
            check=True,
        )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("build did not produce exactly one wheel")
    wheel = wheels[0]
    receipt = {
        "schema": "spoolcache-release/v1", "commit": commit,
        "source_date_epoch": int(epoch), "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "build_requires": ["setuptools==80.9.0", "wheel==0.45.1"],
        "build_python": str(python.resolve()),
    }
    (output / "release.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/release"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="target interpreter path (defaults to this script's Python)")
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.output, python=args.python))
