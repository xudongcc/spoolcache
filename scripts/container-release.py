#!/usr/bin/env python3
"""Authenticate published wheel inputs against a release Git tag and commit."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import zipfile


def verify(directory: Path, tag: str, commit: str) -> dict[str, str]:
    if re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag) is None:
        raise ValueError("expected a stable release Git tag: vMAJOR.MINOR.PATCH")
    version = tag[1:]
    receipt = json.loads((directory / "release.json").read_text())
    if receipt.get("schema") != "spoolcache-release/v1":
        raise ValueError("unknown release receipt schema")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or receipt.get("commit") != commit:
        raise ValueError("release receipt commit differs from the Git tag")
    name = f"spoolcache-{version}-py3-none-any.whl"
    wheels = list(directory.glob("*.whl"))
    if receipt.get("wheel") != name or wheels != [directory / name]:
        raise ValueError("expected exactly the universal wheel for the Git tag")
    wheel = wheels[0]
    sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if sha256 != receipt.get("wheel_sha256"):
        raise ValueError("release wheel SHA-256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(
            archive.read(f"spoolcache-{version}.dist-info/METADATA")
        )
    if metadata["Name"] != "spoolcache" or metadata["Version"] != version:
        raise ValueError("wheel metadata differs from the Git tag")
    return {"tag": tag, "version": version, "commit": commit,
            "wheel": name, "wheel_sha256": sha256}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.tag, args.commit), sort_keys=True))
