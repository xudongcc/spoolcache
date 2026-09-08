"""Authenticate an installed SpoolCache package against the release wheel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import zipfile


def verify(wheel: Path, expected_sha256: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected wheel SHA-256 must be 64 lowercase hex characters")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("wheel SHA-256 mismatch")
    distribution = importlib.metadata.distribution("spoolcache")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if not (name.startswith("spoolcache/") or ".dist-info/" in name):
                raise ValueError(f"unexpected wheel member: {name}")
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError("unsafe wheel member")
            installed = Path(distribution.locate_file(name))
            if installed.read_bytes() != archive.read(name):
                raise ValueError(f"installed bytes differ: {name}")
            checked += 1
        expected_modules = {n for n in archive.namelist() if n.startswith("spoolcache/") and not n.endswith("/")}
    package = Path(distribution.locate_file("spoolcache"))
    actual_modules = {
        "spoolcache/" + str(p.relative_to(package))
        for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    if actual_modules != expected_modules:
        raise ValueError("installed package contains missing or extra files")
    spec = importlib.util.find_spec("spoolcache")
    if spec is None or Path(spec.origin).resolve() != (package / "__init__.py").resolve():
        raise ValueError("installed package is shadowed by another import path")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("archive_info", {}).get("hashes", {}).get("sha256") != expected_sha256:
        raise ValueError("installation is not from the expected immutable wheel")
    return {"schema": "spoolcache-install-verification/v1", "status": "passed",
            "version": distribution.version, "wheel_sha256": expected_sha256,
            "verified_files": checked, "import_path": spec.origin}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.sha256), sort_keys=True))
