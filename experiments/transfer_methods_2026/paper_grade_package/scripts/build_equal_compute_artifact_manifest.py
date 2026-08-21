#!/usr/bin/env python3
"""Build the portable content ledger for the fixed-compute selector screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / "evidence/equal_compute_selector_control_v1"
OUTPUT = ROOT / "artifact_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        artifacts.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not artifacts:
        raise RuntimeError("fixed-compute archive is empty")
    payload = {
        "schema_version": "equal-compute-selector-artifact-manifest-v1",
        "artifacts": artifacts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifacts": len(artifacts), "manifest": str(OUTPUT)}))


if __name__ == "__main__":
    main()
