from __future__ import annotations

import os
from pathlib import Path


def project_root(anchor: str | Path | None = None) -> Path:
    override = os.environ.get("FUSION_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if anchor is not None:
        path = Path(anchor).resolve()
        return path.parents[2]
    return Path(__file__).resolve().parents[1]


def data_root(root: Path | None = None) -> Path:
    override = os.environ.get("FUSION_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (root or project_root()) / "data"
