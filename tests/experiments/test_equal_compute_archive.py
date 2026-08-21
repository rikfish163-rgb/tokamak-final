from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "experiments/transfer_methods_2026/paper_grade_package/scripts"
    / "archive_equal_compute_selector_results.py"
)


def archive_module():
    spec = importlib.util.spec_from_file_location("archive_equal_compute", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def make_complete_bundle(root: Path) -> None:
    archive = archive_module()
    plan = {
        "seeds": list(archive.SEEDS),
        "replicates": 3,
        "commands": [{"seed": seed, "runs": ["x"] * 4}
                     for seed in archive.SEEDS],
    }
    (root / "run_plan.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "run_plan.json").write_text(json.dumps(plan))
    for relative in archive.required_relative_paths():
        path = root / relative
        if path.name == "run_plan.json":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "summary.json":
            path.write_text(json.dumps({"shot_count": 360}))
        elif path.name in {"predictions.jsonl", "query_decisions.jsonl"}:
            path.write_text("{}\n" * 360)
        elif path.suffix == ".json":
            path.write_text("{}\n")
        else:
            path.write_bytes(b"fixture\n")


def test_archive_fails_closed_on_incomplete_bundle(tmp_path: Path) -> None:
    archive = archive_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "run_plan.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="incomplete equal-compute"):
        archive.archive(source, tmp_path / "release", replace=False)


def test_archive_copies_complete_bundle_and_requires_explicit_replace(
    tmp_path: Path,
) -> None:
    archive = archive_module()
    source = tmp_path / "source"
    destination = tmp_path / "release"
    make_complete_bundle(source)

    archive.archive(source, destination, replace=False)
    archive.validate_source(destination)
    assert len([path for path in destination.rglob("*") if path.is_file()]) == len(
        archive.required_relative_paths()
    )

    with pytest.raises(FileExistsError, match="pass --replace"):
        archive.archive(source, destination, replace=False)

    archive.archive(source, destination, replace=True)
    archive.validate_source(destination)
    assert not destination.with_name(destination.name + ".previous").exists()
