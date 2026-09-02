from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/transfer_methods_2026/validate_fpr_episode_v3.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_fpr_episode_v3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_baseline_accepts_sealed_frozen_runs_layout(tmp_path: Path) -> None:
    validator = load_validator()
    run = tmp_path / "seed131"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "stage": "STATIC360_THEN_RESET_ONLINE360_FROZEN_ARCHITECTURE",
    }))
    (run / "predictions.jsonl").write_text("{}\n")
    (run / "query_decisions.jsonl").write_text("{}\n")

    assert validator.resolve_baseline_run(tmp_path, 131) == run.resolve()


def test_resolve_baseline_rejects_unrelated_or_missing_layout(tmp_path: Path) -> None:
    validator = load_validator()
    run = tmp_path / "seed131"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"stage": "other"}))
    (run / "predictions.jsonl").write_text("{}\n")
    (run / "query_decisions.jsonl").write_text("{}\n")

    with pytest.raises(FileNotFoundError, match="no sealed v2.1 frozen baseline"):
        validator.resolve_baseline_run(tmp_path, 131)
