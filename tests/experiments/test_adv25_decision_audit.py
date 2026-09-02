from __future__ import annotations

import importlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "transfer_methods_2026"


def _audit():
    return importlib.import_module("audit_adv25_decision")


def _write_shot(path: Path, *, disruptive: bool, down_time_s: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        signal = data.create_dataset("ip", data=np.zeros(200, dtype=np.float32))
        signal.attrs["StartTime"] = 0.0
        signal.attrs["SampleRate"] = 1000.0
        meta = handle.create_group("meta")
        meta.create_dataset("IsDisrupt", data=np.asarray(disruptive, dtype=np.bool_))
        meta.create_dataset("DownTime", data=np.asarray(down_time_s, dtype=np.float64))


def test_hdf5_meta_is_authoritative_and_endpoint_count_is_causal(tmp_path: Path) -> None:
    audit = _audit()
    path = tmp_path / "J-TEXT/unified_hdf5/100/101.hdf5"
    _write_shot(path, disruptive=True)
    meta = audit.read_hdf5_meta(tmp_path, "J-TEXT", 101)
    assert meta["is_disrupt"] is True
    # The synthetic shot ends at 199 ms; its DownTime endpoint (200 ms) is
    # not a sampled row, so the inclusive intervals contain 25 and 125 rows.
    assert audit.endpoint_count(meta, 25.0) == 25
    assert audit.endpoint_count(meta, 125.0) == 125


def test_role_overlap_is_rejected() -> None:
    audit = _audit()
    with pytest.raises(ValueError, match="split overlap"):
        audit._require_unique_roles({"train": [1, 2], "val": [2]})


def test_adv25_audit_reads_expected_metric_shape(tmp_path: Path) -> None:
    audit = _audit()
    payload = {
        "results": {
            "static_unified": {
                "tpr_at_30ms": 0.8,
                "fpr": 0.2,
                "over_3000ms_count": 1,
            },
            "online_unified": {
                "tpr_at_30ms": 0.85,
                "fpr": 0.15,
                "over_3000ms_count": 2,
            },
        }
    }
    result = audit._metrics_from_summary(payload)
    assert result["static_tpr30"] == pytest.approx(0.8)
    assert result["static_fpr"] == pytest.approx(0.2)
    assert result["static_j"] == pytest.approx(0.6)
    assert result["online_tpr30"] == pytest.approx(0.85)
    assert result["online_fpr"] == pytest.approx(0.15)
    assert result["online_j"] == pytest.approx(0.7)
    assert result["delta_tpr30"] == pytest.approx(0.05)
    assert result["delta_fpr"] == pytest.approx(-0.05)
    assert result["delta_j"] == pytest.approx(0.1)
    assert result["static_over_3000ms_count"] == 1
    assert result["online_over_3000ms_count"] == 2


def test_current_historical_reference_is_rejected_without_lineage() -> None:
    runner = importlib.import_module("run_online_active_a1")
    reference = runner.REFERENCE_DIR / "seed131.json"
    if not reference.is_file():
        pytest.skip("historical reference is not present in this checkout")
    with pytest.raises(RuntimeError, match="no reproduction_contract"):
        runner.reference_contract(131)


def test_audit_script_has_local_output_contract() -> None:
    script = HERE / "audit_adv25_decision.py"
    source = script.read_text(encoding="utf-8")
    assert "adv25-root-cause-audit-v1" in source
    assert "DIAGNOSTIC_ONLY" in source
