from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/transfer_methods_2026/paper_grade_package/scripts"
    / "assemble_dynamic_results.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("assemble_dynamic_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jtext_uses_first_endpoint_threshold_crossing_and_strict_10ms_lead() -> None:
    module = _module()
    records = [
        {"event": "endpoint_predict", "shot": 1, "endpoint": 0, "risk": 0.60},
        {"event": "endpoint_predict", "shot": 1, "endpoint": 8, "risk": 0.80},
        {"event": "endpoint_predict", "shot": 1, "endpoint": 9, "risk": 0.90},
        {"event": "endpoint_predict", "shot": 2, "endpoint": 10, "risk": 0.80},
        {"event": "endpoint_predict", "shot": 3, "endpoint": 1, "risk": 0.75},
        {"event": "endpoint_predict", "shot": 4, "endpoint": 1, "risk": 0.20},
    ]
    truth = {
        1: {"is_disrupt": True, "down_time_s": 1.0},
        2: {"is_disrupt": True, "down_time_s": 1.0},
        3: {"is_disrupt": False, "down_time_s": None},
        4: {"is_disrupt": False, "down_time_s": None},
    }
    timing = {
        shot: {"start_time_s": 0.9, "sample_rate_hz": 100.0}
        for shot in truth
    }

    rows = module.rebuild_jtext_endpoint_rows(
        records, threshold=0.70, truth_by_shot=truth, timing_by_shot=timing,
    )
    metrics = module.compute_metrics(rows, device="JTEXT")

    assert rows[0]["alarm_endpoint"] == 8
    assert rows[0]["alarm_time_s"] == pytest.approx(0.98)
    assert metrics["counts"] == {
        "disruptions": 2,
        "non_disruptions": 2,
        "raw_tp": 2,
        "raw_fn": 0,
        "tardy": 1,
        "operational_tp": 1,
        "operational_fn": 1,
        "fp": 1,
        "tn": 1,
    }
    assert metrics["operational_tpr"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)


def test_east_uses_sealed_alarm_time_and_inclusive_30ms_lead() -> None:
    module = _module()
    truth = {
        11: {"is_disrupt": True, "down_time_s": 1.0},
        12: {"is_disrupt": True, "down_time_s": 1.0},
        13: {"is_disrupt": False, "down_time_s": None},
        14: {"is_disrupt": False, "down_time_s": None},
    }
    predictions = [
        {"shot": 11, "alarmed": True, "alarm_time_s": 0.970},
        {"shot": 12, "alarmed": True, "alarm_time_s": 0.971},
        {"shot": 13, "alarmed": True, "alarm_time_s": 0.5},
        {"shot": 14, "alarmed": False, "alarm_time_s": None},
    ]

    rows = module.rebuild_east_rows(predictions, truth_by_shot=truth)
    metrics = module.compute_metrics(rows, device="EAST")

    assert metrics["counts"]["operational_tp"] == 1
    assert metrics["counts"]["tardy"] == 1
    assert metrics["minimum_lead_rule"] == "lead_ms >= 30"


def test_absolute_paths_are_confined_to_nonportable_provenance() -> None:
    module = _module()
    report = {
        "report_body": {
            "artifacts": [{"path": "east/seed131/predictions.jsonl"}],
        },
        "nonportable_provenance": {
            "east_root": "/tmp/east",
            "jtext_online_root": "/tmp/jtext",
            "data_root": "/tmp/data",
        },
    }

    module.validate_portable_report(report)
    report["report_body"]["bad"] = "/tmp/leak"
    with pytest.raises(ValueError, match="absolute path"):
        module.validate_portable_report(report)
