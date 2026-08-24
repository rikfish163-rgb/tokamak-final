from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT / "experiments/transfer_methods_2026/paper_grade_package/scripts/"
    "validate_dynamic_source_online_gain.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_dynamic_source_online_gain", MODULE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recompute_metrics_uses_hdf5_outcomes_and_30ms_timeliness() -> None:
    module = load_module()
    rows = [
        {"shot": 1, "alarmed": True, "alarm_time_s": 0.90},
        {"shot": 2, "alarmed": True, "alarm_time_s": 0.98},
        {"shot": 3, "alarmed": False, "alarm_time_s": None},
        {"shot": 4, "alarmed": True, "alarm_time_s": 0.50},
        {"shot": 5, "alarmed": False, "alarm_time_s": None},
    ]
    outcomes = {
        1: {"is_disrupt": True, "down_time_s": 1.0},
        2: {"is_disrupt": True, "down_time_s": 1.0},
        3: {"is_disrupt": True, "down_time_s": 1.0},
        4: {"is_disrupt": False, "down_time_s": None},
        5: {"is_disrupt": False, "down_time_s": None},
    }

    metric = module.recompute_metrics(rows, outcomes)
    assert metric["TP"] == 2
    assert metric["FN"] == 1
    assert metric["FP"] == 1
    assert metric["TN"] == 1
    assert metric["tpr_at_30ms"] == pytest.approx(1 / 3)
    assert metric["fpr"] == pytest.approx(1 / 2)


def test_temporal_ledgers_require_query_before_reveal_and_no_label_features() -> None:
    module = load_module()
    static = [
        {"shot": 10, "stream_index": 0},
        {"shot": 20, "stream_index": 1},
    ]
    online = [
        {"shot": 10, "stream_index": 0, "state_shot": 1, "queried": False},
        {"shot": 20, "stream_index": 1, "state_shot": 2, "queried": True},
    ]
    queries = [
        {"shot": 10, "stream_index": 0, "selected": False},
        {"shot": 20, "stream_index": 1, "selected": True},
    ]
    reveals = [{"shot": 20, "stream_index": 1, "revealed_is_disrupt": False}]

    receipt = module.validate_temporal_ledgers(
        static, online, queries, reveals, expected_count=2,
    )
    assert receipt == {"shots": 2, "queries": 1, "reveals": 1}

    leaked = [dict(queries[0], is_disrupt=True), queries[1]]
    with pytest.raises(RuntimeError, match="query decision contains outcome"):
        module.validate_temporal_ledgers(
            static, online, leaked, reveals, expected_count=2,
        )


def test_metric_comparison_rejects_silent_summary_drift() -> None:
    module = load_module()
    expected = {
        "TP": 2, "FN": 1, "FP": 1, "TN": 1,
        "tpr": 2 / 3, "fpr": 1 / 2, "tpr_at_30ms": 1 / 3,
        "median_lead_ms": 60.0, "over_3000ms_count": 0,
    }
    module.assert_metric_equal(expected, dict(expected), name="fixture")
    changed = dict(expected, fpr=0.4)
    with pytest.raises(RuntimeError, match="fixture fpr"):
        module.assert_metric_equal(expected, changed, name="fixture")
