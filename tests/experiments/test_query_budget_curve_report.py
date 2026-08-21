from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments/transfer_methods_2026/build_query_budget_curve_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("query_budget_report", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metric(tpr30: float, fpr: float) -> dict:
    return {
        "TP": 1, "FP": 1, "FN": 1, "TN": 1,
        "tpr": tpr30, "tpr_at_30ms": tpr30, "fpr": fpr,
        "median_lead_ms": 100.0, "over_3000ms_count": 0,
    }


def _online(query_count: int, tpr30: float, fpr: float) -> dict:
    return {
        "query_count": query_count,
        "results": {
            "static_unified": _metric(.80, .20),
            "online_unified": _metric(tpr30, fpr),
        },
    }


def _frontload(online_j_wins: bool) -> dict:
    online = _metric(.84, .18)
    direct = _metric(.82 if online_j_wins else .90, .20)
    return {"metrics_on_label_disjoint_complement": {
        "original_static_same_complement": _metric(.80, .20),
        "original_online_same_complement": online,
        "frontload_frozen_parent_threshold": direct,
        "frontload_support110_recalibrated_threshold": _metric(.81, .20),
    }}


def test_separates_full_stream_curve_from_paired_complements() -> None:
    module = _load_module()
    report = module.build_report([
        {"quota": 4, "online": _online(66, .84, .15),
         "frontload": _frontload(True)},
        {"quota": 2, "online": _online(35, .82, .16),
         "frontload": _frontload(False)},
    ])
    assert [row["query_count"] for row in report["full360_online_curve"]] == [35, 66]
    assert report["paired_label_disjoint_complements"][0]["online_beats_best_direct"] is False
    assert report["paired_label_disjoint_complements"][1]["online_beats_best_direct"] is True
    assert report["cross_budget_complement_comparison_allowed"] is False
