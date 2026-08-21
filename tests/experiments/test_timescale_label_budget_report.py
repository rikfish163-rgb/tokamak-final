from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments/transfer_methods_2026/build_timescale_label_budget_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("timescale_budget_report", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(tpr30: float, fpr: float, *, query_count: int = 30):
    metric = {
        "tpr": tpr30, "tpr_at_30ms": tpr30, "fpr": fpr,
        "median_lead_ms": 100.0,
    }
    return {
        "results": {"static_unified": dict(metric), "online_unified": dict(metric)},
        "query_count": query_count, "fixed_query_manifest": "same.json",
    }


def test_decomposes_timescales_and_compares_equal_label_budget() -> None:
    module = _load_module()
    intra = _summary(0.80, 0.20, query_count=0)
    inter = _summary(0.82, 0.20)
    joint = _summary(0.84, 0.19)
    frontload = {"metrics": {
        "original_static_same_complement": {
            "tpr": .8, "tpr_at_30ms": .8, "fpr": .2, "median_lead_ms": 100,
        },
        "original_online_same_complement": {
            "tpr": .84, "tpr_at_30ms": .84, "fpr": .19, "median_lead_ms": 100,
        },
        "frontload_frozen_parent_threshold": {
            "tpr": .9, "tpr_at_30ms": .9, "fpr": .30, "median_lead_ms": 100,
        },
        "frontload_support110_recalibrated_threshold": {
            "tpr": .88, "tpr_at_30ms": .88, "fpr": .27, "median_lead_ms": 100,
        },
    }}
    report = module.build_report(
        intra=intra, inter=inter, joint=joint, frontload=frontload,
    )
    assert report["effect_decomposition"]["inter_main"] > 0
    assert report["equal_label_budget_verdict"]["online_beats_direct_frontload"]
    assert not report["equal_label_budget_verdict"]["compute_matched"]
