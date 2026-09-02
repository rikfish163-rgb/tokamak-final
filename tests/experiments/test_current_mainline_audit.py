from __future__ import annotations

import pytest

from experiments.transfer_methods_2026 import audit_current_mainline as audit
from experiments.transfer_methods_2026.support_label_audit import summarize_support_labels


def test_audit_normalizes_jtext_operational_counts() -> None:
    row = {
        "counts": {
            "operational_tp": 94,
            "operational_fn": 16,
            "fp": 12,
            "tn": 98,
        },
        "operational_tpr": 94 / 110,
        "fpr": 12 / 110,
        "operational_tp_lead_ms": {"median_ms": 68.4},
    }

    result = audit._metrics(
        row,
        label="synthetic J-TEXT",
        disruptive=110,
        safe=110,
        tpr_key="operational_tpr",
        count_keys={
            "TP": "operational_tp",
            "FP": "fp",
            "FN": "operational_fn",
            "TN": "tn",
        },
    )

    assert result["TP"] == 94
    assert result["FN"] == 16
    assert result["tpr"] == pytest.approx(94 / 110)
    assert result["median_lead_ms"] == pytest.approx(68.4)


def test_audit_checks_raw_east_tpr_separately_from_operational_tpr() -> None:
    row = {
        "TP": 162,
        "FP": 35,
        "FN": 18,
        "TN": 145,
        "tpr": 0.9,
        "tpr_at_30ms": 138 / 180,
        "fpr": 35 / 180,
        "median_lead_ms": 124.0,
    }

    result = audit._metrics(
        row,
        label="synthetic EAST",
        disruptive=180,
        safe=180,
        tpr_key="tpr_at_30ms",
        raw_tpr_key="tpr",
    )

    assert result["tpr"] == pytest.approx(138 / 180)
    assert result["tpr_key"] == "tpr_at_30ms"


def test_audit_rejects_raw_confusion_count_mismatch() -> None:
    row = {
        "TP": 161,
        "FP": 35,
        "FN": 19,
        "TN": 145,
        "tpr": 0.9,
        "tpr_at_30ms": 138 / 180,
        "fpr": 35 / 180,
        "median_lead_ms": 124.0,
    }

    with pytest.raises(RuntimeError, match="raw TPR/count mismatch"):
        audit._metrics(
            row,
            label="bad EAST",
            disruptive=180,
            safe=180,
            tpr_key="tpr_at_30ms",
            raw_tpr_key="tpr",
        )


def test_support_label_audit_keeps_hdf5_authority_explicit() -> None:
    result = summarize_support_labels(
        {"disruption": [101, 102], "safe": [201, 202, 203]},
        {101: 1, 102: 0, 201: 0, 202: 1, 203: 0},
    )

    assert result["declared_counts"] == {
        "disruption": 2,
        "safe": 3,
        "total": 5,
    }
    assert result["authoritative_counts"] == {
        "disruption": 2,
        "safe": 3,
        "total": 5,
    }
    assert result["composition_matches_declared"] is False
    assert result["mismatches"] == [{
        "shot": 102,
        "declared_role": "disruption",
        "declared_is_disrupt": 1,
        "hdf5_is_disrupt": 0,
    }, {
        "shot": 202,
        "declared_role": "safe",
        "declared_is_disrupt": 0,
        "hdf5_is_disrupt": 1,
    }]


def test_support_label_audit_rejects_role_overlap() -> None:
    with pytest.raises(ValueError, match="support role overlap"):
        summarize_support_labels(
            {"disruption": [101], "safe": [101]},
            {101: 1},
        )
