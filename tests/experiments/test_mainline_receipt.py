import pytest

from experiments.transfer_methods_2026.build_mainline_receipt import (
    arithmetic_mean,
    delta,
    metric_row,
    operational_row,
)


def test_operational_row_normalizes_nested_lead() -> None:
    row = operational_row({
        "operational_tpr": 0.8,
        "fpr": 0.1,
        "auc": 0.9,
        "operational_tp_lead_ms": {"median_ms": 42.0},
    })
    assert row["median_lead_ms"] == 42.0


def test_metric_row_keeps_east_optional_auc_and_lead_missing() -> None:
    row = metric_row(
        {"tpr_at_30ms": 0.8, "fpr": 0.1, "TP": 8, "FP": 1},
        source="east",
        stage="static",
        tpr_key="tpr_at_30ms",
        fpr_key="fpr",
    )
    assert row["tpr"] == 0.8
    assert row["J"] == pytest.approx(0.7)
    assert row["auc"] is None
    assert row["median_lead_ms"] is None
    assert row["TP"] == 8


def test_delta_and_mean_do_not_invent_optional_metrics() -> None:
    baseline = {"tpr": 0.8, "fpr": 0.1, "auc": None, "median_lead_ms": None, "J": 0.7}
    candidate = {"tpr": 0.82, "fpr": 0.09, "auc": None, "median_lead_ms": None, "J": 0.73}
    assert delta(candidate, baseline) == pytest.approx(
        {"tpr": 0.02, "fpr": -0.01, "J": 0.03}
    )
    assert arithmetic_mean([baseline, candidate]) == pytest.approx(
        {"tpr": 0.81, "fpr": 0.095, "J": 0.715}
    )
