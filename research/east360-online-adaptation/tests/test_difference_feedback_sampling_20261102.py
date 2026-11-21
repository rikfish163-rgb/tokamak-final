import numpy as np

from experiments.risk_aligned_online.difference_feedback_probe_20261102.run import (
    select_queries,
)


def test_disruption_queries_cover_declared_leads_without_using_late_points():
    selected, skipped = select_queries(np.arange(1000) / 1000, True, 1.0)
    assert [r["query"] for r in selected] == [750, 850, 900, 940, 965]
    assert [r["actual_lead_ms"] for r in selected] == [250, 150, 100, 60, 35]
    assert not skipped


def test_safe_queries_are_uniform_actual_endpoints_after_minimum_context():
    selected, skipped = select_queries(np.arange(1000) / 1000, False, None)
    assert [r["query"] for r in selected] == [160, 369, 579, 789, 999]
    assert not skipped


def test_missing_requested_lead_is_explicit_not_replaced_by_any_nearest_endpoint():
    selected, skipped = select_queries(np.arange(200) / 1000, True, 0.2)
    assert [r["query"] for r in selected] == [165]
    assert len(skipped) == 4
    assert all(r["reason"] == "requested_lead_not_covered" for r in skipped)


def test_short_record_is_explicitly_uncovered():
    selected, skipped = select_queries(np.arange(160) / 1000, False, None)
    assert not selected and skipped[0]["reason"] == "insufficient_prefix"
