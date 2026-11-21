import numpy as np

from experiments.risk_aligned_online.full_path_meta_20261102.feedback_timing import (
    timing,
)


def test_last_record_update_cannot_affect_a_later_saved_prediction():
    t = np.arange(500, dtype=np.float64) / 1000
    events = [
        dict(maturity=k, first_affected=k + 1, accepted=a)
        for k, a in [(159, True), (287, False), (415, True), (499, True)]
    ]
    result = timing(t, events, True, 0.5, 415)
    assert result["accepted_receipts"] == 3
    assert result["accepted_with_future_record"] == 2
    assert result["accepted_with_timely_future"] == 2
    assert result["accepted_first_effect_in_30_125_ms"] == 1
    assert result["last_maturity_before_query"] == 159
    assert result["feedback_age_at_query_ms"] == 256
    assert result["first_effect_offset_ms"] == 160
    assert result["last_effect_offset_ms"] == 416
    assert timing(t, events, True, 0.5, 416)["last_maturity_before_query"] == 415


def test_no_update_and_unsupported_query_are_not_filled_with_zero_age():
    result = timing(np.arange(200, dtype=np.float64) / 1000, [], True, 2.0, None)
    assert result["feedback_age_at_query_ms"] is None
    assert result["last_maturity_before_query"] is None
    assert result["first_effect_offset_ms"] is None


def test_safe_record_uses_record_clock_not_disruption_clock():
    t = 3 + np.arange(1000, dtype=np.float64) / 1000
    events = [
        dict(maturity=159, first_affected=160, accepted=True),
        dict(maturity=799, first_affected=800, accepted=True),
    ]
    result = timing(t, events, False, None, 999)
    assert result["accepted_first_effect_in_30_125_ms"] is None
    assert result["accepted_with_timely_future"] is None
    assert result["first_effect_offset_ms"] == 160
    assert result["last_effect_offset_ms"] == 800
    assert result["feedback_age_at_query_ms"] == 200
