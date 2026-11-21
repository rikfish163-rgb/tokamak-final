import json

import numpy as np
import pytest

from experiments.risk_aligned_online.local_readout_20261029.error_profile import (
    ARMS,
    aggregate,
    causal_feedback_before,
    cosine_stats,
    load_sealed_run,
    profile_shot,
)


def receipt(events=(), updates=None, active=True):
    updates = len(events) if updates is None else updates
    summary = {
        arm: {"opportunities": len(events), "updates": updates}
        for arm in ARMS[1:]
    }
    summary["feedback"] = list(events)
    return {"shot": 1, "index": 21, "adaptation_active": active,
            "outcome_used": False, "updates": summary if active else {}}


def event(clock):
    return {"clock": clock, "calibration_last_target": clock - 2,
            "gradient_first_target": clock - 1,
            "gradient_last_target": clock - 1}


def curves(values):
    return {arm: np.array(values, dtype=float) for arm in ARMS}


def test_cosine_uses_cold_relative_deltas_and_zero_norm_is_undefined():
    value = cosine_stats(np.array([1., -1.]), np.array([-2., 2.]))
    assert value["cosine"] == pytest.approx(-1.)
    assert value["dot"] == pytest.approx(-4.)
    assert cosine_stats(np.zeros(2), np.ones(2))["cosine"] is None
    with pytest.raises(ValueError, match="finite"):
        cosine_stats(np.array([np.nan]), np.array([1.]))


def test_disruption_margin_is_strictly_more_than_30ms_and_post_down_excluded():
    times = np.array([9.969, 9.970, 9.980, 10.001])
    scores = curves([.4, .6, .8, .99])
    result = profile_shot(times, scores, .5, True, 10., receipt([event(2)]))
    cold = result["arms"]["cold"]
    assert result["baseline_category"] == "tardy"
    assert cold["first_crossing"]["lead_ms"] == 30.
    assert cold["first_crossing"]["feedback_before"]["had_update"] is False
    assert cold["timely_peak"]["index"] == 0
    assert cold["timely_peak"]["margin"] == pytest.approx(-.1)
    assert cold["valid_peak"]["index"] == 2
    assert cold["valid_peak"]["feedback_before"]["had_update"] is False
    assert not cold["timely"]


def test_candidate_crossing_and_margin_changes_are_actionable():
    times = np.array([9.950, 9.960, 9.969, 9.970, 9.980])
    scores = curves([.1, .2, .4, .6, .7])
    scores["local_readout"][2:] = [.55, .6, .7]
    result = profile_shot(times, scores, .5, True, 10., receipt([event(2)]))
    pair = result["comparisons"]["cold"]
    assert result["baseline_category"] == "tardy"
    assert result["arms"]["local_readout"]["timely"]
    assert result["arms"]["local_readout"]["first_crossing"]["lead_ms"] == 31.
    assert result["arms"]["local_readout"]["first_crossing"]["feedback_before"]["had_update"] is True
    assert pair["candidate_minus_control_peak_margin"] == pytest.approx(.15)
    assert pair["candidate_minus_control_first_crossing_ms"] == pytest.approx(-1.)
    assert pair["transition"] == "tardy->TP"


def test_no_update_reference_guard_covers_cold20_and_zero_update_shots():
    times = np.arange(4, dtype=float)
    scores = curves([.1, .2, .3, .4])
    scores["local_readout"][3] = .6
    with pytest.raises(ValueError, match="no-update"):
        profile_shot(times, scores, .5, False, None, receipt(active=False))
    with pytest.raises(ValueError, match="no-update"):
        profile_shot(times, scores, .5, False, None, receipt([event(2)], updates=0))


def test_no_update_prefix_guard_also_preserves_threshold_decisions():
    times = np.arange(4, dtype=float)
    scores = curves([.1, .4999999, .3, .4])
    scores["local_hold"][1] = .5000001
    with pytest.raises(ValueError, match="no-update"):
        profile_shot(times, scores, .5, False, None, receipt([event(2)]))


def test_feedback_at_event_clock_is_future_only_and_partial_is_unknown():
    r = receipt([event(2), event(4)], updates=1)
    assert causal_feedback_before(r, "local_readout", 1)["had_update"] is False
    assert causal_feedback_before(r, "local_readout", 2)["had_update"] is None
    assert causal_feedback_before(r, "local_readout", 4)["had_update"] is True


def test_noncausal_feedback_and_recorded_decision_mismatch_are_rejected():
    bad = event(2)
    bad["gradient_last_target"] = 2
    with pytest.raises(ValueError, match="causal"):
        profile_shot(np.arange(4.), curves([.1] * 4), .5, False, None, receipt([bad]))
    records = {a: {"shot": 1, "is_disrupt": False, "alarm": False,
                   "lead_ms": None, "timely": False} for a in ARMS}
    records["cold"]["alarm"] = True
    with pytest.raises(ValueError, match="recorded"):
        profile_shot(np.arange(4.), curves([.1] * 4), .5, False, None,
                     receipt(active=False), records)


def test_no_timely_samples_and_category_stratification():
    times = np.array([9.980, 9.990])
    one = profile_shot(times, curves([.1, .2]), .5, True, 10., receipt(active=False))
    two = profile_shot(times, curves([.1, .8]), .5, False, None, receipt(active=False))
    assert one["baseline_category"] == "FN"
    assert one["arms"]["cold"]["timely_peak"] is None
    assert one["comparisons"]["cold"]["candidate_minus_control_peak_margin"] is None
    totals = aggregate([one, two])
    assert totals["baseline_categories"]["FN"] == 1
    assert totals["baseline_categories"]["FP"] == 1
    assert totals["strata"]["FN"]["candidate_peak_feedback"]["no_peak"] == 1
    assert totals["strata"]["FP"]["candidate_peak_feedback"]["no"] == 1
    assert totals["strata"]["FP"]["margin_by_peak_feedback"]["no"]["cold"]["mean"] == 0.


def test_grouped_margin_changes_and_candidate_only_crossing():
    times = np.arange(4, dtype=float)
    scores = curves([.1, .2, .3, .4])
    scores["local_readout"][2:] = [.6, .7]
    row = profile_shot(times, scores, .5, False, None, receipt([event(2)]))
    assert row["comparisons"]["cold"]["crossing_relation"] == "candidate_only"
    assert row["comparisons"]["cold"]["transition"] == "TN->FP"
    group = aggregate([row])["strata"]["TN"]["margin_by_peak_feedback"]["yes"]["cold"]
    assert group["count"] == 1
    assert group["mean"] == pytest.approx(.3)
    assert group["median"] == pytest.approx(.3)


def test_inconsistent_feedback_counts_and_nonfinite_curves_are_rejected():
    r = receipt([event(2)])
    r["updates"]["local_hold"]["updates"] = 2
    with pytest.raises(ValueError, match="feedback/update"):
        profile_shot(np.arange(4.), curves([.1] * 4), .5, False, None, r)
    scores = curves([.1] * 4)
    scores["cold"][1] = np.nan
    with pytest.raises(ValueError, match="finite probability"):
        profile_shot(np.arange(4.), scores, .5, False, None, receipt(active=False))


def test_missing_or_unsealed_complete_is_refused(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        load_sealed_run(tmp_path)
    (tmp_path / "complete.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="complete"):
        load_sealed_run(tmp_path)


def test_tampered_sealed_manifest_is_refused(tmp_path):
    (tmp_path / "complete.json").write_text(json.dumps({"complete": True, "protocol_sha256": "0" * 64}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed hash mismatch"):
        load_sealed_run(tmp_path)
