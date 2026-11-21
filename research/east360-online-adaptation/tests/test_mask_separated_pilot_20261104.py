import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.mask_separated_intra_20261103 import (
    pilot_core as c,
)


def manifest():
    return [
        dict(
            shot=s,
            query=200 + j * 31,
            origin=174 + j * 31,
            label=999,
            disruptive="UNTRUSTED",
            gradient256=[999],
        )
        for s in range(20)
        for j in range(5)
    ]


def test_coordinates_discard_old_truth_and_old_neural_outputs():
    rows = manifest()
    before = copy.deepcopy(rows)
    result = c.coordinates(rows, list(range(20)))
    assert len(result) == 100 and rows == before
    assert all(set(r) == {"shot", "origin", "query"} for r in result)


@pytest.mark.parametrize("bad", ["duplicate", "foreign", "clock", "short", "bool"])
def test_coordinates_reject_wrong_fixed_development_support(bad):
    rows = manifest()
    if bad == "duplicate":
        rows[-1] = rows[-2].copy()
    elif bad == "foreign":
        rows[-1]["shot"] = 999
    elif bad == "clock":
        rows[-1]["origin"] += 1
    elif bad == "short":
        rows.pop()
    else:
        rows[-1]["query"] = True
    with pytest.raises(ValueError):
        c.coordinates(rows, list(range(20)))


def raw_predictions():
    return [
        dict(
            stem=stem,
            mode=mode,
            shot=sid,
            query=200,
            query_time=when,
            before_logit=0.0,
            after_logit=(-0.2 if not sid else 0.2) if mode == "real" else 0.0,
        )
        for stem in c.STEMS
        for mode in c.MODES
        for sid, when in ((0, 0.8), (1, 0.9), (2, 0.8))
    ]


def test_scoring_gets_hard_truth_and_time_only_from_supplied_hdf_outcomes():
    raw = raw_predictions()
    old = copy.deepcopy(raw)
    scored = c.score(raw, {0: (False, None), 1: (True, 1.0), 2: (True, 1.0)}, 0.5)
    assert raw == old
    real = [r for r in scored if r["stem"] == "reference_bt" and r["mode"] == "real"]
    assert [r["disruptive"] for r in real] == [False, True, True]
    assert [r["point_label125"] for r in real] == [0, 1, 0]
    assert all(r["delta_bce"] < 0 for r in real)
    assert real[-1]["margin_target"] == 1  # Timely danger is not the 125ms point label.
    assert real[0]["after_bce"] == pytest.approx(np.logaddexp(0, -0.2))


@pytest.mark.parametrize(
    "bad", ["untimely", "nan", "outcome", "threshold", "truth_in_input"]
)
def test_score_rejects_invalid_or_leaky_records(bad):
    raw = raw_predictions()
    outcomes, threshold = {0: (False, None), 1: (True, 1.0), 2: (True, 1.0)}, 0.5
    if bad == "untimely":
        raw[1]["query_time"] = 0.970
    elif bad == "nan":
        raw[1]["after_logit"] = float("nan")
    elif bad == "outcome":
        outcomes[1] = (1, 1.0)
    elif bad == "threshold":
        threshold = 1.0
    else:
        raw[1]["label"] = 1
    with pytest.raises(ValueError):
        c.score(raw, outcomes, threshold)


def test_summary_uses_shot_means_not_query_pseudoreplicates():
    rows = [
        dict(
            shot=s,
            disruptive=d,
            actual_lead_ms=lead,
            before_bce=1.0,
            after_bce=1.0 + delta,
            delta_bce=delta,
        )
        for s, d, lead, delta in [(1, False, None, 0.1)] * 5
        + [(2, False, None, -0.2), (3, True, 100, -0.3), (3, True, 200, 0.1)]
    ]
    summary = c.aggregate(rows)
    assert summary["safe"]["mean_delta_bce"] == pytest.approx(-0.05)
    assert summary["disruptive"]["mean_delta_bce"] == pytest.approx(-0.1)
    assert summary["danger_window"]["mean_delta_bce"] == pytest.approx(-0.3)
    assert summary["early_window"]["mean_delta_bce"] == pytest.approx(0.1)
    assert summary["balanced"] == pytest.approx(-0.075)


def test_gate_cannot_pass_from_static_stem_improvement_or_matching_hold():
    def arm(delta):
        return {
            **{
                g: {"mean_delta_bce": delta}
                for g in ("safe", "disruptive", "danger_window")
            },
            "balanced": delta,
        }

    summary = {"reference_bt_" + mode: arm(-0.01) for mode in c.MODES}
    assert c.gate(summary)["passed"] is False
    summary["reference_bt_real"] = arm(-0.02)
    assert c.gate(summary)["passed"] is True
    summary["reference_bt_real"]["safe"]["mean_delta_bce"] = 0.0
    assert c.gate(summary)["passed"] is False
