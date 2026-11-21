"""Regression tests for the frozen, retrospective H25 diagnostic."""

import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.feedback_information_20261029.diagnostic import (
    ADAPTIVE_ARMS,
    HISTORIES,
    bootstrap_summary,
    choose_disruptive_clock,
    greedy_pairs,
    reconstruct_masks,
    validate_feedback,
)


def point(shot, clock, probability, mask, **extras):
    return dict(shot=shot, clock=clock, probability=probability, mask=mask, **extras)


def feedback(observed=50):
    return {
        "clock": 256,
        "arms": {
            arm: dict(
                observed=observed,
                loss_before=0.125 if observed else None,
                accepted_norm=0.0,
                reason="no_acceptable_step" if observed else "no_observed_target",
                factor=0.0,
            )
            for arm in ADAPTIVE_ARMS
        },
    }


def test_rounded_lower_boundary_and_unrounded_upper_and_last_clock():
    clocks = [256, 384, 512, 640]
    leads = [125.0000000001, 125.0, 30.0000000006, 30.0000000001]
    assert np.round(leads[-1], 9) == 30.0
    assert choose_disruptive_clock(clocks, leads, 999) == (512, "selected")
    assert choose_disruptive_clock(clocks, leads, 400) == (384, "selected")
    assert choose_disruptive_clock(clocks, leads, 383) == (None, "cold_already_alarmed")
    assert choose_disruptive_clock([256], [30.0000000001], 999) == (
        None,
        "no_time_window",
    )
    # A crossing at clock c is not before that feedback clock.
    assert choose_disruptive_clock([256], [125.0], 256) == (256, "selected")


def test_mask_requires_each_segment_and_start_plus_25_points():
    active = np.ones(23, dtype=bool)
    active[2] = False
    available = np.ones((256, 23), dtype=bool)
    available[205, 3] = False  # first segment start, c-51
    available[230, 4] = False  # shared end/start, c-26
    available[255, 5] = False  # second segment last target, c-1
    mask = reconstruct_masks(available, active, [256])[0]
    assert mask.shape == (2, 23)
    assert not mask[:, 2].any()
    assert not mask[0, 3] and mask[1, 3]
    assert not mask[:, 4].any()
    assert mask[0, 5] and not mask[1, 5]
    assert mask[:, 0].all()


def test_greedy_order_dedup_and_no_effect_selection():
    mask = np.ones((2, 23), dtype=bool)
    ds = [point(20, 512, 0.5, mask, q=100), point(10, 512, 0.5, mask, q=-100)]
    ss = [
        point(102, 640, 0.5, mask, q=-1),
        point(101, 512, 0.5, mask, q=1),
        point(101, 640, 0.5, mask, q=999),
    ]
    pairs, unmatched = greedy_pairs(ds, ss)
    assert [(d["shot"], s["shot"], s["clock"]) for d, s in pairs] == [
        (10, 101, 512),
        (20, 102, 640),
    ]
    assert unmatched == []
    ds2, ss2 = copy.deepcopy(ds), copy.deepcopy(ss)
    for p in ds2 + ss2:
        p["q"] = -999999 * p["q"]
        p["outcome_effect"] = 1e100
    pairs2, _ = greedy_pairs(ds2, ss2)
    assert [(d["shot"], s["shot"], s["clock"]) for d, s in pairs2] == [
        (10, 101, 512),
        (20, 102, 640),
    ]


def test_mask_equality_not_just_observation_count_and_tie_break():
    dmask = np.zeros((2, 23), dtype=bool)
    dmask[:, 0] = True
    wrong = np.zeros((2, 23), dtype=bool)
    wrong[:, 1] = True
    ds = [point(1, 512, 0.0, dmask)]
    ss = [
        point(90, 512, 0.0, wrong),
        point(10, 640, 0.05, dmask),
        point(9, 640, 0.05, dmask),
        point(9, 384, 0.05, dmask),
    ]
    pairs, unmatched = greedy_pairs(ds, ss)
    assert [(s["shot"], s["clock"]) for _, s in pairs] == [(9, 384)]
    assert unmatched == []
    assert greedy_pairs(ds, [ss[0]])[1] == [1]


@pytest.mark.parametrize("problem", ["missing_arm", "missing_loss", "nan", "infinite", "wrong_count"])
def test_missing_nonfinite_and_mask_count_mismatch_rejected(problem):
    mask = np.zeros((2, 23), dtype=bool)
    mask[:, 0] = True
    record = feedback()
    validate_feedback(record, mask)
    if problem == "missing_arm":
        del record["arms"]["hold_reverse"]
    elif problem == "missing_loss":
        del record["arms"]["hold_reverse"]["loss_before"]
    elif problem == "nan":
        record["arms"]["hold_reverse"]["loss_before"] = float("nan")
    elif problem == "infinite":
        record["arms"]["hold_reverse"]["loss_before"] = float("inf")
    else:
        record["arms"]["hold_reverse"]["observed"] = 75
    with pytest.raises(ValueError):
        validate_feedback(record, mask)


def test_no_observation_requires_explicit_abstention():
    mask = np.zeros((2, 23), dtype=bool)
    record = feedback(observed=0)
    validate_feedback(record, mask)
    record["arms"]["true_true"]["loss_before"] = 0.0
    with pytest.raises(ValueError):
        validate_feedback(record, mask)


def test_bootstrap_uses_same_fixed_paired_draws_for_all_three_contrasts():
    q = np.array([[1.0, -4.0, 2.0], [3.0, 2.0, -1.0], [-2.0, 1.0, 5.0]])
    hold = np.array([[2.0, 3.0, 4.0], [1.0, -3.0, 0.0], [4.0, 2.0, -2.0]])
    result = bootstrap_summary(q, hold)
    assert result == bootstrap_summary(q, hold)
    indices = np.random.default_rng(131).integers(0, 3, size=(2000, 3))
    q_draws = q[indices].mean(axis=1)
    hold_draws = hold[indices].mean(axis=1)
    for i, history in enumerate(HISTORIES):
        assert result["q"][history]["estimate"] == pytest.approx(q[:, i].mean())
        assert result["q"][history]["ci95"] == pytest.approx(
            np.percentile(q_draws[:, i], [2.5, 97.5], method="linear")
        )
        assert result["hold_target"][history]["ci95"] == pytest.approx(
            np.percentile(hold_draws[:, i], [2.5, 97.5], method="linear")
        )
    contrasts = {
        "D_true": q_draws[:, 0],
        "abs_D_true_minus_abs_D_reverse": np.abs(q_draws[:, 0]) - np.abs(q_draws[:, 1]),
        "abs_D_true_minus_abs_D_hold": np.abs(q_draws[:, 0]) - np.abs(q_draws[:, 2]),
    }
    for key, values in contrasts.items():
        assert result["screen_contrasts"][key]["ci95"] == pytest.approx(
            np.percentile(values, [2.5, 97.5], method="linear")
        )
    assert result["screen_status"] == "insufficient_evidence"


def test_screen_enforces_minimum_and_all_three_conditions():
    hold = np.zeros((20, 3))
    supported = np.tile([3.0, 1.0, -0.5], (20, 1))
    assert bootstrap_summary(supported, hold)["screen_status"] == "supported"
    assert bootstrap_summary(supported[:19], hold[:19])["screen_status"] == "insufficient_evidence"
    tied_control = np.tile([3.0, -3.0, -0.5], (20, 1))
    assert bootstrap_summary(tied_control, hold)["screen_status"] == "stop_scalar_route"
    crossing_zero = np.tile([0.0, 0.0, 0.0], (20, 1))
    assert bootstrap_summary(crossing_zero, hold)["screen_status"] == "stop_scalar_route"
