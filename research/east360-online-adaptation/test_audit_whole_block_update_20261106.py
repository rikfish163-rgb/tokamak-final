"""Synthetic-only contract tests for the independent post-seal evaluator."""

import json
import math

import numpy as np
import pytest

import audit_whole_block_update_20261106 as audit
import paired_diagnostic_metrics_20261106 as paired
from verify_confirmation_raw_replay_20261105 import score as native_score


def cube(n, value=-2.0):
    return np.full((n, 3, 6), value, dtype=float)


def make_row(shot, z, *, disruptive=True, down=1.0, times=None, active=None,
             lead=10.0, horizon=125.0, threshold=0.5):
    n = len(z)
    if times is None:
        times = np.linspace(0.9, 0.99, n) if n > 1 else [0.9]
    if active is None:
        active = [True] * n
    row = audit.per_shot(times, z, active, disruptive, down, horizon, threshold, lead)
    row["shot"] = shot
    return row


def entry(row, arm="block", seed=131):
    return row["seeds"][str(seed)][arm]


def simple_outcome(shot, disruptive, states, *, lead=10.0):
    """states maps an arm to three independent synthetic crossing booleans."""
    z = cube(1)
    for arm, hits in states.items():
        z[0, :, audit.ARMS.index(arm)] = np.where(hits, 1.0, -1.0)
    return make_row(shot, z, disruptive=disruptive,
                    down=1.0 if disruptive else None, lead=lead)


def passing_rows():
    all_seeds, none = [True] * 3, [False] * 3
    states = {arm: all_seeds for arm in audit.ARMS}
    rows = [simple_outcome("base", True, states)]
    states = {arm: none for arm in audit.ARMS}
    states["block"] = [True, True, False]
    rows.append(simple_outcome("gain", True, states))
    rows.append(simple_outcome("safe", False, {arm: none for arm in audit.ARMS}))
    return rows


def test_public_order_and_argument_order():
    assert audit.SEEDS == (131, 317, 563)
    assert audit.ARMS == ("static", "old", "point", "block", "reverse", "score")
    row = audit.per_shot([0.9], cube(1), [True], True, 1, 125, 0.5, 10)
    assert (row["threshold"], row["lead_ms"]) == (0.5, 10)
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("cutoff", [10.0, 30.0])
@pytest.mark.parametrize("lead", [0.0, 10.0, 10.00001, 30.0, 30.00001,
                                    3000.0, 3000.00001])
def test_strict_raw_and_explicit_reasonable_boundary(cutoff, lead):
    row = make_row(1, cube(1, 0), times=[10 - lead / 1000], down=10, lead=cutoff)
    value = entry(row)
    assert value["first_lead_ms"] == pytest.approx(lead, abs=1e-9)
    assert value["raw_tp"] is (lead > cutoff)
    assert value["reasonable_tp"] is (lead > max(30, cutoff) and lead <= 3000)
    assert value["native_reasonable_tp"] is (lead > cutoff and lead <= 3000)


@pytest.mark.parametrize("cutoff", [10.0, 30.0])
def test_matches_existing_scorers_on_synthetic_trajectories(cutoff):
    rng = np.random.default_rng(761)
    times = np.array([0.0, 0.6, 0.875, 0.95, 0.97, 0.99, 1.0, 1.0000000005,
                      1.000000002])
    z = rng.normal(size=(len(times), 3, 6))
    active = np.array([False, True, True, False, True, True, True, True, True])
    for disruptive in [False, True]:
        down = 1.0 if disruptive else None
        actual = make_row(1, z, times=times, active=active, disruptive=disruptive,
                          down=down, lead=cutoff, threshold=0.73)
        expected = paired.shot_metrics(times, z, active, disruptive, down, 125,
                                       cutoff, 0.73)
        for i, seed in enumerate(audit.SEEDS):
            for j, arm in enumerate(audit.ARMS):
                observed = entry(actual, arm, seed)
                reference = expected["seeds"][str(seed)][paired.ARMS[j]]
                for key in ("first_index", "first_lead_ms", "raw_tp", "fp", "strata"):
                    assert observed[key] == reference[key]
                assert observed["native_reasonable_tp"] == reference["reasonable_tp"]
                if cutoff == 30:
                    probabilities = 1 / (1 + np.exp(-z[:, i, j]))
                    reference_native = native_score(
                        times, probabilities,
                        {"shot": 1, "disruptive": disruptive, "down": down}, 0.73,
                    )
                    assert observed["first_index"] == reference_native["first_index"]
                    assert observed["first_lead_ms"] == reference_native["lead_ms"]
                    assert observed["raw_tp"] == (reference_native["status"] == "tp")
                    assert observed["fp"] == (reference_native["status"] == "fp")


def test_logit_threshold_matches_paired_but_can_differ_from_float32_probability():
    threshold = 0.2
    logit_threshold = math.log(threshold) - math.log1p(-threshold)
    z = np.full((1, 3, 6), logit_threshold, dtype=np.float32)
    row = make_row(1, z, times=[0.9], threshold=threshold, lead=30)
    prior = paired.shot_metrics([0.9], z, [True], True, 1, 125, 30, threshold)
    assert entry(row)["first_index"] == prior["seeds"]["131"]["real"]["first_index"]
    assert entry(row)["first_index"] is None
    # Native is given rounded float32 probabilities, not the original logits.
    probabilities = 1 / (1 + np.exp(-z[:, 0, 3]))
    native = native_score(np.array([0.9]), probabilities,
                          {"shot": 1, "disruptive": True, "down": 1.0}, threshold)
    assert native["first_index"] == 0


def test_full_trajectory_first_hit_not_active_or_later_rescue():
    z = cube(3)
    z[0, :, :] = 0
    z[1:, :, :] = 2
    row = make_row(1, z, times=[0.96, 0.97, 1.0], active=[False, True, True], lead=30)
    assert entry(row)["first_index"] == 0
    assert entry(row)["raw_tp"] is True
    assert entry(row)["strata"]["positive"]["n_points"] == 0
    safe = make_row(2, z, times=[0, 1, 2], active=[False, True, True],
                    disruptive=False, down=None)
    assert entry(safe)["first_index"] == 0
    assert entry(safe)["fp"] is True


def test_very_early_first_alarm_is_not_replaced_by_later_reasonable_crossing():
    row = make_row(1, cube(2, 1), times=[6.5, 9.9], down=10, lead=30)
    assert entry(row)["first_lead_ms"] == 3500
    assert entry(row)["raw_tp"] is True
    assert entry(row)["reasonable_tp"] is False
    summary = audit.summarize([row])
    assert summary["per_seed"]["131"]["block"]["very_early_tp"] == 1


def test_post_outcome_tolerance_is_applied_before_rounding():
    z = cube(2)
    z[1] = 1
    row = make_row(1, z, times=[0.9999e-9, 1.0004e-9], down=0)
    assert entry(row)["first_index"] is None
    assert entry(row)["strata"]["late"]["n_points"] == 1
    z[0] = 1
    row = make_row(1, z, times=[0.9999e-9, 1.0004e-9], down=0)
    assert entry(row)["first_index"] == 0
    assert entry(row)["first_lead_ms"] == -0.000001


def test_positive_window_horizon_tolerance_and_stable_bce():
    leads = [400, 125.000001001, 125.000001, 125, 80, 30, 10, 0]
    times = [1 - value / 1000 for value in leads]
    z = cube(len(times), 0)
    z[:, :, 3] = 2
    row = make_row(1, z, times=times,
                   active=[True, True, True, True, False, True, True, True], lead=10)
    strata = entry(row)["strata"]
    assert {k: v["n_points"] for k, v in strata.items()} == {
        "positive": 3, "early": 2, "late": 2, "outside": 4, "safe_high": 0,
    }
    assert strata["positive"]["bce"] == pytest.approx(np.logaddexp(0, -2))
    z[:] = 1e308
    huge = make_row(2, z, times=times, lead=10)
    assert math.isfinite(entry(huge)["strata"]["early"]["bce"])
    assert entry(huge)["strata"]["positive"]["bce"] == 0


def test_all_safe_peaks_and_high_slice_use_static_mask_and_all_points():
    z = cube(3, -5)
    z[:, :, 0] = np.array([math.log(0.35 / 0.65), -2, -2])[:, None]
    z[:, :, 3] = np.array([4, -3, 2])[:, None]
    row = make_row("safe", z, times=[0, 1, 2], active=[False, True, True],
                   disruptive=False, down=None)
    value = entry(row)
    assert value["safe_peak"]["index"] == 0
    assert value["safe_peak"]["time"] == 0
    assert value["safe_peak"]["probability"] == pytest.approx(1 / (1 + math.exp(-4)))
    assert value["strata"]["safe_high"]["n_points"] == 0
    result = audit.summarize([row])
    stats = result["per_seed"]["131"]["block"]["safe_peak_probability"]
    assert stats["count"] == 1
    assert stats["max"] == pytest.approx(value["safe_peak"]["probability"])
    assert result["paired"]["block_minus_static"]["bootstrap"]["safe_peak_probability"]["n_shots"] == 1


def test_paired_losses_gains_new_and_removed_false_positives_vs_every_control():
    yes = [True] * 3
    rows = [
        simple_outcome("gain", True, {"block": yes}),
        simple_outcome("loss", True, {"static": yes, "old": yes, "point": yes,
                                       "reverse": yes, "score": yes}),
        simple_outcome("new", False, {"block": yes}),
        simple_outcome("removed", False, {"static": yes, "old": yes, "point": yes,
                                           "reverse": yes, "score": yes}),
    ]
    summary = audit.summarize(rows)
    assert set(summary["paired"]) == {f"block_minus_{a}" for a in audit.ARMS if a != "block"}
    for contrast in summary["paired"].values():
        p = contrast["per_seed"]["131"]
        assert p["timely_gains"] == ["gain"]
        assert p["timely_losses"] == ["loss"]
        assert p["new_fp"] == ["new"]
        assert p["removed_fp"] == ["removed"]
        assert (p["timely_delta"], p["fp_delta"], p["net_delta"]) == (0, 0, 0)
    assert summary["per_seed"]["131"]["block"]["raw_tp"] == 1
    assert summary["per_seed"]["131"]["block"]["fp"] == 1


def test_seed_averaged_shot_paired_bootstrap_and_equal_shot_bce_pooling():
    rows = []
    for shot, values in [(3, [1, -1, -1]), (1, [1, 1, -1]), (2, [-1, -1, -1])]:
        z = cube(1, 0)
        z[0, :, 3] = values
        rows.append(make_row(shot, z))
    first = audit.summarize(rows)
    assert first == audit.summarize(list(reversed(rows)))
    boot = first["paired"]["block_minus_static"]["bootstrap"]["raw_tp"]
    assert boot["n_shots"] == 3
    # Static crosses in all nine seed/shot observations; block crosses in three.
    assert boot["mean_difference"] == pytest.approx(-6 / 9)
    assert boot["ci_95"][0] <= boot["mean_difference"] <= boot["ci_95"][1]
    # The same resampled shot carries all three seed-specific differences.
    shot_differences = np.array([-1 / 3, -1, -2 / 3])  # sorted shot IDs 1, 2, 3
    rng = np.random.default_rng(audit.BOOTSTRAP_SEED)
    indices = rng.integers(0, 3, size=(audit.BOOTSTRAP_RESAMPLES, 3))
    expected_ci = np.quantile(shot_differences[indices].mean(axis=1), [0.025, 0.975])
    assert boot["ci_95"] == pytest.approx(expected_ci)
    positive = first["pooled_strata"]["positive"]["block"]
    expected = np.mean([np.mean(np.logaddexp(0, -np.array(v))) for v in
                        ([1, -1, -1], [1, 1, -1], [-1, -1, -1])])
    assert positive["bce"] == pytest.approx(expected)
    assert positive["n_shots"] == 3
    assert positive["n_seed_shot_observations"] == 9
    assert first["paired"]["block_minus_static"]["bootstrap"]["fp"]["ci_95"] is None


def test_pooled_positive_bce_is_equal_shot_weighted_not_point_weighted():
    a = cube(1, 0)
    a[:, :, 3] = 2
    b = cube(4, 0)
    b[:, :, 3] = -2
    first = make_row(1, a, times=[0.9])
    second = make_row(2, b, times=[0.90, 0.91, 0.92, 0.93])
    summary = audit.summarize([first, second])
    positive = summary["pooled_strata"]["positive"]["block"]
    assert positive["n_points"] == 15  # seed/point exposures, not unique points
    assert positive["bce"] == pytest.approx(
        (np.logaddexp(0, -2) + np.logaddexp(0, 2)) / 2
    )


def test_source_gate_exact_conditions_and_objective():
    summary = audit.summarize(passing_rows())
    gate = summary["source_gate"]
    assert gate["passed"] is True
    assert gate["objective"] == "sum_over_seeds(raw_tp - fp)"
    assert gate["objective_by_arm"]["block"] == 5
    assert gate["objective_by_arm"]["static"] == 3
    assert gate["n_seeds_improved"] == 2
    assert all(gate["criteria"].values())


def test_source_gate_uses_uncapped_first_alarm_not_reasonable_count():
    rows = passing_rows()
    z = cube(1)
    z[0, :2, audit.ARMS.index("block")] = 1
    rows[1] = make_row("gain", z, times=[-2.5], down=1, lead=10)
    summary = audit.summarize(rows)
    assert sum(summary["per_seed"][str(seed)]["block"]["raw_tp"] for seed in audit.SEEDS) == 5
    assert sum(summary["per_seed"][str(seed)]["block"]["reasonable_tp"] for seed in audit.SEEDS) == 3
    assert summary["source_gate"]["passed"] is True


@pytest.mark.parametrize("arm", ["point", "reverse", "score"])
def test_gate_rejects_equal_control_net_even_with_more_timely(arm):
    rows = passing_rows()
    rows[1] = simple_outcome("gain", True, {
        "block": [True, True, False], arm: [True, True, False],
    })
    gate = audit.summarize(rows)["source_gate"]
    assert gate["objective_by_arm"]["block"] == gate["objective_by_arm"][arm]
    assert gate["criteria"][f"block_net_strictly_above_{arm}"] is False
    assert gate["passed"] is False


def test_gate_uses_net_not_raw_count_against_controls():
    rows = passing_rows()
    # The control ties block in timely counts, but also incurs one false alarm.
    rows[1] = simple_outcome("gain", True, {
        "block": [True, True, False], "point": [True, True, False],
    })
    rows[2] = simple_outcome("safe", False, {"point": [True, False, False]})
    gate = audit.summarize(rows)["source_gate"]
    assert gate["objective_by_arm"]["block"] == 5
    assert gate["objective_by_arm"]["point"] == 4
    assert gate["criteria"]["block_net_strictly_above_point"] is True


def test_gate_detects_seed_level_loss_and_new_fp_even_when_pooled_gain():
    rows = passing_rows()
    base = {arm: [True, True, True] for arm in audit.ARMS}
    base["block"] = [True, True, False]
    rows[0] = simple_outcome("base", True, base)
    rows[2] = simple_outcome("safe", False, {"block": [True, False, False]})
    gate = audit.summarize(rows)["source_gate"]
    assert not gate["per_seed"]["563"]["raw_tp_at_least_static"]
    assert not gate["per_seed"]["131"]["fp_at_most_static"]
    assert gate["passed"] is False


def test_gate_rejects_new_fp_in_one_seed_even_if_removed_in_another():
    rows = passing_rows()
    rows[2] = simple_outcome("safe", False, {
        "static": [False, True, False],
        "block": [True, False, False],
    })
    summary = audit.summarize(rows)
    assert sum(summary["per_seed"][str(seed)]["block"]["fp"] for seed in audit.SEEDS) == 1
    assert sum(summary["per_seed"][str(seed)]["static"]["fp"] for seed in audit.SEEDS) == 1
    assert not summary["source_gate"]["criteria"]["all_seeds_fp_at_most_static"]
    assert summary["paired"]["block_minus_static"]["per_seed"]["131"]["new_fp"] == ["safe"]
    assert summary["paired"]["block_minus_static"]["per_seed"]["317"]["removed_fp"] == ["safe"]


def test_gate_requires_two_improving_seeds_and_both_outcome_groups():
    rows = passing_rows()
    rows[1] = simple_outcome("gain", True, {"block": [True, False, False]})
    gate = audit.summarize(rows)["source_gate"]
    assert gate["n_seeds_improved"] == 1
    assert not gate["criteria"]["at_least_two_seeds_improve_raw_tp"]
    assert not audit.summarize(rows[:2])["source_gate"]["passed"]
    assert not audit.summarize([rows[2]])["source_gate"]["passed"]
    assert not audit.summarize([])["source_gate"]["passed"]


def test_prediction_only_auxiliary_diagnostics_respect_common_mask():
    maturity = [63, 95, 127]
    common = [True, False, True]
    norms = np.arange(27, dtype=float).reshape(3, 3, 3)
    cos = np.array([[0.2, -0.1, 0.4], [999, 999, 999], [0.6, 0.3, -0.2]])
    offsets = np.array([[1., 2., 3.], [999, 999, 999], [-1., 4., -3.]])
    diag = audit.prediction_diagnostics(maturity, common, norms, cos, offsets)
    assert (diag["n_maturity"], diag["n_common"]) == (3, 2)
    assert diag["per_seed"]["131"]["mean_norm"] == {
        "old": 9.0, "point": 10.0, "block": 11.0,
    }
    assert diag["per_seed"]["131"]["mean_point_block_cos"] == pytest.approx(0.4)
    assert diag["per_seed"]["131"]["mean_score_offset"] == 0
    assert diag["per_seed"]["131"]["mean_abs_score_offset"] == 1
    empty = audit.prediction_diagnostics([], [], np.zeros((0, 3, 3)),
                                         np.zeros((0, 3)), np.zeros((0, 3)))
    assert empty["per_seed"]["131"]["mean_point_block_cos"] is None
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("which", ["shape", "nonfinite", "mask", "order"])
def test_prediction_diagnostics_rejects_invalid_alignment(which):
    m, c = np.array([63, 95]), np.array([True, False])
    n, q, o = np.ones((2, 3, 3)), np.zeros((2, 3)), np.zeros((2, 3))
    if which == "shape":
        o = o[:, :2]
    elif which == "nonfinite":
        q[1, 0] = np.nan
    elif which == "mask":
        c = np.array([1, 0])
    else:
        m = m[::-1]
    with pytest.raises(ValueError):
        audit.prediction_diagnostics(m, c, n, q, o)


@pytest.mark.parametrize("bad", ["empty", "times", "logits", "active", "outcome",
                                  "down", "horizon", "threshold", "lead"])
def test_per_shot_rejects_malformed_inputs(bad):
    kwargs = dict(times=[0.9, 1.0], logits=cube(2), active=[True, True],
                  disruptive=True, down=1.0, horizon_ms=125, threshold=0.5, lead_ms=10)
    changes = {"empty": {"times": [], "logits": cube(0), "active": []},
               "times": {"times": [1, 1]}, "logits": {"logits": cube(1)},
               "active": {"active": [1, 0]}, "outcome": {"disruptive": 1},
               "down": {"down": float("nan")}, "horizon": {"horizon_ms": 10},
               "threshold": {"threshold": 1}, "lead": {"lead_ms": -1}}
    kwargs.update(changes[bad])
    with pytest.raises(ValueError):
        audit.per_shot(**kwargs)


def test_summarize_rejects_duplicate_missing_or_mixed_configuration():
    row = make_row(1, cube(1))
    with pytest.raises(ValueError, match="duplicate"):
        audit.summarize([row, row])
    with pytest.raises(ValueError, match="shot"):
        audit.summarize([{k: v for k, v in row.items() if k != "shot"}])
    with pytest.raises(ValueError, match="threshold"):
        audit.summarize([row, make_row(2, cube(1), threshold=0.8)])
    with pytest.raises(ValueError, match="lead"):
        audit.summarize([row, make_row(2, cube(1), lead=30)])
    # Horizons legitimately differ between shots under the J-TEXT contract.
    result = audit.summarize([row, make_row(2, cube(1), horizon=250)])
    assert result["horizons_ms"] == [125, 250]
    json.dumps(result, allow_nan=False)
