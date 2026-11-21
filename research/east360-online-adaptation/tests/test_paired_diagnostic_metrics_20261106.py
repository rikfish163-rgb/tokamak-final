"""CPU-only contract tests for the paired, post-reveal diagnostic metrics."""

import json

import numpy as np
import pytest

from paired_diagnostic_metrics_20261106 import (
    ARMS,
    SEEDS,
    aggregate,
    shot_metrics,
    source_gate,
)


def make_row(shot, times, logits, active, disruptive, down, horizon=125, lead=30):
    row = shot_metrics(times, logits, active, disruptive, down, horizon, lead, 0.5)
    row["shot"] = shot
    return row


def constant_logits(n, value=0.0):
    return np.full((n, len(SEEDS), len(ARMS)), value, dtype=float)


def view(row, seed=131, arm="static"):
    return row["seeds"][str(seed)][arm]


def test_contract_and_identical_arms_have_zero_paired_differences():
    assert ARMS == ("static", "real", "reverse", "unsigned", "zero_target", "score")
    assert SEEDS == (131, 317, 563)
    logits = constant_logits(3, -1.0)
    logits[1:] = 0.0
    row = make_row(7, [0.8, 0.9, 0.99], logits, [False, True, True], True, 1.0)
    for seed in SEEDS:
        for arm in ARMS:
            assert view(row, seed, arm) == view(row, seed, "static")
            assert view(row, seed, arm)["first_index"] == 1
            assert view(row, seed, arm)["first_lead_ms"] == 100

    result = aggregate([row], 0.5, 30)
    for contrast in result["paired_bootstrap"].values():
        for metric in ("positive", "raw_tp", "reasonable_tp"):
            assert contrast[metric]["mean_difference"] == 0.0
            assert contrast[metric]["ci_95"] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("lead", "raw", "reasonable"),
    [
        (30, False, False),
        (30.00001, True, True),
        (3000, True, True),
        (3000.00001, True, False),
        (0, False, False),
    ],
)
def test_strict_first_alarm_timing_and_reasonable_boundaries(lead, raw, reasonable):
    row = make_row(1, [10 - lead / 1000], constant_logits(1), [False], True, 10)
    observed = view(row)
    assert observed["first_index"] == 0
    assert observed["first_lead_ms"] == pytest.approx(lead, abs=1e-9)
    assert observed["raw_tp"] is raw
    assert observed["reasonable_tp"] is reasonable
    # Inactive points still participate in the whole-shot alarm.
    assert observed["strata"]["positive"]["n_points"] == 0


def test_post_quench_exclusion_and_tolerance():
    logits = constant_logits(3, -1)
    logits[1:] = 1
    times = [0, 0.9999e-9, 2e-9]
    row = make_row(1, times, logits, [True, True, True], True, 0)
    assert view(row)["first_index"] == 1
    assert view(row)["first_lead_ms"] == -0.000001
    assert view(row)["raw_tp"] is False
    assert view(row)["strata"]["outside"]["n_points"] == 2
    assert view(row)["strata"]["late"]["n_points"] == 2

    logits[1] = -1
    assert view(make_row(2, times, logits, [True] * 3, True, 0))["first_index"] is None


def test_post_quench_tolerance_is_applied_before_lead_rounding():
    # Both displayed leads round to -1e-6 ms, but only the first is eligible.
    logits = constant_logits(2, -1)
    logits[1] = 1
    row = make_row(1, [0.9999e-9, 1.0004e-9], logits, [True, True], True, 0)
    assert view(row)["first_index"] is None
    assert view(row)["strata"]["late"]["n_points"] == 1


def test_alarm_threshold_is_converted_from_probability_to_logit():
    logits = constant_logits(2, 1)
    logits[1] = 1.4
    row = shot_metrics([0, 1], logits, [True, True], False, None, 125, 30, 0.8)
    assert view(row)["first_index"] == 1
    assert view(row)["fp"] is True


def test_supported_positive_early_late_outside_masks_and_bce():
    # Leads are 400, 125, 80, 30 and 0 ms. The 80-ms point is inactive.
    times = [0.6, 0.875, 0.92, 0.97, 1.0]
    logits = constant_logits(5)
    logits[:, :, ARMS.index("real")] = 2.0
    row = make_row(1, times, logits, [True, True, False, True, True], True, 1.0)
    strata = view(row, arm="real")["strata"]
    assert {name: data["n_points"] for name, data in strata.items()} == {
        "positive": 1,
        "early": 1,
        "late": 2,
        "outside": 3,
        "safe_high": 0,
    }
    assert strata["positive"]["bce"] == pytest.approx(np.logaddexp(0, -2))
    assert strata["early"]["bce"] == pytest.approx(np.logaddexp(0, 2))
    assert strata["outside"]["bce"] == pytest.approx(np.logaddexp(0, 2))
    assert strata["safe_high"]["bce"] is None


def test_positive_horizon_tolerance_is_applied_after_lead_rounding():
    # The first lead exceeds the tolerated horizon by 1e-9 ms; the second is
    # exactly at its inclusive upper boundary, and the third is at the horizon.
    leads = [125.000001001, 125.000001, 125.0]
    times = [1 - value / 1000 for value in leads]
    logits = constant_logits(3)
    logits[:, :, ARMS.index("real")] = np.array([2.0, 1.0, 0.0])[:, None]
    row = make_row(1, times, logits, [True] * 3, True, 1, horizon=125, lead=10)
    strata = view(row, arm="real")["strata"]
    assert strata["positive"]["n_points"] == 2
    assert strata["positive"]["bce"] == pytest.approx(
        (np.logaddexp(0, -1) + np.log(2)) / 2
    )
    assert strata["early"]["n_points"] == 1
    assert strata["early"]["bce"] == pytest.approx(np.logaddexp(0, 2))
    assert strata["outside"]["n_points"] == 1


def test_safe_high_uses_seed_specific_static_scores_shared_by_all_arms():
    logits = constant_logits(4, -5)
    logits[0, :, 0] = 0.0  # High static probability, but inactive.
    logits[1, :, 0] = np.log(0.4 / 0.6)
    logits[2, 0, 0] = np.log(0.4 / 0.6)
    logits[:, :, ARMS.index("real")] = 3.0
    row = make_row(1, [0, 1, 2, 3], logits, [False, True, True, True], False, None)
    assert view(row, 131, "real")["strata"]["safe_high"]["n_points"] == 2
    assert view(row, 317, "real")["strata"]["safe_high"]["n_points"] == 1
    assert view(row, 131, "real")["strata"]["safe_high"]["bce"] == pytest.approx(
        np.logaddexp(0, 3)
    )
    assert view(row, 131, "real")["fp"] is True
    assert view(row, 131, "real")["first_index"] == 0
    assert view(row, 131, "real")["first_lead_ms"] is None


def test_safe_high_includes_the_threshold_minus_point_15_boundary():
    boundary = 0.5 - 0.15
    logits = constant_logits(2)
    logits[0, :, 0] = np.log(boundary / (1 - boundary))
    below = boundary - 1e-8
    logits[1, :, 0] = np.log(below / (1 - below))
    row = make_row(1, [0, 1], logits, [True, True], False, None)
    assert view(row)["strata"]["safe_high"]["n_points"] == 1


def test_stratum_pooling_gives_equal_weight_to_shots_then_available_seeds():
    first = constant_logits(1)
    first[:, :, 1] = 2
    second = constant_logits(3)
    a = make_row(1, [0.9], first, [True], True, 1)
    b = make_row(2, [0.89, 0.9, 0.91], second, [True] * 3, True, 1)

    safe_a = constant_logits(1, -1)
    safe_a[:, :, 0] = 0  # Three supported seed/shot observations.
    safe_b = constant_logits(1, -1)
    safe_b[:, 0, 0] = 0  # One supported seed/shot observation.
    safe_b[:, :, 1] = 2
    c = make_row(3, [0], safe_a, [True], False, None)
    d = make_row(4, [0], safe_b, [True], False, None)

    result = aggregate([a, b, c, d], 0.5, 30)
    positive = result["pooled_strata"]["positive"]["real"]
    assert positive["n_shots"] == 2
    assert positive["n_seed_shot_observations"] == 6
    assert positive["n_points"] == 12
    assert positive["bce"] == pytest.approx((np.logaddexp(0, -2) + np.log(2)) / 2)
    high = result["pooled_strata"]["safe_high"]["real"]
    assert (high["n_shots"], high["n_seed_shot_observations"]) == (2, 4)
    assert high["bce"] == pytest.approx((np.logaddexp(0, -1) + np.logaddexp(0, 2)) / 2)


def test_per_seed_counts_and_exact_paired_gains_losses():
    def one(shot, lead, static, real, disruptive=True):
        logits = constant_logits(1, -1)
        logits[:, :, 0] = static
        logits[:, :, 1] = real
        return make_row(
            shot,
            [10 - lead / 1000],
            logits,
            [True],
            disruptive,
            10 if disruptive else None,
        )

    rows = [
        one(1, 100, 0, -1),
        one(2, 100, -1, 0),
        one(3, 3500, 0, 0),
        one(4, 0, -1, 0, False),
        one(5, 0, 0, -1, False),
    ]
    real = aggregate(rows, 0.5, 30)["per_seed"]["131"]["real"]
    assert (real["n_disruptive"], real["n_safe"]) == (3, 2)
    assert (real["raw_tp"], real["reasonable_tp"], real["early_tp"], real["fp"]) == (
        2,
        1,
        1,
        1,
    )
    assert real["raw_tpr"] == pytest.approx(2 / 3)
    assert real["reasonable_tpr"] == pytest.approx(1 / 3)
    assert real["fpr"] == 0.5
    assert real["vs_static"]["raw_tp"] == {
        "gained_count": 1,
        "gained_shots": [2],
        "lost_count": 1,
        "lost_shots": [1],
    }
    assert real["vs_static"]["reasonable_tp"]["gained_shots"] == [2]
    assert real["vs_static"]["reasonable_tp"]["lost_shots"] == [1]
    assert real["vs_static"]["fp"]["gained_shots"] == [4]
    assert real["vs_static"]["fp"]["lost_shots"] == [5]


def test_bootstrap_is_seed_fixed_shot_paired_and_input_order_invariant():
    rows = []
    for shot, value in enumerate([-2.0, 1.0, 2.0], 1):
        logits = constant_logits(1)
        logits[:, :, 1] = value
        rows.append(make_row(shot, [0.9], logits, [True], True, 1))
    first = aggregate(rows, 0.5, 30)["paired_bootstrap"]
    second = aggregate(list(reversed(rows)), 0.5, 30)["paired_bootstrap"]
    assert first == second
    comparison = first["real_minus_static"]["positive"]
    expected = np.mean([np.logaddexp(0, -x) - np.log(2) for x in [-2, 1, 2]])
    assert comparison["n_shots"] == 3
    assert comparison["mean_difference"] == pytest.approx(expected)
    assert comparison["ci_95"][0] <= expected <= comparison["ci_95"][1]
    assert comparison["resamples"] == 2000
    assert first["real_minus_reverse"]["positive"] == comparison


def gate_rows(n_positive=20, n_safe=20, real_positive=0.1, real_safe=-0.2, lead=30):
    rows = []
    for shot in range(n_positive):
        logits = constant_logits(1)
        logits[:, :, 1] = real_positive
        rows.append(make_row(shot, [0.9], logits, [True], True, 1, lead=lead))
    for shot in range(n_safe):
        logits = constant_logits(1, -0.2)
        logits[:, :, 1] = real_safe
        rows.append(make_row(100 + shot, [0], logits, [True], False, None, lead=lead))
    return rows


def test_gate_requires_all_controls_and_twenty_distinct_supported_shots():
    result = aggregate(gate_rows(), 0.5, 30)
    verdict = source_gate(result)
    assert verdict["passed"] is True
    assert all(verdict["criteria"].values())
    assert source_gate(aggregate(gate_rows(n_positive=19), 0.5, 30))["passed"] is False
    assert source_gate(aggregate(gate_rows(n_safe=19), 0.5, 30))["passed"] is False
    assert (
        source_gate(aggregate(gate_rows(real_positive=1e-7), 0.5, 30))["passed"]
        is False
    )
    assert source_gate(aggregate(gate_rows(real_safe=0.2), 0.5, 30))["passed"] is False


@pytest.mark.parametrize(
    "control", ["static", "reverse", "unsigned", "zero_target", "score"]
)
def test_gate_positive_requires_beating_every_control(control):
    rows = gate_rows()
    for shot in range(20):
        logits = constant_logits(1)
        logits[:, :, 1] = 0.1
        logits[:, :, ARMS.index(control)] = 0.2
        rows[shot] = make_row(shot, [0.9], logits, [True], True, 1)
    verdict = source_gate(aggregate(rows, 0.5, 30))
    assert not verdict["criteria"][f"positive_real_better_than_{control}_by_over_1e-6"]
    assert verdict["passed"] is False


def test_gate_positive_margin_is_strictly_greater_than_one_micro_bce():
    static_bce = np.log(2.0)
    for difference, expected in [(0.999e-6, False), (1.001e-6, True)]:
        real_logit = -np.log(np.expm1(static_bce - difference))
        verdict = source_gate(aggregate(gate_rows(real_positive=real_logit), 0.5, 30))
        assert (
            verdict["criteria"]["positive_real_better_than_static_by_over_1e-6"]
            is expected
        )


def test_gate_uses_mean_raw_counts_across_seeds_not_best_seed():
    rows = gate_rows()
    logits = constant_logits(1)
    logits[:, :, 1] = 0.1
    logits[:, 2, 1] = -0.1
    rows[0] = make_row(0, [0.9], logits, [True], True, 1)
    result = aggregate(rows, 0.5, 30)
    assert result["per_seed"]["131"]["real"]["reasonable_tp"] == 20
    assert result["per_seed"]["563"]["real"]["reasonable_tp"] == 19
    verdict = source_gate(result)
    assert verdict["criteria"]["mean_raw_tp_at_least_static"] is False
    assert verdict["passed"] is False


def test_gate_compares_raw_tp_including_alarms_earlier_than_3000_ms():
    rows = gate_rows(lead=10)
    logits = constant_logits(2, -1)
    logits[0, :, 0] = 0.1
    logits[1, :, 0] = 0.0
    logits[1, :, 1] = -0.1
    # Static first crosses at 3500 ms; real never crosses, despite having
    # positive-window BCE that is pooled with the other supported shots.
    rows[0] = make_row(0, [0.5, 3.9], logits, [True, True], True, 4, lead=10)
    result = aggregate(rows, 0.5, 10)
    assert result["mean_counts"]["static"]["raw_tp"] == 20
    assert result["mean_counts"]["real"]["raw_tp"] == 19
    assert result["mean_counts"]["static"]["reasonable_tp"] == 19
    assert result["mean_counts"]["real"]["reasonable_tp"] == 19
    verdict = source_gate(result)
    assert verdict["criteria"]["mean_raw_tp_at_least_static"] is False
    assert all(
        passed
        for criterion, passed in verdict["criteria"].items()
        if criterion != "mean_raw_tp_at_least_static"
    )
    assert verdict["passed"] is False


@pytest.mark.parametrize(("lead", "expected"), [(10.0, False), (10.00001, True)])
def test_raw_tp_uses_configured_strict_ten_ms_boundary(lead, expected):
    row = make_row(1, [1 - lead / 1000], constant_logits(1), [True], True, 1, lead=10)
    assert view(row)["raw_tp"] is expected


def test_gate_false_positive_allowance_uses_mean_across_all_seeds():
    for added_in_one_seed, allowed in [(3, True), (4, False)]:
        rows = gate_rows()
        for index in range(added_in_one_seed):
            logits = constant_logits(1, -0.2)
            logits[:, 2, 1] = 0.01
            rows[20 + index] = make_row(100 + index, [0], logits, [True], False, None)
        result = aggregate(rows, 0.5, 30)
        verdict = source_gate(result)
        assert verdict["criteria"]["mean_fp_at_most_static_plus_1"] is allowed
        assert verdict["passed"] is allowed


def test_gate_false_positive_allowance_compares_exact_seed_count_totals():
    rows = gate_rows()
    # Static FP by seed: (0, 0, 2); real FP: (0, 1, 4). Their mean
    # difference is exactly one, despite binary floating-point rounding.
    for index in range(4):
        logits = constant_logits(1, -0.2)
        if index < 2:
            logits[:, 2, 0] = 0
        logits[:, 2, 1] = 0
        if index == 0:
            logits[:, 1, 1] = 0
        rows[20 + index] = make_row(100 + index, [0], logits, [True], False, None)
    result = aggregate(rows, 0.5, 30)
    assert result["mean_counts"]["static"]["fp"] == pytest.approx(2 / 3)
    assert result["mean_counts"]["real"]["fp"] == pytest.approx(5 / 3)
    verdict = source_gate(result)
    assert verdict["criteria"]["mean_fp_at_most_static_plus_1"] is True
    assert verdict["passed"] is True


def test_gate_equal_mean_raw_counts_do_not_fail_from_float_rounding():
    rows = gate_rows()
    # Static timely by seed: (0, 0, 6); real: (0, 1, 5).
    for index in range(20):
        logits = constant_logits(1, -0.2)
        logits[:, :, 1] = -0.1
        if index < 6:
            logits[:, 2, 0] = 0.1
        if index < 5:
            logits[:, 2, 1] = 0.2
        if index == 0:
            logits[:, 1, 1] = 0.1
        rows[index] = make_row(index, [0.9], logits, [True], True, 1)
    result = aggregate(rows, 0.5, 30)
    assert result["mean_counts"]["static"]["reasonable_tp"] == pytest.approx(2)
    assert result["mean_counts"]["real"]["reasonable_tp"] == pytest.approx(2)
    verdict = source_gate(result)
    assert verdict["criteria"]["mean_raw_tp_at_least_static"] is True
    assert verdict["passed"] is True


def test_absent_support_is_null_and_cannot_pass():
    row = make_row(1, [0.9], constant_logits(1), [False], True, 1)
    result = aggregate([row], 0.5, 30)
    assert result["pooled_strata"]["positive"]["real"]["bce"] is None
    assert result["paired_bootstrap"]["real_minus_static"]["positive"]["ci_95"] is None
    assert source_gate(result)["passed"] is False
    empty = aggregate([], 0.5, 30)
    assert empty["per_seed"]["131"]["static"]["raw_tpr"] is None
    assert source_gate(empty)["passed"] is False


def test_strict_json_encoding_with_extreme_logits_and_empty_strata():
    logits = constant_logits(2, -1000)
    logits[0, :, 1] = 1000
    row = make_row(1, [0.9, 1.1], logits, [True, True], True, 1)
    result = aggregate([row], 0.5, 30)
    for value in (row, result, source_gate(result)):
        json.dumps(value, allow_nan=False)


def test_extreme_finite_logits_and_mixed_bootstrap_differences_stay_json_safe():
    rows = []
    largest = np.finfo(np.float64).max
    for shot, sign in enumerate((1, -1)):
        logits = constant_logits(1, -sign * largest)
        logits[:, :, 1] = sign * largest
        rows.append(make_row(shot, [0.9], logits, [True], True, 1))
    result = aggregate(rows, 0.5, 30)
    assert (
        result["paired_bootstrap"]["real_minus_static"]["positive"]["mean_difference"]
        == 0
    )
    json.dumps(result, allow_nan=False)


def test_json_round_trip_and_per_shot_horizons_are_supported():
    a = make_row(1, [0.9], constant_logits(1), [True], True, 1, horizon=125)
    b = make_row(2, [0.9], constant_logits(1), [True], True, 1, horizon=200)
    rows = json.loads(json.dumps([a, b], allow_nan=False))
    result = aggregate(rows, 0.5, 30)
    assert result["horizons_ms"] == [125, 200]
    assert result["pooled_strata"]["positive"]["real"]["n_shots"] == 2
    json.dumps(source_gate(result), allow_nan=False)


def test_score_does_not_mutate_inputs_and_low_threshold_includes_safe_points():
    times = np.array([0.0, 1.0])
    logits = constant_logits(2, -1000)
    active = np.array([True, False])
    original = (times.copy(), logits.copy(), active.copy())
    row = shot_metrics(times, logits, active, False, np.nan, 125, 30, 0.1)
    assert row["down"] is None
    assert view(row)["strata"]["safe_high"]["n_points"] == 1
    for observed, expected in zip((times, logits, active), original, strict=True):
        np.testing.assert_array_equal(observed, expected)
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    ("times", "logits", "active", "disruptive", "down", "horizon", "lead", "threshold"),
    [
        ([], np.zeros((0, 3, 6)), [], True, 1, 125, 30, 0.5),
        ([0], np.zeros((1, 3, 5)), [True], True, 1, 125, 30, 0.5),
        ([0], constant_logits(1), [1], True, 1, 125, 30, 0.5),
        ([1, 0], constant_logits(2), [True, True], True, 1, 125, 30, 0.5),
        ([0], constant_logits(1, np.nan), [True], True, 1, 125, 30, 0.5),
        ([np.nan], constant_logits(1), [True], True, 1, 125, 30, 0.5),
        ([0], constant_logits(1), [True], True, np.nan, 125, 30, 0.5),
        ([0], constant_logits(1), [True], True, 1, 30, 30, 0.5),
        ([0], constant_logits(1), [True], True, 1, 125, -1, 0.5),
        ([0], constant_logits(1), [True], True, 1, 125, 30, 0),
        ([0], constant_logits(1), [True], True, 1, 125, 30, 1),
    ],
)
def test_invalid_inputs_are_rejected(
    times, logits, active, disruptive, down, horizon, lead, threshold
):
    with pytest.raises(ValueError):
        shot_metrics(times, logits, active, disruptive, down, horizon, lead, threshold)


def test_aggregate_rejects_duplicate_shots_and_config_mismatch():
    row = make_row(1, [0.9], constant_logits(1), [True], True, 1)
    with pytest.raises(ValueError, match="shot"):
        aggregate([row, row], 0.5, 30)
    with pytest.raises(ValueError, match="threshold"):
        aggregate([row], 0.6, 30)
    with pytest.raises(ValueError, match="lead"):
        aggregate([row], 0.5, 20)
    with pytest.raises(ValueError, match="shot"):
        aggregate(
            [{key: value for key, value in row.items() if key != "shot"}], 0.5, 30
        )
