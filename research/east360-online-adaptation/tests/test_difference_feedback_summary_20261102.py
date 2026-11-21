"""CPU-only regression tests for the difference-feedback screening summary."""

from copy import deepcopy
import math
import sys

import pytest

from experiments.risk_aligned_online.difference_feedback_probe_20261102.summary import (
    summarize,
)


ARMS = (
    "centered_real",
    "difference_real",
    "difference_reverse",
    "difference_hold",
)


def make_row(
    shot,
    query,
    disruptive,
    label,
    *,
    real=-4e-5,
    centered=0.0,
    reverse=0.0,
    hold=0.0,
):
    values = (centered, real, reverse, hold)
    return {
        "shot": shot,
        "query": query,
        "disruptive": disruptive,
        "label": label,
        "results": {
            arm: {"delta_bce": value, "unused": "must not affect summary"}
            for arm, value in zip(ARMS, values)
        },
    }


def covered_rows(*, pos=-4e-5, neg=0.0, centered=0.0, reverse=0.0):
    return [
        make_row(shot, 0, True, 1, real=pos, centered=centered, reverse=reverse)
        for shot in range(100, 106)
    ] + [
        make_row(shot, 0, False, 0, real=neg, centered=centered, reverse=reverse)
        for shot in range(200, 208)
    ]


def test_groups_use_shot_not_point_weights_then_balance_labels():
    rows = [
        make_row(1, 1, True, 1, real=-0.2, centered=-0.4),
        make_row(1, 2, True, 1, real=-0.4, centered=-0.8),
        make_row(1, 3, True, 1, real=-0.6, centered=-1.2),
        make_row(2, 1, True, 1, real=0.2, centered=0.4),
        make_row(1, 4, True, 0, real=0.6, centered=1.2),
        make_row(10, 1, False, 0, real=0.1, centered=0.2),
        make_row(10, 2, False, 0, real=0.1, centered=0.2),
        make_row(10, 3, False, 0, real=0.1, centered=0.2),
        make_row(11, 1, False, 0, real=-0.2, centered=-0.4),
    ]
    original = deepcopy(rows)

    result = summarize(rows)
    real = result["arms"]["difference_real"]
    assert real["class1"] == {
        "n_points": 4,
        "n_shots": 2,
        "mean_delta_bce": pytest.approx(-0.1),
    }
    assert real["class0"] == {
        "n_points": 5,
        "n_shots": 3,
        "mean_delta_bce": pytest.approx(1 / 6),
    }
    assert real["safe"] == {
        "n_points": 4,
        "n_shots": 2,
        "mean_delta_bce": pytest.approx(-0.05),
    }
    assert real["balanced_delta_bce"] == pytest.approx((-0.1 + 1 / 6) / 2)
    centered = result["arms"]["centered_real"]
    assert centered["class1"]["mean_delta_bce"] == pytest.approx(-0.2)
    assert centered["safe"]["mean_delta_bce"] == pytest.approx(-0.1)
    assert centered["balanced_delta_bce"] == pytest.approx((-0.2 + 1 / 3) / 2)
    assert rows == original


def test_fully_covered_synthetic_case_passes_all_gates():
    result = summarize(covered_rows())
    assert result["coverage"] == {
        "positive_disruptive_n_shots": 6,
        "negative_safe_n_shots": 8,
        "positive_disruptive": True,
        "negative_safe": True,
        "passed": True,
    }
    assert all(result["gates"].values())
    assert result["decision"] == "pass"
    for arm in ARMS:
        assert result["arms"][arm]["class0"]["n_shots"] == 8
        assert result["arms"][arm]["class1"]["n_shots"] == 6
        assert result["arms"][arm]["safe"]["n_shots"] == 8


def test_safe_guard_is_independent_from_all_negative_label_shots():
    rows = covered_rows(pos=-1e-3, neg=2e-6)
    rows.extend(make_row(shot, 1, True, 0, real=-1e-3) for shot in range(100, 106))
    result = summarize(rows)
    assert result["arms"]["difference_real"]["class0"]["mean_delta_bce"] < 1e-6
    assert result["arms"]["difference_real"]["safe"]["mean_delta_bce"] == pytest.approx(
        2e-6
    )
    assert result["gates"]["class0"] is True
    assert result["gates"]["safe"] is False
    assert result["gates"]["passed"] is False
    assert result["decision"] == "screen_failed"


def test_class0_guard_is_independent_from_safe_guard():
    rows = covered_rows(pos=-1e-3, neg=0.0)
    rows.extend(make_row(shot, 1, True, 0, real=1e-3) for shot in range(100, 106))
    result = summarize(rows)
    assert result["gates"]["safe"] is True
    assert result["gates"]["class0"] is False
    assert result["decision"] == "screen_failed"


@pytest.mark.parametrize("n_disruptive,n_safe", [(5, 8), (6, 7), (5, 7)])
def test_coverage_counts_distinct_eligible_shots_not_points(n_disruptive, n_safe):
    rows = [
        make_row(100 + shot, query, True, 1)
        for shot in range(n_disruptive)
        for query in range(4)
    ] + [
        make_row(200 + shot, query, False, 0, real=0.0)
        for shot in range(n_safe)
        for query in range(3)
    ]
    result = summarize(rows)
    assert result["coverage"]["positive_disruptive_n_shots"] == n_disruptive
    assert result["coverage"]["negative_safe_n_shots"] == n_safe
    assert result["gates"]["coverage"] is False
    assert result["gates"]["passed"] is False
    assert result["decision"] == "coverage_insufficient"


def test_coverage_counts_only_disruptive_positive_and_safe_negative():
    rows = [make_row(100 + shot, 0, True, 0) for shot in range(8)]
    rows += [make_row(200 + shot, 0, False, 1) for shot in range(8)]
    result = summarize(rows)
    assert result["coverage"]["positive_disruptive_n_shots"] == 0
    assert result["coverage"]["negative_safe_n_shots"] == 0
    assert result["arms"]["difference_real"]["safe"] == {
        "n_points": 0,
        "n_shots": 0,
        "mean_delta_bce": None,
    }
    assert result["decision"] == "coverage_insufficient"


@pytest.mark.parametrize(
    "positive,passes",
    [
        (-1e-5, False),
        (math.nextafter(-1e-5, -math.inf), True),
        (math.nextafter(-1e-5, math.inf), False),
    ],
)
def test_positive_gate_is_strict(positive, passes):
    assert summarize(covered_rows(pos=positive))["gates"]["class1"] is passes


@pytest.mark.parametrize(
    "negative,passes",
    [
        (1e-6, True),
        (math.nextafter(1e-6, -math.inf), True),
        (math.nextafter(1e-6, math.inf), False),
    ],
)
def test_negative_and_safe_gates_are_inclusive(negative, passes):
    gates = summarize(covered_rows(neg=negative))["gates"]
    assert gates["class0"] is passes
    assert gates["safe"] is passes


@pytest.mark.parametrize(
    "positive,passes",
    [(-2e-5, False), (math.nextafter(-2e-5, -math.inf), True)],
)
def test_balanced_gate_is_strict(positive, passes):
    assert summarize(covered_rows(pos=positive))["gates"]["balanced"] is passes


@pytest.mark.parametrize(
    "arm,gate", [("centered", "vs_centered_real"), ("reverse", "vs_difference_reverse")]
)
def test_comparator_minimum_difference_is_inclusive(arm, gate):
    real = -2e-5
    boundary = real + 1e-6
    kwargs = {arm: boundary}
    assert summarize(covered_rows(pos=real, neg=real, **kwargs))["gates"][gate] is True
    kwargs[arm] = math.nextafter(boundary, -math.inf)
    assert summarize(covered_rows(pos=real, neg=real, **kwargs))["gates"][gate] is False


def test_hold_is_checked_per_point_not_just_its_average():
    rows = covered_rows()
    rows[0]["results"]["difference_hold"]["delta_bce"] = 2e-10
    rows[1]["results"]["difference_hold"]["delta_bce"] = -2e-10
    result = summarize(rows)
    assert result["arms"]["difference_hold"]["class1"]["mean_delta_bce"] == 0.0
    assert result["gates"]["hold_zero"] is False
    assert result["gates"]["passed"] is False
    assert result["decision"] == "implementation_check_failed"


@pytest.mark.parametrize(
    "hold,passes",
    [(1e-10, True), (-1e-10, True), (math.nextafter(1e-10, math.inf), False)],
)
def test_hold_tolerance_is_inclusive(hold, passes):
    rows = covered_rows()
    rows[0]["results"]["difference_hold"]["delta_bce"] = hold
    assert summarize(rows)["gates"]["hold_zero"] is passes


def test_missing_groups_and_no_points_are_explicitly_not_passing():
    result = summarize([])
    empty = {"n_points": 0, "n_shots": 0, "mean_delta_bce": None}
    for arm in ARMS:
        assert result["arms"][arm]["class0"] == empty
        assert result["arms"][arm]["class1"] == empty
        assert result["arms"][arm]["safe"] == empty
        assert result["arms"][arm]["balanced_delta_bce"] is None
    assert not any(result["gates"].values())
    assert result["decision"] == "coverage_insufficient"

    positive_only = summarize([make_row(1, 1, True, 1)])
    assert positive_only["arms"]["difference_real"]["class0"] == empty
    for gate in (
        "class0",
        "safe",
        "balanced",
        "vs_centered_real",
        "vs_difference_reverse",
        "passed",
    ):
        assert positive_only["gates"][gate] is False


def test_coverage_failure_does_not_hide_hold_implementation_failure():
    rows = [make_row(1, 1, True, 1, hold=2e-10)]
    result = summarize(rows)
    assert result["decision"] == "coverage_insufficient"
    assert result["gates"]["coverage"] is False
    assert result["gates"]["hold_zero"] is False
    assert result["gates"]["passed"] is False


def test_rejects_duplicate_shot_query_even_when_rows_are_identical():
    row = make_row(1, 2, True, 1)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, deepcopy(row)])


@pytest.mark.parametrize("arm", ARMS)
def test_rejects_each_missing_arm(arm):
    row = make_row(1, 2, True, 1)
    del row["results"][arm]
    with pytest.raises(ValueError, match=arm):
        summarize([row])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_values_from_any_arm(arm, value):
    row = make_row(1, 2, True, 1)
    row["results"][arm]["delta_bce"] = value
    with pytest.raises(ValueError, match=arm):
        summarize([row])


def test_rejects_missing_delta_and_inconsistent_shot_status():
    row = make_row(1, 2, True, 1)
    del row["results"]["difference_real"]["delta_bce"]
    with pytest.raises(ValueError, match="difference_real"):
        summarize([row])
    with pytest.raises(ValueError, match="disruptive"):
        summarize([make_row(1, 2, True, 1), make_row(1, 3, False, 0)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("shot", True),
        ("shot", "1"),
        ("query", 2.0),
        ("disruptive", 1),
        ("label", True),
        ("label", 2),
    ],
)
def test_rejects_invalid_identity_and_group_fields(field, value):
    row = make_row(1, 2, True, 1)
    row[field] = value
    with pytest.raises(ValueError):
        summarize([row])


def test_rejects_non_numeric_delta_or_missing_row_fields():
    row = make_row(1, 2, True, 1)
    row["results"]["difference_real"]["delta_bce"] = "0.0"
    with pytest.raises(ValueError, match="difference_real"):
        summarize([row])
    del row["query"]
    with pytest.raises(ValueError, match="query"):
        summarize([row])


def test_record_order_and_single_pass_iterable_do_not_change_results():
    rows = covered_rows()
    rows += [
        make_row(100, 1, True, 1, real=-0.2),
        make_row(100, 2, True, 0, real=0.3),
        make_row(200, 1, False, 0, real=0.1),
    ]
    assert summarize(iter(rows)) == summarize(iter(reversed(rows)))


def test_finite_extreme_values_do_not_overflow_group_means():
    largest = sys.float_info.max
    result = summarize(covered_rows(pos=largest, neg=largest))
    real = result["arms"]["difference_real"]
    for group in ("class0", "class1", "safe"):
        assert real[group]["mean_delta_bce"] == largest
    assert real["balanced_delta_bce"] == largest
    assert result["gates"]["passed"] is False


def test_subnormal_finite_values_survive_shot_and_label_averaging():
    smallest = math.ulp(0.0)
    rows = covered_rows(pos=smallest, neg=smallest)
    rows += [
        make_row(100, 1, True, 1, real=smallest),
        make_row(200, 1, False, 0, real=smallest),
    ]
    real = summarize(rows)["arms"]["difference_real"]
    for group in ("class0", "class1", "safe"):
        assert real[group]["mean_delta_bce"] == smallest
    assert real["balanced_delta_bce"] == smallest
