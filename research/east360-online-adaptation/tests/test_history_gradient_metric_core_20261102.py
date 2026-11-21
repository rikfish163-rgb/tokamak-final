import numpy as np
import pytest

from experiments.risk_aligned_online.history_gradient_metric_20261102.core import (
    coefficients,
    evaluate,
    fit,
    group_summary,
    row_weights,
    solve,
)


def row(shot, query, label, signal, task, *, disruptive=True):
    return dict(
        shot=shot,
        query=query,
        label=label,
        disruptive=disruptive,
        task_gradient256=task,
        results={"difference_real": {"receipt": {"gradient256": signal}}},
    )


def test_label_then_shot_weights_not_number_of_points():
    rows = [
        row(1, 1, 0, [1, 0], [1, 0]),
        row(1, 2, 0, [1, 0], [1, 0]),
        row(2, 1, 0, [1, 0], [1, 0]),
        row(3, 1, 1, [1, 0], [1, 0]),
    ]
    assert np.allclose(row_weights(rows), [0.125, 0.125, 0.25, 0.5])


def test_coefficient_formula_and_solver_improve_the_declared_objective():
    s = np.asarray([[1.0, 0], [0, 1.0]])
    r = np.asarray([[0.0, 1], [0, 1.0]])
    a, b = coefficients(s, r, np.array([0.5, 0.5]))
    assert np.allclose(a, [1.0, 1.0])
    assert np.allclose(b, [0.5, 1.0])
    p, mu = solve(a, b)
    assert np.allclose(p, [0.75, 1.25])
    assert mu == pytest.approx(-0.25)
    assert np.sum(a * p * p - 2 * b * p) <= np.sum(a - 2 * b)


def test_fit_normalizes_gradient_pairs_and_reports_training_identity_only():
    rows = [
        row(10, 1, 0, [2, 0], [0, 5], disruptive=False),
        row(20, 1, 1, [0, 3], [0, 2]),
    ]
    result = fit(rows)
    assert result["training_ids"] == [10, 20]
    assert result["n_defined"] == result["n_total"] == 2
    assert np.allclose(result["weights"], [0.75, 1.25])
    assert result["objective_fit"] <= result["objective_identity"]


def test_zero_gradients_are_reported_not_given_a_fabricated_direction():
    rows = [
        row(10, 1, 0, [1, 0], [0, 1], disruptive=False),
        row(20, 1, 1, [0, 1], [0, 1]),
        row(30, 1, 0, [0, 0], [1, 0]),
    ]
    result = fit(rows)
    assert result["n_total"] == 3 and result["n_defined"] == 2
    assert result["undefined_pairs"] == [{"shot": 30, "query": 1}]
    test = evaluate(rows[-1], np.ones(2))
    assert test["before_cosine"] is None and test["after_cosine"] is None


def test_absent_label_group_is_insufficient_not_an_identity_solution():
    with pytest.raises(ValueError):
        fit([row(10, 1, 0, [1, 0], [1, 0])])


def test_group_summary_keeps_safe_separate_and_does_not_hide_undefined():
    rows = [
        dict(
            shot=1,
            query=1,
            label=0,
            disruptive=False,
            before_cosine=0.1,
            after_cosine=0.2,
        ),
        dict(
            shot=1,
            query=2,
            label=0,
            disruptive=False,
            before_cosine=0.1,
            after_cosine=0.2,
        ),
        dict(
            shot=2,
            query=1,
            label=0,
            disruptive=True,
            before_cosine=0.3,
            after_cosine=0.5,
        ),
        dict(
            shot=3,
            query=1,
            label=1,
            disruptive=True,
            before_cosine=0.1,
            after_cosine=0.3,
        ),
    ]
    result = group_summary(rows)
    assert result["groups"]["class0"]["before_mean_cosine"] == pytest.approx(0.2)
    assert result["groups"]["safe"]["after_mean_cosine"] == pytest.approx(0.2)
    assert result["gates"]["passed"]
    rows[0]["after_cosine"] = None
    changed = group_summary(rows)
    assert changed["groups"]["safe"]["n_total"] == 2
    assert changed["groups"]["safe"]["after_defined"] == 1
    assert not changed["gates"]["coverage"] and not changed["gates"]["passed"]
