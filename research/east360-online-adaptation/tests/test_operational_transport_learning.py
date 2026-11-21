import math

import pytest
import torch

from experiments.risk_aligned_online.operational_transport_20261030.learning import (
    bounded_step,
    fit_rule,
    predict_rule,
)


def _unit_rows(values):
    values = values.to(torch.float64)
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / torch.where(norms == 0, torch.ones_like(norms), norms)


def _example():
    signal = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [-2.0, 0.0, 2.0],
         [0.0, 0.0, 0.0], [1.0, -1.0, 1.0]],
        dtype=torch.float64,
    )
    task = torch.tensor(
        [[0.0, 2.0, 1.0], [2.0, 0.0, -1.0], [1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.2, 0.4, 0.0, 0.6, 0.8], dtype=torch.float64)
    return signal, task, weights


def test_weighted_dual_matches_primal_ridge_with_unpenalized_intercept():
    signal, task, weights = _example()
    ridge = 0.37
    rule = fit_rule(signal, task, weights, ridge=ridge)
    x, y = _unit_rows(signal), _unit_rows(task)
    w = weights / weights.sum()
    mx = (w[:, None] * x).sum(dim=0)
    my = (w[:, None] * y).sum(dim=0)
    xc, yc = x - mx, y - my
    primal = torch.linalg.solve(
        xc.T @ (w[:, None] * xc) + ridge * torch.eye(x.shape[1], dtype=x.dtype),
        xc.T @ (w[:, None] * yc),
    )
    intercept = my - mx @ primal
    query = torch.tensor([[1.0, -2.0, 3.0], [-2.0, 4.0, 1.0]], dtype=torch.float64)
    expected = _unit_rows(query) @ primal + intercept

    assert set(rule) == {"keys", "coefficients", "center", "mean", "ridge"}
    assert rule["keys"].shape == signal.shape
    assert rule["coefficients"].shape == task.shape
    assert rule["ridge"] == ridge
    assert all(rule[key].dtype == torch.float64 for key in ("keys", "coefficients", "center", "mean"))
    torch.testing.assert_close(rule["center"], mx, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(rule["mean"], my, rtol=1e-13, atol=1e-13)
    expected_keys = torch.sqrt(w)[:, None] * xc
    torch.testing.assert_close(rule["keys"], expected_keys, rtol=1e-13, atol=1e-13)
    dual_system = expected_keys @ expected_keys.T + ridge * torch.eye(signal.shape[0], dtype=x.dtype)
    torch.testing.assert_close(
        dual_system @ rule["coefficients"], torch.sqrt(w)[:, None] * yc, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        rule["mean"] - rule["center"] @ rule["keys"].T @ rule["coefficients"],
        intercept,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.linalg.vector_norm(intercept) > 0.1
    torch.testing.assert_close(predict_rule(query, rule), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 3.5, 1e308])
def test_permutation_and_weight_rescaling_leave_predictions_unchanged(scale):
    signal, task, weights = _example()
    query = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 2.0, 0.5]], dtype=torch.float64)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    reference = predict_rule(query, fit_rule(signal, task, weights))
    transformed = fit_rule(signal[permutation], task[permutation], weights[permutation] * scale)
    torch.testing.assert_close(predict_rule(query, transformed), reference, rtol=1e-12, atol=1e-12)


def test_zero_query_is_exact_zero_even_when_intercept_is_nonzero():
    signal = torch.eye(2, dtype=torch.float64)
    task = torch.tensor([[1.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    rule = fit_rule(signal, task, torch.ones(2))
    query = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    result = predict_rule(query, rule)
    assert result.dtype == query.dtype and result.device == query.device
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    torch.testing.assert_close(result[1], torch.tensor([1.0, 0.0]))
    assert torch.equal(predict_rule(query[0], rule), torch.zeros_like(query[0]))


def test_zero_task_is_a_valid_target_and_single_sample_retains_intercept():
    rule = fit_rule(torch.tensor([[1.0, 2.0]]), torch.zeros((1, 2)), torch.tensor([2]))
    assert torch.equal(predict_rule(torch.tensor([2.0, 1.0]), rule), torch.zeros(2))
    rule = fit_rule(torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 2.0]]), torch.tensor([2]))
    torch.testing.assert_close(predict_rule(torch.tensor([2.0, 1.0]), rule), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_predict_preserves_dtype_for_batched_queries(dtype):
    signal, task, weights = _example()
    rule = fit_rule(signal, task, weights)
    query = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                          [[2.0, 4.0, 6.0], [-1.0, 2.0, 3.0]]], dtype=dtype)
    result = predict_rule(query, rule)
    assert result.shape == query.shape
    assert result.dtype == dtype and result.device == query.device
    assert torch.equal(result[0, 1], torch.zeros_like(result[0, 1]))
    assert torch.equal(result[0, 0], result[1, 0])


def test_helpers_do_not_mutate_inputs_or_rule():
    signal, task, weights = _example()
    inputs = (signal, task, weights)
    originals = tuple(value.clone() for value in inputs)
    rule = fit_rule(*inputs)
    query = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    original_query = query.clone()
    original_buffers = {key: value.clone() for key, value in rule.items() if isinstance(value, torch.Tensor)}
    mapped = predict_rule(query, rule)
    original_mapped = mapped.clone()
    bounded_step(mapped)
    assert all(torch.equal(value, original) for value, original in zip(inputs, originals))
    assert torch.equal(query, original_query)
    assert torch.equal(mapped, original_mapped)
    assert all(torch.equal(rule[key], original) for key, original in original_buffers.items())


def test_predictions_have_sample_dependent_noncollinear_directions():
    signal = torch.eye(2, dtype=torch.float64)
    rule = fit_rule(signal, signal, torch.ones(2))
    mapped = predict_rule(signal, rule)
    assert torch.linalg.det(mapped).abs() > 0.5
    assert mapped[0, 0] > mapped[0, 1]
    assert mapped[1, 1] > mapped[1, 0]


def test_high_dimensional_small_sample_cpu_fixture():
    generator = torch.Generator().manual_seed(1234)
    signal = torch.randn((5, 8192), generator=generator)
    task = torch.randn((5, 8192), generator=generator)
    query = torch.randn((3, 8192), generator=generator)
    rule = fit_rule(signal, task, torch.tensor([1.0, 2.0, 3.0, 0.0, 1.0]))
    result = predict_rule(query, rule)
    assert rule["keys"].shape == (5, 8192)
    assert rule["coefficients"].shape == (5, 8192)
    assert result.shape == query.shape
    assert result.dtype == query.dtype and result.device == query.device
    assert torch.isfinite(result).all()


def test_normalization_handles_extreme_finite_magnitudes():
    signal = torch.tensor([[1e308, 1e308], [1e-320, 0.0]], dtype=torch.float64)
    task = torch.tensor([[1e308, -1e308], [0.0, 1e-320]], dtype=torch.float64)
    unit_signal = torch.tensor([[1 / math.sqrt(2), 1 / math.sqrt(2)], [1.0, 0.0]], dtype=torch.float64)
    unit_task = torch.tensor([[1 / math.sqrt(2), -1 / math.sqrt(2)], [0.0, 1.0]], dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    expected = fit_rule(unit_signal, unit_task, weights)
    actual = fit_rule(signal, task, weights)
    torch.testing.assert_close(predict_rule(signal, actual), predict_rule(unit_signal, expected))


@pytest.mark.parametrize(
    ("mapped", "radius", "expected"),
    [
        ([0.0, 0.0], 0.05, [0.0, 0.0]),
        ([0.3, 0.4], 0.05, [-0.015, -0.02]),
        ([3.0, 4.0], 0.05, [-0.03, -0.04]),
        ([3.0, 4.0], 0.0, [0.0, 0.0]),
    ],
)
def test_bounded_step_exact_formula_and_norm_bound(mapped, radius, expected):
    mapped = torch.tensor(mapped, dtype=torch.float64)
    step = bounded_step(mapped, radius=radius)
    torch.testing.assert_close(step, torch.tensor(expected, dtype=mapped.dtype), rtol=1e-14, atol=1e-14)
    assert step.dtype == mapped.dtype and step.device == mapped.device
    assert torch.linalg.vector_norm(step) <= radius + 1e-15


def test_bounded_step_handles_extreme_finite_values_and_flattens_norm():
    maximum = torch.finfo(torch.float64).max
    mapped = torch.tensor([[maximum, -maximum], [0.0, 0.0]], dtype=torch.float64)
    step = bounded_step(mapped)
    expected = torch.tensor([[-0.05 / math.sqrt(2), 0.05 / math.sqrt(2)], [0.0, 0.0]], dtype=torch.float64)
    assert torch.isfinite(step).all()
    torch.testing.assert_close(step, expected, rtol=1e-14, atol=0.0)
    assert torch.linalg.vector_norm(step) <= 0.05 + 1e-15


def test_bounded_step_rejects_a_result_unrepresentable_in_input_dtype():
    with pytest.raises(ValueError):
        bounded_step(torch.tensor([1.0, 0.0], dtype=torch.float16), radius=1e100)


@pytest.mark.parametrize(
    ("signal", "task", "weights", "ridge"),
    [
        (None, torch.ones((2, 2)), torch.ones(2), 0.1),
        (torch.ones(2), torch.ones((2, 2)), torch.ones(2), 0.1),
        (torch.empty((0, 2)), torch.empty((0, 2)), torch.empty(0), 0.1),
        (torch.empty((2, 0)), torch.empty((2, 0)), torch.ones(2), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 3)), torch.ones(2), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones((2, 1)), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones(3), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.tensor([1.0, -1.0]), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.zeros(2), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.tensor([1.0, float("nan")]), 0.1),
        (torch.tensor([[float("inf"), 1.0], [1.0, 1.0]]), torch.ones((2, 2)), torch.ones(2), 0.1),
        (torch.ones((2, 2)), torch.full((2, 2), float("nan")), torch.ones(2), 0.1),
        (torch.ones((2, 2), dtype=torch.complex64), torch.ones((2, 2)), torch.ones(2), 0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones(2), 0.0),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones(2), -0.1),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones(2), float("inf")),
        (torch.ones((2, 2)), torch.ones((2, 2)), torch.ones(2), float("nan")),
    ],
)
def test_fit_rejects_invalid_arguments(signal, task, weights, ridge):
    with pytest.raises((TypeError, ValueError)):
        fit_rule(signal, task, weights, ridge)


@pytest.mark.parametrize(
    "query",
    [None, torch.tensor(1.0), torch.ones(3), torch.ones(2, dtype=torch.int64),
     torch.ones(2, dtype=torch.complex64), torch.tensor([float("nan"), 1.0])],
)
def test_predict_rejects_invalid_queries(query):
    rule = fit_rule(torch.eye(2), torch.eye(2), torch.ones(2))
    with pytest.raises((TypeError, ValueError)):
        predict_rule(query, rule)


@pytest.mark.parametrize("defect", ["not_dict", "missing", "keys_shape", "coefficient_shape", "nonfinite"])
def test_predict_rejects_invalid_rules(defect):
    rule = fit_rule(torch.eye(2), torch.eye(2), torch.ones(2))
    if defect == "not_dict":
        rule = None
    elif defect == "missing":
        del rule["mean"]
    elif defect == "keys_shape":
        rule["keys"] = torch.ones(2)
    elif defect == "coefficient_shape":
        rule["coefficients"] = torch.ones((1, 2))
    elif defect == "nonfinite":
        rule["center"] = torch.tensor([float("inf"), 0.0])
    with pytest.raises((TypeError, ValueError)):
        predict_rule(torch.ones(2), rule)


@pytest.mark.parametrize("radius", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_bounded_step_rejects_invalid_radius(radius):
    with pytest.raises((TypeError, ValueError)):
        bounded_step(torch.ones(2), radius)


@pytest.mark.parametrize(
    "mapped",
    [None, torch.empty(0), torch.ones(2, dtype=torch.int64),
     torch.ones(2, dtype=torch.complex64), torch.tensor([float("inf"), 0.0])],
)
def test_bounded_step_rejects_invalid_mapped_values(mapped):
    with pytest.raises((TypeError, ValueError)):
        bounded_step(mapped)
