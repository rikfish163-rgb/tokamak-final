import math

import pytest
import torch

from risk_meta_adam_20261108 import adam_step


@pytest.fixture
def direction_step():
    from risk_meta_adam_20261108 import direction_step

    return direction_step


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_multistep_matches_adam_state_and_length(direction_step, dtype):
    weight = torch.tensor([[0.25, -0.5, 0.125], [0.75, 0.0, -0.25]], dtype=dtype)
    gradients = (
        torch.tensor([[0.3, -0.6, 0.2], [0.1, 0.4, -0.2]], dtype=dtype),
        torch.tensor([[3.0, -1.5, 2.0], [-0.5, 0.25, 1.0]], dtype=dtype),
        torch.zeros((2, 3), dtype=dtype),
        torch.tensor([[-0.2, 0.1, -0.3], [0.5, -0.1, 0.2]], dtype=dtype),
    )
    history = ()
    for gradient in gradients:
        weight_before, gradient_before = weight.clone(), gradient.clone()
        history_before = tuple(item.clone() for item in history)
        expected = adam_step(weight, gradient, history)
        actual = direction_step(weight, gradient, history)

        assert set(actual) == set(expected) | {"adam_weight", "adam_step_norm", "momentum_norm"}
        assert len(actual["history"]) == len(expected["history"])
        for index, (got, ref) in enumerate(zip(actual["history"], expected["history"])):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            if index < len(history):
                assert got is history[index]
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["adam_weight"], expected["weight"], rtol=0, atol=0)
        torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
        torch.testing.assert_close(gradient, gradient_before, rtol=0, atol=0)
        for item, before in zip(history, history_before):
            torch.testing.assert_close(item, before, rtol=0, atol=0)

        old_length = torch.linalg.vector_norm(expected["weight"] - weight)
        momentum_length = torch.linalg.vector_norm(expected["exp_avg"])
        torch.testing.assert_close(actual["adam_step_norm"], old_length, rtol=0, atol=1e-7)
        torch.testing.assert_close(actual["momentum_norm"], momentum_length, rtol=0, atol=1e-7)
        expected_weight = weight - old_length * expected["exp_avg"] / momentum_length
        torch.testing.assert_close(actual["weight"], expected_weight, rtol=0, atol=1e-7)
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual["weight"] - weight), old_length, rtol=0, atol=1e-7,
        )
        weight, history = actual["weight"], actual["history"]


def test_direction_is_negative_momentum_not_adam_coordinate_direction(direction_step):
    weight = torch.zeros(4, dtype=torch.float64)
    gradient = torch.tensor([0.7, -0.2, 0.05, -0.4], dtype=torch.float64)
    actual = direction_step(weight, gradient, lr=0.003)
    direction = actual["exp_avg"] / actual["momentum_norm"]
    displacement = actual["weight"] - weight
    expected = -actual["adam_step_norm"] * direction
    torch.testing.assert_close(displacement, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        torch.dot(displacement, -direction) / torch.linalg.vector_norm(displacement),
        torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-12,
    )
    assert not torch.allclose(displacement, actual["adam_weight"] - weight, rtol=1e-2, atol=1e-8)


def test_8192_scalar_step_uses_raw_gradient_and_keeps_length(direction_step):
    weight = torch.zeros((128, 64), dtype=torch.float32)
    gradient = torch.linspace(-0.8, 1.2, weight.numel(), dtype=weight.dtype).reshape_as(weight)
    before = gradient.clone()
    adam = adam_step(weight, gradient)
    actual = direction_step(weight, gradient)
    torch.testing.assert_close(gradient, before, rtol=0, atol=0)
    torch.testing.assert_close(actual["history"][0], adam["history"][0], rtol=0, atol=0)
    assert torch.linalg.vector_norm(gradient) > 1
    torch.testing.assert_close(
        torch.linalg.vector_norm(actual["weight"] - weight),
        torch.linalg.vector_norm(adam["weight"] - weight), rtol=0, atol=1e-7,
    )


@pytest.mark.parametrize("dtype,scale", ((torch.float32, 1e-25), (torch.float64, 1e-200)))
def test_tiny_nonzero_momentum_is_not_smoothed_to_zero(direction_step, dtype, scale):
    weight = torch.zeros(3, dtype=dtype)
    gradient = torch.tensor([1.0, -2.0, 3.0], dtype=dtype) * scale
    result = direction_step(weight, gradient)
    old_delta = result["adam_weight"] - weight
    expected_length = old_delta.abs().max() * (math.sqrt(14) / 3)
    direction = torch.tensor([1.0, -2.0, 3.0], dtype=dtype) / math.sqrt(14)
    assert result["momentum_norm"].item() > 0
    assert result["adam_step_norm"].item() > 0
    torch.testing.assert_close(result["adam_step_norm"], expected_length, rtol=2e-6, atol=0)
    torch.testing.assert_close(result["weight"], -expected_length * direction, rtol=2e-6, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_is_exactly_no_update_and_has_finite_first_derivative(direction_step, dtype):
    weight = torch.tensor([[0.25, -0.5], [0.75, -0.25]], dtype=dtype, requires_grad=True)
    gradients = tuple(torch.zeros_like(weight, requires_grad=True) for _ in range(2))
    history = ()
    original = weight.clone()
    for gradient in gradients:
        result = direction_step(weight, gradient, history)
        assert torch.equal(result["weight"], original)
        assert torch.equal(result["adam_weight"], original)
        assert result["adam_step_norm"].item() == 0
        assert result["momentum_norm"].item() == 0
        weight, history = result["weight"], result["history"]

    derivatives = torch.autograd.grad(weight.sum(), gradients)
    for derivative, multiplier in zip(derivatives, (1 + 0.9 / 1.9, 1 / 1.9)):
        assert torch.isfinite(derivative).all()
        expected = torch.full_like(derivative, -1e-4 / 1e-8 * multiplier)
        torch.testing.assert_close(derivative, expected, rtol=2e-6, atol=1e-3)


def test_zero_is_not_generally_twice_differentiable(direction_step):
    # In one dimension the exact forward is -lr*g/(abs(g)+eps), which has
    # opposite one-sided second derivatives at zero; no C2 claim is possible.
    zero = torch.zeros(1, dtype=torch.float64)
    gradient = zero.clone().requires_grad_()
    at_zero = direction_step(zero, gradient)["weight"]
    first, = torch.autograd.grad(at_zero.sum(), gradient)
    h = 1e-10
    plus = direction_step(zero, torch.tensor([h], dtype=torch.float64))["weight"]
    minus = direction_step(zero, torch.tensor([-h], dtype=torch.float64))["weight"]
    one_sided_plus = (plus - at_zero - first * h) / h**2
    one_sided_minus = (minus - at_zero + first * h) / h**2
    assert one_sided_plus.item() > 0
    assert one_sided_minus.item() < 0


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), 3e38))
def test_nonfinite_gradient_fails_via_adam_without_mutation(direction_step, bad_value):
    weight = torch.ones(2, dtype=torch.float32)
    gradient = torch.tensor([bad_value, bad_value], dtype=torch.float32)
    before = gradient.clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        direction_step(weight, gradient)
    torch.testing.assert_close(weight, torch.ones_like(weight), rtol=0, atol=0)
    torch.testing.assert_close(gradient, before, rtol=0, atol=0, equal_nan=True)


def test_ordinary_multistep_first_and_mixed_derivatives(direction_step):
    weight = torch.tensor([0.12, -0.18, 0.21], dtype=torch.float64, requires_grad=True)
    first = torch.tensor([0.20, -0.30, 0.40], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([-0.31, 0.23, 0.11], dtype=torch.float64, requires_grad=True)

    def update(w, g1, g2):
        one = direction_step(w, g1, lr=0.003)
        return direction_step(one["weight"], g2, one["history"], lr=0.003)["weight"]

    assert torch.autograd.gradcheck(update, (weight, first, second), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(update, (weight, first, second), eps=1e-6, atol=1e-7)
    output = update(weight, first, second)
    first_gradient, = torch.autograd.grad(output.sum(), first, create_graph=True)
    mixed, = torch.autograd.grad(first_gradient.sum(), second)
    assert torch.isfinite(first_gradient).all()
    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0
