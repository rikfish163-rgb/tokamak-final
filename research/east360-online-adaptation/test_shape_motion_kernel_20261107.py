import math

import pytest
import torch
from torch import nn

from motion_metric_kernel_20261107 import potential as level_potential
from shape_motion_kernel_20261107 import ShapeMetric, lagged_motion, potential


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_metric_architecture_initialization_and_detached_context(dtype):
    metric = ShapeMetric().to(dtype=dtype)
    layers = [module for module in metric.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(69, 32), (32, 46)]
    assert any(isinstance(module, nn.SiLU) for module in metric.modules())
    assert torch.count_nonzero(layers[-1].weight) == 0
    torch.testing.assert_close(
        layers[-1].bias[:23], torch.full_like(layers[-1].bias[:23], math.atanh(0.5)),
        rtol=0, atol=2e-8,
    )
    assert torch.count_nonzero(layers[-1].bias[23:]) == 0

    context = torch.randn(3, 69, dtype=dtype, requires_grad=True)
    weights = metric(context)
    assert weights.shape == (3, 2, 23)
    torch.testing.assert_close(weights[:, 0], torch.ones_like(weights[:, 0]), rtol=0, atol=2e-8)
    assert torch.equal(weights[:, 1], torch.zeros_like(weights[:, 1]))

    # The detachment must hold even after the initially zero final layer learns.
    with torch.no_grad():
        layers[-1].weight.fill_(0.01)
    metric(context).sum().backward()
    assert context.grad is None
    assert layers[-1].weight.grad is not None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_both_weight_groups_stay_in_open_bounds_when_saturated(dtype):
    metric = ShapeMetric(context_dim=3, channels=5).to(dtype=dtype)
    with torch.no_grad():
        metric.net[-1].bias.copy_(torch.tensor(
            [-100, -1, 0, 1, 100, 100, 1, 0, -1, -100], dtype=dtype,
        ))
    weights = metric(torch.ones(2, 3, dtype=dtype))
    assert weights.shape == (2, 2, 5)
    assert (weights.abs() < 2).all()
    assert (weights[:, :, 0] * weights[:, :, 4] < 0).all()
    torch.testing.assert_close(weights[0, :, 1:4], 2 * torch.tanh(metric.net[-1].bias.reshape(2, 5)[:, 1:4]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_level_only_matches_old_potential_and_gradients(dtype):
    generator = torch.Generator().manual_seed(71)
    prediction = torch.randn(3, 25, 23, generator=generator, dtype=dtype)
    motion = torch.randn(3, 25, 23, generator=generator, dtype=dtype)
    mask = torch.rand(3, 25, 23, generator=generator) > 0.35
    prediction[~mask] = float('nan')
    motion[~mask] = float('nan')
    prediction.requires_grad_()
    motion.requires_grad_()
    weights = torch.zeros(3, 2, 23, dtype=dtype)
    weights[:, 0] = torch.randn(3, 23, generator=generator, dtype=dtype)
    weights.requires_grad_()

    actual = potential(prediction, motion, mask, weights, shape_scale=2.5)
    expected = level_potential(prediction, motion, mask, weights[:, 0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), (prediction, motion, weights), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (prediction, motion, weights))
    for got, want in zip(actual_grad[:2], expected_grad[:2]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert torch.isfinite(got).all()
    torch.testing.assert_close(actual_grad[2][:, 0], expected_grad[2][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_constant_motion_has_zero_shape_term(dtype):
    generator = torch.Generator().manual_seed(5)
    prediction = torch.randn(2, 25, 23, generator=generator, dtype=dtype)
    motion = torch.randn(2, 1, 23, generator=generator, dtype=dtype).expand_as(prediction)
    mask = torch.rand(2, 25, 23, generator=generator) > 0.25
    weights = torch.randn(2, 2, 23, generator=generator, dtype=dtype)
    actual = potential(prediction, motion, mask, weights, shape_scale=0.7)
    expected = level_potential(prediction, motion, mask, weights[:, 0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, motion_difference, pair_mask = lagged_motion(prediction, motion, mask)
    assert torch.count_nonzero(motion_difference[pair_mask]) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shape_uses_only_pairs_with_both_endpoints_and_per_row_count(dtype):
    prediction = torch.full((2, 25, 23), float('nan'), dtype=dtype)
    motion = torch.full_like(prediction, float('nan'))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    for row, time, channel, pred, move in [
        (0, 0, 0, 2, 5), (0, 8, 0, 6, 8), (0, 16, 0, 11, 6),
        (0, 8, 1, 2, 7), (0, 16, 1, 4, 4),
        (0, 9, 2, 1e6, -1e6), (1, 2, 4, 0, 1), (1, 10, 4, 4, 4),
    ]:
        prediction[row, time, channel] = pred
        motion[row, time, channel] = move
        mask[row, time, channel] = True
    weights = torch.zeros(2, 2, 23, dtype=dtype)
    weights[0, 1, 0], weights[0, 1, 1], weights[1, 1, 4] = 0.5, -2, 1
    prediction.requires_grad_()
    motion.requires_grad_()
    weights.requires_grad_()

    pred_diff, motion_diff, pair_mask = lagged_motion(prediction, motion, mask)
    assert pred_diff.shape == motion_diff.shape == pair_mask.shape == (2, 17, 23)
    assert torch.equal(pair_mask.sum(dim=(1, 2)), torch.tensor([3, 1]))
    assert torch.isfinite(pred_diff).all() and torch.isfinite(motion_diff).all()
    assert torch.count_nonzero(pred_diff[~pair_mask]) == 0
    assert torch.count_nonzero(motion_diff[~pair_mask]) == 0
    assert pred_diff[0, 0, 0] == 4 and motion_diff[0, 0, 0] == 3
    result = potential(prediction, motion, mask, weights, shape_scale=3.0)
    torch.testing.assert_close(result, torch.tensor([-13, -36], dtype=dtype), rtol=0, atol=0)
    gradients = torch.autograd.grad(result.sum(), (prediction, motion, weights))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[0][~mask]) == 0
    assert torch.count_nonzero(gradients[1][~mask]) == 0


def test_shape_scale_is_applied_after_the_pair_mean():
    prediction = (torch.arange(25, dtype=torch.float32) / 8).reshape(1, 25, 1).expand(-1, -1, 23)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    weights = torch.zeros(1, 2, 23)
    weights[:, 1] = 1
    result = potential(prediction, prediction, mask, weights, shape_scale=1e37)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([-1e37]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_no_valid_pairs_give_exact_zero_shape_and_finite_gradients(dtype):
    prediction = torch.full((2, 25, 23), float('nan'), dtype=dtype)
    motion = torch.full_like(prediction, float('nan'))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, :8, :2] = True
    prediction[mask] = torch.arange(mask.sum(), dtype=dtype)
    motion[mask] = 2
    prediction.requires_grad_()
    motion.requires_grad_()
    weights = torch.zeros(2, 2, 23, dtype=dtype)
    weights[:, 1] = 1
    weights.requires_grad_()
    assert not lagged_motion(prediction, motion, mask)[2].any()

    result = potential(prediction, motion, mask, weights, shape_scale=1.0)
    assert torch.equal(result, torch.zeros_like(result))
    gradients = torch.autograd.grad(result.sum(), (prediction, motion, weights))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradients[0]) == 0
    assert torch.count_nonzero(gradients[1]) == 0
    assert torch.count_nonzero(gradients[2][:, 1]) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_joint_time_reordering_changes_shape_gradient(dtype):
    generator = torch.Generator().manual_seed(11)
    prediction = torch.randn(1, 25, 23, generator=generator, dtype=dtype, requires_grad=True)
    motion = torch.randn(1, 25, 23, generator=generator, dtype=dtype)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    weights = torch.zeros(1, 2, 23, dtype=dtype)
    weights[:, 1] = 1
    ordered_gradient, = torch.autograd.grad(
        potential(prediction, motion, mask, weights, shape_scale=1.0).sum(), prediction,
    )
    order = torch.tensor(list(range(0, 25, 2)) + list(range(1, 25, 2)))
    reordered = prediction.detach()[:, order].clone().requires_grad_()
    reordered_gradient, = torch.autograd.grad(
        potential(reordered, motion[:, order], mask[:, order], weights, shape_scale=1.0).sum(),
        reordered,
    )
    restored_gradient = torch.empty_like(reordered_gradient)
    restored_gradient[:, order] = reordered_gradient
    assert not torch.allclose(ordered_gradient, restored_gradient)


@pytest.mark.parametrize('input_name', ['prediction', 'motion'])
def test_observed_nan_is_rejected_in_both_public_functions(input_name):
    inputs = {
        'prediction': torch.zeros(1, 25, 23),
        'motion': torch.zeros(1, 25, 23),
        'mask': torch.zeros(1, 25, 23, dtype=torch.bool),
    }
    inputs['mask'][0, 4, 5] = True
    inputs[input_name][0, 4, 5] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        lagged_motion(**inputs)
    with pytest.raises(ValueError, match='finite'):
        potential(**inputs, weights=torch.ones(1, 2, 23), shape_scale=1.0)


@pytest.mark.parametrize('scale', [0.0, -1.0, float('nan'), float('inf')])
def test_shape_scale_must_be_positive_and_finite(scale):
    prediction = torch.zeros(1, 25, 23)
    with pytest.raises(ValueError, match='scale'):
        potential(prediction, prediction, torch.ones_like(prediction, dtype=torch.bool),
                  torch.ones(1, 2, 23), shape_scale=scale)


def test_shape_weights_must_be_finite():
    prediction = torch.zeros(1, 25, 23)
    weights = torch.ones(1, 2, 23)
    weights[0, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        potential(prediction, prediction, torch.ones_like(prediction, dtype=torch.bool),
                  weights, shape_scale=1.0)


def test_float64_gradcheck_and_gradgradcheck():
    generator = torch.Generator().manual_seed(37)
    prediction = torch.randn(1, 25, 23, generator=generator, dtype=torch.float64, requires_grad=True)
    motion = torch.randn(1, 25, 23, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(1, 2, 23, generator=generator, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(1, 25, 23, generator=generator) > 0.3

    def evaluate(pred, move, weight):
        return potential(pred, move, mask, weight, shape_scale=1.25)

    assert torch.autograd.gradcheck(evaluate, (prediction, motion, weights), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (prediction, motion, weights), fast_mode=True)


def test_float64_meta_gradient_reaches_shape_weights_through_prediction_update():
    generator = torch.Generator().manual_seed(53)
    metric = ShapeMetric().double()
    context = torch.randn(2, 69, generator=generator, dtype=torch.float64, requires_grad=True)
    state = torch.randn(2, 4, 64, generator=generator, dtype=torch.float64, requires_grad=True)
    projection = torch.randn(256, 25 * 23, generator=generator, dtype=torch.float64) / 16
    motion = torch.randn(2, 25, 23, generator=generator, dtype=torch.float64)
    mask = torch.rand(2, 25, 23, generator=generator) > 0.2

    def outer_value(current_state, create_graph):
        prediction = (current_state.flatten(1) @ projection).reshape(2, 25, 23)
        gradient, = torch.autograd.grad(
            potential(prediction, motion, mask, metric(context), shape_scale=1.5).sum(),
            current_state, create_graph=create_graph,
        )
        return (current_state - 0.1 * gradient).square().sum()

    bias = metric.net[-1].bias
    bias_gradient, context_gradient = torch.autograd.grad(
        outer_value(state, create_graph=True), (bias, context), allow_unused=True,
    )
    assert context_gradient is None
    assert torch.isfinite(bias_gradient).all()
    assert torch.count_nonzero(bias_gradient[23:]) > 0

    index = 23 + bias_gradient[23:].abs().argmax().item()
    original = bias[index].item()
    epsilon = 1e-4
    try:
        with torch.no_grad():
            bias[index] = original + epsilon
        plus = outer_value(state.detach().requires_grad_(), create_graph=False).item()
        with torch.no_grad():
            bias[index] = original - epsilon
        minus = outer_value(state.detach().requires_grad_(), create_graph=False).item()
    finally:
        with torch.no_grad():
            bias[index] = original
    assert bias_gradient[index].item() == pytest.approx((plus - minus) / (2 * epsilon), rel=1e-5, abs=1e-7)
