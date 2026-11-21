import math

import pytest
import torch
from torch import nn

from motion_metric_kernel_20261107 import MotionMetric, descent, match_state, potential


def test_magnitude_update_retains_small_motion_and_caps_large_motion():
    from motion_metric_kernel_20261107 import magnitude_descent
    gradient = torch.zeros(3, 4, 64, dtype=torch.float64)
    gradient[0, 0, 0] = .1
    gradient[1, 0, 0] = 10
    actual = magnitude_descent(gradient, reference_norm=1., step=.35)
    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([-.035, -.35, 0.], dtype=torch.float64))
    assert not actual[2].any()


def test_magnitude_update_has_finite_meta_gradient_at_quiet_feedback():
    from motion_metric_kernel_20261107 import magnitude_descent
    gradient = torch.zeros(2, 4, 64, dtype=torch.float64, requires_grad=True)
    output = magnitude_descent(gradient, reference_norm=.5, step=.35)
    derivative, = torch.autograd.grad(output.sum(), gradient)
    torch.testing.assert_close(derivative, torch.full_like(gradient, -.7))


def _head(metric):
    return [module for module in metric.modules() if isinstance(module, nn.Linear)][-1]


def test_metric_architecture_initialization_and_detached_context():
    metric = MotionMetric(context_dim=7, channels=5)
    linear = [module for module in metric.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(7, 32), (32, 5)]
    assert any(isinstance(module, nn.SiLU) for module in metric.modules())
    assert torch.count_nonzero(linear[-1].weight) == 0
    torch.testing.assert_close(
        linear[-1].bias, torch.full((5,), math.atanh(0.5)), rtol=0, atol=0
    )

    context = torch.randn(4, 7, requires_grad=True)
    result = metric(context)
    assert torch.equal(result, torch.ones(4, 5))
    result.sum().backward()
    assert context.grad is None
    assert linear[-1].bias.grad is not None


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_metric_weights_are_signed_and_strictly_bounded_even_when_saturated(dtype):
    metric = MotionMetric(context_dim=3, channels=5).to(dtype=dtype)
    with torch.no_grad():
        _head(metric).bias.copy_(torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=dtype))
    result = metric(torch.ones(2, 3, dtype=dtype))
    assert (result[:, :2] < 0).all()
    assert (result[:, 3:] > 0).all()
    assert (result.abs() < 2).all()
    torch.testing.assert_close(result[:, 1:4], 2 * torch.tanh(_head(metric).bias[1:4]).expand(2, -1))


def test_metric_float64_conversion_preserves_tanh_formula_near_one():
    metric = MotionMetric().double()
    result = metric(torch.zeros(2, 69, dtype=torch.float64))
    expected = (2 * torch.tanh(_head(metric).bias)).expand(2, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=2e-8)


def test_potential_matches_masked_difference_of_half_mse_gradients():
    generator = torch.Generator().manual_seed(71)
    shape = (3, 25, 23)
    prediction = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    hold = torch.randn(shape, generator=generator, dtype=torch.float64)
    motion = torch.randn(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator) > 0.45
    weights = torch.ones(3, 23, dtype=torch.float64)

    actual = potential(prediction, motion, mask, weights)
    actual_grad = torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0]
    mse_difference = sum(
        0.5 * (prediction[row][mask[row]] - (hold[row][mask[row]] + motion[row][mask[row]])).square().mean()
        - 0.5 * (prediction[row][mask[row]] - hold[row][mask[row]]).square().mean()
        for row in range(3)
    )
    expected_grad = torch.autograd.grad(mse_difference, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_potential_values_use_per_row_masked_mean_and_signed_weights():
    prediction = torch.zeros(2, 25, 23, dtype=torch.float64)
    motion = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    weights = torch.ones(2, 23, dtype=torch.float64)
    prediction[0, 0, 0], motion[0, 0, 0], weights[0, 0] = 2, 3, -1
    prediction[0, 1, 1], motion[0, 1, 1], weights[0, 1] = 4, 2, 2
    prediction[1, 4, 2], motion[1, 4, 2], weights[1, 2] = -2, 3, 0.5
    mask[0, 0, 0] = mask[0, 1, 1] = mask[1, 4, 2] = True
    result = potential(prediction, motion, mask, weights)
    torch.testing.assert_close(result, torch.tensor([-5.0, 3.0], dtype=torch.float64))


def test_large_prediction_has_no_independent_quadratic_gradient():
    motion = torch.linspace(-3, 4, 25 * 23, dtype=torch.float64).reshape(1, 25, 23)
    mask = torch.ones_like(motion, dtype=torch.bool)
    mask[:, ::2, ::3] = False
    weights = torch.linspace(-1.5, 1.5, 23, dtype=torch.float64).reshape(1, 23)
    small = torch.zeros_like(motion, requires_grad=True)
    large = torch.full_like(motion, 1e12, requires_grad=True)
    small_grad = torch.autograd.grad(potential(small, motion, mask, weights).sum(), small)[0]
    large_grad = torch.autograd.grad(potential(large, motion, mask, weights).sum(), large)[0]
    assert torch.equal(small_grad, large_grad)
    assert torch.count_nonzero(small_grad) > 0


def test_unobserved_nonfinite_values_do_not_leak_into_outputs_or_gradients():
    mask = torch.zeros(2, 25, 23, dtype=torch.bool)
    mask[0, 0, 0] = mask[0, 0, 1] = True
    prediction = torch.full((2, 25, 23), float('nan'), dtype=torch.float64)
    motion = torch.full_like(prediction, float('inf'))
    prediction[0, 0, :2] = torch.tensor([2.0, -1.0])
    motion[0, 0, :2] = torch.tensor([3.0, 4.0])
    prediction.requires_grad_()
    motion.requires_grad_()
    weights = torch.ones(2, 23, dtype=torch.float64, requires_grad=True)

    value = potential(prediction, motion, mask, weights)
    torch.testing.assert_close(value, torch.tensor([-1.0, 0.0], dtype=torch.float64))
    gradients = torch.autograd.grad(value.sum(), (prediction, motion, weights))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[0][~mask]) == 0
    assert torch.count_nonzero(gradients[1][~mask]) == 0
    assert torch.count_nonzero(gradients[2][1]) == 0


@pytest.mark.parametrize('input_name', ['prediction', 'motion'])
@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_observed_nonfinite_value_raises(input_name, bad_value):
    inputs = {
        'prediction': torch.zeros(1, 25, 23),
        'motion': torch.zeros(1, 25, 23),
        'mask': torch.zeros(1, 25, 23, dtype=torch.bool),
        'weights': torch.ones(1, 23),
    }
    inputs['mask'][0, 4, 5] = True
    inputs[input_name][0, 4, 5] = bad_value
    with pytest.raises(ValueError, match='finite'):
        potential(**inputs)


def test_zero_motion_produces_exactly_zero_state_update():
    state = torch.randn(2, 4, 64, requires_grad=True)
    prediction = state.flatten(1)[:, :23].unsqueeze(1).expand(-1, 25, -1)
    motion = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    weights = torch.randn(2, 23, requires_grad=True)
    gradient = torch.autograd.grad(
        potential(prediction, motion, mask, weights).sum(), state, create_graph=True
    )[0]
    update = descent(gradient)
    assert torch.equal(update, torch.zeros_like(update))
    assert torch.equal(state + update, state)
    weight_grad = torch.autograd.grad(update.sum(), weights)[0]
    assert torch.equal(weight_grad, torch.zeros_like(weight_grad))


def test_descent_is_per_row_normalized_and_supports_noncontiguous_inputs():
    generator = torch.Generator().manual_seed(19)
    gradient = torch.randn(3, 64, 4, generator=generator, dtype=torch.float64).transpose(1, 2)
    gradient[1].zero_()
    assert not gradient.is_contiguous()
    update = descent(gradient, step=0.42)
    assert torch.equal(update[1], torch.zeros_like(update[1]))
    torch.testing.assert_close(torch.linalg.vector_norm(update[[0, 2]], dim=(1, 2)), torch.tensor([0.42, 0.42], dtype=torch.float64))
    for row in (0, 2):
        torch.testing.assert_close(update[row] / 0.42, -gradient[row] / torch.linalg.vector_norm(gradient[row]))


def test_meta_gradient_flows_through_descent_to_metric():
    generator = torch.Generator().manual_seed(53)
    metric = MotionMetric().double()
    context = torch.randn(2, 69, generator=generator, dtype=torch.float64, requires_grad=True)
    state = torch.randn(2, 4, 64, generator=generator, dtype=torch.float64, requires_grad=True)
    projection = torch.randn(256, 25 * 23, generator=generator, dtype=torch.float64) / 16
    prediction = (state.flatten(1) @ projection).reshape(2, 25, 23)
    motion = torch.randn(2, 25, 23, generator=generator, dtype=torch.float64)
    mask = torch.rand(2, 25, 23, generator=generator) > 0.2
    gradient = torch.autograd.grad(
        potential(prediction, motion, mask, metric(context)).sum(), state, create_graph=True
    )[0]
    next_state = state + descent(gradient)
    metric_grad, context_grad = torch.autograd.grad(
        next_state.square().sum(), (_head(metric).bias, context), allow_unused=True
    )
    assert metric_grad is not None and torch.isfinite(metric_grad).all()
    assert torch.count_nonzero(metric_grad) > 0
    assert context_grad is None

    # An independent finite difference checks the full state-update path.
    index = metric_grad.abs().argmax().item()
    bias = _head(metric).bias
    original = bias[index].item()

    def outer_value(offset):
        with torch.no_grad():
            bias[index] = original + offset
        sample_state = state.detach().requires_grad_()
        sample_prediction = (sample_state.flatten(1) @ projection).reshape(2, 25, 23)
        sample_gradient = torch.autograd.grad(
            potential(sample_prediction, motion, mask, metric(context)).sum(), sample_state
        )[0]
        return (sample_state + descent(sample_gradient)).square().sum().item()

    try:
        epsilon = 1e-4
        finite_difference = (outer_value(epsilon) - outer_value(-epsilon)) / (2 * epsilon)
    finally:
        with torch.no_grad():
            bias[index] = original
    assert metric_grad[index].item() == pytest.approx(finite_difference, rel=1e-5, abs=1e-7)


def test_match_state_preserves_direction_and_handles_zero_vectors():
    state = torch.zeros(4, 4, 64, dtype=torch.float64)
    reference = torch.zeros_like(state)
    state[0, 0, :2] = torch.tensor([3.0, 4.0])
    reference[0, 1, 0] = 10
    reference[1, 2, 0] = 7
    state[2, 0, 0] = -5
    state[3, 0, 0] = -2
    reference[3, 0, :2] = torch.tensor([3.0, 4.0])
    matched = match_state(state, reference)
    assert torch.equal(matched[0, 0, :2], torch.tensor([6.0, 8.0], dtype=torch.float64))
    assert torch.equal(matched[1], state[1])
    assert torch.equal(matched[2], torch.zeros_like(state[2]))
    assert matched[3, 0, 0] == -5


def test_zero_vectors_remain_zero_for_float16():
    zeros = torch.zeros(2, 4, 64, dtype=torch.float16)
    assert torch.equal(descent(zeros), zeros)
    assert torch.equal(match_state(zeros, torch.ones_like(zeros)), zeros)


def test_small_nonzero_float16_gradient_still_gets_the_requested_step():
    gradient = torch.zeros(1, 4, 64, dtype=torch.float16)
    gradient[0, 0, 0] = torch.finfo(torch.float16).tiny / 1024
    update = descent(gradient)
    assert update[0, 0, 0].item() == pytest.approx(-0.35, abs=2e-4)


def test_descent_retains_clamp_derivative_at_zero_in_float64():
    gradient = torch.zeros(1, 4, 64, dtype=torch.float64, requires_grad=True)
    update = descent(gradient)
    derivative = torch.autograd.grad(update.sum(), gradient)[0]
    torch.testing.assert_close(derivative, torch.full_like(derivative, -0.35 / 1e-12))


def test_match_state_preserves_norm_for_small_nonzero_states():
    state = torch.zeros(1, 4, 64, dtype=torch.float64)
    reference = torch.zeros_like(state)
    state[0, 0, 0] = 1e-13
    reference[0, 0, 1] = 2.0
    matched = match_state(state, reference)
    assert matched[0, 0, 0] == 2.0
    assert torch.linalg.vector_norm(matched) == 2.0


def test_public_shape_and_mask_boundaries_reject_invalid_inputs():
    prediction = torch.zeros(2, 25, 23)
    motion = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    weights = torch.ones(2, 23)
    with pytest.raises((TypeError, ValueError)):
        potential(prediction, motion, mask.float(), weights)
    with pytest.raises(ValueError):
        potential(prediction[:, :-1], motion[:, :-1], mask[:, :-1], weights)
    with pytest.raises(ValueError):
        potential(prediction, motion, mask, weights[:1])
    with pytest.raises(ValueError):
        descent(torch.zeros(2, 4, 63))
    with pytest.raises(ValueError):
        match_state(torch.zeros(2, 4, 64), torch.zeros(1, 4, 64))
