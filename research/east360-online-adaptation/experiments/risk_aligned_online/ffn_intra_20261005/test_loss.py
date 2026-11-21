"""CPU-only signal-loss contracts; these are not evidence of online gain."""

import math

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.ffn_intra_20261005.loss import (
    SignalRule,
    context_features,
    signal_components,
    weighted_signal_loss,
)


def inputs(batch=2, horizon=4, dtype=torch.float32):
    prediction = torch.zeros(batch, horizon, 23, dtype=dtype)
    truth = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    return prediction, truth, mask, torch.zeros(23, dtype=dtype), torch.ones(23, dtype=dtype)


def sparse_inputs():
    prediction, truth, mask, center, scale = inputs()
    center = torch.linspace(-2., 2., 23)
    scale = torch.linspace(.5, 2., 23)
    truth.fill_(float('nan'))
    prediction.fill_(float('nan'))
    mask.zero_()
    # Channel 0: residuals [0, 1, 2] and adjacent residual changes [1, 1].
    truth[0, :3, 0] = center[0] + scale[0] * torch.sinh(torch.tensor([0., 1., 3.]))
    prediction[0, :3, 0] = torch.tensor([0., 2., 5.])
    mask[0, :3, 0] = True
    # Channel 1: two separate observations must not form an adjacent pair.
    truth[0, [0, 2], 1] = center[1]
    prediction[0, [0, 2], 1] = torch.tensor([.5, -.5])
    mask[0, [0, 2], 1] = True
    # Nonfinite truth is not a finite observation, even if the mask is true.
    mask[0, :, 2] = True
    prediction[0, :, 2] = 7.
    truth[0, :, 2] = torch.tensor([float('nan'), float('inf'), -float('inf'), float('nan')])
    # A second example has one valid channel, not a batch-wide denominator.
    truth[1, 1, 5], prediction[1, 1, 5], mask[1, 1, 5] = center[5], 2., True
    return prediction, truth, mask, center, scale


def test_components_use_source_asinh_and_only_finite_observed_adjacent_pairs():
    args = sparse_inputs()
    result = signal_components(*args)
    assert set(result) == {'level', 'difference', 'valid', 'difference_valid'}
    assert all(value.shape == (2, 23) for value in result.values())
    assert result['valid'].dtype == result['difference_valid'].dtype == torch.bool
    expected_level = torch.zeros(2, 23)
    expected_level[0, 0], expected_level[0, 1], expected_level[1, 5] = 2. / 3., .125, 1.5
    expected_difference = torch.zeros(2, 23)
    expected_difference[0, 0] = .5
    torch.testing.assert_close(result['level'], expected_level)
    torch.testing.assert_close(result['difference'], expected_difference)
    assert torch.equal(result['valid'], expected_level > 0)
    assert torch.equal(result['difference_valid'], expected_difference > 0)


def test_zero_error_channels_are_still_valid():
    result = signal_components(*inputs(horizon=2))
    assert result['valid'].all() and result['difference_valid'].all()
    assert not result['level'].any() and not result['difference'].any()


def test_truth_detaches_and_unavailable_prediction_gradients_are_finite_zero():
    prediction, truth, mask, center, scale = sparse_inputs()
    prediction.requires_grad_()
    truth.requires_grad_()
    result = signal_components(prediction, truth, mask, center, scale)
    weighted_signal_loss(result).sum().backward()
    assert truth.grad is None
    assert torch.isfinite(prediction.grad).all()
    observed = mask & torch.isfinite(truth)
    assert not prediction.grad[~observed].any()
    assert prediction.grad[observed].abs().sum() > 0


@pytest.mark.parametrize('horizon', [0, 1, 4])
@pytest.mark.parametrize('learned', [False, True])
def test_all_missing_is_graph_connected_zero_even_with_nonfinite_predictions(horizon, learned):
    prediction, truth, mask, center, scale = inputs(horizon=horizon)
    prediction.fill_(float('nan')).requires_grad_()
    truth.fill_(float('inf')).requires_grad_()
    mask.zero_()
    weights = torch.ones(2, 2, 23, requires_grad=True) if learned else None
    components = signal_components(prediction, truth, mask, center, scale)
    loss = weighted_signal_loss(components, weights)
    assert loss.shape == (2,) and loss.requires_grad
    assert torch.equal(loss, torch.zeros_like(loss))
    assert not components['valid'].any() and not components['difference_valid'].any()
    loss.sum().backward()
    assert prediction.grad is not None and torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert truth.grad is None
    if learned:
        assert weights.grad is not None and not weights.grad.any()
    context = context_features(prediction, truth, mask, center, scale)
    assert torch.equal(context, torch.zeros(2, 92))


def test_single_step_has_level_but_no_difference():
    prediction, truth, mask, center, scale = inputs(horizon=1)
    prediction.fill_(.5).requires_grad_()
    components = signal_components(prediction, truth, mask, center, scale)
    assert components['valid'].all() and not components['difference_valid'].any()
    assert components['difference'].requires_grad and not components['difference'].any()
    torch.testing.assert_close(weighted_signal_loss(components), torch.full((2,), .125))


@pytest.mark.parametrize('function', [signal_components, context_features])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_observed_prediction_raises(function, bad):
    args = inputs()
    args[0][0, 1, 3] = bad
    with pytest.raises(FloatingPointError, match='prediction|forecast'):
        function(*args)


@pytest.mark.parametrize('function', [signal_components, context_features])
def test_masked_nonfinite_values_do_not_change_results(function):
    prediction, truth, mask, center, scale = inputs()
    prediction.fill_(.3)
    mask[:, 1:3, 4:9] = False
    before = function(prediction, truth, mask, center, scale)
    prediction[~mask], truth[~mask] = float('inf'), float('nan')
    after = function(prediction, truth, mask, center, scale)
    if isinstance(before, dict):
        for key in before:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    else:
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_context_block_order_values_and_complete_detachment():
    prediction, truth, mask, center, scale = sparse_inputs()
    for tensor in (prediction, truth, center, scale):
        tensor.requires_grad_()
    context = context_features(prediction, truth, mask, center, scale)
    expected = torch.zeros(2, 4, 23)
    expected[0, :, 0] = torch.tensor([1., 1., 1.5, .75])
    expected[0, :, 1] = torch.tensor([0., .5, 0., .5])
    expected[1, :, 5] = torch.tensor([2., 2., 0., .25])
    assert context.shape == (2, 92) and context.dtype == prediction.dtype
    torch.testing.assert_close(context, expected.flatten(1))
    assert not context.requires_grad and context.grad_fn is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_context_explicitly_bounds_extreme_finite_values_without_losing_cancellation(dtype):
    prediction, truth, mask, center, scale = inputs(batch=1, dtype=dtype)
    largest = torch.finfo(prediction.dtype).max
    prediction[:, :, 0] = largest
    prediction[:, :, 1] = -largest
    prediction[0, :, 2] = torch.tensor([largest, largest, -largest, -largest], dtype=dtype)
    truth[0, :, 3] = torch.tensor([largest, -largest, largest, -largest], dtype=dtype)
    context = context_features(prediction, truth, mask, center, scale).reshape(1, 4, 23)
    assert torch.isfinite(context).all() and context.abs().max() <= 10.
    torch.testing.assert_close(context[0, 0, :3], torch.tensor([10., -10., 0.], dtype=dtype))
    assert torch.equal(context[0, 1, :3], torch.full((3,), 10., dtype=dtype))
    assert context[0, 2, 3] == 10. and context[0, 3].eq(1.).all()


def test_weighted_loss_uses_valid_channel_count_not_weight_sum_or_all_channels():
    components = signal_components(*sparse_inputs())
    weights = torch.ones(2, 2, 23)
    weights[0, 0, 0], weights[0, 0, 1], weights[0, 1, 0] = 2., .5, 1.5
    expected = torch.tensor([(2. * (2. / 3.) + .5 * .125 + .25 * 1.5 * .5) / 2., 1.5])
    torch.testing.assert_close(weighted_signal_loss(components, weights), expected)
    uniform = weighted_signal_loss(components)
    torch.testing.assert_close(uniform, torch.tensor([(2. / 3. + .125 + .25 * .5) / 2., 1.5]))
    torch.testing.assert_close(weighted_signal_loss(components, torch.ones_like(weights)), uniform, rtol=0, atol=0)


def test_weighted_loss_excludes_invalid_channels_and_missing_differences():
    level = torch.full((1, 23), float('nan'))
    difference = torch.full_like(level, float('nan'))
    valid = torch.zeros(1, 23, dtype=torch.bool)
    valid[0, 0], level[0, 0] = True, 2.
    components = dict(level=level.requires_grad_(), difference=difference.requires_grad_(),
                      valid=valid, difference_valid=torch.zeros_like(valid))
    weights = torch.ones(1, 2, 23, requires_grad=True)
    loss = weighted_signal_loss(components, weights)
    torch.testing.assert_close(loss, torch.tensor([2.]))
    loss.sum().backward()
    assert torch.isfinite(level.grad).all() and torch.isfinite(difference.grad).all()
    assert not difference.grad.any() and not weights.grad[:, 1].any()


def test_rule_architecture_bias_and_bit_exact_initial_weights():
    rule = SignalRule()
    layers = list(rule.modules())
    linear = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [(92, 32), (32, 46)]
    assert sum(isinstance(layer, nn.SiLU) for layer in layers) == 1
    assert not linear[-1].weight.any()
    torch.testing.assert_close(linear[-1].bias,
                               torch.full_like(linear[-1].bias, math.log(3. / 4.)), rtol=0, atol=0)
    context = torch.linspace(-10., 10., 3 * 92).reshape(3, 92)
    weights = rule(context)
    assert weights.shape == (3, 2, 23)
    assert torch.equal(weights, torch.ones_like(weights))


def test_rule_weights_follow_sigmoid_formula_and_remain_bounded():
    rule = SignalRule()
    final = [layer for layer in rule.modules() if isinstance(layer, nn.Linear)][-1]
    logits = torch.linspace(-1000., 1000., 46)
    with torch.no_grad():
        final.bias.copy_(logits)
    weights = rule(torch.zeros(2, 92))
    expected = (.25 + 1.75 * logits.sigmoid()).reshape(1, 2, 23).expand(2, -1, -1)
    torch.testing.assert_close(weights, expected)
    assert (weights >= .25).all() and (weights <= 2.).all()
    assert weights.min() == .25 and weights.max() == 2.


def test_rule_has_nonzero_mixed_derivative_through_forecast_gradient():
    prediction, truth, mask, center, scale = inputs(batch=1, horizon=3)
    prediction[:] = torch.tensor([.1, .3, .6])[None, :, None]
    prediction.requires_grad_()
    truth.requires_grad_()
    context = context_features(prediction, truth, mask, center, scale)
    rule = SignalRule()
    weights = rule(context)
    components = signal_components(prediction, truth, mask, center, scale)
    forecast_gradient, = torch.autograd.grad(weighted_signal_loss(components, weights).sum(),
                                             prediction, create_graph=True)
    final = [layer for layer in rule.modules() if isinstance(layer, nn.Linear)][-1]
    mixed, = torch.autograd.grad(forecast_gradient.square().sum(), final.bias)
    assert torch.isfinite(mixed).all()
    assert mixed[:23].abs().sum() > 0 and mixed[23:].abs().sum() > 0
    assert truth.grad is None and not context.requires_grad


def test_weighted_components_pass_first_and_second_order_numerical_gradient_checks():
    prediction, truth, mask, center, scale = inputs(batch=1, horizon=3, dtype=torch.float64)
    prediction[:] = torch.linspace(.1, .6, prediction.numel()).reshape_as(prediction)
    prediction.requires_grad_()
    mask[:, 1, 0] = False
    weights = torch.ones(1, 2, 23, dtype=torch.float64, requires_grad=True)

    def loss(prediction, weights):
        return weighted_signal_loss(signal_components(prediction, truth, mask, center, scale), weights)

    assert torch.autograd.gradcheck(loss, (prediction, weights), fast_mode=True)
    assert torch.autograd.gradgradcheck(loss, (prediction, weights), fast_mode=True)


@pytest.mark.parametrize('function', [signal_components, context_features])
@pytest.mark.parametrize('problem', ['prediction_shape', 'truth_shape', 'mask_shape', 'mask_dtype',
                                     'center_shape', 'scale_shape', 'center_nonfinite',
                                     'scale_nonfinite', 'scale_zero', 'scale_negative'])
def test_invalid_signal_contract_raises(function, problem):
    prediction, truth, mask, center, scale = inputs()
    if problem == 'prediction_shape':
        prediction = prediction[..., :22]
    elif problem == 'truth_shape':
        truth = truth[:, :1]
    elif problem == 'mask_shape':
        mask = mask[:, :1]
    elif problem == 'mask_dtype':
        mask = mask.float()
    elif problem == 'center_shape':
        center = center[None]
    elif problem == 'scale_shape':
        scale = scale[:22]
    elif problem == 'center_nonfinite':
        center[0] = float('nan')
    elif problem == 'scale_nonfinite':
        scale[0] = float('inf')
    elif problem == 'scale_zero':
        scale[0] = 0.
    else:
        scale[0] = -1.
    with pytest.raises(ValueError):
        function(prediction, truth, mask, center, scale)


def test_weights_require_batch_two_by_23_shape():
    components = signal_components(*inputs())
    with pytest.raises(ValueError, match='weights'):
        weighted_signal_loss(components, torch.ones(23))
