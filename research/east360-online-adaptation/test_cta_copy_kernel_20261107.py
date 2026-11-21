"""CPU contract tests for the source-aligned encoder-copy research probe."""

import math

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F

from cta_copy_kernel_20261107 import (
    copy_for_training,
    encoder_parameter_norm,
    forward_features,
    inner_gradient,
    inner_step,
    paired_steps,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)


torch.set_num_threads(1)


@pytest.fixture(params=[torch.float32, torch.float64])
def inputs(request):
    torch.manual_seed(824)
    cfg = Config(window=8, patch=4, dim=8, heads=2, layers=1, horizon=3,
                 fast_site='tokens', fast_rule='normalized')
    base = Simple23(cfg, torch.linspace(-1., 1., 23), torch.linspace(.5, 2., 23))
    base = base.to(dtype=request.param).eval().requires_grad_(False)
    model = copy_for_training(base)
    x = torch.randn(1, cfg.window, 23, dtype=request.param)
    available = torch.rand(1, cfg.window, 23) > .25
    truth = torch.randn(1, cfg.horizon, 23, dtype=request.param)
    mask = torch.rand(1, cfg.horizon, 23) > .3
    return base, model, x, available, truth, mask


def bytes_of(state):
    return {name: value.detach().clone().contiguous().view(torch.uint8) for name, value in state.items()}


def assert_same_bytes(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_copy_identity_trainable_scope_and_differentiable_features(inputs):
    base, model, x, available, _, _ = inputs
    assert_same_bytes(bytes_of(base.state_dict()), bytes_of(model.state_dict()))
    for name, parameter in model.named_parameters():
        expected = (name in ('position', 'cls') or
                    name.startswith(('embedding.', 'encoder.', 'norm.', 'forecast_head.')))
        assert parameter.requires_grad == expected, name
    assert not any(parameter.requires_grad for parameter in base.parameters())
    assert not model.center.requires_grad and not model.scale.requires_grad
    for name, parameter in model.named_parameters():
        assert parameter.data_ptr() != dict(base.named_parameters())[name].data_ptr()

    # Enabling input gradients makes both models use the same attention path.
    x.requires_grad_()
    original = base(x, available)
    result = forward_features(model, x, available, return_tokens=True)
    assert torch.equal(result['logit'], original['logit'])
    assert torch.equal(result['forecast'], original['forecast'])
    assert torch.equal(result['risk_features'], result['tokens'][:, -1])
    assert result['tokens'].shape == (1, 3, 8)
    risk_grad, = torch.autograd.grad(result['logit'].sum(), model.embedding.weight)
    feature_grad, = torch.autograd.grad(
        forward_features(model, x, available)['risk_features'].square().sum(), model.embedding.weight)
    assert torch.isfinite(risk_grad).all() and risk_grad.abs().sum() > 0
    assert torch.isfinite(feature_grad).all() and feature_grad.abs().sum() > 0
    assert not model.risk_head.weight.requires_grad


def test_features_preserve_missing_input_and_token_masks(inputs):
    _, model, x, available, _, _ = inputs
    available[:, :model.cfg.patch] = False
    changed = x.clone()
    changed[~available] = float('nan')
    first = forward_features(model, x, available, return_tokens=True)
    second = forward_features(model, changed, available, return_tokens=True)
    for key in ('logit', 'forecast', 'risk_features', 'tokens'):
        assert torch.equal(first[key], second[key]), key
    assert torch.count_nonzero(first['tokens'][:, 0]) == 0


def test_raw_masked_forecast_gradient_bounded_step_and_no_mutation(inputs):
    base, model, x, available, truth, mask = inputs
    before_base, before_copy = bytes_of(base.state_dict()), bytes_of(model.state_dict())
    gradients = inner_gradient(model, x, available, truth, mask)
    prediction = functional_call(model, gradients['parameters'], (x, available))['forecast']
    target = torch.asinh((truth - model.center) / model.scale)
    expected_loss = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~mask, 0.).sum() / mask.sum()
    torch.testing.assert_close(gradients['loss'], expected_loss.detach(), rtol=0, atol=0)
    assert isinstance(gradients['gradients'], dict)
    assert float(gradients['gradient_norm']) > 0
    assert float(encoder_parameter_norm(model)) > 0

    reference, budget = float(gradients['gradient_norm']) * 2., .025
    step = inner_step(model, x, available, truth, mask, reference_norm=reference, budget=budget)
    assert step['delta'].keys() == gradients['gradients'].keys()
    for name, gradient in gradients['gradients'].items():
        assert torch.equal(step['delta'][name], -budget / reference * gradient), name
    assert float(step['update_norm']) <= budget
    torch.testing.assert_close(step['update_norm'], step['gradient_norm'] * (budget / reference),
                               rtol=1e-6, atol=1e-12)
    for name, value in model.named_parameters():
        if name not in step['delta']:
            assert torch.equal(step['parameters'][name], value)
        assert step['parameters'][name].data_ptr() != value.data_ptr(), name
    query = functional_call(model, step['parameters'], (x, available))['logit']
    query_gradient, = torch.autograd.grad(query.sum(), step['parameters']['embedding.weight'])
    assert torch.isfinite(query_gradient).all() and query_gradient.abs().sum() > 0
    assert_same_bytes(before_base, bytes_of(base.state_dict()))
    assert_same_bytes(before_copy, bytes_of(model.state_dict()))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_masked_future_values_do_not_affect_loss_gradients_or_update(inputs):
    _, model, x, available, truth, mask = inputs
    altered = truth.clone()
    altered[~mask] = float('nan')
    first = inner_gradient(model, x, available, truth, mask)
    second = inner_gradient(model, x, available, altered, mask)
    assert torch.equal(first['loss'], second['loss'])
    for name in first['gradients']:
        assert torch.equal(first['gradients'][name], second['gradients'][name]), name
    step1 = inner_step(model, x, available, truth, mask, reference_norm=1., budget=.1)
    step2 = inner_step(model, x, available, altered, mask, reference_norm=1., budget=.1)
    for name in step1['delta']:
        assert torch.equal(step1['delta'][name], step2['delta'][name]), name


def test_empty_observation_is_bitwise_no_op(inputs):
    _, model, x, available, truth, mask = inputs
    mask.zero_()
    truth.fill_(float('nan'))
    before = bytes_of(model.state_dict())
    with torch.no_grad():
        step = inner_step(model, x, available, truth, mask, reference_norm=1., budget=.5)
    assert step['loss'] == 0 and step['gradient_norm'] == 0 and step['update_norm'] == 0
    assert all(torch.count_nonzero(value) == 0 for value in step['delta'].values())
    for name, value in model.named_parameters():
        assert torch.equal(step['parameters'][name].contiguous().view(torch.uint8),
                           value.detach().contiguous().view(torch.uint8)), name
    assert_same_bytes(before, bytes_of(model.state_dict()))


def test_wrong_feedback_matches_real_norm_without_changing_real(inputs):
    _, model, x, available, truth, mask = inputs
    wrong = -2 * truth + 1
    reference, budget = 2., .1
    single = inner_step(model, x, available, truth, mask, reference_norm=reference, budget=budget)
    pair = paired_steps(model, x, available, truth, wrong, mask,
                        reference_norm=reference, budget=budget)
    assert pair['unmatched_control'] is False
    for name in single['parameters']:
        assert torch.equal(single['parameters'][name], pair['real']['parameters'][name]), name
    for name in single['delta']:
        assert torch.equal(single['delta'][name], pair['real']['delta'][name]), name
    assert pair['real']['update_norm'] > 0 and pair['wrong']['update_norm'] > 0
    torch.testing.assert_close(pair['wrong']['update_norm'], pair['real']['update_norm'],
                               rtol=1e-6, atol=1e-12)
    wrong_gradient = inner_gradient(model, x, available, wrong, mask)['gradients']
    for name, gradient in wrong_gradient.items():
        # Matching may rescale the wrong direction, but never reverses it.
        assert (pair['wrong']['delta'][name] * gradient).sum() <= 0


def test_zero_wrong_gradient_is_reported_and_real_step_is_retained(inputs):
    _, model, x, available, truth, mask = inputs
    # Zero features and forecast give an exactly solved center-valued target,
    # while a different target still has a nonzero norm.bias gradient.
    with torch.no_grad():
        model.norm.weight.zero_()
        model.norm.bias.zero_()
        model.forecast_head.bias.zero_()
    wrong = model.center.reshape(1, 1, 23).expand_as(truth).clone()
    truth = wrong + model.scale.reshape(1, 1, 23)
    mask.fill_(True)
    single = inner_step(model, x, available, truth, mask, reference_norm=1., budget=.1)
    pair = paired_steps(model, x, available, truth, wrong, mask, reference_norm=1., budget=.1)
    assert single['update_norm'] > 0
    assert pair['wrong']['gradient_norm'] == 0 and pair['wrong']['update_norm'] == 0
    assert pair['unmatched_control'] is True
    for name in single['delta']:
        assert torch.equal(single['delta'][name], pair['real']['delta'][name]), name
    for name, value in model.named_parameters():
        assert torch.equal(pair['wrong']['parameters'][name], value), name


@pytest.mark.parametrize('reference,budget', [(0., 1.), (-1., 1.), (math.inf, 1.), (1., -1.), (1., math.nan)])
def test_invalid_update_budgets_are_rejected(inputs, reference, budget):
    _, model, x, available, truth, mask = inputs
    with pytest.raises(ValueError):
        inner_step(model, x, available, truth, mask, reference_norm=reference, budget=budget)


def test_episode_and_feedback_contracts_are_enforced(inputs):
    _, model, x, available, truth, mask = inputs
    with pytest.raises(ValueError, match='single'):
        inner_gradient(model, x.expand(2, -1, -1), available.expand(2, -1, -1),
                       truth.expand(2, -1, -1), mask.expand(2, -1, -1))
    wrong = truth.clone()
    wrong[mask.nonzero(as_tuple=True)[0][0], mask.nonzero(as_tuple=True)[1][0],
          mask.nonzero(as_tuple=True)[2][0]] = float('nan')
    with pytest.raises(ValueError, match='same effective mask'):
        paired_steps(model, x, available, truth, wrong, mask, reference_norm=1., budget=.1)


def test_feature_hooks_are_removed_after_forward_failure(inputs):
    _, model, x, available, _, _ = inputs
    before = (len(model.risk_head._forward_pre_hooks), len(model.forecast_head._forward_pre_hooks))
    with pytest.raises(ValueError):
        forward_features(model, x[:, :-1], available[:, :-1], return_tokens=True)
    assert before == (len(model.risk_head._forward_pre_hooks), len(model.forecast_head._forward_pre_hooks))


def test_qualified_hash_locked_source_initial_identity():
    base, _ = load_parent(DEFAULT_PARENT)
    model = copy_for_training(base)
    assert_same_bytes(bytes_of(base.state_dict()), bytes_of(model.state_dict()))
    torch.manual_seed(119)
    x = torch.randn(1, base.cfg.window, 23, requires_grad=True)
    available = torch.rand_like(x) > .2
    original = base(x, available)
    copy = forward_features(model, x, available, return_tokens=True)
    assert torch.equal(original['logit'], copy['logit'])
    assert torch.equal(original['forecast'], copy['forecast'])
    assert torch.equal(copy['risk_features'], copy['tokens'][:, -1])


def test_qualified_hash_locked_source_functional_step():
    base, _ = load_parent(DEFAULT_PARENT)
    model = copy_for_training(base)
    before_base, before_copy = bytes_of(base.state_dict()), bytes_of(model.state_dict())
    torch.manual_seed(541)
    x = torch.randn(1, base.cfg.window, 23)
    available = torch.rand_like(x) > .2
    available[:, :, 2] = False
    truth = torch.randn(1, base.cfg.horizon, 23)
    mask = torch.rand_like(truth) > .3
    mask[:, :, 2] = False
    gradients = inner_gradient(model, x, available, truth, mask)
    reference = float(gradients['gradient_norm']) * 2
    budget = float(encoder_parameter_norm(model)) * .01
    step = inner_step(model, x, available, truth, mask, reference_norm=reference, budget=budget)
    assert 0 < step['update_norm'] <= budget
    result = forward_features(model, x, available, parameters=step['parameters'], return_tokens=True)
    assert torch.equal(result['risk_features'], result['tokens'][:, -1])
    risk_grad, = torch.autograd.grad(result['logit'].sum(), step['parameters']['embedding.weight'])
    assert torch.isfinite(risk_grad).all() and risk_grad.abs().sum() > 0
    assert_same_bytes(before_base, bytes_of(base.state_dict()))
    assert_same_bytes(before_copy, bytes_of(model.state_dict()))
