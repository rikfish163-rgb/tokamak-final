"""CPU contracts for ordered feedback and a differentiable representation step."""

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.context_vector_20261030.learning import (
    ContextTarget,
    ordered_context,
    representation_step,
)


def context_inputs(batch=2):
    h_cls = torch.arange(batch * 64, dtype=torch.float64).reshape(batch, 64) / 100
    h_last = h_cls + .5
    residual = torch.arange(batch * 25 * 23, dtype=torch.float64).reshape(batch, 25, 23) / 100
    available = torch.ones_like(residual, dtype=torch.bool)
    return h_cls, h_last, residual, available


def test_context_order_masked_means_and_proportions():
    h_cls, h_last, residual, available = context_inputs()
    available[0, :5, 0] = False
    available[0, :2, 1] = False
    residual[~available] = float('nan')
    context, support = ordered_context(h_cls, h_last, residual, available)
    assert context.shape == (2, 358) and context.dtype == h_cls.dtype
    assert support.shape == (2,) and support.dtype == torch.bool and support.all()
    torch.testing.assert_close(context[:, :64], h_cls, rtol=0, atol=0)
    torch.testing.assert_close(context[:, 64:128], h_last, rtol=0, atol=0)
    blocks = context[:, 128:].reshape(2, 5, 46)
    for b in range(2):
        for segment in range(5):
            for channel in range(23):
                mask = available[b, 5 * segment:5 * segment + 5, channel]
                values = residual[b, 5 * segment:5 * segment + 5, channel][mask]
                mean = float(values.mean()) if len(values) else 0.
                assert blocks[b, segment, channel] == pytest.approx(mean)
                assert blocks[b, segment, 23 + channel] == pytest.approx(float(mask.sum()) / 5)
    reversed_context, _ = ordered_context(h_cls, h_last, residual.flip(1), available.flip(1))
    assert not torch.equal(context[:, 128:], reversed_context[:, 128:])
    torch.testing.assert_close(reversed_context[:, 128:].reshape(2, 5, 46), blocks.flip(1))


def test_all_missing_feedback_is_zero_and_never_activates_target():
    h_cls, h_last, residual, available = context_inputs()
    residual.fill_(float('nan'))
    available.fill_(False)
    context, support = ordered_context(h_cls, h_last, residual, available)
    assert not support.any() and torch.isfinite(context).all()
    assert torch.count_nonzero(context[:, 128:]) == 0
    target = ContextTarget().double()
    with torch.no_grad():
        target.network[3].bias.fill_(.7)
    assert torch.count_nonzero(target(context, support)) == 0


def test_unavailable_nonfinite_values_are_masked_without_mutation():
    args = list(context_inputs())
    args[3][:, :3, :3] = False
    args[2][:, 0, :3] = float('nan')
    args[2][:, 1, :3] = float('inf')
    args[2][:, 2, :3] = -float('inf')
    originals = [v.clone() for v in args]
    context, _ = ordered_context(*args)
    assert torch.isfinite(context).all()
    for value, original in zip(args, originals):
        torch.testing.assert_close(value, original, equal_nan=True, rtol=0, atol=0)


def test_context_and_target_cannot_backpropagate_into_observations():
    args = list(context_inputs())
    for value in args[:3]:
        value.requires_grad_()
    context, support = ordered_context(*args)
    assert not context.requires_grad and context.grad_fn is None
    # Check the target boundary independently; nonzero weights make detachment observable.
    supplied_context = context.clone().requires_grad_()
    target = ContextTarget().double()
    with torch.no_grad():
        target.network[3].weight.fill_(.1)
    target(supplied_context, support).sum().backward()
    assert supplied_context.grad is None
    assert all(value.grad is None for value in args[:3])
    assert target.network[3].bias.grad.abs().sum() > 0


def test_target_architecture_zero_initialization_and_support_gate():
    target = ContextTarget().double()
    assert [type(module) for module in target.network] == [
        nn.LayerNorm, nn.Linear, nn.SiLU, nn.Linear, nn.Tanh]
    assert target.network[0].normalized_shape == (358,)
    assert (target.network[1].in_features, target.network[1].out_features) == (358, 32)
    assert (target.network[3].in_features, target.network[3].out_features) == (32, 64)
    assert torch.count_nonzero(target.network[3].weight) == 0
    assert torch.count_nonzero(target.network[3].bias) == 0
    context, _ = ordered_context(*context_inputs())
    support = torch.tensor([True, False])
    assert torch.equal(target(context, support), torch.zeros((2, 64), dtype=torch.float64))
    with torch.no_grad():
        target.network[3].bias.fill_(.7)
    result = target(context, support)
    torch.testing.assert_close(result[0], torch.full((64,), .7, dtype=torch.float64).tanh())
    assert torch.count_nonzero(result[1]) == 0


class AnalyticModel(nn.Module):
    """h(s) = bias + scale * s[:64], independently for each batch row."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(.5, 1.5, 64, dtype=torch.float64), requires_grad=False)
        self.bias = nn.Parameter(torch.linspace(-.2, .2, 64, dtype=torch.float64), requires_grad=False)

    def forward(self, state):
        return self.bias + self.scale * state[:, :64]


def test_zero_initial_step_has_analytic_nonzero_outer_parameter_gradient():
    torch.manual_seed(41)
    model = AnalyticModel()
    source_before = {name: value.clone() for name, value in model.state_dict().items()}
    target = ContextTarget().double()
    context, support = ordered_context(*context_inputs())
    context.requires_grad_()
    delta = target(context, support)
    state = torch.zeros((2, 8192), dtype=torch.float64, requires_grad=True)
    h = model(state)
    reference = h.detach().clone().requires_grad_()
    update = representation_step(h, reference, delta, state, create_graph=True)
    assert update.requires_grad and torch.equal(update, torch.zeros_like(state))
    assert torch.count_nonzero(state) == 0
    desired = torch.full_like(h, .9)
    outer = .5 * (model(update) - desired).square().sum()
    outer.backward()
    expected = (.1 * model.scale.square() * (model.bias - desired)).sum(0)
    torch.testing.assert_close(target.network[3].bias.grad, expected, rtol=1e-12, atol=1e-12)
    assert target.network[3].weight.grad.abs().sum() > 0
    assert torch.isfinite(target.network[3].weight.grad).all()
    assert context.grad is None and reference.grad is None
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, source_before[name]) for name, value in model.state_dict().items())


def test_step_is_negative_gradient_not_normalized_accumulated_or_batch_averaged():
    state = torch.full((3, 8192), .02, dtype=torch.float64, requires_grad=True)
    reference = torch.zeros((3, 64), dtype=torch.float64, requires_grad=True)
    delta = torch.full_like(reference, .01, requires_grad=True)
    update = representation_step(state[:, :64] * 2, reference, delta, state)
    expected = torch.zeros_like(state)
    expected[:, :64] = -.006
    torch.testing.assert_close(update, expected, rtol=1e-12, atol=1e-12)
    assert not update.requires_grad
    assert torch.equal(state, torch.full_like(state, .02))
    assert state.grad is None and reference.grad is None and delta.grad is None


def test_per_sample_radius_zero_gradient_and_batch_independence():
    state = torch.zeros((3, 8192), dtype=torch.float64, requires_grad=True)
    delta = torch.zeros((3, 64), dtype=torch.float64)
    delta[0, 0] = .2
    delta[1, :2] = torch.tensor([3., 4.])
    delta.requires_grad_()
    update = representation_step(state[:, :64], torch.zeros_like(delta), delta, state,
                                 create_graph=True)
    expected = torch.zeros_like(state)
    expected[0, 0] = .02
    expected[1, :2] = torch.tensor([.06, .08], dtype=torch.float64)
    torch.testing.assert_close(update, expected, rtol=1e-12, atol=1e-12)
    assert bool((update.norm(dim=1) <= .1 + 1e-15).all())
    gradient, = torch.autograd.grad(update[0].sum(), delta)
    torch.testing.assert_close(gradient[0], torch.full_like(gradient[0], .1))
    assert torch.count_nonzero(gradient[1:]) == 0


@pytest.mark.parametrize('eta,radius', [(0., .1), (.1, 0.), (0., 0.)])
def test_zero_step_or_radius_is_finite_and_exact_zero(eta, radius):
    state = torch.zeros((1, 8192), dtype=torch.float64, requires_grad=True)
    delta = torch.ones((1, 64), dtype=torch.float64, requires_grad=True)
    update = representation_step(state[:, :64], torch.zeros_like(delta), delta, state,
                                 eta=eta, radius=radius, create_graph=True)
    assert torch.equal(update, torch.zeros_like(state))
    gradient, = torch.autograd.grad(update.sum(), delta)
    assert torch.equal(gradient, torch.zeros_like(delta))


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_available_nonfinite_residual_and_nonfinite_representation_are_rejected(bad_value):
    args = list(context_inputs())
    args[2][0, 0, 0] = bad_value
    with pytest.raises(ValueError, match='finite'):
        ordered_context(*args)
    args = list(context_inputs())
    args[0][0, 0] = bad_value
    with pytest.raises(ValueError, match='finite'):
        ordered_context(*args)


@pytest.mark.parametrize('defect', ['h_shape', 'residual_shape', 'mask_shape', 'mask_dtype',
                                   'residual_dtype', 'mixed_dtype'])
def test_context_rejects_incompatible_inputs(defect):
    args = list(context_inputs())
    if defect == 'h_shape':
        args[1] = args[1][:, :63]
    elif defect == 'residual_shape':
        args[2] = args[2][:, :24]
    elif defect == 'mask_shape':
        args[3] = args[3][:1]
    elif defect == 'mask_dtype':
        args[3] = args[3].float()
    elif defect == 'residual_dtype':
        args[2] = args[2].long()
    else:
        args[1] = args[1].float()
    with pytest.raises((TypeError, ValueError)):
        ordered_context(*args)


@pytest.mark.parametrize('defect', ['context_shape', 'nonfinite', 'integer', 'support_shape',
                                   'support_dtype'])
def test_target_rejects_incompatible_inputs(defect):
    context = torch.zeros((2, 358))
    support = torch.ones(2, dtype=torch.bool)
    if defect == 'context_shape':
        context = context[:, :357]
    elif defect == 'nonfinite':
        context[0, 0] = float('nan')
    elif defect == 'integer':
        context = context.long()
    elif defect == 'support_shape':
        support = support[:, None]
    else:
        support = support.float()
    with pytest.raises((TypeError, ValueError)):
        ContextTarget()(context, support)


@pytest.mark.parametrize('defect', ['state_shape', 'h_shape', 'delta_shape', 'reference_shape',
                                   'state_no_grad', 'mixed_dtype', 'state_nan', 'h_inf',
                                   'reference_nan', 'delta_inf'])
def test_step_rejects_invalid_tensor_boundaries(defect):
    state = torch.zeros((2, 8192), dtype=torch.float64, requires_grad=True)
    h = state[:, :64] + 1
    reference, delta = torch.zeros_like(h), torch.zeros_like(h)
    if defect == 'state_shape':
        state = state[:, :8191]
    elif defect == 'h_shape':
        h = h[:, :63]
    elif defect == 'delta_shape':
        delta = delta[:1]
    elif defect == 'reference_shape':
        reference = reference[:, :63]
    elif defect == 'state_no_grad':
        state = state.detach()
    elif defect == 'mixed_dtype':
        delta = delta.float()
    elif defect == 'state_nan':
        with torch.no_grad():
            state[0, 0] = float('nan')
    elif defect == 'h_inf':
        h = h * float('inf')
    elif defect == 'reference_nan':
        reference[0, 0] = float('nan')
    else:
        delta[0, 0] = float('inf')
    with pytest.raises((TypeError, ValueError)):
        representation_step(h, reference, delta, state)


@pytest.mark.parametrize('name,value', [('eta', -1.), ('radius', -1.), ('eta', float('nan')),
                                       ('radius', float('inf')), ('eta', True), ('radius', '.1')])
def test_step_rejects_invalid_hyperparameters(name, value):
    state = torch.zeros((1, 8192), requires_grad=True)
    h = state[:, :64]
    with pytest.raises((TypeError, ValueError)):
        representation_step(h, torch.zeros_like(h), torch.zeros_like(h), state, **{name: value})
