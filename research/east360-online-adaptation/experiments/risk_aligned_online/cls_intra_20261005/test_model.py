"""CPU contracts for the CLS-only readout and the frozen source-coordinate risk."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
from experiments.risk_aligned_online.cls_intra_20261005.model import ClsModel


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32', 'float64'])
def case(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                        torch.linspace(-2., 3., 23), torch.logspace(-2., 1., 23))
        base = base.to(device='cpu', dtype=request.param)
        # A learned nonzero origin catches omission of the final source affine.
        with torch.no_grad():
            base.fast_origin.copy_(torch.linspace(-.7, .8, 128))
        model = ClsModel(base)
        values = model.center + model.scale * torch.randn(2, 128, 23, dtype=request.param)
        available = torch.rand(values.shape) > .2
        try:
            yield model, values, available
        finally:
            torch.set_num_threads(threads)


def source_coordinates(model, values, available):
    observed = available & torch.isfinite(values)
    clean = torch.where(observed, values, model.center)
    return torch.asinh((clean - model.center) / model.scale), observed


def assert_source_frozen(model):
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())


def test_constructor_reuses_source_without_changing_any_tensor(case):
    model, _, _ = case
    source = deepcopy(model.base).train().requires_grad_(True)
    state = {name: value.clone() for name, value in source.state_dict().items()}
    parameters = dict(source.named_parameters())
    wrapped = ClsModel(source)
    assert wrapped.base is source
    assert set(dict(wrapped.named_children())) == {'base', 'decoder'}
    assert {name for name, p in wrapped.named_parameters() if p.requires_grad} == {
        'decoder.weight', 'decoder.bias'}
    assert not wrapped.training and not wrapped.decoder.training
    assert_source_frozen(wrapped)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(p is parameters[name] for name, p in source.named_parameters())


@pytest.mark.parametrize('magnitude', [0., .02])
def test_risk_matches_original_ffn_and_preserves_source(case, magnitude):
    model, values, available = case
    original = FfnModel(deepcopy(model.base))
    state = {name: value.clone() for name, value in model.base.state_dict().items()}
    parameters = dict(model.base.named_parameters())
    delta = model.initial_delta() + magnitude * torch.randn_like(model.initial_delta())
    actual = model(values, available, delta)
    expected = original(values, available, delta)

    torch.testing.assert_close(actual['logit'], expected['logit'], rtol=0, atol=0)
    assert set(actual) == {'logit', 'forecast', 'cls'}
    assert actual['logit'].shape == (2,)
    assert actual['forecast'].shape == (2, 25, 23)
    assert actual['cls'].shape == (2, 64)
    assert type(model.base) is Simple23
    assert model.delta_count == 16576
    assert model.parameter_names == original.parameter_names
    for name, value in model.base.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(p is parameters[name] for name, p in model.base.named_parameters())
    assert_source_frozen(model)


def test_cls_is_the_actual_risk_head_and_decoder_input(case):
    model, values, available = case
    risk_inputs, forecast_inputs = [], []
    risk_hook = model.base.risk_head.register_forward_pre_hook(
        lambda module, args: risk_inputs.append(args[0]))
    forecast_hook = model.decoder.register_forward_pre_hook(
        lambda module, args: forecast_inputs.append(args[0]))
    try:
        out = model(values, available, model.initial_delta())
    finally:
        risk_hook.remove()
        forecast_hook.remove()
    assert len(risk_inputs) == len(forecast_inputs) == 1
    assert out['cls'] is risk_inputs[0] is forecast_inputs[0]
    expected = model.base.risk_head(out['cls']).squeeze(-1)
    torch.testing.assert_close(out['logit'], expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(out['logit'].sum(), out['cls'])
    torch.testing.assert_close(gradient, model.base.risk_head.weight.expand(2, -1),
                               rtol=0, atol=0)
    assert type(model.decoder) is nn.Linear
    assert (model.decoder.in_features, model.decoder.out_features) == (64, 25 * 23)
    assert sum(p.numel() for p in model.decoder.parameters()) == 37375
    assert model.decoder.weight.dtype == model.center.dtype
    assert model.decoder.weight.device == model.center.device


def test_decoder_changes_neither_risk_nor_cls(case):
    model, values, available = case
    delta = model.initial_delta() + .01
    before = model(values, available, delta)
    z, observed = source_coordinates(model, values, available)
    static = model.risk_from_source_coordinates(z, observed)
    with torch.no_grad():
        model.decoder.weight.add_(.5)
        model.decoder.bias.add_(3.)
    after = model(values, available, delta)
    for key in ('logit', 'cls'):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    assert not torch.equal(before['forecast'], after['forecast'])
    torch.testing.assert_close(model.risk_from_source_coordinates(z, observed), static,
                               rtol=0, atol=0)
    risk_grads = torch.autograd.grad(after['logit'].sum(), tuple(model.decoder.parameters()),
                                     allow_unused=True)
    assert risk_grads == (None, None)


@pytest.mark.parametrize('mask_case', ['complete', 'channel_patch', 'nonfinite',
                                      'left_padding', 'all_missing'])
def test_raw_and_coordinate_risk_and_input_gradients_match(case, mask_case):
    model, values, available = case
    available = torch.ones_like(available)
    if mask_case == 'channel_patch':
        available[:, :, 5] = False
        available[:, 32:40] = False
        available[0, 13, 9] = False
    elif mask_case == 'nonfinite':
        values[0, 10, 0] = float('nan')
        values[0, 11, 1] = float('inf')
        values[1, 12, 2] = -float('inf')
        available[1, 13, 3] = False
    elif mask_case == 'left_padding':
        available[0, :120] = False
        available[1, :123] = False
    elif mask_case == 'all_missing':
        available.zero_()
    values.requires_grad_()
    z, observed = source_coordinates(model, values, available)
    expected = FfnModel(deepcopy(model.base))(values, available, model.initial_delta())['logit']
    actual = model.risk_from_source_coordinates(z, observed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    raw_gradient, = torch.autograd.grad(expected.sum(), values)
    z_gradient, = torch.autograd.grad(actual.sum(), z, retain_graph=True)
    input_gradient, = torch.autograd.grad(actual.sum(), values)
    torch.testing.assert_close(input_gradient, raw_gradient, rtol=0, atol=0)
    assert torch.isfinite(z_gradient).all() and torch.isfinite(input_gradient).all()
    assert not z_gradient[~observed].any()
    assert not input_gradient[~observed].any()
    if mask_case != 'all_missing':
        assert input_gradient[observed].norm() > 0
    assert_source_frozen(model)


def test_nonfinite_coordinates_follow_raw_missing_value_contract(case):
    model, values, available = case
    available.fill_(True)
    values[0, 2, 3] = float('nan')
    values[0, 3, 4] = float('inf')
    values[1, 4, 5] = -float('inf')
    available[1, :17] = False
    z = torch.asinh((values - model.center) / model.scale).requires_grad_()
    expected = FfnModel(deepcopy(model.base))(values, available, model.initial_delta())['logit']
    actual = model.risk_from_source_coordinates(z, available)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), z)
    assert torch.isfinite(gradient).all()
    assert not gradient[~(available & torch.isfinite(z))].any()


def test_coordinate_entry_has_no_inverse_or_renormalization(case, monkeypatch):
    model, values, available = case
    z = torch.full_like(values, 120., requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('already in source coordinates')

    monkeypatch.setattr(torch, 'sinh', forbidden)
    monkeypatch.setattr(torch.Tensor, 'sinh', forbidden)
    monkeypatch.setattr(torch, 'asinh', forbidden)
    monkeypatch.setattr(torch.Tensor, 'asinh', forbidden)
    risk = model.risk_from_source_coordinates(z, available)
    gradient, = torch.autograd.grad(risk.sum(), z)
    assert torch.isfinite(risk).all() and torch.isfinite(gradient).all()


def test_forecast_gradients_reach_decoder_and_external_delta_only(case):
    model, values, available = case
    delta = model.initial_delta()
    forecast = model(values, available, delta)['forecast']
    forecast.square().mean().backward()
    for parameter in (delta, *model.decoder.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert_source_frozen(model)
    assert not model.initial_delta().any()
    assert delta.is_leaf and delta.shape == (16576,)
    assert not any('delta' in name for name in model.state_dict())


@pytest.mark.parametrize('task_metric', [False, True], ids=['signal', 'source_risk'])
def test_second_order_outer_risk_reaches_decoder_through_delta(case, task_metric):
    model, values, available = case
    delta = model.initial_delta()
    forecast = model(values, available, delta)['forecast']
    if task_metric:
        z, observed = source_coordinates(model, values, available)
        predicted_window = torch.cat((z[:, 25:], forecast), dim=1)
        arrived_mask = torch.cat((observed[:, 25:], torch.ones_like(available[:, :25])), dim=1)
        inner = model.risk_from_source_coordinates(predicted_window, arrived_mask).square().mean()
    else:
        inner = forecast.square().mean()
    gradient, = torch.autograd.grad(inner, delta, create_graph=True)
    updated = delta - .1 * gradient
    outer = model(values * .9, available, updated)['logit'].square().mean()
    meta_gradients = torch.autograd.grad(outer, tuple(model.decoder.parameters()))
    for derivative in (gradient, *meta_gradients):
        assert torch.isfinite(derivative).all() and derivative.norm() > 0
    assert_source_frozen(model)


def test_coordinate_input_double_backward(case):
    model, values, available = case
    z, observed = source_coordinates(model, values, available)
    z.requires_grad_()
    risk = model.risk_from_source_coordinates(z, observed)
    first, = torch.autograd.grad(risk.square().sum(), z, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), z)
    assert torch.isfinite(second).all() and second.norm() > 0
    assert not second[~observed].any()
    assert_source_frozen(model)


def test_train_keeps_original_source_frozen_and_in_eval(case):
    model, values, available = case
    base = model.base
    model.train()
    assert model.training and model.decoder.training
    assert all(not module.training for module in base.modules())
    assert_source_frozen(model)
    model.requires_grad_(False)
    delta = model.initial_delta()
    gradient, = torch.autograd.grad(model(values, available, delta)['forecast'].sum(), delta)
    assert gradient.norm() > 0
    assert all(not p.requires_grad for p in model.parameters())
    model.decoder.requires_grad_(True)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        'decoder.weight', 'decoder.bias'}
    assert_source_frozen(model)
    assert model.base is base
    assert model.cfg is base.cfg and model.center is base.center and model.scale is base.scale


@pytest.mark.parametrize('entry', ['raw', 'coordinates'])
def test_early_attention_always_uses_differentiable_dispatch(case, entry):
    model, values, available = case
    seen = []
    hook = model.base.encoder.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].requires_grad))
    try:
        if entry == 'raw':
            model(values, available, model.initial_delta())
        else:
            z, observed = source_coordinates(model, values, available)
            model.risk_from_source_coordinates(z, observed)
    finally:
        hook.remove()
    assert seen == [True]


@pytest.mark.parametrize('entry', ['raw', 'coordinates'])
@pytest.mark.parametrize('invalid', ['rank', 'window', 'channels', 'mask_shape', 'mask_dtype'])
def test_invalid_input_shapes_and_masks(case, entry, invalid):
    model, values, available = case
    if invalid == 'rank':
        values = values[0]
    elif invalid == 'window':
        values = values[:, :-1]
    elif invalid == 'channels':
        values = values[:, :, :-1]
    elif invalid == 'mask_shape':
        available = available[:1]
    else:
        available = available.to(values.dtype)
    with pytest.raises(ValueError):
        if entry == 'raw':
            model(values, available, model.initial_delta())
        else:
            model.risk_from_source_coordinates(values, available)


@pytest.mark.parametrize('invalid', ['batch', 'short', 'dtype', 'nan', 'inf'])
def test_invalid_delta_rejected(case, invalid):
    model, values, available = case
    delta = model.initial_delta().detach()
    if invalid == 'batch':
        delta = delta.expand(2, -1)
    elif invalid == 'short':
        delta = delta[:-1]
    elif invalid == 'dtype':
        delta = delta.to(torch.float32 if delta.dtype == torch.float64 else torch.float64)
    else:
        delta[0] = float(invalid)
    with pytest.raises(ValueError, match='delta'):
        model(values, available, delta)


@pytest.mark.parametrize('entry', ['raw', 'coordinates'])
@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_disabled_autograd_is_explicitly_rejected(case, entry, context):
    model, values, available = case
    delta = model.initial_delta()
    with context(), pytest.raises(RuntimeError, match='autograd'):
        if entry == 'raw':
            model(values, available, delta)
        else:
            model.risk_from_source_coordinates(values, available)


@pytest.mark.parametrize('failure_site', ['encoder', 'risk_head', 'decoder'])
def test_failed_forward_restores_source_parameters_and_temporary_hooks(case, failure_site):
    model, values, available = case
    parameters = dict(model.base.named_parameters())
    hooks = dict(model.base.risk_head._forward_pre_hooks)

    def fail(*args):
        raise RuntimeError('intentional source failure')

    module = model.decoder if failure_site == 'decoder' else getattr(model.base, failure_site)
    handle = module.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='intentional source failure'):
            model(values, available, model.initial_delta() + .01)
    finally:
        handle.remove()
    assert dict(model.base.risk_head._forward_pre_hooks) == hooks
    assert all(p is parameters[name] for name, p in model.base.named_parameters())
    model(values, available, model.initial_delta())
    assert dict(model.base.risk_head._forward_pre_hooks) == hooks


@pytest.mark.parametrize('configuration', [dict(fast_site='output'), dict(input_view='window'),
                                           dict(dim=32), dict(window=64), dict(horizon=5)])
def test_unqualified_source_rejected(configuration):
    config = dict(fast_site='tokens', input_view='source')
    config.update(configuration)
    base = Simple23(Config(**config), torch.zeros(23), torch.ones(23))
    with pytest.raises(ValueError, match='qualified'):
        ClsModel(base)


def test_non_simple_source_rejected():
    with pytest.raises(ValueError, match='qualified'):
        ClsModel(nn.Linear(64, 1))
