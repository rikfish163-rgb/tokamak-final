import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from experiments.risk_aligned_online.anticipatory_readout import future_window, AnticipatoryReadout, complete_context
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot


class Toy(nn.Module):
    adaptation_site = 'test internal scalar; not a qualified source model'
    temporary_dim = 1

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=128, horizon=25, fast_site='tokens')
        self.register_buffer('center', torch.zeros(23))
        self.register_buffer('scale', torch.ones(23))
        self.memory = nn.Parameter(torch.zeros(1))

    def initial_fast(self, batch):
        return self.center.new_zeros(batch, 1)

    def persistent_parameters(self):
        return iter((self.memory,))

    def summarize_fast(self, tensor):
        return tensor.detach().reshape(-1).tolist()

    def forward(self, values, available, fast=None):
        fast = self.initial_fast(len(values)) if fast is None else fast
        level = torch.asinh(values[:, -1, 0]) + self.memory + fast[:, 0]
        return dict(logit=level, forecast=level[:, None, None].expand(-1, 25, 23))

    def forecast_loss(self, prediction, truth, available):
        error = (prediction - torch.asinh(truth)).square()
        return error.masked_fill(~available, 0.).sum()/available.sum().clamp_min(1)


def inputs():
    x = torch.arange(128, dtype=torch.float32)[None, :, None].expand(1, 128, 23).clone()
    return x, torch.ones_like(x, dtype=torch.bool)


def test_forecast_window_uses_103_real_points_and_25_direct_predictions():
    x, a = inputs()
    center = torch.arange(23, dtype=torch.float32)
    scale = torch.arange(1, 24, dtype=torch.float32)
    desired = torch.arange(25, dtype=torch.float32)[None, :, None] + center[None, None]
    prediction = torch.asinh((desired-center)/scale).requires_grad_()
    a[:, -1, 2] = False
    values, mask = future_window(x, a, prediction, center, scale)
    torch.testing.assert_close(values[:, :103], x[:, 25:], rtol=0, atol=0)
    torch.testing.assert_close(values[:, 103:, 0], desired[:, :, 0])
    assert mask[:, 103:, 2].sum() == 0
    assert mask[:, 103:, 0].all()
    values[:, -1, 0].sum().backward()
    assert prediction.grad[0, -1, 0] > 0


def test_persistence_control_repeats_current_observation_not_forecast():
    x, a = inputs()
    prediction = torch.full((1, 25, 23), 4.)
    values, mask = future_window(x, a, prediction, torch.zeros(23), torch.ones(23), mode='hold')
    torch.testing.assert_close(values[:, -25:], x[:, -1:].expand(-1, 25, -1))
    assert mask.all()


def test_readout_keeps_original_direct_forecast_and_has_no_new_parameters():
    internal = Toy()
    model = AnticipatoryReadout(internal)
    x, a = inputs()
    fast = model.initial_fast(1).requires_grad_()
    original = internal(x, a, fast)
    actual = model(x, a, fast)
    torch.testing.assert_close(actual['forecast'], original['forecast'], rtol=0, atol=0)
    torch.testing.assert_close(actual['observed_logit'], original['logit'], rtol=0, atol=0)
    assert {id(p) for p in model.parameters()} == {id(p) for p in internal.parameters()}
    assert model.memory is internal.memory
    assert torch.autograd.grad(actual['logit'].sum(), fast)[0].norm() > 0


@pytest.mark.parametrize('bad', ['nan', 'overflow', 'shape', 'mode'])
def test_invalid_prediction_or_shape_is_rejected_not_clamped(bad):
    x, a = inputs()
    prediction = torch.zeros(1, 25, 23)
    mode = 'forecast'
    if bad == 'nan': prediction[0, 0, 0] = float('nan')
    if bad == 'overflow': prediction.fill_(1000.)
    if bad == 'shape': prediction = prediction[:, :24]
    if bad == 'mode': mode = 'unknown'
    with pytest.raises((ValueError, FloatingPointError)):
        future_window(x, a, prediction, torch.zeros(23), torch.ones(23), mode=mode)


def test_online_readout_cannot_read_unarrived_signals_and_fast_is_discarded():
    model = AnticipatoryReadout(Toy())
    x = np.linspace(1, 2, 61, dtype=np.float32)[:, None].repeat(23, axis=1)
    shot = dict(device='EAST', shot=3, values=x, available=np.ones_like(x, bool),
                times=np.arange(61)/1000.)
    first, audit = predict_shot(model, shot, intra=True)
    future = dict(shot, values=x.copy(), y=np.ones(61), down=.061, is_disrupt=True)
    future['values'][40:] *= 10
    second, _ = predict_shot(model, future, intra=True)
    np.testing.assert_array_equal(first[:40], second[:40])
    assert audit['attempts'][0]['maturity_index'] == 25
    assert audit['attempts'][0]['effective_index'] == 26
    assert audit['fast_discarded'] and model.memory.item() == 0


def test_context_requires_real_rows_not_all_23_channels_or_nonzero_values():
    x = torch.zeros(3, 128, 23)
    a = torch.zeros_like(x, dtype=torch.bool)
    a[:, :, 0] = True
    a[1, 0, 0] = False
    x[2, 31, 0] = float('nan')
    assert complete_context(x, a).tolist() == [True, False, False]


def test_full_context_route_retains_observed_risk_on_short_history():
    internal = Toy()
    model = AnticipatoryReadout(internal, require_full_context=True)
    x, a = inputs()
    x = x.expand(2, -1, -1).clone(); a = a.expand_as(x).clone()
    a[0, :7] = False
    fast = torch.tensor([[.1], [.2]], requires_grad=True)
    original = internal(x, a, fast)
    unconstrained = AnticipatoryReadout(internal)(x, a, fast)
    actual = model(x, a, fast)
    torch.testing.assert_close(actual['logit'][0], original['logit'][0], rtol=0, atol=0)
    torch.testing.assert_close(actual['logit'][1], unconstrained['logit'][1], rtol=0, atol=0)
    torch.testing.assert_close(actual['forecast'], original['forecast'], rtol=0, atol=0)
    assert torch.autograd.grad(actual['logit'].sum(), fast)[0].abs().min() > 0


def test_no_future_pass_is_called_for_a_wholly_short_context():
    internal = Toy()
    model = AnticipatoryReadout(internal, require_full_context=True)
    x, a = inputs(); a[:, 0] = False
    calls = []
    handle = internal.register_forward_hook(lambda *args: calls.append(True))
    try: output = model(x, a)
    finally: handle.remove()
    assert len(calls) == 1
    torch.testing.assert_close(output['logit'], output['observed_logit'], rtol=0, atol=0)
