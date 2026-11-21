"""A matured state loss changes temporary gradients, never risk inference."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_dynamics import (
    EventDynamicsFfnModel, arrived_windows, forecast_windows,
)
from experiments.risk_aligned_online.event_gradient_projection import make_directions
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.redesign_20261004.test_intra import (
    MockContext, OpaqueOutcomeShot, PrefixData, shot,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_model():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    basis = make_directions(torch.randn(6, 192), torch.ones(192), mode='event')
    return EventDynamicsFfnModel(base, basis, torch.zeros(192), torch.ones(192), mode='event').eval()


def test_windows_use_each_direct_subhorizon_and_matching_observed_channels():
    x = torch.arange(128.)[None, :, None].expand(1, 128, 23).clone()
    truth = torch.arange(128., 153.)[None, :, None].expand(1, 25, 23).clone()
    a = torch.ones_like(x, dtype=torch.bool)
    b = torch.ones_like(truth, dtype=torch.bool)
    a[:, -1, 0] = False
    b[:, 5, 1] = False
    values, mask, common = arrived_windows(x, a, truth, b, torch.zeros(23))
    assert values.shape == mask.shape == (5, 128, 23)
    assert not common[..., 0].any() and not common[:, 5, 1].any()
    for j, horizon in enumerate((5, 10, 15, 20, 25)):
        torch.testing.assert_close(values[j, :, 2], torch.arange(horizon, horizon + 128.))
        assert not mask[j, -horizon:, 0].any()
    predicted, p_mask = forecast_windows(x, a, truth.asinh(), common,
                                          torch.zeros(23), torch.ones(23))
    torch.testing.assert_close(predicted, values, rtol=1e-6, atol=1e-5)
    assert torch.equal(p_mask, mask)


def test_risk_and_forecast_are_exactly_original_with_zero_fast_state():
    m = make_model()
    control = DecoderFfnModel(deepcopy(m.base)).eval()
    x = torch.randn(2, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    fast = m.initial_fast(2).requires_grad_()
    for k, v in control(x, a, fast).items():
        torch.testing.assert_close(m(x, a, fast)[k], v, rtol=0, atol=0)


def test_matured_target_is_detached_and_perfect_direct_prediction_has_zero_loss():
    m = make_model().requires_grad_(False)
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    truth = torch.randn(1, 25, 23, requires_grad=True)
    b = torch.ones_like(truth, dtype=torch.bool)
    prediction = truth.detach().asinh().requires_grad_()
    loss, receipt = m.prepare_matured_forecast_loss(prediction.detach(), truth, b, values=x, observed=a)
    error = loss(prediction, truth, b)
    assert float(error.detach()) < 1e-10
    wrong = (prediction.detach() + .4).requires_grad_()
    err = loss(wrong, truth, b)
    err.backward()
    assert float(err.detach()) > 0 and wrong.grad.norm() > 0
    assert truth.grad is None and all(p.grad is None for p in m.parameters())
    assert receipt['horizons'] == [5, 10, 15, 20, 25]
    assert receipt['valid_common_targets'] == 575
    assert not receipt['outcome_access'] and not receipt['predicted_signals_used_for_risk']


def test_actual_fast_parameters_receive_gradient_without_changing_teacher():
    m = make_model().requires_grad_(False)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    x, truth = torch.randn(1, 128, 23), torch.randn(1, 25, 23)
    a, b = torch.ones_like(x, dtype=torch.bool), torch.ones_like(truth, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    out = m(x, a, fast)
    loss, _ = m.prepare_matured_forecast_loss(out['forecast'].detach(), truth, b, values=x, observed=a)
    grad, = torch.autograd.grad(loss(out['forecast'], truth, b), fast)
    assert grad.norm() > 0 and torch.isfinite(grad).all()
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    assert not m.teacher.base.forecast_head._forward_pre_hooks


def test_unavailable_targets_do_not_become_supervision_or_depend_on_hidden_values():
    m = make_model().requires_grad_(False)
    x, truth = torch.randn(1, 128, 23), torch.randn(1, 25, 23)
    a, b = torch.ones_like(x, dtype=torch.bool), torch.zeros_like(truth, dtype=torch.bool)
    forecast = torch.randn_like(truth, requires_grad=True)
    loss, receipt = m.prepare_matured_forecast_loss(forecast.detach(), truth, b, values=x, observed=a)
    assert receipt['valid_common_targets'] == 0
    result = loss(forecast, truth * 1000, b)
    assert float(result.detach()) == 0
    result.backward()
    assert not forecast.grad.any()


class RecordingMatured(MockContext):
    def __init__(self):
        super().__init__()
        self.matured_calls = []

    def prepare_matured_forecast_loss(self, issued, truth, available, *, values, observed):
        self.matured_calls.append((values.detach().clone(), truth.detach().clone()))
        return self.forecast_loss, dict(test_identity=True, outcome_access=False)


def test_runtime_calls_context_hook_at_maturity_without_rewriting_previous_predictions():
    m = RecordingMatured()
    stream = OpaqueOutcomeShot(shot(80))
    scores, audit = predict_shot(m, stream, intra=True, data_api=PrefixData())
    expected, _ = predict_shot(MockContext(), stream, intra=True, data_api=PrefixData())
    np.testing.assert_array_equal(scores, expected)
    assert len(m.matured_calls) == 1
    assert m.matured_calls[0][0].shape == (1, 128, 23)
    torch.testing.assert_close(m.matured_calls[0][1], torch.ones(1, 25, 23))
    r = audit['attempts'][0]
    assert r['matured_objective']['test_identity']
    assert r['predictions_sealed_through_index'] == 25 and r['effective_index'] == 26


def test_causal_stream_is_repeatable_and_discards_all_temporary_changes():
    m = make_model()
    stream = OpaqueOutcomeShot(shot(80))
    stream['values'] = np.random.default_rng(9).normal(size=(80, 23)).astype('float32')
    future = deepcopy(stream)
    future['values'][50:] = 10000.
    first, audit = predict_shot(m, stream, intra=True, data_api=PrefixData())
    changed, _ = predict_shot(m, future, intra=True, data_api=PrefixData())
    repeat, _ = predict_shot(m, stream, intra=True, data_api=PrefixData())
    np.testing.assert_array_equal(first[:50], changed[:50])
    np.testing.assert_array_equal(first, repeat)
    assert audit['fast_discarded'] and audit['model_parameter_change_norm'] == 0
    assert audit['accepted_updates'] == 1
    assert not m.memory.any() and not m.teacher.memory.any()


def test_invalid_dimensions_and_overflow_fail_explicitly():
    x = torch.ones(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    y = torch.ones(1, 25, 23)
    b = torch.ones_like(y, dtype=torch.bool)
    with pytest.raises(ValueError):
        arrived_windows(x[:, :-1], a[:, :-1], y, b, torch.zeros(23))
    with pytest.raises(FloatingPointError, match='overflow'):
        forecast_windows(x, a, y * 1000, b, torch.zeros(23), torch.ones(23))
