"""CPU-only contract/regression tests for the independent INTRA redesign."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.redesign_20261004 import intra as online


@pytest.fixture(autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class MockContext(nn.Module):
    """Fixed ContextModel API; temporary deltas do not replace its baseline."""

    def __init__(self, *, numerical_trap=False, forecast_kind='normal'):
        super().__init__()
        self.cfg = SimpleNamespace(window=128, horizon=25, dim=64,
                                   fast_site='tokens', input_view='source')
        self.base = nn.Linear(23, 1)
        nn.init.zeros_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        self.base.requires_grad_(False)
        self.persistent = nn.Parameter(torch.tensor(.2))
        self.register_buffer('center', torch.zeros(23))
        self.register_buffer('scale', torch.ones(23))
        self.register_buffer('temporary_zero', torch.zeros(1, 128))
        self.calls, self.loss_calls = [], []
        self.initial_calls = 0
        self.numerical_trap = numerical_trap
        self.forecast_kind = forecast_kind

    def initial_fast(self, batch):
        self.initial_calls += 1
        # Deliberately aliased: the predictor must not mutate this buffer.
        return self.temporary_zero.expand(batch, -1)

    def persistent_parameters(self):
        return [self.persistent]

    def forward(self, values, available, fast=None):
        fast = self.initial_fast(len(values)) if fast is None else fast
        self.calls.append(dict(values=values.detach().clone(),
                               available=available.detach().clone(),
                               fast=fast.detach().clone(),
                               grad=torch.is_grad_enabled(),
                               fast_grad=fast.requires_grad,
                               flags=[p.requires_grad for p in self.parameters()],
                               modes=[m.training for m in self.modules()]))
        clean = torch.where(available & torch.isfinite(values), values, 0.)
        logit = self.base(clean[:, -1]).squeeze(-1) + self.persistent + fast[:, 0]
        forecast = (self.persistent + fast[:, None, :23]).expand(-1, 25, -1)
        if self.numerical_trap and (not torch.is_grad_enabled() or not fast.requires_grad):
            forecast = forecast + 1000.
        if self.forecast_kind == 'nan':
            forecast = forecast * float('nan')
        if self.forecast_kind == 'zero_gradient':
            forecast = self.persistent + fast[:, None, :23].square().expand(-1, 25, -1)
        if self.forecast_kind == 'no_gradient':
            forecast = torch.full((len(values), 25, 23), .2)
        if self.forecast_kind == 'bad_candidate':
            forecast = torch.where(fast[:, None, :23] != 0.,
                                   torch.full_like(forecast, float('nan')), forecast)
        if self.forecast_kind == 'bad_candidate_risk':
            logit = torch.where(fast[:, 0] != 0., torch.full_like(logit, float('nan')), logit)
        if self.forecast_kind == 'nonfinite_gradient':
            forecast = InfiniteGradient.apply(forecast)
        return dict(logit=logit, forecast=forecast)

    def forecast_loss(self, prediction, truth, available):
        self.loss_calls.append((truth.detach().clone(), available.detach().clone()))
        if self.forecast_kind == 'nonfinite_loss':
            return prediction.sum() * float('nan')
        if self.forecast_kind == 'nonfinite_candidate_loss' and bool((prediction != .2).any()):
            return prediction.sum() * float('nan')
        target = torch.asinh((truth.detach() - self.center) / self.scale)
        return F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(
            ~available, 0.).sum() / available.sum().clamp_min(1)


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float('inf'))


class PrefixData:
    def __init__(self):
        self.ends = []

    def window(self, shot, end, length):
        assert len(shot['values']) == end + 1, 'window API saw unarrived signals'
        assert not {'y', 'valid', 'down', 'is_disrupt'} & set(shot)
        self.ends.append(end)
        return simple_data.window(shot, end, length)

    def attach_outcome(self, *args, **kwargs):
        raise AssertionError('INTRA requested a target label')


class OpaqueOutcomeShot(dict):
    forbidden = {'y', 'valid', 'down', 'is_disrupt', 'horizon'}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f'INTRA read outcome field {key}')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f'INTRA read outcome field {key}')
        return super().get(key, default)


def shot(n=80):
    return dict(shot=7, device='EAST', values=np.ones((n, 23), np.float32),
                available=np.ones((n, 23), bool), times=np.arange(n) / 1000.)


def predict(model=None, signals=None, **kwargs):
    return online.predict_shot(model or MockContext(), signals if signals is not None else shot(),
                               intra=True, device='cpu', data_api=PrefixData(), **kwargs)


def test_maturity_seals_every_endpoint_before_update_and_audits_times():
    data, model, signals = PrefixData(), MockContext(), shot()
    # Record times, not an assumed uniform 25 ms conversion.
    signals['times'] = np.cumsum(np.linspace(.001, .002, 80))
    scores, audit = online.predict_shot(model, signals, True, stride=30, data_api=data)
    assert data.ends == list(range(80))
    assert scores.shape == (80,) and scores.dtype == np.float32
    assert audit['attempted_updates'] == audit['accepted_updates'] == 2
    first, second = audit['attempts']
    assert [r['origin_index'] for r in audit['attempts']] == [0, 30]
    assert [r['maturity_index'] for r in audit['attempts']] == [25, 55]
    assert [r['effective_index'] for r in audit['attempts']] == [26, 56]
    assert first['maturity_time'] == signals['times'][25]
    assert first['effective_time'] == signals['times'][26]
    assert first['predictions_sealed_through_index'] == 25
    np.testing.assert_array_equal(scores[:26], np.full(26, scores[0], np.float32))
    assert scores[26] > scores[25] and scores[56] > scores[55]
    assert second['loss_after'] < second['loss_before']
    assert len(first['gradient']) == len(first['fast_before']) == len(first['fast_after']) == 128
    assert first['grad_norm'] > 0 and first['fast_change_norm'] == pytest.approx(.05)
    assert audit['pending'][0]['origin_index'] == 60
    assert audit['pending'][0]['maturity_index'] == 85
    assert audit['pending'][0]['reason'] == 'unmatured'


def test_future_diagnostics_cannot_change_scores_through_maturity():
    left, right = shot(40), shot(40)
    right['values'][1:26] = -1.
    a, aa = predict(signals=left, stride=100)
    b, ba = predict(signals=right, stride=100)
    np.testing.assert_array_equal(a[:26], b[:26])
    assert a[26] > b[26]
    assert aa['attempts'][0]['effective_index'] == ba['attempts'][0]['effective_index'] == 26


def test_default_stride_is_forecast_cadence_not_score_downsampling():
    data, model, signals = PrefixData(), MockContext(), shot(140)
    scores, audit = online.predict_shot(model, signals, True, data_api=data)
    assert len(scores) == 140 and data.ends == list(range(140))
    assert [r['origin_index'] for r in audit['attempts']] == [0, 100]
    assert audit['accepted_updates'] == 2


def test_overlapping_forecasts_mature_once_and_never_use_predicted_inputs():
    signals, model = shot(52), MockContext()
    signals['values'] = np.arange(52 * 23, dtype=np.float32).reshape(52, 23) / 100.
    scores, audit = predict(model, signals, stride=10)
    assert len(scores) == 52
    assert [r['origin_index'] for r in audit['attempts']] == [0, 10, 20]
    assert [r['maturity_index'] for r in audit['attempts']] == [25, 35, 45]
    for call in model.calls:
        # Every supplied token is a real measurement or masked left padding.
        valid_values = call['values'][call['available']].numpy()
        assert np.isin(valid_values, signals['values']).all()
    for record in audit['attempts']:
        assert record['predictions_sealed_through_index'] == record['maturity_index']
        assert record['effective_index'] == record['maturity_index'] + 1


def test_fast_discard_preserves_persistent_baseline_parameters_grads_and_modes():
    model = MockContext()
    model.train()
    model.base.eval()  # Mixed child modes must be restored exactly.
    model.persistent.grad = torch.tensor(3.)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    first, audit = predict(model, stride=30)
    second, _ = predict(model, stride=30)
    np.testing.assert_array_equal(first, second)
    assert model.initial_calls == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [m.training for m in model.modules()] == modes
    assert model.persistent.grad.item() == 3.
    assert all(not any(c['flags']) and not any(c['modes']) for c in model.calls)
    assert audit['model_parameter_change_norm'] == audit['model_buffer_change_norm'] == 0.
    assert audit['fast_discarded'] is True
    assert first[0] == pytest.approx(torch.tensor(.2).sigmoid().item())


def test_outcome_keys_are_allowed_but_are_never_read_or_passed_to_data_api():
    signals = OpaqueOutcomeShot(shot(40), y=object(), valid=object(), down=object(),
                                is_disrupt=object(), horizon=object())
    actual, audit = predict(signals=signals)
    reference, _ = predict(signals=shot(40))
    np.testing.assert_array_equal(actual, reference)
    assert audit['accepted_updates'] == 1


def test_acceptance_uses_identical_autograd_enabled_forward_mode_inside_no_grad():
    model = MockContext(numerical_trap=True)
    with torch.no_grad():
        scores, audit = predict(model, stride=30)
        assert not torch.is_grad_enabled()
    assert audit['accepted_updates'] == 2 and scores[26] > scores[25]
    assert all(c['grad'] and c['fast_grad'] for c in model.calls)


def test_level_objective_is_original_loss_over_all_valid_flat23_channels():
    signals, model = shot(27), MockContext()
    signals['available'][:] = False
    signals['available'][:, 22] = True
    _, audit = predict(model, signals)
    record = audit['attempts'][0]
    assert record['valid_targets'] == 25 and record['accepted']
    expected = F.smooth_l1_loss(torch.tensor(.2), torch.asinh(torch.tensor(1.))).item()
    assert record['loss_before'] == pytest.approx(expected)
    assert record['gradient'][22] != 0. and record['gradient'][0] == 0.
    assert all(int(mask.sum()) == 25 for _, mask in model.loss_calls)


def test_innovation_uses_future_first_differences_and_a_low_weight_origin_interval():
    signals, model = shot(27), MockContext()
    model.center.fill_(2.)
    model.scale.fill_(3.)
    signals['values'][:] = 2.
    signals['values'][0] = 5.
    signals['values'][1:26] = np.arange(1, 26, dtype=np.float32)[:, None] + 2.
    _, audit = predict(model, signals, objective='innovation')
    target = torch.asinh(torch.from_numpy(signals['values'][1:26]) / 3. - 2. / 3.)
    forecast = torch.full((25, 23), .2)
    anchor = torch.asinh(torch.ones(23))
    pairs = F.smooth_l1_loss(torch.diff(forecast, dim=0), torch.diff(target, dim=0))
    first_interval = F.smooth_l1_loss(forecast[0] - anchor, target[0] - anchor)
    expected = pairs + .1 * first_interval
    record = audit['attempts'][0]
    assert record['loss_before'] == pytest.approx(expected.item())
    assert record['valid_innovations'] == 24 * 23
    assert record['valid_origin_intervals'] == 23
    assert record['loss_components_before']['future_first_difference'] == pytest.approx(pairs.item())
    assert record['loss_components_before']['origin_first_interval'] == pytest.approx(first_interval.item())
    assert audit['loss_weights'] == dict(level=0., future_first_difference=1., origin_first_interval=.1)
    # Absolute values/masks are still inputs, not replaced with differences.
    torch.testing.assert_close(model.calls[0]['values'][0, -1], torch.full((23,), 5.))


def test_innovation_only_uses_last_available_causal_origin_and_adjacent_real_pairs():
    signals, model = shot(56), MockContext()
    signals['values'][28] = 3.
    signals['values'][29:31] = 999.
    signals['available'][29:31] = False
    signals['available'][32] = False
    _, audit = predict(model, signals, stride=30, objective='innovation')
    record = audit['attempts'][1]
    assert record['valid_targets'] == 24 * 23
    # At origin 30 the anchor is point 28. Pairs 31->32 and 32->33 are masked.
    assert record['valid_innovations'] == 22 * 23
    assert record['valid_origin_intervals'] == 23
    expected_level = F.smooth_l1_loss(torch.tensor(record['fast_before'][:23]) + .2,
                                    torch.asinh(torch.ones(23))).item()
    assert record['loss_components_before']['level'] == pytest.approx(expected_level)
    assert record['loss_before'] == pytest.approx(
        record['loss_components_before']['future_first_difference']
        + .1 * record['loss_components_before']['origin_first_interval'])


def test_future_difference_component_does_not_duplicate_absolute_level_loss():
    model = MockContext()
    forecast = torch.zeros(1, 25, 23, requires_grad=True)
    truth = torch.arange(25, dtype=torch.float32)[None, :, None].expand(1, -1, 23)
    available = torch.ones_like(truth, dtype=torch.bool)
    anchor = torch.ones(1, 23)
    # No origin anchor: adding a constant domain offset changes level, not the
    # actual within-future difference objective.
    anchor_seen = torch.zeros_like(anchor, dtype=torch.bool)
    a, ac = online._loss(model, forecast, truth, available, 'innovation', anchor, anchor_seen)
    b, bc = online._loss(model, forecast + 2., truth, available, 'innovation', anchor, anchor_seen)
    assert a.item() == b.item()
    assert ac['future_first_difference'] == bc['future_first_difference']
    assert ac['level'] != bc['level']
    assert ac['valid_origin_intervals'] == bc['valid_origin_intervals'] == 0


def test_radius_bounds_only_temporary_delta_not_persistent_baseline():
    _, audit = predict(stride=26, signals=shot(105), radius=.06)
    assert audit['accepted_updates'] >= 2
    for record in audit['attempts']:
        assert np.linalg.norm(record['fast_after']) <= .060001
        assert record['fast_change_norm'] <= .050001


@pytest.mark.parametrize('kind, reason', [('nan', 'nonfinite_forecast'),
                                       ('zero_gradient', 'zero_gradient'),
                                       ('no_gradient', 'no_fast_gradient'),
                                       ('bad_candidate', 'nonfinite_candidate_forecast'),
                                       ('bad_candidate_risk', 'nonfinite_candidate_logit'),
                                       ('nonfinite_gradient', 'nonfinite_gradient'),
                                       ('nonfinite_loss', 'nonfinite_loss'),
                                       ('nonfinite_candidate_loss', 'nonfinite_candidate_loss')])
def test_invalid_or_ineffective_attempts_are_rejected_without_state_change(kind, reason):
    scores, audit = predict(MockContext(forecast_kind=kind), shot(40))
    record = audit['attempts'][0]
    assert audit['accepted_updates'] == 0 and audit['attempted_updates'] == 1
    assert record['reason'] == reason and not record['accepted']
    assert record['fast_change_norm'] == 0. and record['effective_index'] is None
    np.testing.assert_array_equal(scores, np.full(40, scores[0], np.float32))
    import json

    json.dumps(audit, allow_nan=False)


def test_finite_but_overshooting_step_is_not_a_useful_update():
    signals = shot(40)
    signals['values'][:] = np.sinh(.20001)
    scores, audit = predict(signals=signals)
    record = audit['attempts'][0]
    assert not record['accepted'] and record['reason'] == 'loss_not_improved'
    assert record['loss_after'] > record['loss_before']
    np.testing.assert_array_equal(scores, np.full(40, scores[0], np.float32))


def test_unavailable_and_nonfinite_truth_is_never_fabricated_into_supervision():
    signals = shot(55)
    signals['available'][1:26] = False
    signals['values'][1:26] = np.nan
    _, audit = predict(signals=signals)
    assert audit['attempts'][0]['reason'] == 'no_valid_targets'
    assert audit['accepted_updates'] == 0


def test_incomplete_horizon_and_last_endpoint_have_no_premature_effect():
    _, incomplete = predict(signals=shot(25))
    assert incomplete['attempted_updates'] == 0 and len(incomplete['pending']) == 1
    scores, exact = predict(signals=shot(26))
    assert exact['attempted_updates'] == exact['accepted_updates'] == 1
    assert exact['attempts'][0]['effective_index'] is None
    assert exact['attempts'][0]['maturity_index'] == 25
    np.testing.assert_array_equal(scores, np.full(26, scores[0], np.float32))


def test_static_and_empty_shots_return_all_endpoints_without_updates():
    model = MockContext()
    scores, audit = online.predict_shot(model, shot(300), False, data_api=PrefixData())
    assert len(scores) == 300 and audit['attempted_updates'] == audit['accepted_updates'] == 0
    assert not audit['attempts'] and not audit['pending']
    empty, audit = predict(signals=shot(0))
    assert empty.shape == (0,) and audit['score_count'] == 0


@pytest.mark.parametrize('kwargs', [dict(stride=0), dict(stride=1.5), dict(step=0.),
                                  dict(step=float('nan')), dict(radius=-1.),
                                  dict(radius=float('inf')), dict(objective='risk')])
def test_invalid_update_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        predict(**kwargs)


def test_wrong_device_feature_or_fast_contract_is_rejected():
    signals = shot()
    signals['device'] = 'J-TEXT'
    with pytest.raises(ValueError, match='EAST'):
        predict(signals=signals)
    signals = shot()
    signals['values'] = signals['values'][:, :22]
    with pytest.raises(ValueError, match='23'):
        predict(signals=signals)
    model = MockContext()
    model.initial_fast = lambda batch: torch.ones(batch, 128)
    with pytest.raises(ValueError, match='zero'):
        predict(model)


def test_forward_exception_restores_modes_requires_grad_and_existing_grads():
    model = MockContext()
    model.train()
    model.base.eval()
    model.persistent.grad = torch.tensor(9.)
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    def fail(*args, **kwargs):
        raise RuntimeError('mock forward failure')
    model.forward = fail
    with pytest.raises(RuntimeError, match='mock forward failure'):
        predict(model)
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    assert model.persistent.grad.item() == 9.


def test_fixed_public_api():
    import inspect

    assert list(inspect.signature(online.predict_shot).parameters) == [
        'model', 'shot', 'intra', 'device', 'stride', 'step', 'radius', 'objective', 'data_api']


def test_real_context_wrapper_keeps_source_origin_and_persistent_baseline_unchanged():
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.redesign_20261004.model import ContextModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                        torch.zeros(23), torch.ones(23))
    with torch.no_grad():
        base.fast_origin.copy_(torch.linspace(-.2, .3, 128))
    model = ContextModel(base).eval()
    with torch.no_grad():
        model.context_map.weight.fill_(.01)
    signals = shot(36)
    signals['values'] = np.random.default_rng(131).normal(size=(36, 23)).astype(np.float32)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scores, audit = predict(model, signals)
    assert len(scores) == 36 and audit['accepted_updates'] == 1
    record = audit['attempts'][0]
    x, mask = simple_data.window(signals, 0, 128)
    truth = torch.from_numpy(signals['values'][1:26])[None]
    available = torch.from_numpy(signals['available'][1:26])[None]
    for name, key in [('fast_before', 'loss_before'), ('proposed_fast', 'loss_after')]:
        fast = torch.tensor(record[name])[None].requires_grad_()
        out = model(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], fast)
        assert model.forecast_loss(out['forecast'], truth, available).item() == record[key]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_acceptance_path_is_also_enabled_and_restored_inside_inference_mode():
    model = MockContext(numerical_trap=True)
    with torch.inference_mode():
        scores, audit = predict(model, shot(40))
        assert torch.is_inference_mode_enabled()
    assert audit['accepted_updates'] == 1 and scores[26] > scores[25]
    assert all(call['grad'] and call['fast_grad'] for call in model.calls)


@pytest.mark.parametrize('change, match', [
    ('window', 'W128'), ('horizon', 'H25'), ('site', 'tokens'),
    ('scale', 'normalization'), ('mask', 'boolean'), ('times', 'increasing'),
    ('fast_shape', 'zero'), ('fast_nan', 'zero'), ('forward_shape', 'forecast')])
def test_malformed_fixed_contract_is_rejected_without_mutating_parameters(change, match):
    model, signals = MockContext(), shot(40)
    if change == 'window':
        model.cfg.window = 127
    elif change == 'horizon':
        model.cfg.horizon = 24
    elif change == 'site':
        model.cfg.fast_site = 'output'
    elif change == 'scale':
        model.scale[0] = 0.
    elif change == 'mask':
        signals['available'] = np.ones_like(signals['values'])
    elif change == 'times':
        signals['times'][3] = signals['times'][2]
    elif change == 'fast_shape':
        model.initial_fast = lambda batch: torch.zeros(batch, 64)
    elif change == 'fast_nan':
        model.initial_fast = lambda batch: torch.full((batch, 128), float('nan'))
    elif change == 'forward_shape':
        model.forward = lambda x, a, fast=None: dict(logit=torch.zeros(len(x)),
                                                    forecast=torch.zeros(len(x), 24, 23))
    before = [p.detach().clone() for p in model.parameters()]
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    with pytest.raises(ValueError, match=match):
        predict(model, signals)
    for actual, expected in zip(model.parameters(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [m.training for m in model.modules()] == modes


def test_all_attempt_audits_are_strict_json_and_losses_decrease_only_when_accepted():
    import json

    for objective in ('level', 'innovation'):
        _, audit = predict(signals=shot(105), stride=26, objective=objective)
        assert json.loads(json.dumps(audit, allow_nan=False))['loss_weights'] == audit['loss_weights']
        for record in audit['attempts']:
            if record['accepted']:
                assert record['loss_gain'] > 0.
                assert record['loss_after'] < record['loss_before']
                assert record['fast_change_norm'] > 0.
            else:
                assert record['fast_change_norm'] == 0. and record['effective_index'] is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('current_value, direction', [(0., 1.), (.99, -1.)])
@pytest.mark.parametrize('gradient_norm', [0., 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.])
def test_matured_candidate_matches_actual_source_normalized_step_norm_grid(
        dtype, current_value, direction, gradient_norm):
    from experiments.risk_aligned_online.fast_update import normalized_step

    model = MockContext().to(dtype=dtype).eval().requires_grad_(False)
    # Prescribe a finite gradient without adding a large scalar that would
    # erase near-zero derivatives through float32 rounding.
    model.forecast_loss = lambda prediction, truth, available: (
        prediction[:, 0, 0] * (direction * gradient_norm)).sum()
    signals = shot(27)
    values, available = simple_data.window(signals, 0, 128)
    values = torch.as_tensor(values, dtype=dtype)[None]
    available = torch.as_tensor(available)[None]
    current = model.initial_fast(1)
    current[0, 0] = current_value
    current.requires_grad_(True)
    issued = model(values, available, current)['forecast'].detach()
    pending = (online._record(0, signals['times'], current), values, available, issued)
    active, record = online._attempt(model, pending, signals, signals['times'], current,
                                     .05, 1., 'level', 'cpu')
    gradient = torch.as_tensor(record['gradient'], dtype=dtype)[None]
    expected = normalized_step(current.detach(), torch.zeros_like(current), gradient,
                               step=.05, radius=1.)
    if gradient_norm == 0.:
        assert record['reason'] == 'zero_gradient' and record['proposed_fast'] is None
        assert not record['accepted']  # A zero loss/gradient is not a useful update.
        torch.testing.assert_close(active, expected, rtol=0, atol=0)
    else:
        proposed = torch.as_tensor(record['proposed_fast'], dtype=dtype)[None]
        torch.testing.assert_close(proposed, expected, rtol=0, atol=0)
        if record['accepted']:
            torch.testing.assert_close(active, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(active, current, rtol=0, atol=0)
