"""Independent CPU contract tests; no imports from another test module."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.redesign_20261004.intra import _record
from experiments.risk_aligned_online.redesign_20261004.shape_intra import attempt


@pytest.mark.parametrize('requested,actual,expected', [('cuda', 'cuda:0', True),
                                                      ('cuda', 'cuda:1', True),
                                                      ('cuda:1', 'cuda:0', False),
                                                      ('cpu', 'cpu', True),
                                                      ('cuda', 'cpu', False)])
def test_device_alias_check_never_requires_a_gpu(requested, actual, expected):
    from experiments.risk_aligned_online.redesign_20261004.shape_intra import _device_matches
    assert _device_matches(torch.device(requested), torch.device(actual)) is expected


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class ShapeModel(nn.Module):
    """Time-dependent forecast with a separate constant-offset fast coordinate."""
    def __init__(self, kind='normal', dtype=torch.float32):
        super().__init__()
        self.kind, self.calls = kind, []
        self.weight = nn.Parameter(torch.tensor(1., dtype=dtype))
        self.register_buffer('center', torch.linspace(-.2, .2, 23, dtype=dtype))
        self.register_buffer('scale', torch.linspace(.5, 2., 23, dtype=dtype))

    def forward(self, values, observed, fast):
        self.calls.append((values.clone(), observed.clone(), fast.detach().clone(),
                           torch.is_grad_enabled(), fast.requires_grad))
        t = torch.linspace(-1., 1., 25, dtype=fast.dtype)[None, :, None]
        slope = values[:, -1:, :1] + fast[:, None, :1] * self.weight
        if self.kind == 'nonlinear':
            slope = slope + 40 * fast[:, None, :1].square() * (values[:, -1:, :1] < .5)
        forecast = (t * slope + fast[:, None, 1:2]).expand(-1, -1, 23)
        if self.kind == 'constant':
            forecast = fast[:, None, :1].expand(-1, 25, 23)
        if self.kind == 'unused':
            forecast = self.weight.expand(len(values), 25, 23)
        if self.kind == 'nan':
            forecast = forecast * float('nan')
        if self.kind == 'bad_candidate' and bool(fast.detach().any()):
            forecast = forecast * float('nan')
        if self.kind == 'bad_gradient':
            forecast = InfiniteGradient.apply(forecast)
        if self.kind == 'orthogonal' and bool(values[0, -1, 0] < .5):
            forecast = (t * (values[:, -1:, :1] + fast[:, None, 2:3])).expand(-1, -1, 23)
        if self.kind == 'wrong_output':
            forecast = forecast[:, :-1]
        logit = fast[:, 0]
        if self.kind == 'bad_logit':
            logit = logit * float('nan')
        if self.kind == 'bad_candidate_logit' and bool(fast.detach().any()):
            logit = logit * float('nan')
        return dict(logit=logit, forecast=forecast)


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float('inf'))


class Signals(dict):
    def __getitem__(self, key):
        assert key in ('values', 'available'), 'outcome read'
        return super().__getitem__(key)


def case(kind='normal', slopes=(.1, .8), n=132, dtype=torch.float32):
    model = ShapeModel(kind, dtype)
    fast = torch.zeros(1, 128, dtype=dtype, requires_grad=True)
    times = np.cumsum(np.linspace(.001, .002, n))
    raw = model.center.numpy()[None].repeat(n, 0)
    signals = Signals(values=raw, available=np.ones((n, 23), bool), y=object())
    blocks = []
    for origin, slope in zip((0, 100), slopes):
        values = torch.full((1, 128, 23), slope, dtype=dtype)
        observed = torch.ones_like(values, dtype=torch.bool)
        issued = model(values, observed, fast)['forecast'].detach().clone()
        blocks.append((_record(origin, times, fast), values, observed, issued))
    model.calls.clear()
    return model, blocks[1], blocks[0], signals, times, fast


def run(data, **kwargs):
    model, pending, previous, signals, times, fast = data
    return attempt(model, pending, previous, signals, times, fast,
                   kwargs.get('step', .05), kwargs.get('radius', 1.), 'cpu')


def test_accepts_exact_new_gradient_step_and_full_json_audit():
    import json
    data = case()
    updated, record = run(data)
    assert record['accepted'] and record['pair_origins'] == [0, 100]
    assert record['cos'] == pytest.approx(1.)
    grad = torch.tensor(record['gradient'])[None]
    expected = normalized_step(data[-1].detach(), torch.zeros_like(grad), grad)
    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
    assert updated.shape == (1, 128) and updated.requires_grad and updated.is_leaf
    assert record['effective_index'] == 126
    assert record['effective_time'] == data[4][126]
    assert record['predictions_sealed_through_index'] == 125
    assert record['fast_change_norm'] == pytest.approx(.05)
    assert record['gradient'][1] == pytest.approx(0., abs=1e-8)
    assert all(a < b for a, b in zip(record['pair_losses_after'], record['pair_losses_before']))
    for key in ('fast_at_origin', 'fast_before', 'fast_after', 'proposed_fast',
                'issued_loss', 'loss_before', 'loss_after', 'loss_gain',
                'valid_targets', 'valid_innovations', 'valid_origin_intervals',
                'loss_components_before', 'loss_components_after', 'gradient_finite',
                'fast_norm_before', 'fast_norm_after', 'candidate_change_norm',
                'shape_components', 'source_asinh_units'):
        assert key in record
    json.dumps(record, allow_nan=False)


def test_masked_residual_mean_ignores_holes_and_singletons_in_source_units():
    data = case()
    model, pending, _, signals, _, _ = data
    mask = signals['available']
    mask[101:126] = False
    mask[[101, 110, 125], 22] = True
    mask[115, 0] = True  # Singleton channel must not contribute.
    signals['values'][101:126] = np.nan
    z = torch.tensor([.4, -.2, .1])
    signals['values'][[101, 110, 125], 22] = (
        model.center[22] + model.scale[22] * z.sinh()).numpy()
    signals['values'][115, 0] = 99.
    _, record = run(data)
    residual = pending[3][0, [0, 9, 24], 22] - z
    expected = F.smooth_l1_loss(residual - residual.mean(), torch.zeros(3)).item()
    assert record['loss_before'] == pytest.approx(expected)
    assert record['shape_components']['new']['valid_shape_times'] == 3
    assert record['shape_components']['new']['valid_shape_channels'] == 1
    assert record['valid_targets'] == 4
    assert record['gradient'][22] == 0.  # Slope coordinate is fast[0], not a per-channel head.


def test_recomputes_old_and_new_gradients_at_same_current_fast():
    data = case()
    data[-1].data[0, 0] = .02
    _, record = run(data)
    assert record['accepted']
    calls = data[0].calls
    assert len(calls) == 4
    for call in calls[:2]:
        torch.testing.assert_close(call[2], data[-1])
    assert all(call[3] and call[4] for call in calls)
    assert record['loss_before'] != record['issued_loss']


@pytest.mark.parametrize('slopes,reason', [((-.8, .8), 'nonpositive_cosine'),
                                         ((0., .8), 'zero_gradient')])
def test_pair_direction_and_nonzero_requirement(slopes, reason):
    data = case(slopes=slopes)
    updated, record = run(data)
    assert updated is data[-1] and not record['accepted']
    assert record['reason'] == reason


def test_positive_cosine_is_insufficient_when_only_new_loss_improves():
    data = case('nonlinear')
    _, record = run(data)
    assert record['cos'] > 0 and not record['accepted']
    assert record['pair_losses_after'][0] > record['pair_losses_before'][0]
    assert record['pair_losses_after'][1] < record['pair_losses_before'][1]
    assert record['reason'] == 'loss_not_improved'


def test_relative_guard_rejects_tiny_old_gain():
    data = case(slopes=(.025000001, .8))
    _, record = run(data)
    assert not record['accepted'] and record['reason'] == 'loss_not_improved'


@pytest.mark.parametrize('kind,reason', [('constant', 'zero_gradient'),
                                       ('unused', 'no_fast_gradient'),
                                       ('nan', 'nonfinite_forecast'),
                                       ('bad_logit', 'nonfinite_logit'),
                                       ('bad_candidate', 'nonfinite_candidate_forecast'),
                                       ('bad_gradient', 'nonfinite_gradient'),
                                       ('wrong_output', 'invalid_model_output'),
                                       ('bad_candidate_logit', 'nonfinite_candidate_logit')])
def test_invalid_model_evidence_is_explicit(kind, reason):
    data = case() if kind == 'wrong_output' else case(kind)
    data[0].kind = kind
    updated, record = run(data)
    assert updated is data[-1] and record['reason'] == reason
    assert not record['accepted'] and record['pair_reset']


def test_zero_cosine_is_rejected():
    _, record = run(case('orthogonal'))
    assert not record['accepted'] and record['cos'] == 0.
    assert record['reason'] == 'nonpositive_cosine'


def test_previous_holes_reset_pair_and_warmup_requires_valid_new_evidence():
    data = list(case())
    data[3]['available'][1:26] = False
    _, record = run(data)
    assert record['reason'] == 'no_shape_evidence' and record['pair_reset']
    data[2] = None
    _, record = run(data)
    assert record['reason'] == 'needs_previous' and not record['pair_reset']
    data[3]['available'][101:126] = False
    _, record = run(data)
    assert record['reason'] == 'no_shape_evidence' and record['pair_reset']


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_small_nonzero_gradient_keeps_source_epsilon_and_radius(dtype):
    data = case(slopes=(.1, .8), dtype=dtype)
    with torch.no_grad():
        data[0].weight.fill_(1e-8)
        data[-1][0, 3] = .99999
    updated, record = run(data)
    gradient = torch.tensor(record['gradient'], dtype=dtype)[None]
    expected = normalized_step(data[-1].detach(), torch.zeros_like(gradient), gradient)
    torch.testing.assert_close(torch.tensor(record['proposed_fast'], dtype=dtype)[None],
                               expected, rtol=0, atol=0)
    assert record['grad_norm'] > 0
    if record['accepted']:
        assert updated.double().norm() <= 1.000001


def test_channel_constant_offsets_do_not_change_shape_loss_or_gradient():
    data = case(dtype=torch.float64)
    _, before = run(data)
    with torch.no_grad():
        data[-1][0, 1] = .3
    _, after = run(data)
    assert after['loss_before'] == pytest.approx(before['loss_before'], abs=1e-14)
    np.testing.assert_allclose(after['gradient'], before['gradient'], atol=1e-14)


@pytest.mark.parametrize('fault', ['missing_previous', 'overlap', 'mismatch', 'hole',
                                   'unmatured', 'bad_times', 'bad_mask', 'bad_fast',
                                   'bad_scale', 'bad_issued'])
def test_contract_rejections_do_not_mutate_fast(fault):
    data = list(case())
    if fault == 'missing_previous':
        data[2] = None
    elif fault == 'overlap':
        data[2][0].update(origin_index=75, maturity_index=100,
                          origin_time=float(data[4][75]), maturity_time=float(data[4][100]))
    elif fault == 'mismatch':
        data[1] = (data[1][0], data[1][1][:, :-1], *data[1][2:])
    elif fault == 'hole':
        data[3]['available'][101:126] = False
    elif fault == 'unmatured':
        data[4] = data[4][:125]
    elif fault == 'bad_times':
        data[4][120] = data[4][119]
    elif fault == 'bad_mask':
        data[3]['available'] = data[3]['available'].astype(np.int32)
    elif fault == 'bad_fast':
        data[5] = torch.zeros(1, 127, requires_grad=True)
    elif fault == 'bad_scale':
        data[0].scale[0] = 0
    else:
        data[1][3][0, 0, 0] = float('nan')
    snapshot = data[-1].detach().clone()
    updated, record = run(data)
    assert updated is data[-1] and not record['accepted']
    assert record['reason'] not in ('unmatured', 'loss_improved')
    assert record['effective_index'] is None and record['fast_change_norm'] == 0
    torch.testing.assert_close(data[-1], snapshot)


def test_only_own_horizons_are_read_and_future_times_are_not_validated():
    data = case()
    _, left = run(data)
    data[3]['values'][26:101] = float('nan')
    data[3]['values'][126:] = float('nan')
    data[3]['available'][126:] = False
    data[4][127:] = float('nan')
    _, right = run(data)
    assert left == right


def test_state_inputs_records_and_existing_grads_remain_unchanged():
    data = case()
    model = data[0]
    model.weight.grad = torch.tensor(7.)
    data[-1].grad = torch.full_like(data[-1], 3.)
    state = deepcopy(model.state_dict())
    records = deepcopy([data[1][0], data[2][0]])
    inputs = [x.clone() for block in data[1:3] for x in block[1:]]
    flags = [p.requires_grad for p in model.parameters()]
    with torch.inference_mode():
        updated, record = run(data)
    assert record['accepted'] and updated.requires_grad
    for name, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert records == [data[1][0], data[2][0]]
    for old, new in zip(inputs, [x for block in data[1:3] for x in block[1:]]):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert model.weight.grad.item() == 7.
    assert bool((data[-1].grad == 3.).all()) and not bool(data[-1].any())
    assert flags == [p.requires_grad for p in model.parameters()]


@pytest.mark.parametrize('step,radius', [(.1, 1.), (.05, 2.), (float('nan'), 1.)])
def test_bounded_hypothesis_refuses_configuration_drift(step, radius):
    data = case()
    updated, record = run(data, step=step, radius=radius)
    assert updated is data[-1] and record['reason'] == 'invalid_configuration'


def test_no_next_endpoint_is_reported_at_last_maturity():
    data = case(n=126)
    _, record = run(data)
    assert record['accepted'] and record['effective_index'] is None
    assert record['effective_time'] is None
