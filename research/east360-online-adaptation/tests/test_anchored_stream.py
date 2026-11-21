"""CPU-only contracts for the signal-only, two-clock anchored stream."""

import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.anchored_stream_20261031 import stream
from experiments.risk_aligned_online.anchored_stream_20261031 import core
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import batch


class ToyNet:
    def __init__(self):
        self.base = type('Base', (), {'center': torch.zeros(23)})()


class DifferentiableNet(torch.nn.Module):
    """Small CPU stand-in with a genuine Q/K-state gradient."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.register_buffer('center', torch.zeros(23))
        self.base.register_buffer('scale', torch.ones(23))

    def first(self, x, mask, state, site):
        assert site == 'qk'
        safe = torch.where(mask, x, 0.)
        features = safe.mean(dim=1).repeat(1, 3)[:, :64] * .01
        change = state[:, :64] * (1 + safe[:, -1, :1] / 100)
        return torch.stack((features * .9 + change, features + change), dim=1), None, None

    def forward(self, x, mask, state, slow, site, forecast=False):
        safe = torch.where(mask, x, 0.)
        return dict(logit=safe[:, -1, 0] * .001 + state[:, :64].sum(dim=1))


def signal(size=210):
    values = np.repeat(np.arange(size, dtype=np.float32)[:, None], 23, axis=1)
    return dict(values=values, available=np.ones_like(values, dtype=bool))


def policies():
    def policy(name):
        origins = torch.zeros(3, 128)
        origins[:2, 0] = 4  # exact tie, broken by fit-row order
        feedback = torch.arange(3 * 230, dtype=torch.float32).reshape(3, 230) / 100
        return dict(controller=name, mean_context=torch.zeros(358),
                    scale=dict(hcenter=torch.zeros(128), hscale=torch.full((128,), 2.),
                               change_scale=torch.full((23,), 2.)),
                    donor_origins=origins, donor_feedback=feedback, radius=.1)
    return {name: policy(name) for name in ('origin', 'mature')}


@pytest.fixture
def replay_mocks(monkeypatch):
    events = []

    def forward(net, signal, start, stop, states, slow):
        assert len(signal['values']) == len(signal['available']) == stop
        assert states.shape == (9, 8192)
        assert not states[[0, 3, 5, 6, 7, 8]].any()
        events.append(('score', start, stop))
        return (np.arange(start, stop, dtype=np.float32)[None, :] +
                states[:, :1].numpy())

    def update(net, prefix, origin, slow, policies):
        assert set(prefix) == {'values', 'available'}
        assert len(prefix['values']) == origin + 26
        events.append(('update', origin + 25))
        supported = bool(prefix['available'][origin + 1:origin + 26].any())
        matrices = {}
        norms = {name: 0. for name in stream.NAMES}
        for variant in stream.VARIANTS:
            matrix = torch.zeros(9, 8192)
            if supported:
                matrix[[1, 2, 4], 0] = .05
            matrices[variant] = matrix
            for i, arm in enumerate(stream.ARMS, 1):
                norms[variant + '__' + arm] = float(matrix[i].norm())
        return matrices, dict(state_norm=norms, cos_real={n: None for n in stream.NAMES},
                              max_bound=max(norms.values()), feedback_supported=supported)

    monkeypatch.setattr(stream, '_forward', forward)
    monkeypatch.setattr(stream, 'new_states', update)
    return events


def test_score_before_each_replacement_and_reset_per_shot(replay_mocks):
    shot = signal()
    shot['available'][153:178] = False
    net = ToyNet()
    curves, receipts = stream.predict(net, shot, None, policies())
    assert set(curves) == set(stream.NAMES)
    assert [(r['origin'], r['maturity'], r['first_affected'], r['apply_count'])
            for r in receipts] == [(127, 152, 153, 25), (152, 177, 178, 25),
                                    (177, 202, 203, 7)]
    for maturity in (152, 177, 202):
        index = replay_mocks.index(('update', maturity))
        assert replay_mocks[index - 2:index] == [('score', maturity - 24, maturity + 1)] * 2
    for variant in stream.VARIANTS:
        real = curves[variant + '__real']
        assert real[152] == curves['cold'][152]
        assert real[153] > curves['cold'][153]
        assert real[177] > curves['cold'][177]
        np.testing.assert_array_equal(real[178:203], curves['cold'][178:203])
        np.testing.assert_array_equal(curves[variant + '__mean_feedback'], curves['cold'])
    assert [r['feedback_supported'] for r in receipts] == [True, False, True]
    again, again_receipts = stream.predict(net, shot, None, policies())
    for name in stream.NAMES:
        np.testing.assert_array_equal(again[name], curves[name])
    assert again_receipts == receipts


def test_future_and_suffix_cannot_change_prior_predictions(replay_mocks):
    original = signal()
    changed = copy.deepcopy(original)
    changed['values'][165:] += 1000
    changed['available'][165:] = False
    full, records = stream.predict(ToyNet(), original, None, policies())
    altered, changed_records = stream.predict(ToyNet(), changed, None, policies())
    short, _ = stream.predict(ToyNet(), {k: v[:165] for k, v in original.items()}, None, policies())
    for name in stream.NAMES:
        np.testing.assert_array_equal(full[name][:165], altered[name][:165])
        np.testing.assert_array_equal(full[name][:165], short[name])
    assert records[0] == changed_records[0]


def test_invalid_future_is_not_checked_before_it_arrives(replay_mocks):
    shot = signal(180)
    shot['values'][160, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        stream.predict(ToyNet(), shot, None, policies())
    assert replay_mocks[-3:] == [('score', 128, 153), ('score', 128, 153), ('update', 152)]
    short, _ = stream.predict(ToyNet(), {k: v[:160] for k, v in shot.items()}, None, policies())
    assert len(short['cold']) == 160


def test_direct_update_rejects_invalid_observed_prefix(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('invalid signal must not reach the observation')
    monkeypatch.setattr(stream, 'observation', unexpected)
    prefix = signal(153)
    prefix['values'][10, 2] = np.nan
    with pytest.raises(ValueError, match='finite'):
        stream.new_states(ToyNet(), prefix, 127, None, policies())
    prefix['values'][10, 2] = 0
    prefix['available'] = prefix['available'].astype(int)
    with pytest.raises(ValueError):
        stream.new_states(ToyNet(), prefix, 127, None, policies())


def test_selected_windows_and_feedback_controls_are_causal(monkeypatch):
    shot = signal(178)
    shot['available'][128:133, 0] = False
    shot['available'][138:143, 1] = False
    observed_calls, step_calls = [], []

    def observe(net, prefix, origin):
        assert set(prefix) == {'values', 'available'}
        assert len(prefix['values']) == origin + 26
        observed_calls.append(origin)
        x, mask = batch(prefix, [origin], net.base.center.device)
        context = torch.zeros(1, 358)
        context[0, 0] = 8
        blocks = context[:, 128:].reshape(1, 5, 46)
        for i in range(5):
            fractions = torch.tensor(prefix['available'][origin + 1 + i*5:origin + 6 + i*5]
                                     .mean(axis=0), dtype=torch.float32)
            blocks[0, i, :23] = (i + 1) * 2 * (fractions > 0)
            blocks[0, i, 23:] = fractions
        return dict(x=x, available=mask, context=context,
                    support=torch.tensor([bool(prefix['available'][origin + 1:].any())]))

    def state_for(net, controller, kind, data, context, mean, *, outer=False):
        assert kind == 'anchored' and not outer
        step_calls.append((controller, data['x'].clone(), data['mask'].clone(),
                           context.clone(), data['support'].clone()))
        out = torch.zeros(1, 8192)
        if data['support'][0]:
            out[0, 0] = (context[0, 128] - mean[128]) * .001
            out[0, 1] = (context[0, 174] - mean[174]) * .001
        return out

    monkeypatch.setattr(stream, 'observation', observe)
    monkeypatch.setattr(stream, 'state_for', state_for)
    ps = policies()
    ps['mature']['donor_origins'][0, 0] = 100
    prefix = {k: v[:153] for k, v in shot.items()}
    states, details = stream.new_states(ToyNet(), prefix, 127, None, ps)
    assert observed_calls == [127]  # both variants share the original observation
    assert details['donor_index'] == {'origin': 0, 'mature': 1}
    assert details['feedback_supported'] is True
    assert set(details['state_norm']) == set(details['cos_real']) == set(stream.NAMES)
    assert details['max_bound'] <= .100001
    assert details['cos_real']['cold'] is None and details['cos_real']['identical'] is None

    origin_x, origin_mask = batch(prefix, [127], 'cpu')
    mature_x, mature_mask = batch(prefix, [152], 'cpu')
    for variant, expected_x, expected_mask in (('origin', origin_x, origin_mask),
                                               ('mature', mature_x, mature_mask)):
        calls = [row for row in step_calls if row[0] == variant]
        assert len(calls) == 4
        for _, x, mask, _, support in calls:
            assert torch.equal(x, expected_x) and torch.equal(mask, expected_mask)
            assert support.item()
        real, reverse, mean, donor = [row[3] for row in calls]
        assert real[0, 0] == 4  # policy's fitted scaling is used
        for context in (reverse, mean, donor):
            assert torch.equal(context[:, :128], real[:, :128])
        assert torch.equal(reverse[:, 128:].reshape(1, 5, 46),
                           real[:, 128:].reshape(1, 5, 46).flip(1))
        assert torch.equal(mean[0, 128:], ps[variant]['mean_context'][128:])
        assert torch.equal(donor[0, 128:], ps[variant]['donor_feedback'][details['donor_index'][variant]])
        matrix = states[variant]
        assert torch.count_nonzero(matrix[[0, 3, 5, 6, 7, 8]]) == 0
        assert matrix[1].norm() > 0
        torch.testing.assert_close(matrix[2].norm(), matrix[1].norm())
        torch.testing.assert_close(matrix[4].norm(), matrix[1].norm())
        assert details['cos_real'][variant + '__mean_feedback'] is None

    # Feedback (including its masks) does not participate in donor selection.
    changed = copy.deepcopy(ps)
    changed['origin']['donor_feedback'] += 100
    _, other = stream.new_states(ToyNet(), prefix, 127, None, changed)
    assert other['donor_index'] == details['donor_index']


def test_real_anchored_step_and_padded_forward_on_cpu(monkeypatch):
    from experiments.risk_aligned_online.anchored_feedback_20261030.learning import AnchoredTarget

    net = DifferentiableNet()
    controller = AnchoredTarget().eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(73)
    with torch.no_grad():
        controller.network[3].weight.copy_(torch.randn(controller.network[3].weight.shape,
                                                       generator=generator) * .2)
    ps = policies()
    for policy in ps.values():
        policy['controller'] = controller
    shot = signal(190)
    shot['values'] /= 100
    shot['available'][128:133, 0] = False
    shot['values'][128:133, 0] = np.nan  # missing values are ignored, not treated as observations
    shot['available'][138:143, 1] = False
    shot['available'][127, 2] = False  # an unavailable origin masks this channel's feedback
    shot['available'][153:178] = False  # second update must replace the first with zero
    calls = []
    actual_state_for = stream.state_for

    def record_step(net, controller, kind, data, context, mean, *, outer=False):
        result = actual_state_for(net, controller, kind, data, context, mean, outer=outer)
        calls.append((context.detach().clone(), result.detach().clone()))
        return result

    monkeypatch.setattr(stream, 'state_for', record_step)
    curves, receipts = stream.predict(net, shot, None, ps)
    assert stream.NAMES == core.NAMES
    core.validate_receipts(receipts, len(shot['values']))
    assert [record['feedback_supported'] for record in receipts] == [True, False]
    assert receipts[0]['state_norm']['origin__real'] > 0
    assert receipts[0]['state_norm']['mature__real'] > 0
    assert all(value == 0 for value in receipts[1]['state_norm'].values())
    assert all(value is None for value in receipts[1]['cos_real'].values())
    for name in stream.NAMES:
        np.testing.assert_array_equal(curves[name][178:], curves['cold'][178:])
    np.testing.assert_array_equal(curves['cold'], curves['identical'])
    for variant in stream.VARIANTS:
        np.testing.assert_array_equal(curves[variant + '__mean_feedback'], curves['cold'])
        assert curves[variant + '__real'][152] == curves['cold'][152]
    # The original observation constructs masked 5x46 blocks; reversing
    # moves each block's means and its availability fractions together.
    real, reverse, mean = (calls[index][0][:, 128:].reshape(1, 5, 46) for index in (0, 1, 2))
    assert torch.equal(reverse, real.flip(1))
    assert real[0, 0, 23] == 0 and real[0, 2, 24] == 0
    assert torch.count_nonzero(real[0, :, 25]) == 0  # missing anchor, channel 2
    assert torch.count_nonzero(real[0, :, 33]) == 0  # excluded channel 10
    assert torch.count_nonzero(mean) == 0
    assert torch.count_nonzero(calls[2][1]) == 0
    # This particular next sample is unavailable, so the toy forward masks it.
    expected = torch.sigmoid(calls[0][1][0, :64].sum())
    assert curves['origin__real'][153] == pytest.approx(float(expected), abs=1e-7)
    assert all(parameter.grad is None for parameter in controller.parameters())


@pytest.mark.parametrize('size,applied', [(1, []), (152, []), (153, [0]),
                                         (154, [1]), (178, [25, 0])])
def test_partial_shots(size, applied, replay_mocks):
    curves, receipts = stream.predict(ToyNet(), signal(size), None, policies())
    assert [r['apply_count'] for r in receipts] == applied
    assert all(len(curve) == size for curve in curves.values())


def test_only_signal_fields_finite_observations_and_exact_policies(replay_mocks):
    shot = signal(10)
    with pytest.raises(ValueError):
        stream.predict(ToyNet(), dict(shot, outcome=1), None, policies())
    with pytest.raises(ValueError):
        stream.predict(ToyNet(), {**shot, 'available': shot['available'].astype(int)}, None, policies())
    shot['values'][0, 0] = np.nan
    with pytest.raises(ValueError):
        stream.predict(ToyNet(), shot, None, policies())
    shot['available'][0, 0] = False
    stream.predict(ToyNet(), shot, None, policies())  # missing values may be nonfinite
    for bad in ({'origin': policies()['origin']}, {**policies(), 'extra': policies()['origin']}):
        with pytest.raises(ValueError):
            stream.predict(ToyNet(), shot, None, bad)
