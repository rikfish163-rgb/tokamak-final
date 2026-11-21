"""CPU regressions for the independent sealed-shot persistent-token updater."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.redesign_20261004 import inter


class ContextStub(nn.Module):
    """Two conditional internal coordinates behind a frozen linear risk head."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=128, dim=64, horizon=25)
        self.memory = nn.Parameter(torch.zeros(2))
        self.base = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(torch.tensor([[1., 1.]]))
        self.base.requires_grad_(False)
        self.register_buffer('center', torch.zeros(23))
        self.register_buffer('scale', torch.ones(23))
        self.fast_seen = []

    def persistent_parameters(self):
        return [self.memory]

    def initial_fast(self, batch):
        # A deliberately nonzero implementation catches forwarding it as fast.
        return torch.ones(batch, 128, device=self.memory.device)

    def forward(self, values, available, fast=None):
        assert fast is not None and not fast.any()
        self.fast_seen.append(fast.detach().clone())
        endpoint = torch.where(available[:, -1], values[:, -1], 0.)
        internal = torch.stack((endpoint[:, 0] + endpoint[:, 1] * self.memory[0],
                                endpoint[:, 2] * self.memory[1]), dim=1)
        return dict(logit=self.base(internal).squeeze(-1),
                    forecast=values.new_zeros(len(values), 25, 23))


def shot(sid, leads=None, *, safe=False, logit=0., context=1., sealed=None):
    leads = np.asarray([200., 125., 80., 31., 30., 10.] if leads is None else leads)
    times = 1. - leads / 1000.
    values = np.zeros((len(times), 23), np.float32)
    values[:, 0] = logit
    values[:, 1] = context
    valid = leads >= 0 if not safe else np.ones(len(times), bool)
    labels = ((leads <= 125.) & valid).astype(np.float32)
    if safe:
        labels[:] = 0.
    scores = np.full(len(times), .5, dtype=np.float32)
    if sealed is not None:
        scores[:] = sealed
    return dict(shot=sid, device='EAST', values=values,
                available=np.ones_like(values, bool), times=times, y=labels,
                valid=valid, is_disrupt=not safe, down=None if safe else 1.,
                horizon=125., _sealed_scores=scores)


def update(model, current, history=(), **kwargs):
    return inter.postshot_update(model, current, history, 'cpu',
                                 np.random.default_rng(13), .5, **kwargs)


def test_fixed_public_api():
    import inspect

    names = list(inspect.signature(inter.postshot_update).parameters)
    assert names == ['model', 'shot', 'replay', 'device', 'rng', 'threshold',
                     'lr', 'radius', 'steps', 'data_api']


def test_label_strata_preserve_125_but_exclude_late_and_early_disruptive():
    current = shot(1, [126., 125., 30.0001, 30., 0., -1.])
    original_y = current['y'].copy()
    result = update(ContextStub(), current, [shot(0, safe=True)], steps=0)
    counts = result['pool_counts']['current']
    assert counts == dict(positive=2, negative=0, excluded_late_positive=2,
                          excluded_early_negative=1)
    assert result['pool_counts']['history']['negative'] == 6
    assert result['sampling']['positive'] == result['sampling']['negative'] == 32
    assert result['sampling']['current_positive'] == 32
    assert result['sampling']['history_positive'] == 0
    np.testing.assert_array_equal(current['y'], original_y)


def test_current_and_history_positives_and_negatives_are_counted():
    result = update(ContextStub(), shot(3),
                    [shot(1), shot(2, safe=True)], steps=0)
    assert result['pool_counts']['current']['positive'] == 3
    assert result['pool_counts']['history']['positive'] == 3
    assert result['sampling']['strategy'] == 'stratified_50_50'
    assert result['sampling']['current_positive'] == 16
    assert result['sampling']['history_positive'] == 16
    assert result['sampling']['history_negative'] == 32
    assert result['sampling']['current_negative'] == 0


def test_stratification_does_not_let_many_safe_shots_swamp_a_positive():
    history = deque([shot(i, safe=True) for i in range(32)], maxlen=32)
    result = update(ContextStub(), shot(100), history, steps=0)
    assert result['pool_counts']['history']['negative'] == 32 * 6
    assert result['sampling']['positive'] == result['sampling']['negative'] == 32


@pytest.mark.parametrize('safe', [False, True])
def test_absent_class_has_no_fabricated_examples(safe):
    result = update(ContextStub(), shot(1, safe=safe))
    assert result['sampling']['strategy'] == 'single_class'
    assert result['sampling']['positive' if safe else 'negative'] == 0
    assert result['guard']['windows'] == 0
    assert result['accepted'] and result['learned_neural_update']


def test_short_disruptive_record_with_no_timely_endpoint_is_explicit():
    result = update(ContextStub(), shot(2, [30., 10., 0.]),
                    [shot(1, [30., 10., 0.])])
    assert not result['accepted'] and result['reason'] == 'no_primary_examples'
    assert result['guard']['no_timely_endpoint_shot_ids'] == [1]
    assert result['guard']['windows'] == 0
    assert result['sampling']['positive'] == result['sampling']['negative'] == 0


def test_only_persistent_parameters_change_and_gradients_modes_are_preserved():
    model = ContextStub().train()
    model.base.eval()
    model.base.weight.requires_grad_(True)  # Prove autograd's selective scope.
    model.base.weight.grad = torch.full_like(model.base.weight, 7.)
    model.memory.grad = torch.full_like(model.memory, 9.)
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items()
              if name != 'memory'}
    result = update(model, shot(1), lr=.1)
    assert result['accepted'] and not torch.equal(model.memory, torch.zeros(2))
    for name, tensor in frozen.items():
        assert torch.equal(model.state_dict()[name], tensor), name
    assert torch.equal(model.base.weight.grad, torch.full_like(model.base.weight, 7.))
    assert torch.equal(model.memory.grad, torch.full_like(model.memory, 9.))
    assert model.training and not model.base.training
    assert result['scope']['parameter_names'] == ['memory']
    assert result['drift']['anchor'] == 'zero'
    assert all(not fast.any() for fast in model.fast_seen)


def test_radius_bounds_total_memory_from_zero_not_each_step():
    model = ContextStub()
    for sid in range(1, 7):
        result = update(model, shot(sid), radius=.04, lr=.5)
        assert model.memory.norm() <= .040001
        assert result['drift']['after'] <= .040001
    assert model.memory.norm() > .03


@pytest.mark.parametrize('failure', ['loss', 'gradient', 'candidate'])
def test_nonfinite_proposals_roll_back(failure, monkeypatch):
    model = ContextStub()
    before = model.memory.detach().clone()
    if failure == 'gradient':
        monkeypatch.setattr(torch.autograd, 'grad',
                            lambda *args, **kwargs: (torch.full_like(model.memory, float('nan')),))
    else:
        forward = model.forward

        def corrupted(*args, **kwargs):
            out = forward(*args, **kwargs)
            if failure == 'loss' or model.memory.detach().norm() > 0:
                out['logit'] = out['logit'] * float('nan')
            return out

        monkeypatch.setattr(model, 'forward', corrupted)
    result = update(model, shot(1))
    assert not result['accepted']
    assert any('nonfinite' in reason for reason in result['reasons'])
    assert not result['learned_neural_update']
    assert torch.equal(model.memory, before)


@pytest.mark.parametrize(('kind', 'expected_reason'), [
    ('timely', 'timely_tp_lost'),
    ('safe', 'new_safe_false_alarm'),
    ('margin', 'historical_margin_degradation'),
])
def test_harmful_history_rejected_even_when_training_bce_improves(kind, expected_reason):
    model = ContextStub()
    if kind == 'safe':
        current = shot(2, logit=-1., context=3.)
        old = shot(1, safe=True, logit=-.005, context=1., sealed=.49)
    else:
        current = shot(2, safe=True, logit=1., context=3.)
        old = shot(1, logit=.01 if kind == 'timely' else 1.,
                   context=1., sealed=.6)
    result = update(model, current, [old], lr=.4)
    assert not result['accepted'] and expected_reason in result['reasons']
    assert result['loss']['candidate_bce'] < result['loss']['before_bce']
    assert torch.equal(model.memory, torch.zeros(2))
    assert result['drift']['candidate'] > 0 and result['drift']['after'] == 0


def test_guard_is_history_only_not_current_or_future_prediction():
    result = update(ContextStub(), shot(2), [shot(1, safe=True)], steps=0)
    assert result['guard']['shot_ids'] == [1]
    assert {row['shot'] for row in result['guard']['endpoints']} == {1}
    assert result['guard']['scope'] == 'sampled_endpoints_fast0'
    assert result['guard']['full_shot_operational'] is False


def test_guard_keeps_sealed_timely_peak_and_safe_hard_negative_peak_bounded():
    old_positive = shot(1, np.linspace(125., 31., 200), sealed=.3)
    old_positive['_sealed_scores'][173] = .9
    old_safe = shot(2, np.linspace(300., 0., 200), safe=True, sealed=.2)
    old_safe['_sealed_scores'][181] = .8
    result = update(ContextStub(), shot(3), [old_positive, old_safe], steps=0)
    identities = {(row['shot'], row['end']) for row in result['guard']['endpoints']}
    assert (1, 173) in identities and (2, 181) in identities
    assert result['guard']['windows'] <= 8
    assert result['retained_error_identities'] == [
        dict(shot=2, end=181, kind='safe_false_alarm')]


def test_cached_peak_selection_does_not_rescan_history_or_run_full_trajectories():
    class CountWindows:
        def __init__(self):
            self.calls = []

        def window(self, current, end, length):
            self.calls.append((current['shot'], end))
            return simple_data.window(current, end, length)

    model, data = ContextStub(), CountWindows()
    history = deque([shot(i, np.linspace(125., 31., 1000)) for i in range(40)], maxlen=32)
    for old in history:
        old['_sealed_scores'][700] = .9
    current = shot(100)
    first = update(model, current, history, steps=0, data_api=data)
    assert first['history']['used'] == 32
    assert first['guard']['windows'] <= 128
    assert len(data.calls) <= 64 + 128
    # Once selected, the exact same retained identities are used again.
    for old in history:
        old['_sealed_scores'][900] = .99
    history.append(current)  # The runner, not the updater, owns the deque.
    second = update(model, shot(101), history, steps=0, data_api=data)
    common = set(first['guard']['shot_ids']) & set(second['guard']['shot_ids'])
    left = [(r['shot'], r['end']) for r in first['guard']['endpoints'] if r['shot'] in common]
    right = [(r['shot'], r['end']) for r in second['guard']['endpoints'] if r['shot'] in common]
    assert left == right
    assert len(model._inter_guard_cache) <= 32


def test_missing_seal_or_outcome_rejected_before_label_use():
    model = ContextStub()
    for key in ('_sealed_scores', 'y', 'is_disrupt'):
        current = shot(1)
        del current[key]
        with pytest.raises(ValueError, match='sealed|revealed|outcome'):
            update(model, current)
    assert not model.fast_seen
    assert not hasattr(model, '_inter_guard_cache')


@pytest.mark.parametrize('change', [dict(device='J-TEXT'), dict(horizon=30.)])
def test_source_online_and_changed_label_horizon_rejected(change):
    current = shot(1)
    current.update(change)
    with pytest.raises(ValueError, match='EAST|125'):
        update(ContextStub(), current)


def test_replay_must_not_include_current_and_is_not_mutated():
    model, current, old = ContextStub(), shot(2), shot(1)
    history = deque([old], maxlen=32)
    update(model, current, history, steps=0)
    assert list(history) == [old]
    with pytest.raises(ValueError, match='current|duplicate'):
        update(model, current, [current])


def test_fixed_threshold_cannot_be_changed_between_updates():
    model = ContextStub()
    update(model, shot(1), steps=0)
    with pytest.raises(ValueError, match='threshold'):
        inter.postshot_update(model, shot(2), [], 'cpu', np.random.default_rng(1), .6)


def test_deterministic_sampling_and_candidate_for_same_seed():
    left, right = ContextStub(), ContextStub()
    history = [shot(1, context=.2), shot(2, safe=True, context=.1)]
    a = update(left, shot(3), history)
    b = update(right, shot(3), history)
    assert a == b
    assert torch.equal(left.memory, right.memory)


def test_sampled_guard_can_miss_an_unsampled_new_safe_alarm():
    model = ContextStub()
    old = shot(1, np.linspace(300., 0., 10), safe=True,
               logit=-2., context=0., sealed=.1)
    old['values'][5, :2] = [-.01, 2.]
    result = update(model, shot(2), [old])
    assert result['accepted']
    assert 5 not in [row['end'] for row in result['guard']['endpoints']]
    x, mask = simple_data.window(old, 5, 128)
    with torch.no_grad():
        risk = model(torch.from_numpy(x[None]), torch.from_numpy(mask[None]),
                     fast=torch.zeros(1, 128))['logit'].sigmoid().item()
    assert risk >= .5  # The full safe trajectory now contains a new false alarm.
    assert result['guard']['full_shot_operational'] is False


def test_nonfinite_candidate_audit_retains_finite_attempted_drift(monkeypatch):
    model = ContextStub()
    forward = model.forward

    def invalid_candidate(*args, **kwargs):
        result = forward(*args, **kwargs)
        if model.memory.detach().norm() > 0:
            result['logit'] = result['logit'] * float('inf')
        return result

    monkeypatch.setattr(model, 'forward', invalid_candidate)
    result = update(model, shot(1))
    assert not result['accepted'] and result['drift']['after'] == 0
    assert result['drift']['candidate'] > 0
    assert result['drift']['candidate_step_l2'] > 0


def test_nonfinite_initial_state_is_not_repaired_and_audit_is_strict_json():
    import json

    model = ContextStub()
    with torch.no_grad():
        model.memory[0] = float('nan')
    result = update(model, shot(1))
    assert not result['accepted'] and torch.isnan(model.memory[0])
    assert result['drift']['after'] is None
    assert result['drift']['step_l2'] is None
    json.dumps(result, allow_nan=False)


def test_runtime_error_after_partial_step_restores_parameters_and_modes(monkeypatch):
    model = ContextStub().train()
    forward = model.forward

    def fail_later(*args, **kwargs):
        if model.memory.detach().norm() > 0:
            raise RuntimeError('deliberate runtime failure')
        return forward(*args, **kwargs)

    monkeypatch.setattr(model, 'forward', fail_later)
    with pytest.raises(RuntimeError, match='deliberate'):
        update(model, shot(1))
    assert torch.equal(model.memory, torch.zeros(2)) and model.training


def test_real_contextmodel_and_simple23_base_update_only_context_map():
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.redesign_20261004.model import ContextModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        base = Simple23(Config(fast_site='tokens'), np.zeros(23), np.ones(23))
        with torch.no_grad():
            base.fast_origin.copy_(torch.linspace(-.01, .01, 128))
        model = ContextModel(base)
        frozen = {name: value.clone() for name, value in base.state_dict().items()}
        result = update(model, shot(1), lr=.1)
    assert result['accepted'] and result['learned_neural_update']
    assert result['scope']['parameter_names'] == ['context_map.weight']
    assert model.context_map.weight.norm() > 0
    for name, value in frozen.items():
        assert torch.equal(base.state_dict()[name], value), name
    assert all(p.grad is None for p in base.parameters())


def test_history_bce_is_checked_separately_from_training_and_risk_margin():
    result = update(ContextStub(), shot(2, safe=True, logit=1., context=3.),
                    [shot(1, logit=-3., context=1., sealed=.8)])
    assert not result['accepted'] and 'historical_bce_degraded' in result['reasons']
    assert result['guard']['max_margin_degradation'] < inter.MAX_RISK_MARGIN_DEGRADATION
    assert result['loss']['candidate_bce'] < result['loss']['before_bce']
