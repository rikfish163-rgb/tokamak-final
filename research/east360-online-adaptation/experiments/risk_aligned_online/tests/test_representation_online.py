"""Regression tests for internal, EAST-only causal adaptation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online import representation_online as online

torch.set_num_threads(1)


def model():
    torch.manual_seed(31)
    return Simple23(Config(window=8, patch=2, dim=8, heads=2, layers=2,
                           horizon=25, fast_site='tokens'), np.zeros(23), np.ones(23)).eval()


def shot(n=90):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(n, 23)).astype(np.float32)
    return dict(shot=1, device='EAST', values=values, available=np.ones_like(values, bool),
                times=np.arange(n)/1000.)


def test_updates_only_last_block_norm_and_restore_after_shot():
    parent = model()
    before = deepcopy(parent.state_dict())
    scores, audit = online.predict_shot(parent, shot(), intra=True, device='cpu', stride=26)
    assert np.isfinite(scores).all()
    assert audit['accepted_updates'] > 0
    assert audit['max_parameter_change'] > 0
    assert audit['adaptation_parameter_count'] == 32
    assert all(name.startswith('encoder.layers.1.norm') for name in audit['adapted_names'])
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(p.requires_grad for p in parent.parameters())
    again, _ = online.predict_shot(parent, shot(), intra=True, device='cpu', stride=26)
    np.testing.assert_array_equal(scores, again)


def test_future_cannot_change_emitted_prefix_and_no_update_before_maturity():
    parent = model()
    s = shot()
    future_changed = deepcopy(s)
    future_changed['values'][51:] *= -100
    a, _ = online.predict_shot(parent, s, intra=True, device='cpu', stride=26)
    b, _ = online.predict_shot(parent, future_changed, intra=True, device='cpu', stride=26)
    np.testing.assert_array_equal(a[:51], b[:51])
    raw, _ = online.predict_shot(parent, s, intra=False, device='cpu')
    np.testing.assert_allclose(a[:26], raw[:26], rtol=0, atol=1e-7)


def test_unobserved_target_does_not_update_and_incomplete_tail_does_not_mature():
    s = shot(25)
    _, audit = online.predict_shot(model(), s, intra=True, device='cpu', stride=26)
    assert audit['attempted_updates'] == 0
    s = shot(80)
    s['available'][:] = False
    scores, audit = online.predict_shot(model(), s, intra=True, device='cpu', stride=26)
    assert np.isfinite(scores).all()
    assert audit['attempted_updates'] == 0


def test_exception_restores_temporary_state(monkeypatch):
    parent = model()
    before = deepcopy(parent.state_dict())
    def broken(*args, **kwargs):
        raise FloatingPointError('test failure')
    monkeypatch.setattr(online, 'forecast_objective', broken)
    with pytest.raises(FloatingPointError, match='test failure'):
        online.predict_shot(parent, shot(), intra=True, device='cpu', stride=26)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_partial_step_failure_restores_parameters_flags_and_training_mode(monkeypatch):
    parent = model().train()
    parent.encoder.layers[0].eval()  # Preserve a deliberately mixed mode.
    before = deepcopy(parent.state_dict())
    modes = [m.training for m in parent.modules()]
    move = online._move
    def partial(*args, **kwargs):
        move(*args, **kwargs)
        raise FloatingPointError('failure after mutation')
    monkeypatch.setattr(online, '_move', partial)
    with pytest.raises(FloatingPointError, match='after mutation'):
        online.predict_shot(parent, shot(), intra=True, device='cpu', stride=26)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert [m.training for m in parent.modules()] == modes
    assert all(p.requires_grad for p in parent.parameters())


def test_historical_guard_rejects_step_and_restores_all_non_adaptive_weights(monkeypatch):
    parent = model().train()
    s = shot()
    s.update(y=(np.arange(90) > 50).astype(np.float32), valid=np.ones(90, bool))
    before = deepcopy(parent.state_dict())
    original = online.F.binary_cross_entropy_with_logits
    calls = []
    def controlled(logits, labels):
        calls.append(len(labels))
        if len(calls) in (1, 3):
            return logits.new_tensor(0. if len(calls) == 1 else 1.)
        return original(logits, labels)
    monkeypatch.setattr(online.F, 'binary_cross_entropy_with_logits', controlled)
    result = online.postshot_update(parent, s, [s], device='cpu', rng=np.random.default_rng(31))
    assert not result['accepted']
    assert result['parameter_change'] == 0.
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert parent.training


def test_postshot_update_changes_internal_parameters_not_risk_head():
    parent = model()
    s = shot()
    s.update(y=(np.arange(90) > 50).astype(np.float32), valid=np.ones(90, bool))
    before = deepcopy(parent.state_dict())
    audit = online.postshot_update(parent, s, [], device='cpu', rng=np.random.default_rng(31))
    assert audit['accepted']
    assert audit['parameter_change'] > 0
    for name, value in parent.state_dict().items():
        if name not in online.adaptation_names(parent):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_zero_learning_rate_rejected_and_non_east_rejected():
    with pytest.raises(ValueError):
        online.predict_shot(model(), shot(), intra=True, device='cpu', lr=0.)
    s = shot()
    s['device'] = 'J-TEXT'
    with pytest.raises(ValueError, match='EAST'):
        online.predict_shot(model(), s, intra=True, device='cpu')


def test_runner_seals_before_natural_reveal_and_never_rescores(tmp_path):
    from experiments.risk_aligned_online import simple_data
    import json
    output = tmp_path/'run'
    class Data:
        window = staticmethod(simple_data.window)
        def read_shot(self, device, sid):
            return shot(90)
        def attach_outcome(self, s, device, advanced):
            receipt = json.loads((output/'0001_1_seal.json').read_text())
            assert receipt['arm'] == 'both'
            with np.load(output/receipt['npz']) as trace:
                self.sealed = trace['scores'].copy()
            return dict(s, y=np.zeros(90, np.float32), valid=np.ones(90, bool),
                        is_disrupt=False, down=None)
    data = Data()
    online.run_arm(model(), order=[dict(shot=1)], arm='both', threshold=.7,
                   output_dir=output, device='cpu', stride=26, data_api=data)
    with np.load(output/'0001_1_sealed.npz') as sealed:
        np.testing.assert_array_equal(sealed['scores'], data.sealed)


def test_statistical_alignment_uses_only_arrived_prefix_and_same_forecast_coordinate():
    parent = model()
    s = shot(140)
    view = online.PrefixDataView(parent.center.cpu().numpy(), parent.scale.cpu().numpy(), channels=['Bt'])
    before, _ = view.window(s, 98, 8)
    np.testing.assert_array_equal(before, s['values'][91:99])
    after, _ = view.window(s, 110, 8)
    median = np.median(s['values'][:100, 0])
    q25, q75 = np.quantile(s['values'][:100, 0], [.25, .75])
    np.testing.assert_allclose(after[:, 0], (s['values'][103:111, 0]-median)/(q75-q25), rtol=1e-6)
    truth = view.forecast_truth(s, 100, 25)
    np.testing.assert_allclose(truth[:, 0], (s['values'][101:126, 0]-median)/(q75-q25), rtol=1e-6)
    future_changed = deepcopy(s)
    future_changed['values'][120:, 0] *= 1000
    other = online.PrefixDataView(parent.center.cpu().numpy(), parent.scale.cpu().numpy(), channels=['Bt'])
    equal, _ = other.window(future_changed, 110, 8)
    np.testing.assert_array_equal(equal, after)
