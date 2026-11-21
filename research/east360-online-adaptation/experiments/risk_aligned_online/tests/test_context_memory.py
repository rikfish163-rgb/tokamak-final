import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextMemory, ContextAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23


def memory():
    torch.manual_seed(131)
    return ContextMemory(64)


def observed(n=16):
    t = torch.arange(n, dtype=torch.float32)[:, None]
    x = torch.sin(t * torch.linspace(.03, .8, 23)[None]) + t.square() * .002
    return x, torch.ones_like(x, dtype=torch.bool)


def test_context_zero_signal_change_and_shot_reset():
    m = memory()
    x, a = observed()
    main = m.trace(x, a, mode='memory')
    assert main['context'].shape == (16, 8)
    assert main['context'].abs().max() > 0
    stable = m.trace(torch.ones_like(x), a, mode='memory')
    assert not stable['context'].count_nonzero()
    assert not stable['delta'].count_nonzero()
    assert torch.equal(m.trace(x, a, mode='memory')['delta'], main['delta'])


def test_context_prefix_invariance_and_no_offline_target_leak():
    m = memory()
    x, a = observed(20)
    full = m.trace(x, a, mode='memory', return_before=True)
    short = m.trace(x[:10], a[:10], mode='memory', return_before=True)
    for key in ('delta', 'forecast', 'context', 'before_delta'):
        assert torch.allclose(full[key][:10], short[key], atol=2e-7, rtol=1e-6)
    changed = x.clone(); changed[10:] += 100
    later = m.trace(changed, a, mode='memory')
    assert torch.equal(full['delta'][:10], later['delta'][:10])


def test_context_write_is_normalized_observed_least_squares_gradient():
    m = memory()
    torch.manual_seed(4)
    c = torch.randn(8, requires_grad=True) * .02
    basis = torch.randn(4, 23, 8) * .1
    target = torch.randn(4, 23) * .2
    mask = torch.rand(4, 23) > .3
    loss = .5 * (((basis @ c - target) * mask).square().sum())
    g, = torch.autograd.grad(loss, c)
    scale = 1 + (basis.square() * mask[..., None]).sum()
    expected = c - m.eta * g / scale
    expected = expected / expected.norm().clamp_min(1)
    actual = m.write(c, basis, target, mask)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert .5 * (((basis @ actual - target) * mask).square().sum()) <= loss
    assert torch.equal(m.write(c, basis, target, torch.zeros_like(mask)), c)


def test_context_missing_values_do_not_influence_keys_targets_or_updates():
    m = memory()
    x, a = observed()
    a[:, 3] = False
    poisoned = x.clone(); poisoned[:, 3] = float('nan')
    first = m.trace(x, a, mode='memory')
    second = m.trace(poisoned, a, mode='memory')
    assert torch.equal(first['delta'], second['delta'])
    empty = m.trace(poisoned, torch.zeros_like(a), mode='memory')
    assert empty['updates'] == 0 and not empty['delta'].count_nonzero()


def test_context_reverse_changes_target_assignment_not_gradient_sum_order():
    m = memory()
    x, a = observed()
    main = m.trace(x, a, mode='memory')
    reverse = m.trace(x, a, mode='reverse', reference_norm=main['norm'])
    assert not torch.allclose(main['delta'][5:], reverse['delta'][5:])
    assert torch.allclose(main['norm'], reverse['norm'], atol=2e-7)
    with pytest.raises(ValueError):
        m.trace(x, a, mode='reverse')


def test_context_source_gradients_and_adapter_freezing():
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), np.zeros(23), np.ones(23))
    model = ContextAdapter(base).eval()
    assert all(not p.requires_grad for p in base.parameters())
    model.memory.requires_grad_(True)
    with torch.no_grad():
        model.memory.output.weight.normal_(0, .02)
    x, a = observed()
    trace = model.memory.trace(x, a, mode='memory')
    loss = model.memory.output(trace['delta']).square().mean()
    loss.backward()
    for name in ('key.weight', 'kernel', 'rate', 'output.weight'):
        p = dict(model.memory.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert all(p.grad is None for p in base.parameters())
    assert trace['norm'].max() <= 1 + 1e-7


def test_context_empty_and_invalid_inputs():
    m = memory()
    t = m.trace(torch.zeros(0, 23), torch.zeros(0, 23, dtype=torch.bool), mode='memory')
    assert t['delta'].shape == (0, 92) and t['context'].shape == (0, 8)
    x, a = observed()
    with pytest.raises(ValueError):
        m.trace(x, a.float(), mode='memory')
    with pytest.raises(ValueError):
        m.trace(x, a, mode='future_oracle')


def test_context_checkpoint_cannot_load_wrong_source_roles_or_architecture():
    from experiments.risk_aligned_online.context_memory_20261028.checkpoint import validate_metadata
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    roles = dict(train=list(range(1353)), val=list(range(2000, 2160)), test=list(range(3000, 3220)))
    data = dict(profile='context8_v1', kind='context8', context_dim=8, epoch=2,
                feedback='progressive', readout='full', source_sha256=PARENT_SHA256,
                train_ids=roles['train'], val_ids=roles['val'], objective='detached',
                memory=memory().state_dict())
    validate_metadata(data, roles)
    for bad in (dict(data, profile='zero_prior_ablation_v3'), dict(data, context_dim=9),
                dict(data, source_sha256='bad'), dict(data, train_ids=roles['test']),
                dict(data, objective='bag'), dict(data, epoch=1)):
        with pytest.raises(ValueError):
            validate_metadata(bad, roles)


def test_context_source_training_requires_every_active_parameter_to_learn():
    from experiments.risk_aligned_online.context_memory_20261028.checkpoint import verify_gradients
    names = ('key.weight', 'key.bias', 'kernel', 'rate', 'output.weight')
    rows = [dict(epoch=e, ids=[1, 2], gradients={n: .2 for n in names}) for e in (1, 2)]
    result = verify_gradients(rows, [1, 2])
    assert result == {n: 2 for n in names}
    rows[0]['gradients']['kernel'] = None
    with pytest.raises(ValueError):
        verify_gradients(rows, [1, 2])


def test_stream_adapter_selection_is_explicit_and_preserves_legacy_default():
    from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter
    from experiments.risk_aligned_online.lifecycle_stream_20261025.model import LifecycleAdapter
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), np.zeros(23), np.ones(23))
    torch.manual_seed(41)
    expected = LifecycleAdapter(base, readout='full', feedback='progressive').eval()
    torch.manual_seed(41)
    actual = build_adapter(base, readout='full', feedback='progressive', context=False)
    assert type(actual) is LifecycleAdapter
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())
    selected = build_adapter(base, readout='full', feedback='progressive', context=True)
    assert type(selected) is ContextAdapter
    with pytest.raises(ValueError):
        build_adapter(base, readout='shape', feedback='progressive', context=True)


def test_context_architecture_pair_keeps_alarm_and_learning_contract():
    from experiments.risk_aligned_online.context_memory_20261028.compare import validate_pair
    a = dict(threshold=.7, seed=131, feedback='progressive', readout='full',
             source_deployment_feedback_match=True, code_sha256={'a': 'old'},
             source_memory=dict(profile='zero_prior_ablation_v3', kind='zero_prior', prior_policy='fixed_zero'))
    b = dict(a, code_sha256={'a': 'new'}, source_memory=dict(profile='context8_v1', kind='context8', context_dim=8))
    validate_pair(a, b)
    for changed in (dict(b, threshold=.6), dict(b, seed=33),
                    dict(b, source_memory=dict(profile='context8_v1', kind='context8', context_dim=9))):
        with pytest.raises(ValueError):
            validate_pair(a, changed)
