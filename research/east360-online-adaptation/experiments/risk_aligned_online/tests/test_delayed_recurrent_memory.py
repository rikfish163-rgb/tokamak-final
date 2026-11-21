import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import (
    DiagnosticMemory, RecurrentAdapter, patch_signal, endpoint_corrections,
)


def setup():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)).eval()
    return RecurrentAdapter(base), torch.randn(3, 128, 23)


def test_delta_write_is_exact_masked_square_gradient_without_projection():
    torch.manual_seed(5)
    m = DiagnosticMemory(64).double()
    k = torch.randn(9, dtype=torch.double) * .1
    y = torch.randn(92, dtype=torch.double) * .1
    mask = torch.arange(92) % 3 != 0
    w = torch.zeros(9, 92, dtype=torch.double, requires_grad=True)
    loss = .5 * ((k @ w-y).square()*mask).sum()
    grad = torch.autograd.grad(loss, w)[0]
    actual = m.write(w, k, y, mask)
    expected = w-m.eta*grad/(1+k.square().sum())
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    y[~mask] = 10000
    torch.testing.assert_close(actual, m.write(w, k, y, mask), atol=0, rtol=0)


def test_delayed_targets_prefix_invariance_and_no_cross_shot_state():
    torch.manual_seed(11)
    m = DiagnosticMemory(64)
    means = torch.randn(18, 23)
    valid = torch.ones_like(means, dtype=torch.bool)
    before = copy.deepcopy(m.state_dict())
    a = m.trace(means, valid, mode='memory')
    changed = means.clone(); changed[9:] += 9
    b = m.trace(changed, valid, mode='memory')
    torch.testing.assert_close(a['delta'][:9], b['delta'][:9], atol=0, rtol=0)
    assert not a['delta'][:4].any() and a['delta'][4].any()
    assert a['forecast'].shape == (18, 4, 23)
    assert a['updates'] == 14
    c = m.trace(means, valid, mode='memory')
    torch.testing.assert_close(a['delta'], c['delta'], atol=0, rtol=0)
    for k, v in before.items(): assert torch.equal(v, m.state_dict()[k])
    endpoints = endpoint_corrections(a['delta'], torch.arange(80))
    assert not endpoints[:40].any() and endpoints[40].any()
    with pytest.raises(ValueError): m.trace(means, valid, mode='future')


def test_missing_targets_do_not_update_and_controls_match_norm():
    torch.manual_seed(21)
    m = DiagnosticMemory(64)
    means = torch.randn(12, 23)
    valid = torch.ones_like(means, dtype=torch.bool)
    empty = m.trace(means, torch.zeros_like(valid), mode='memory')
    assert empty['updates'] == 0 and not empty['delta'].any()
    main = m.trace(means, valid, mode='memory')
    assert main['norm'].max() <= 1.000001
    for arm in ('local', 'hold', 'reverse'):
        control = m.trace(means, valid, mode=arm, reference_norm=main['norm'])
        ok = ~control['zero_direction']
        torch.testing.assert_close(control['norm'][ok], main['norm'][ok], atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError): patch_signal({'values': np.ones((20,23)), 'available': np.ones((20,23),bool), 'y': 1}, torch.zeros(23), torch.ones(23))


def test_internal_off_identity_and_slow_gradient_through_memory():
    adapter, x = setup(); available = torch.ones_like(x, dtype=torch.bool)
    expected = adapter.base(x, available)['logit']
    h, valid = adapter.tokens(x, available)
    zero = torch.zeros(3, 92)
    actual = adapter.risk(h, valid, zero)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    with torch.no_grad(): adapter.memory.output.weight.normal_(0,.02)
    means = torch.randn(20,23); mask = torch.ones_like(means,dtype=torch.bool)
    trace = adapter.memory.trace(means,mask,mode='memory')
    z = adapter.risk(h,valid,trace['delta'][-3:])
    z.sum().backward()
    assert adapter.memory.key.weight.grad.abs().max() > 0
    assert adapter.memory.prior.grad.abs().max() > 0
    assert not any(p.grad is not None for p in adapter.base.parameters())
    torch.testing.assert_close(adapter.risk(h.detach(),valid,zero),expected,atol=1e-6,rtol=1e-6)


def test_patch_observation_semantics_and_forecast_targets():
    values = np.arange(96*23,dtype=np.float32).reshape(96,23)/100
    available = np.ones_like(values,dtype=bool);available[3,5]=False
    q,m = patch_signal(dict(values=values,available=available),torch.zeros(23),torch.ones(23))
    assert q.shape == (12,23) and not m[0,5] and m[1,5]
    layer = DiagnosticMemory(64)
    t = layer.trace(q,m,mode='memory')
    target = q[1:5]-q[0]
    assert torch.equal(t['first_target'],target)
    assert not t['first_mask'][:,5].any()


def test_nonfull_batch_direct_parent_exact():
    from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import original_predictions,predict
    adapter,_=setup()
    rng=np.random.default_rng(11)
    signal=dict(values=rng.normal(size=(75,23)).astype(np.float32),available=np.ones((75,23),bool))
    curves,_,_=predict(adapter,signal)
    np.testing.assert_array_equal(curves['off'],original_predictions(adapter.base,signal))
