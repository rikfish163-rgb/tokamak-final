import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.aligned_state_20261024.model import first_tokens, fixed_readout
from experiments.risk_aligned_online.latent_completion_20261028.model import (
    Imputer, tokens, masked, latent_loss, risk_kl, readout, step, predict,
)


def setup():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(fast_site='tokens'), np.zeros(23), np.ones(23)).eval().requires_grad_(False)
    return base, Imputer().eval().requires_grad_(False)


def test_zero_state_exact_original_and_frozen_readout():
    base, _ = setup()
    x = torch.randn(3, 128, 23); a = torch.rand_like(x) > .1
    h, v = tokens(base, x, a, torch.zeros(128))
    old, ov = first_tokens(base, x, a)
    torch.testing.assert_close(h, old, rtol=0, atol=0)
    assert torch.equal(v, ov)
    torch.testing.assert_close(fixed_readout(base, h, v), base(x, a)['logit'], rtol=0, atol=0)


def test_masks_cover_observed_blocks_without_using_missing_values():
    base, _ = setup()
    x = torch.randn(2, 128, 23); a = torch.rand_like(x) > .2
    xx, aa = masked(x, a)
    assert torch.equal(aa[0::2] | aa[1::2], a)
    assert not (aa[0::2] & aa[1::2]).any()
    assert torch.equal(xx[0::2], x)
    bad = x.clone(); bad[~a] = float('nan')
    h, v = tokens(base, bad, a, torch.zeros(128))
    assert torch.isfinite(h).all() and v[:, -1].all()


def test_update_changes_only_disposable_state_and_has_finite_gradient():
    base, net = setup(); x = torch.randn(1,128,23); a = torch.ones_like(x, dtype=torch.bool)
    saved = {k: v.clone() for k,v in base.state_dict().items()}
    state, info = step(base, net, x, a, torch.zeros(128))
    assert info['gradient_norm'] > 0 and info['updates'] == 1
    assert 0 < state.norm() <= 1
    assert not state.requires_grad
    assert all(torch.equal(v, saved[k]) for k,v in base.state_dict().items())
    assert all(p.grad is None for p in net.parameters())
    h = torch.randn(2,17,64,requires_grad=True); target=torch.zeros_like(h); valid=torch.ones(2,17,dtype=torch.bool)
    assert torch.isfinite(latent_loss(h,target,valid)).all()


def test_stream_is_prefix_invariant_reset_and_update_affects_only_later_points():
    base, net = setup()
    signal = dict(values=np.random.default_rng(13).normal(size=(160,23)).astype('float32'),
                  available=np.ones((160,23),bool))
    curves, info = predict(base, {'plain':net}, signal, None)
    short={k:v[:145].copy() for k,v in signal.items()}
    prefix,_ = predict(base, {'plain':net}, short, None)
    torch.testing.assert_close(torch.tensor(curves['plain'][:145]),torch.tensor(prefix['plain']),rtol=0,atol=0)
    assert np.array_equal(curves['plain'][:128],curves['cold'][:128])
    assert info['plain']['updates'] == 1
    again,_=predict(base, {'plain':net}, signal, None)
    assert np.array_equal(again['plain'],curves['plain'])


def test_future_values_cannot_change_earlier_predictions_and_reversal_is_not_identity():
    base,net=setup()
    rng=np.random.default_rng(22)
    signal=dict(values=rng.normal(size=(288,23)).astype('float32'),available=np.ones((288,23),bool))
    initial,_=predict(base,{'task':net},signal,None,reversed_arm=True)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][160:]*=100
    after,_=predict(base,{'task':net},changed,None,reversed_arm=True)
    for name in initial:assert np.array_equal(initial[name][:160],after[name][:160])
    assert not np.array_equal(initial['task'][128:],initial['task_reverse'][128:])


def test_source_distillation_gradient_trains_only_the_imputer():
    base,net=setup();net.requires_grad_(True)
    x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():
        h,v=tokens(base,x,a,torch.zeros(128));teacher=readout(base,h,v,None)
        xx,aa=masked(x,a);mh,mv=tokens(base,xx,aa,torch.zeros(128))
    out=net(mh,mv);valid=v.repeat_interleave(2,0)
    kl=risk_kl(readout(base,out,valid,None),teacher.repeat_interleave(2)).mean()
    kl.backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in net.parameters())
    assert all(p.grad is None for p in base.parameters())
