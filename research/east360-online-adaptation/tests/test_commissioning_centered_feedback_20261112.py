import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_centered_feedback_20261112 as centered
from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_finite_injection_keeps_padding_and_cls_fixed_and_matches_the_real_token_norm(monkeypatch):
    torch.manual_seed(13)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2,16,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    mask[0,:4] = False
    captured = []
    hook = base.encoder.layers[0].register_forward_pre_hook(lambda _m,args:captured.append(args[0].clone()))
    monkeypatch.setattr(group.layer,'patcher',lambda *_:lambda x:x*.8)
    with torch.no_grad():
        a, _ = centered.hidden(base,x,mask,group.layer.patcher(None,None))
        steps = torch.randn(2,2,4,8)
        budget = torch.tensor([.02,.03])
        result, error = centered.forward_with_writes(base,x,mask,group.layer.patcher(None,None),
                                                    steps,steps.flip(2),torch.tensor([.3,1.]),budget)
    hook.remove()
    before, after = captured[0], captured[-1].reshape(2,2,5,8)
    delta = after-before
    torch.testing.assert_close(delta[:,0,0],torch.zeros_like(delta[:,0,0]),atol=0.,rtol=0.)
    torch.testing.assert_close(delta[:,:,-1],torch.zeros_like(delta[:,:,-1]),atol=0.,rtol=0.)
    torch.testing.assert_close(delta[:,:,:-1].sum(2),torch.zeros(2,2,8),atol=1e-6,rtol=0.)
    torch.testing.assert_close(delta.flatten(2).norm(dim=2),budget.expand(2,2),atol=1e-6,rtol=0.)
    assert (result-a).abs().max()>1e-6 and error<1e-6


def test_real_feedback_perturbation_can_only_change_the_current_state_at_its_release(monkeypatch):
    torch.manual_seed(41)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    q = np.array([15,47,79,111,143])
    width = len(group.lag.CHANNELS)
    x = np.random.default_rng(41).normal(size=(170,23)).astype(np.float32)
    signal = dict(values=x,available=np.ones_like(x,bool))
    monkeypatch.setattr(group.layer,'patcher',lambda *_:lambda raw:raw*.8)
    fields = (torch.zeros(len(q),width),torch.zeros(len(q),width),
              torch.ones(len(q),width),torch.ones(width))
    phase = correction = torch.zeros(len(q)+1)
    prior = torch.zeros(len(q),group.lag.H,width)
    decoder = (torch.zeros(8),torch.ones(8),torch.randn(group.lag.H*width,8)*.3,torch.zeros(group.lag.H*width))
    waveform = torch.randn_like(prior)
    def run(w):
        return centered.feedback_directions(base,signal,q,np.ones(len(q),bool),
            phase,correction,prior,fields,w,decoder,torch.ones(2,4))[0]
    altered = waveform.clone()
    altered[3] = -3.*altered[3]
    a,b = run(waveform),run(altered)
    torch.testing.assert_close(a[:,:4],b[:,:4],atol=0.,rtol=0.)
    assert (a[0,4]-b[0,4]).abs().max()>1e-6
    torch.testing.assert_close(a[1,4],b[1,4],atol=0.,rtol=0.)
    assert all(p.grad is None for p in base.parameters())


def test_mean_and_full_writes_are_distinct_equal_dose_actions_on_real_encoder():
    torch.manual_seed(20)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2,16,23)
    mask = torch.ones_like(x,dtype=torch.bool)
    mask[0,:4] = False
    steps = torch.randn(3,2,4,8)
    budget = torch.tensor([.21,.24])
    seen = []
    hook = base.encoder.layers[0].register_forward_pre_hook(lambda _m,p:seen.append(p[0].clone()))
    centered.hidden(base,x,mask,lambda v:v)
    scores, error = centered.forward_with_writes(base,x,mask,lambda v:v,steps,steps.flip(2),
        torch.tensor([.3,1.]),budget,spaces=('free','mean','full'))
    hook.remove()
    delta = seen[-1].reshape(3,2,5,8)-seen[0]
    assert error<1e-6
    torch.testing.assert_close(delta.flatten(2).norm(dim=-1),budget.expand(3,2),atol=1e-6,rtol=0.)
    assert not delta[:,0,0].any() and not delta[:,:,-1].any()
    torch.testing.assert_close(delta[0,:,:-1].sum(1),torch.zeros(2,8),atol=1e-6,rtol=0.)
    torch.testing.assert_close(delta[1,:,1],delta[1,:,2],atol=1e-6,rtol=0.)
    assert delta[1,:,:-1].sum(1).norm()>.1 and delta[2,:,:-1].sum(1).norm()>.1
    assert (scores[1]-scores[2]).abs().max()>1e-6


def test_level_states_use_only_matured_feedback(monkeypatch):
    torch.manual_seed(30)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    q = np.arange(15,144,32)
    width = len(group.lag.CHANNELS)
    x = np.random.default_rng(30).normal(size=(170,23)).astype(np.float32)
    mask = np.ones_like(x,bool)
    monkeypatch.setattr(group.layer,'patcher',lambda *_:lambda v:v)
    fields = (torch.zeros(len(q),width),torch.zeros(len(q),width),
              torch.ones(len(q),width),torch.ones(width))
    correction = torch.zeros(len(q)+1)
    prior = torch.zeros(len(q),group.lag.H,width)
    decoder = (torch.zeros(8),torch.ones(8),torch.randn(group.lag.H*width,8)*.3,torch.zeros(group.lag.H*width))
    waveform = torch.randn_like(prior)
    changed = waveform.clone()
    changed[3] *= -2
    def run(w):
        return centered.feedback_directions(base,dict(values=x,available=mask),q,np.ones(len(q),bool),
            correction,correction,prior,fields,w,decoder,torch.ones(2,4),spaces=('free','mean','full'))[0]
    a,b = run(waveform),run(changed)
    torch.testing.assert_close(a[:,:4],b[:,:4],atol=0.,rtol=0.)
    torch.testing.assert_close(a[3::2,4],b[3::2,4],atol=0.,rtol=0.)
    assert (a[2,4]-b[2,4]).abs().max()>1e-6
    torch.testing.assert_close(a[2,4,1],a[2,4,2],atol=1e-6,rtol=0.)
    torch.testing.assert_close(a[0,4]+a[2,4],a[4,4],atol=1e-6,rtol=0.)
