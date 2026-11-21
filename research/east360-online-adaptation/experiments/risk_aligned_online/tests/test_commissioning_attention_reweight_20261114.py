import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_attention_reweight_20261114 as study
from experiments.risk_aligned_online.simple23 import Config, Simple23


def model():
    torch.manual_seed(41)
    return Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)


def test_tilt_rebalances_only_existing_time_keys_and_keeps_cls_at_zero():
    valid = torch.tensor([[True]*16, [False]*10+[True]*6])
    start = torch.ones(2,4,2)
    bias = study.attention_bias(valid,start).reshape(2,4,17,17)
    np.testing.assert_allclose(bias[..., :-1].sum(-1).numpy(),0.,atol=1e-6)
    assert torch.count_nonzero(bias[..., -1]) == 0
    assert torch.count_nonzero(bias[1,...,:10]) == 0
    zero = study.attention_bias(valid,start*0)
    assert torch.count_nonzero(zero) == 0


def test_actual_frozen_attention_is_differentiable_in_temporary_coefficients():
    net = model()
    x = torch.randn(4,128,23)
    m = torch.ones_like(x,dtype=torch.bool)
    state = torch.zeros(4,4,2,requires_grad=True)
    h = study.hidden(net,x,m,lambda y:y,state)
    torch.testing.assert_close(net.risk_head(h).flatten(),net(x,m)['logit'],atol=2e-7,rtol=0.)
    g, = torch.autograd.grad(net.risk_head(h).sum(),state)
    assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    shifted = study.hidden(net,x,m,lambda y:y,state.detach()+.25)
    assert float((shifted-h.detach()).abs().max())>1e-5
    assert all(p.grad is None for p in net.parameters())


def test_controls_match_each_real_logit_dose_and_respect_joint_support():
    a = torch.randn(3,5,4,2)
    b = torch.randn_like(a)
    b[1,2] = 0.
    f = a.new_tensor([0.,.25,0.,.75,1.])
    observed = torch.ones((5,16),dtype=torch.bool)
    clock = a.new_tensor([1.,0.,1.,1.,1.])
    out,dose,err = study.realized_tilts(a,b,np.ones(5,bool),f,clock,observed,torch.zeros(4))
    assert err<1e-7
    torch.testing.assert_close(out[:3],out[4:])
    assert not out[3].any()
    assert dose[1]==dose[2]==0.
    assert float(dose.max())<=study.LOGIT_RADIUS+1e-6


def test_teacher_releases_only_after_query_block_and_uses_both_outcome_types():
    gains = np.array([.1,-.2,.05,.03])
    pair = np.stack((gains,-gains))
    history = [(41,True,pair,np.array([.7,.2])),(81,False,pair,np.array([.6,.8]))]
    assert not study.teacher(history,80).any()
    w = study.teacher(history,81)
    assert w[0]>0. and w[1]<0.


def test_teaching_rewards_safe_reduction_and_uses_only_timely_positive_window():
    t = np.arange(180)*.001
    base = np.full(len(t),.40)
    candidate = np.full(len(t),.30)
    candidate[151] = .99
    # 29 ms lead does not make positive orientation beneficial.
    assert study.preference(t,base,candidate,True,.18,.50)<0.
    assert study.preference(t,base,candidate,False,None,.50)<0.


def test_history_is_separate_from_temporary_feedback():
    w = torch.tensor([.3,.4,.5,.6])
    static = study.historical_prior(w,torch.zeros(5,4,2))
    torch.testing.assert_close(static[...,0],w.expand(5,-1))
    assert not static[...,1].any()
