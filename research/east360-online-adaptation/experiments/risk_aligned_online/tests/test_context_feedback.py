"""Conditional feedback is an internal update, never a supervised online gate."""
import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.context_feedback_20261022.model import (
    ContextFeedback, update_observed, operational_loss,
)
from experiments.risk_aligned_online.context_feedback_20261022.stream import predict_pair
from experiments.risk_aligned_online.context_feedback_20261022.east import recovered_trace


@pytest.fixture
def model():
    torch.set_num_threads(2); torch.manual_seed(42)
    base = Simple23(Config(window=32,patch=8,dim=16,heads=2,fast_site='tokens'),
                    np.zeros(23),np.ones(23))
    m = ContextFeedback(base,np.ones(23)); m.freeze_base()
    return m


def signals(n=64):
    rng=np.random.default_rng(222)
    return dict(values=rng.normal(size=(n,23)).astype('float32'),
                available=np.ones((n,23),bool))


def tensors():
    s=signals(32)
    return torch.tensor(s['values'])[None],torch.tensor(s['available'])[None]


def test_zero_controller_equals_uniform_and_preserves_source(model):
    x,m=tensors(); old=copy.deepcopy(model.state_dict())
    a,aa=update_observed(model,x,m,use_event=True,uniform=False)
    b,bb=update_observed(model,x,m,use_event=False,uniform=True)
    assert aa['delta']>0 and aa['steps']==2
    for k in a: torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    for k in old: assert torch.equal(old[k],model.state_dict()[k])
    assert not any(p.requires_grad for p in model.base.parameters())


def test_meta_gradient_reaches_controller(model):
    x,m=tensors()
    fast,_=update_observed(model,x,m,use_event=False,create_graph=True)
    loss=model.predict_fast(x,m,fast)['logit'].square().mean()
    g=torch.autograd.grad(loss,tuple(model.controller.parameters()))
    assert all(torch.isfinite(v).all() for v in g)
    assert sum(float(v.abs().sum()) for v in g)>1e-8


def test_controller_meta_gradient_matches_finite_difference(model):
    x,m=tensors(); parameter=model.controller[-1].bias
    direction=torch.randn_like(parameter); direction/=direction.norm()
    def objective(graph):
        fast,_=update_observed(model,x,m,use_event=True,create_graph=graph)
        return model.predict_fast(x,m,fast)['logit'].square().mean()
    exact=(torch.autograd.grad(objective(True),parameter)[0]*direction).sum()
    old=parameter.detach().clone();eps=.01
    with torch.no_grad(): parameter.copy_(old+eps*direction)
    plus=objective(False).detach()
    with torch.no_grad(): parameter.copy_(old-eps*direction)
    minus=objective(False).detach()
    with torch.no_grad(): parameter.copy_(old)
    torch.testing.assert_close(exact,(plus-minus)/(2*eps),rtol=.08,atol=2e-6)


def test_event_context_is_inferred_not_an_online_label(model):
    x,m=tensors()
    with pytest.raises(TypeError): update_observed(model,x,m,use_event=True,event_target=3)
    with pytest.raises(TypeError): update_observed(model,x,m,use_event=True,y=1)
    with pytest.raises(ValueError): update_observed(model,x.repeat(2,1,1),m.repeat(2,1,1),use_event=True)


def test_empty_observations_do_not_update(model):
    x,m=tensors(); _,info=update_observed(model,x,m&False,use_event=True)
    assert info['steps']==0 and info['delta']==0


def test_categorical_mode_is_not_a_regression_target(model):
    x,m=tensors(); m[:]=False; m[:,:,model.mode_channel]=True
    _,info=update_observed(model,x,m,use_event=False)
    assert info['observations']==0 and info['steps']==0


def test_whole_shot_objective_keeps_early_timely_alarm_not_point_labels():
    logits=torch.tensor([3.,-4.,6.],requires_grad=True)
    times=np.array([.800,.990,1.000]); valid=np.ones(3,bool)
    # A first alarm200ms before disruption is timely, despite a25ms point label.
    loss=operational_loss(logits,times,valid,is_disrupt=True,down=1.,threshold=.5)
    loss.backward()
    assert logits.grad[0]<0 and logits.grad[1]==0 and logits.grad[2]==0
    safe=operational_loss(logits,times,valid,is_disrupt=False,down=None,threshold=.5)
    assert safe>loss


def test_strict_deadline_and_no_eligible_times():
    with pytest.raises(ValueError,match='eligible'):
        operational_loss(torch.ones(1),np.array([.990]),np.ones(1,bool),
                         is_disrupt=True,down=1.,threshold=.5)


def test_stream_cannot_rewrite_past_or_carry_state_between_shots(model):
    s=signals(96); a,info=predict_pair(model,s,device='cpu',use_event=False,interval=16)
    changed=copy.deepcopy(s); changed['values'][55:]+=3
    b,_=predict_pair(model,changed,device='cpu',use_event=False,interval=16)
    c,_=predict_pair(model,s,device='cpu',use_event=False,interval=16)
    for arm in a:
        np.testing.assert_array_equal(a[arm][:55],b[arm][:55])
        np.testing.assert_array_equal(a[arm],c[arm])
    for r in info['updates']:
        assert r['first_affected_prediction']==r['last_observation']+1
    with pytest.raises(ValueError,match='outcome'):
        predict_pair(model,dict(s,is_disrupt=False),device='cpu',use_event=False)


def test_recovery_accepts_only_complete_finite_clock_matching_traces(tmp_path):
    times=np.arange(40,dtype=float)/1000
    file=tmp_path/'shot.npz'
    np.savez_compressed(file,times=times,static=np.zeros(40),intra=np.ones(40))
    result=recovered_trace(file,times)
    assert set(result)=={'static','intra'}
    with pytest.raises(ValueError,match='clock'):
        recovered_trace(file,times+.001)
    np.savez_compressed(file,times=times,static=np.zeros(40),intra=np.ones(39))
    with pytest.raises(ValueError,match='complete'):
        recovered_trace(file,times)
    np.savez_compressed(file,times=times,static=np.zeros(40),intra=np.full(40,np.nan))
    with pytest.raises(ValueError,match='finite'):
        recovered_trace(file,times)
