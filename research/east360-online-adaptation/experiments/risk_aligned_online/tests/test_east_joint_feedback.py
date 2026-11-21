import numpy as np
import pytest
import torch

from ..credit_horizon_20261023.learning import episode as detached_episode
from ..east_joint_feedback_20261023.learning import enable, episode, objective
from .test_credit_horizon import fixture


def test_exact_initial_gradient_and_deployed_forward():
    m,s=fixture();enable(m)
    r=episode(m,s,384,[384,392],True,boundary=.4,track=True,device='cpu')
    ref=detached_episode(m,s,384,[384,392],True,boundary=.4,tail=8,device='cpu')
    torch.testing.assert_close(r['after'],ref['risk'],rtol=0,atol=0)
    for k in ref['fast']:torch.testing.assert_close(r['fast'][k],ref['fast'][k],rtol=0,atol=0)
    encoder=tuple(p for k,p in m.named_parameters() if k in m.encoder_names())
    head=tuple(m.base.forecast_head.parameters())
    gs=torch.autograd.grad(r['after'],encoder+head)
    assert all(torch.isfinite(g).all() for g in gs)
    assert sum(g.square().sum() for g in gs[:len(encoder)])>0
    assert sum(g.square().sum() for g in gs[len(encoder):])>0
    assert not any(p.requires_grad for p in m.base.risk_head.parameters())
    assert not m.base.center.requires_grad


def test_full_initial_gradient_finite_difference():
    m,s=fixture();m=m.double();s['values']=s['values'].astype('float64');enable(m)
    def value():return episode(m,s,384,[384,392],True,boundary=.4,track=True,device='cpu')['after']
    p=m.base.embedding.weight; v=value();g,=torch.autograd.grad(v,p)
    d=g/g.norm();old=p.detach().clone();eps=1e-4;values=[]
    for sign in (1,-1):
        with torch.no_grad():p.copy_(old+sign*eps*d)
        values.append(float(value().detach()))
    with torch.no_grad():p.copy_(old)
    assert (values[0]-values[1])/(2*eps)==pytest.approx(float((g*d).sum()),rel=2e-3,abs=1e-7)


def test_training_target_never_enters_signal_update_and_only_future_risk_changes():
    m,s=fixture();enable(m)
    a=episode(m,s,384,[384,392],False,boundary=.4,track=False,device='cpu')
    s['values'][384:]=1000.
    b=episode(m,s,384,[384,392],True,boundary=.4,track=False,device='cpu')
    for k in a['fast']:torch.testing.assert_close(a['fast'][k],b['fast'][k],rtol=0,atol=0)
    torch.testing.assert_close(a['forecast'],b['forecast'],rtol=0,atol=0)
    with pytest.raises(ValueError):episode(m,dict(s,y=1),384,[384],True,boundary=.4,track=True,device='cpu')
    torch.testing.assert_close(objective(a,'joint'),a['before']+.2*a['forecast'])
    torch.testing.assert_close(objective(a,'meta'),.5*(a['before']+a['after'])+.2*a['forecast'])
    with pytest.raises(ValueError):objective(a,'unknown')
