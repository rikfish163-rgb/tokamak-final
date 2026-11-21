import copy
import numpy as np
import pytest
import torch

from ..asymmetric_feedback_20261023.learning import decoder_forecast, routed_episode, gradient_relation
from ..east_joint_feedback_20261023.learning import enable,episode
from ..deepcoral_prefix_20261022.learning import forecast_examples
from .test_credit_horizon import fixture


def test_forecast_value_and_head_gradient_identical_encoder_gradient_removed():
    m,s=fixture();enable(m)
    x,mask,t,tm,_=forecast_examples({k:v[:384] for k,v in s.items()},384,m.cfg,'cpu')
    from ..increment_forecast_20261023.learning import feedback_loss
    old=feedback_loss(m,m(x,mask)['forecast'],x,mask,t,tm)
    new=decoder_forecast(m,s,384,device='cpu')
    torch.testing.assert_close(old,new,rtol=0,atol=0)
    head=tuple(m.base.forecast_head.parameters())
    a=torch.autograd.grad(old,head,retain_graph=True);b=torch.autograd.grad(new,head,retain_graph=True)
    # Detaching encoder tensors can change floating-point accumulation paths.
    # Measured float32 difference is <1e-9, not an altered head objective;
    # retain a tight absolute bound instead of demanding bitwise GEMM equality.
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-9)
    enc=tuple(p for k,p in m.named_parameters() if k in m.encoder_names())
    assert all(v is None for v in torch.autograd.grad(new,enc,allow_unused=True))


def test_routing_does_not_change_risk_or_inner_update_but_risk_still_trains_encoder():
    m,s=fixture();enable(m)
    old=episode(m,s,384,[384,392],True,boundary=.4,track=True,device='cpu')
    new=routed_episode(m,s,384,[384,392],True,boundary=.4,track=True,device='cpu')
    for k in ('before','after','forecast'):torch.testing.assert_close(old[k],new[k],rtol=0,atol=0)
    for k in old['fast']:torch.testing.assert_close(old['fast'][k],new['fast'][k],rtol=0,atol=0)
    enc=tuple(p for k,p in m.named_parameters() if k in m.encoder_names())
    grads=torch.autograd.grad(new['after'],enc+tuple(m.base.forecast_head.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and sum(g.square().sum() for g in grads)>0
    changed=copy.deepcopy(s);changed['values'][384:]=1e5
    torch.testing.assert_close(decoder_forecast(m,s,384,device='cpu'),decoder_forecast(m,changed,384,device='cpu'),rtol=0,atol=0)
    with pytest.raises(ValueError):decoder_forecast(m,dict(s,y=1),384,device='cpu')


def test_gradient_relation_sign_zero_and_invalid():
    a=torch.tensor([1.,2.]);r=gradient_relation(a,-a)
    assert r['dot']==-5 and r['cosine']==pytest.approx(-1)
    assert gradient_relation(a,torch.zeros(2))['cosine'] is None
    with pytest.raises(ValueError):gradient_relation(a,torch.ones(3))
    with pytest.raises(FloatingPointError):gradient_relation(a,torch.tensor([float('nan'),1.]))


def test_routed_head_derivative_finite_difference():
    m,s=fixture();m=m.double();enable(m)
    p=m.base.forecast_head.weight;v=decoder_forecast(m,s,384,device='cpu')
    g,=torch.autograd.grad(v,p);d=g/g.norm();saved=p.detach().clone();vals=[];eps=1e-4
    for sign in (1,-1):
        with torch.no_grad():p.copy_(saved+sign*eps*d)
        vals.append(float(decoder_forecast(m,s,384,device='cpu').detach()))
    with torch.no_grad():p.copy_(saved)
    assert (vals[0]-vals[1])/(2*eps)==pytest.approx(float((g*d).sum()),rel=1e-5,abs=1e-8)
