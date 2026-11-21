from copy import deepcopy

import torch

from experiments.risk_aligned_online import joint_meta
from experiments.risk_aligned_online.joint_meta import alignment_loss,train_step
from experiments.risk_aligned_online.tests.test_joint_meta import setup


def test_alignment_uses_risk_direction_as_detached_target():
    signal=torch.tensor([[1.,0.]],requires_grad=True)
    risk=torch.tensor([[0.,1.]],requires_grad=True)
    loss=alignment_loss(signal,risk)
    loss.backward()
    assert signal.grad[0,1]<0
    assert risk.grad is None
    assert alignment_loss(torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]])).item()==0.
    assert alignment_loss(torch.tensor([[1.,0.]]),torch.tensor([[-1.,0.]])).item()==2.


def test_alignment_zero_risk_gradient_has_no_direction_target():
    signal=torch.tensor([[.1,.2]],requires_grad=True)
    loss=alignment_loss(signal,torch.zeros_like(signal))
    loss.backward()
    assert loss.item()==0.
    assert torch.equal(signal.grad,torch.zeros_like(signal))


def test_alignment_training_reaches_forecast_head_without_forecast_loss():
    net,e=setup()
    old=deepcopy(net.state_dict())
    log=train_step(net,torch.optim.SGD(net.parameters(),lr=.001),e,
                   mode='aligned',signal_weight=0.)
    assert 0<=log['alignment_loss']<=2
    assert log['head_grad_norm']>0
    assert not torch.equal(net.forecast_head.weight,old['forecast_head.weight'])
    assert torch.equal(net.fast_origin,old['fast_origin'])


def test_aligned_objective_really_optimizes_post_adaptation_risk_not_only_logs_it():
    net,e=setup()
    log=train_step(net,torch.optim.SGD(net.parameters(),lr=0.),e,
                   mode='aligned',signal_weight=.2)
    expected=.5*(log['risk_before']+log['risk_after'])+.2*log['signal_loss']+.1*log['alignment_loss']
    assert abs(log['objective']-expected)<1e-6
    assert log['risk_after_weight']==.5


def test_after_adaptation_task_gradient_reaches_forecast_head_without_proxy_penalties(monkeypatch):
    # With BOTH direct signal loss and alignment penalty removed, only the
    # post-adaptation risk can connect this head to the training objective.
    monkeypatch.setattr(joint_meta,'alignment_loss',lambda signal,risk: signal.sum()*0.)
    net,e=setup()
    log=train_step(net,torch.optim.SGD(net.parameters(),lr=0.),e,
                   mode='aligned',signal_weight=0.)
    assert log['alignment_loss']==0.
    assert log['head_grad_norm']>0.
