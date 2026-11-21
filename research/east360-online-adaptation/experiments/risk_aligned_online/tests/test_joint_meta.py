from copy import deepcopy
from dataclasses import replace

import torch

from experiments.risk_aligned_online.tests.test_fast_meta import fixture
from experiments.risk_aligned_online.joint_meta import train_step


def setup():
    net,e=fixture()
    net.cfg=replace(net.cfg,fast_site='tokens')
    return net,e


def test_joint_meta_updates_encoder_and_risk_but_preserves_origin_and_scaler():
    net,e=setup()
    before=deepcopy(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0.)
    log=train_step(net,opt,e,mode='meta',signal_weight=.2)
    assert log['encoder_grad_norm']>0
    assert log['head_grad_norm']>0
    assert not torch.equal(net.embedding.weight,before['embedding.weight'])
    assert not torch.equal(net.risk_head.weight,before['risk_head.weight'])
    for name in ('fast_origin','center','scale'):
        assert torch.equal(net.state_dict()[name],before[name])
    assert all(p.requires_grad for p in net.parameters())


def test_meta_risk_gradient_reaches_forecast_head_without_auxiliary_loss():
    net,e=setup()
    other=deepcopy(net)
    meta=train_step(net,torch.optim.SGD(net.parameters(),lr=.01),e,
                    mode='meta',signal_weight=0.)
    joint=train_step(other,torch.optim.SGD(other.parameters(),lr=.01),e,
                     mode='joint',signal_weight=0.)
    assert meta['head_grad_norm']>0
    assert joint['head_grad_norm']==0
    assert not torch.equal(net.forecast_head.weight,other.forecast_head.weight)


def test_joint_ignores_inner_update_for_risk_objective():
    net,e=setup()
    log=train_step(net,torch.optim.SGD(net.parameters(),lr=0.),e,
                    mode='joint',signal_weight=.2)
    assert abs(log['objective']-(log['risk_before']+.2*log['signal_loss']))<1e-6
