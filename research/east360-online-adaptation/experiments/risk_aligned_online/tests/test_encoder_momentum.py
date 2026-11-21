import torch
from ..encoder_momentum_20261023.learning import optimizer
from ..east_joint_feedback_20261023.learning import enable
from .test_credit_horizon import fixture


def test_only_encoder_first_moment_is_disabled_and_learning_rates_unchanged():
    m,_=fixture();enable(m);opt=optimizer(m)
    enc,head=opt.param_groups
    assert enc['betas']==(0.,.999) and head['betas']==(.9,.999)
    assert enc['lr']==1e-5 and head['lr']==1e-4
    assert {id(p) for p in enc['params']}=={id(p) for k,p in m.named_parameters() if k in m.encoder_names()}
    assert {id(p) for p in head['params']}=={id(p) for p in m.base.forecast_head.parameters()}


def test_zero_new_gradient_has_no_encoder_momentum_but_head_keeps_its_history():
    m,_=fixture();enable(m);opt=optimizer(m)
    for group in opt.param_groups:
        for p in group['params']:p.grad=torch.ones_like(p)
    opt.step()
    states=[[p.detach().clone() for p in g['params']] for g in opt.param_groups]
    for group in opt.param_groups:
        for p in group['params']:p.grad.zero_()
    opt.step()
    for p,s in zip(opt.param_groups[0]['params'],states[0]):torch.testing.assert_close(p,s,rtol=0,atol=0)
    assert any(not torch.equal(p,s) for p,s in zip(opt.param_groups[1]['params'],states[1]))
