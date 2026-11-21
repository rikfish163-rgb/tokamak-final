import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.internal_fast_window_20261023.model import InnerLayer,WindowAdapter


def setup():
    torch.set_num_threads(1);torch.manual_seed(131)
    base=Simple23(Config(fast_site='tokens'),torch.zeros(23),torch.ones(23)).eval()
    return WindowAdapter(base),torch.randn(3,128,23)


def test_zero_initialization_matches_parent_and_inner_update_has_no_writeback():
    m,x=setup();mask=torch.ones_like(x,dtype=torch.bool);saved=copy.deepcopy(m.state_dict())
    expected=m.base(x,mask)['logit']
    for mode in ('bypass','off','on','reverse'):
        z,info=m(x,mask,mode=mode)
        torch.testing.assert_close(z,expected,atol=1e-6,rtol=1e-6)
    for k,v in saved.items():assert torch.equal(v,m.state_dict()[k])
    assert not any(p.requires_grad for p in m.base.parameters())


def test_analytic_step_equals_autograd_of_masked_square_loss():
    torch.manual_seed(33);layer=InnerLayer(64).double()
    h=torch.randn(2,17,64,dtype=torch.double);valid=torch.ones(2,17,dtype=torch.bool)
    valid[0,:4]=False
    q,k,v=layer.coordinates(h)
    w=layer.origin.expand(2,-1,-1).detach().clone().requires_grad_(True)
    observed=valid.clone();observed[:,-1]=False
    loss=((k@w-v).square().sum(-1)*observed).sum(-1)/(2*observed.sum(-1))
    g=torch.autograd.grad(loss.sum(),w)[0]
    actual,info=layer.fast_weights(h,valid,mode='on')
    expected=w-layer.eta*g
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_inner_has_task_gradient_and_no_batch_or_missing_target_contamination():
    m,x=setup();mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():m.layer.output.weight.normal_(0,.03)
    z,_=m(x,mask,mode='on');F.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.,0.])).backward()
    assert m.layer.value.weight.grad.abs().max()>0
    assert not any(p.grad is not None for p in m.base.parameters())
    one,_=m(x[:1],mask[:1],mode='on')
    torch.testing.assert_close(one,z[:1],atol=2e-6,rtol=2e-6)
    layer=m.layer;h=torch.randn(1,17,64);valid=torch.ones(1,17,dtype=torch.bool);valid[:,2:5]=False
    a,_=layer.fast_weights(h,valid,mode='on');h[:,2:5]=1000
    b,_=layer.fast_weights(h,valid,mode='on');torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_no_effective_observations_means_no_inner_step_and_reset():
    m,x=setup();h=torch.randn(2,17,64);valid=torch.zeros(2,17,dtype=torch.bool);valid[:,-1]=True
    w,receipt=m.layer.fast_weights(h,valid,mode='on')
    torch.testing.assert_close(w,m.layer.origin.expand(2,-1,-1),atol=0,rtol=0)
    assert receipt['updated'].tolist()==[False,False]
    with torch.no_grad():m.layer.output.weight.normal_(0,.03)
    a,_=m(x,torch.ones_like(x,dtype=torch.bool),mode='on')
    b,_=m(x,torch.ones_like(x,dtype=torch.bool),mode='on')
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    with pytest.raises(ValueError):m(x,torch.ones_like(x,dtype=torch.bool),mode='unknown')


def test_capacity_control_can_learn_from_identical_parent_initialization():
    m,x=setup();mask=torch.ones_like(x,dtype=torch.bool)
    z,_=m(x,mask,mode='off')
    F.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.,0.])).backward()
    assert m.layer.output.weight.grad.abs().max()>1e-8
