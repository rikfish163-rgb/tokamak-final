import numpy as np
import torch
from experiments.risk_aligned_online.internal_fast_window_20261023.model import InnerLayer
from experiments.risk_aligned_online.curvature_fast_window_20261023.model import CurvatureLayer


def test_reproduces_correlated_key_overshoot_and_controls_it_without_labels():
    torch.manual_seed(131);raw=InnerLayer(64).double();safe=CurvatureLayer(64).double()
    with torch.no_grad():
        raw.key.weight.zero_();raw.key.weight[:,0]=8;raw.value.weight.zero_()
    safe.load_state_dict(raw.state_dict())
    h=torch.zeros(1,17,64,dtype=torch.double);h[:,:,0]=1
    valid=torch.ones(1,17,dtype=torch.bool)
    _,old=raw.fast_weights(h,valid,mode='on');_,new=safe.fast_weights(h,valid,mode='on')
    assert old['loss_after']>old['loss_before']
    assert new['loss_after']<new['loss_before']
    assert new['effective_eta']<.125


def test_curvature_step_matches_explicit_autograd_and_masked_loss():
    torch.manual_seed(44);layer=CurvatureLayer(64).double()
    h=torch.randn(3,17,64,dtype=torch.double);valid=torch.ones(3,17,dtype=torch.bool);valid[0,:5]=False
    _,k,v=layer.coordinates(h);observed=valid.clone();observed[:,-1]=False
    count=observed.sum(-1);w=layer.origin.expand(3,-1,-1).detach().clone().requires_grad_(True)
    loss=((k@w-v).square()*observed[:,:,None]).sum((1,2))/(2*count)
    g=torch.autograd.grad(loss.sum(),w,create_graph=True)[0]
    trace=(k.square().sum(-1)*observed).sum(-1)/count
    eta=layer.eta/(1+trace)
    result,receipt=layer.fast_weights(h,valid,mode='on')
    torch.testing.assert_close(result,w-eta[:,None,None]*g,atol=1e-12,rtol=1e-12)
    explicit=((k@result-v).square()*observed[:,:,None]).sum((1,2))/(2*count)
    torch.testing.assert_close(receipt['loss_after'],explicit,atol=1e-12,rtol=1e-12)
    assert (receipt['loss_after']<=receipt['loss_before']).all()
    for mode in ('off','reverse'):
        _,r=layer.fast_weights(h,valid,mode=mode)
        assert (r['loss_after']<=r['loss_before']+1e-12).all()


def test_zero_keys_no_division_by_zero_and_outer_gradient_exists():
    layer=CurvatureLayer(64)
    h=torch.randn(2,17,64);valid=torch.ones(2,17,dtype=torch.bool)
    with torch.no_grad():layer.key.weight.zero_()
    w,r=layer.fast_weights(h,valid,mode='on')
    assert torch.isfinite(w).all();torch.testing.assert_close(w,layer.origin.expand(2,-1,-1))
    with torch.no_grad():layer.key.weight.normal_(0,.1)
    w,r=layer.fast_weights(h,valid,mode='on');w.square().sum().backward()
    assert layer.key.weight.grad.abs().max()>0 and layer.value.weight.grad.abs().max()>0


def test_off_control_has_identical_forward_and_backward_to_original():
    torch.manual_seed(131);old=InnerLayer(64);new=CurvatureLayer(64)
    with torch.no_grad():old.output.weight.normal_(0,.03)
    new.load_state_dict(old.state_dict())
    h=torch.randn(32,17,64);valid=torch.ones(32,17,dtype=torch.bool)
    a,_=old(h,valid,mode='off');b,_=new(h,valid,mode='off')
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.square().sum().backward();b.square().sum().backward()
    for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
        if p.grad is None:assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
