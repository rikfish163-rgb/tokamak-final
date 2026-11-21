import copy
import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.change_conditioned_20261023.model import ChangeConditioned,patches,fixed_scale


def make(kind='change'):
    torch.set_num_threads(2);torch.manual_seed(101)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval()
    return ChangeConditioned(base,torch.ones(23),kind).eval()


def test_scaling_missing_and_exact_three_patch_support():
    x=torch.arange(128,dtype=torch.float32)[None,:,None].expand(1,-1,23).clone()
    m=torch.ones_like(x,dtype=torch.bool);m[:,:,2]=False
    model=make();p=patches(model.base,x,m)
    scale,count=fixed_scale([p])
    assert scale[2]==1 and count[2]==0 and (scale>=1e-3).all()
    assert count[0]==15
    with pytest.raises(ValueError):ChangeConditioned(model.base,torch.zeros(23),'change')
    for key in ('level','change'):
        model=make(key)
        with torch.no_grad():model.readout.weight.normal_(0,.05)
        a=model(x,m,mode='fit');changed=x.clone();changed[:,-8:]+=5
        b=model(changed,m,mode='fit')
        torch.testing.assert_close(a['fast'],b['fast'],rtol=0,atol=0)
        assert a['local_observations']==13*22
        assert not a['target_mask'][:,2].any()
        missing=m.clone();missing[:,:120]=False
        r=model(x,missing,mode='fit')
        assert not r['target_mask'].any() and not r['correction'].any()
        torch.testing.assert_close(r['fast'],model.prior[None],rtol=0,atol=0)


def test_initial_identity_scope_and_actual_risk_gradients():
    model=make();x=torch.randn(4,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2055
    from experiments.risk_aligned_online.local_transition_20261023.model import LocalTransition
    original=LocalTransition(copy.deepcopy(model.base)).eval()
    torch.testing.assert_close(model(x,mask,mode='fit')['logit'],original(x,mask,mode='prior')['logit'],rtol=0,atol=0)
    with torch.no_grad():model.readout.weight.normal_(0,.05)
    result=model(x,mask,mode='fit')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(result['logit'],torch.tensor([0.,1.,0.,1.]))
    loss.backward()
    for name,param in model.named_parameters():
        if name.startswith('base.'):assert param.grad is None
        else:assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name


def test_rms_only_depends_on_passed_training_statistics():
    model=make();x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    p=patches(model.base,x,mask)
    a,n=fixed_scale([p]);b,nn=fixed_scale([p,p])
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(nn,2*n)


def test_stream_is_causal_resets_and_rejects_outcomes():
    from experiments.risk_aligned_online.change_conditioned_20261023.run import predict
    model=make();model.requires_grad_(False)
    with torch.no_grad():model.readout.weight.normal_(0,.03)
    models={'level':copy.deepcopy(model),'change':model};models['level'].key_kind='level'
    rng=np.random.default_rng(101);s={'values':rng.normal(size=(160,23)).astype(np.float32),'available':np.ones((160,23),bool)}
    a,_=predict(models,s,device='cpu');other={k:v.copy() for k,v in s.items()};other['values'][120:]+=8
    b,_=predict(models,other,device='cpu');c,_=predict(models,s,device='cpu')
    for key in a:
        np.testing.assert_array_equal(a[key][:120],b[key][:120]);np.testing.assert_array_equal(a[key],c[key])
        assert np.isfinite(a[key]).all()
    with pytest.raises(ValueError):predict(models,{**s,'outcome':1},device='cpu')
