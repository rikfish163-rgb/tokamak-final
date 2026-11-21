import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.local_transition_20261023.model import fit_local, LocalTransition
from experiments.risk_aligned_online.simple23 import Config,Simple23


def test_masked_closed_form_objective_and_unobserved_prior():
    torch.manual_seed(1)
    k=torch.randn(2,14,3,dtype=torch.double,requires_grad=True)
    y=torch.randn(2,14,4,dtype=torch.double)
    m=torch.rand(2,14,4)>.3; m[:,:,3]=False
    prior=torch.randn(3,4,dtype=torch.double,requires_grad=True)
    w=fit_local(k,y,m,prior)
    for b in range(2):
        for c in range(4):
            take=m[b,:,c]; n=int(take.sum())
            if not n:
                torch.testing.assert_close(w[b,:,c],prior[:,c],rtol=0,atol=0)
            else:
                residual=k[b,take]@w[b,:,c]-y[b,take,c]
                stationary=k[b,take].T@residual/n+.1*(w[b,:,c]-prior[:,c])
                torch.testing.assert_close(stationary,torch.zeros(3,dtype=torch.double),atol=1e-12,rtol=0)
    assert torch.autograd.gradcheck(lambda kk,pp:fit_local(kk,y,m,pp),(k,prior))


def test_zero_projection_equals_parent_and_last_patch_not_used_for_fitting():
    torch.set_num_threads(2); torch.manual_seed(3)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval()
    model=LocalTransition(base)
    # The archived paired predictor uses a grad-enabled unfused attention path.
    x=torch.randn(3,128,23,requires_grad=True); m=torch.ones_like(x,dtype=torch.bool)
    parent=base(x,m)['logit']; out=model(x,m,mode='fit')
    torch.testing.assert_close(out['logit'],parent,rtol=0,atol=0)
    altered=x.clone(); altered[:,-8:]+=99
    fresh=model(altered,m,mode='fit')
    torch.testing.assert_close(fresh['fast'],out['fast'],rtol=0,atol=0)
    assert not torch.equal(fresh['innovation'],out['innovation'])
    m[:,:,5]=False
    missing=model(x,m,mode='fit')
    torch.testing.assert_close(missing['fast'][:,:,5],model.prior[:,5].expand(3,-1),rtol=0,atol=0)
    assert not missing['target_mask'][:,5].any() and not missing['innovation'][:,5].any()


def test_outer_loss_reaches_new_parameters_and_leaves_base_frozen():
    torch.manual_seed(5)
    model=LocalTransition(Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)))
    with torch.no_grad():model.readout.weight.normal_(0,.01)
    before={k:v.clone() for k,v in model.base.state_dict().items()}
    out=model(torch.randn(4,128,23),torch.ones(4,128,23,dtype=torch.bool),mode='fit')
    loss=out['logit'].square().mean()+out['forecast_loss']; loss.backward()
    assert model.key.weight.grad.abs().sum()>0 and model.prior.grad.abs().sum()>0
    assert model.readout.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    for k,v in before.items():torch.testing.assert_close(v,model.base.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError):model(torch.randn(4,128,23),torch.ones(4,128,23,dtype=torch.bool),mode='unknown')


def test_stream_prefix_causality_and_no_cross_shot_state():
    from experiments.risk_aligned_online.local_transition_20261023.run import predict
    torch.set_num_threads(2); torch.manual_seed(13)
    model=LocalTransition(Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)))
    with torch.no_grad():model.readout.weight.normal_(0,.02)
    model.eval().requires_grad_(False)
    models={'static_trained':model,'adaptive_trained':model}
    before={k:v.clone() for k,v in model.state_dict().items()}
    rng=np.random.default_rng(13)
    signal={'values':rng.normal(size=(96,23)).astype(np.float32),
            'available':np.ones((96,23),dtype=bool)}
    first,_=predict(models,signal,device='cpu')
    changed={k:v.copy() for k,v in signal.items()}; changed['values'][64:]+=20
    second,_=predict(models,changed,device='cpu')
    repeated,_=predict(models,signal,device='cpu')
    for arm in first:
        np.testing.assert_array_equal(first[arm][:64],second[arm][:64])
        np.testing.assert_array_equal(first[arm],repeated[arm])
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(models,{**signal,'is_disrupt':True},device='cpu')
