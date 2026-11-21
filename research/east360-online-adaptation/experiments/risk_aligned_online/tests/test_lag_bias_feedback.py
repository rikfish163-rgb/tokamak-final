import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.lag_bias_feedback_20261022.learning import split_residuals,feedback_loss,adapt_arrived
from experiments.risk_aligned_online.simple23 import Simple23,Config
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback,initial_encoder


def test_constant_offset_removed_but_later_growth_remains():
    p=torch.full((4,25,23),2.,requires_grad=True);truth=torch.zeros_like(p);mask=torch.ones_like(p,dtype=torch.bool)
    bias,valid=split_residuals(p,truth,mask)
    assert not bias.requires_grad
    recent=feedback_loss(p,truth,mask,bias,valid,correct=False)
    corrected=feedback_loss(p,truth,mask,bias,valid,correct=True)
    assert recent.item()==1.5 and corrected.item()==0
    q=p.detach().clone();q[2:]+=1;q.requires_grad_(True)
    b,v=split_residuals(q,truth,mask)
    loss=feedback_loss(q,truth,mask,b,v,correct=True)
    assert loss.item()==.5
    gradient,=torch.autograd.grad(loss,q)
    assert not gradient[:2].any() and gradient[2:].abs().sum()>0


def test_masked_old_targets_do_not_create_false_bias_or_supervision():
    p=torch.full((4,25,23),2.);truth=torch.zeros_like(p);mask=torch.ones_like(p,dtype=torch.bool)
    mask[:2,:,0]=False;p[:2,:,0]=999
    mask[0,0,1]=False;p[0,0,1]=999
    b,v=split_residuals(p,truth,mask)
    assert not v[0] and b[1].item()==2
    assert feedback_loss(p,truth,mask,b,v,correct=True).item()==0
    mask[:2]=False
    b,v=split_residuals(p,truth,mask);assert not v.any()
    with pytest.raises(ValueError):feedback_loss(p,truth,mask,b,v,correct=True)


def test_new_ramp_survives_but_sustained_old_offset_is_not_identifiable():
    p=torch.full((4,25,23),2.);truth=torch.zeros_like(p);mask=torch.ones_like(p,dtype=torch.bool)
    p[2:]+=torch.linspace(0,1,50).reshape(2,25,1)
    b,v=split_residuals(p,truth,mask)
    expected=torch.nn.functional.smooth_l1_loss(torch.linspace(0,1,50),torch.zeros(50))
    torch.testing.assert_close(feedback_loss(p,truth,mask,b,v,correct=True),expected)
    # A constant dangerous residual already present in OLD is canceled too.
    # Algebra cannot identify whether its origin is physical or instrumental.
    p.fill_(5.);b,v=split_residuals(p,truth,mask)
    assert feedback_loss(p,truth,mask,b,v,correct=True).item()==0


def fixture():
    torch.set_num_threads(2);torch.manual_seed(3)
    base=Simple23(Config(dim=16,heads=2,fast_site='tokens',fast_rule='normalized'),np.zeros(23),np.ones(23))
    model=PairedFeedback(base,np.ones(23));model.set_stage('frozen')
    x=np.random.default_rng(3).normal(size=(550,23)).astype(np.float32)
    return model,dict(values=x,available=np.ones_like(x,bool))


def test_update_only_uses_arrived_data_and_temporary_parameters():
    model,s=fixture();before={k:v.clone() for k,v in model.state_dict().items()};fast=initial_encoder(model)
    new,r=adapt_arrived(model,s,256,fast,device='cpu',correct=True)
    assert r['calibration_origins']==[155,180] and r['gradient_origins']==[205,230]
    assert r['calibration_last_target']==205 and r['first_gradient_target']==206
    assert r['first_affected_prediction']==256 and r['last_observation']==255
    assert 0<r['parameter_delta_norm']<=.001001
    changed=dict(s,values=s['values'].copy());changed['values'][256:]=1e5
    new2,r2=adapt_arrived(model,changed,256,fast,device='cpu',correct=True)
    assert r==r2
    for k in new:torch.testing.assert_close(new[k],new2[k],rtol=0,atol=0)
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError):adapt_arrived(model,dict(s,y=0),256,fast,device='cpu',correct=True)
    with pytest.raises(ValueError):adapt_arrived(model,s,128,fast,device='cpu',correct=True)


def test_missing_feedback_explicitly_returns_unchanged_state():
    model,s=fixture();s['available'][156:206]=False;fast=initial_encoder(model)
    after,r=adapt_arrived(model,s,256,fast,device='cpu',correct=True)
    assert after is fast and r['gradient_steps']==0 and r['reason']=='no_shared_observed_targets'


def test_full_prediction_causal_prefix_and_reset_between_shots():
    from experiments.risk_aligned_online.lag_bias_feedback_20261022.run import predict_arms
    model,s=fixture()
    first,updates=predict_arms(model,s,device='cpu')
    changed=dict(s,values=s['values'].copy());changed['values'][400:]+=2
    other,_=predict_arms(model,changed,device='cpu')
    repeated,updates2=predict_arms(model,s,device='cpu')
    for k in first:
        np.testing.assert_array_equal(first[k][:400],other[k][:400])
        np.testing.assert_array_equal(first[k],repeated[k])
    assert updates==updates2
    assert [r['first_affected_prediction'] for r in updates['lag_bias']]==[256,384,512]
    assert np.max(np.abs(first['lag_bias']-first['static']))>0
