import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.increment_forecast_20261023.learning import (
    IncrementFeedback, prepare_head, feedback_loss, update, predict, train_head_step,
)
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def model(mode):
    original=small_model();result=IncrementFeedback(copy.deepcopy(original.base),mode)
    result.load_state_dict(original.state_dict());prepare_head(result)
    return original,result


def test_zero_increment_is_persistence_without_any_classifier_change():
    original,m=model('increment');x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    a,b=original(x,mask),m(x,mask)
    torch.testing.assert_close(a['logit'],b['logit'],rtol=0,atol=0)
    expected=torch.asinh(x[:,-1:]).expand(-1,25,-1)
    torch.testing.assert_close(b['forecast'],expected,rtol=0,atol=0)
    _,abs_model=model('absolute')
    assert not abs_model(x,mask)['forecast'].any()
    with pytest.raises(ValueError):IncrementFeedback(original.base,'unknown')


def test_head_fit_keeps_every_other_weight_and_filters_missing_last_value():
    _,m=model('increment');before={k:v.clone() for k,v in m.state_dict().items()}
    x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool);mask[:,-1,0]=False
    target=torch.randn(2,25,23);tm=torch.ones_like(target,dtype=torch.bool)
    pred=m(x,mask)['forecast'];first=feedback_loss(m,pred,x,mask,target,tm)
    target[:,:,0]=999
    torch.testing.assert_close(first,feedback_loss(m,pred,x,mask,target,tm),rtol=0,atol=0)
    opt=torch.optim.Adam(m.base.forecast_head.parameters(),lr=3e-4)
    train_head_step(m,opt,x,mask,target,tm)
    assert not torch.equal(m.base.forecast_head.weight,before['base.forecast_head.weight'])
    for k,v in m.state_dict().items():
        if not k.startswith('base.forecast_head.'):torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_updates_are_past_only_and_repeating_shot_resets_fast_weights():
    _,m=model('increment')
    with torch.no_grad():m.base.forecast_head.weight.normal_(0,.01)
    s=dict(values=np.random.default_rng(5).normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    frozen={k:v.clone() for k,v in m.state_dict().items()}
    r,u,p=predict(m,s,device='cpu');r2,u2,p2=predict(m,s,device='cpu')
    assert u==u2 and len(u)==2 and p['counts'].tolist()==[256]
    for k in r:np.testing.assert_array_equal(r[k],r2[k])
    np.testing.assert_array_equal(r['static'][:256],r['intra'][:256])
    s['values'][256:]=999
    changed,cu,cp=predict(m,s,device='cpu')
    assert cu[0]==u[0]
    np.testing.assert_array_equal(changed['intra'][:256],r['intra'][:256])
    np.testing.assert_array_equal(cp['intra'][0],p['intra'][0])
    for k,v in m.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(m,dict(s,y=1),device='cpu')


def test_no_observations_means_explicit_no_update_not_zero_improvement():
    _,m=model('increment');fast=initial_encoder(m)
    s=dict(values=np.zeros((256,23),np.float32),available=np.zeros((256,23),bool))
    after,r=update(m,s,256,fast,device='cpu')
    assert r['gradient_steps']==0 and r['observed_targets']==0
    for k in fast:torch.testing.assert_close(after[k],fast[k],rtol=0,atol=0)
