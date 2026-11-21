from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.cross_channel_prefix import (
    CrossChannelModel, channel_mask, episode_windows, predict_shot,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.fast_update import normalized_step


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_model(site='embedding'):
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23))
    return CrossChannelModel(base, site=site).eval().requires_grad_(False)


def signals(n=100):
    x = np.random.default_rng(13).normal(size=(n,23)).astype(np.float32)
    return dict(values=x, available=np.ones_like(x,bool), times=np.arange(n)/1000., shot=1,device='EAST')


def test_channel_masks_partition_observations_even_with_one_arrived_point():
    a=torch.zeros(2,128,23,dtype=torch.bool);a[:,-1]=True;a[0,-1,2]=False
    masks=[channel_mask(a,phase=p) for p in range(4)]
    assert torch.equal(torch.stack(masks).sum(0),a.int())
    assert all(m.any() for m in masks)
    with pytest.raises(ValueError):channel_mask(a,phase=4)


@pytest.mark.parametrize('site',['embedding','ffn'])
def test_zero_state_is_original_classifier_and_auxiliary_head_cannot_change_risk(site):
    m=make_model(site);x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool)
    expected=m.base(x,a,m.base.initial_fast(2).detach().requires_grad_())['logit']
    torch.testing.assert_close(m(x,a)['logit'],expected,atol=0,rtol=0)
    with torch.no_grad():m.decoder.weight.add_(20)
    torch.testing.assert_close(m(x,a)['logit'],expected,atol=0,rtol=0)


def test_embedding_update_changes_only_value_columns_not_availability_or_bias():
    m=make_model('embedding');fast=torch.ones(1,m.temporary_dim)*.01
    replacements=m.overrides(fast[0])
    assert set(replacements)=={'embedding.weight'} and m.temporary_dim==11776
    diff=(replacements['embedding.weight']-m.base.embedding.weight).reshape(64,8,46)
    assert diff[:,:,:23].abs().sum()>0 and torch.equal(diff[:,:,23:],torch.zeros_like(diff[:,:,23:]))


@pytest.mark.parametrize('site',['embedding','ffn'])
def test_hidden_values_are_targets_only_and_meta_gradient_exists(site):
    m=make_model(site);m.decoder.requires_grad_(True)
    x=torch.randn(1,128,23);a=torch.zeros_like(x,dtype=torch.bool);a[:,-8:]=True
    hidden=channel_mask(a,phase=1);changed=x.clone();changed[hidden]+=10
    fast=m.initial_fast(1).requires_grad_()
    p,t,h=m.reconstruction(x,a,fast,phase=1)
    p2,t2,h2=m.reconstruction(changed,a,fast,phase=1)
    torch.testing.assert_close(p,p2,atol=0,rtol=0)
    assert not t.requires_grad and not torch.equal(t,t2) and torch.equal(h,h2)
    loss=m.reconstruction_loss(x,a,fast,phase=1)
    grad,=torch.autograd.grad(loss,fast,create_graph=True)
    assert grad.norm()>0
    candidate=normalized_step(fast,fast.detach(),grad)
    outer=F.binary_cross_entropy_with_logits(m(x,a,candidate)['logit'],torch.ones(1))
    derivatives=torch.autograd.grad(outer,tuple(m.decoder.parameters()))
    assert all(torch.isfinite(d).all() for d in derivatives)
    assert sum(float(d.norm()) for d in derivatives)>0
    assert all(p.grad is None for p in m.base.parameters())


def test_source_episode_query_has_only_the_same_record_history_plus_one_point():
    e=dict(x=torch.zeros(1,128,23),a=torch.ones(1,128,23,dtype=torch.bool),
           truth=torch.ones(1,25,23),mask=torch.ones(1,25,23,dtype=torch.bool),
           qx=torch.full((1,128,23),2.),qa=torch.ones(1,128,23,dtype=torch.bool))
    x,a,qx,qa=episode_windows(e,8)
    assert a[:,:-8].sum()==0 and a[:,-8:].all()
    assert qa[:,:-9].sum()==0 and qa[:,-9:].all()
    assert torch.equal(x[:,-8:],e['truth'][:,-8:])
    assert torch.equal(qx,e['qx'])


def test_early_feedback_is_causal_discarded_and_only_changes_subsequent_predictions():
    m=make_model();s=signals();initial={k:v.clone() for k,v in m.state_dict().items()}
    p,a=predict_shot(m,s,intra=True)
    static,_=predict_shot(m,s,intra=False)
    changed=deepcopy(s);changed['values'][60:]+=100
    p2,a2=predict_shot(m,changed,intra=True)
    repeat,again=predict_shot(m,s,intra=True)
    np.testing.assert_array_equal(p[:8],static[:8])
    np.testing.assert_array_equal(p[:60],p2[:60])
    np.testing.assert_array_equal(p,repeat)
    assert [r['update_index'] for r in a['attempts']]==[7,39,71]
    assert a['attempts'][:2]==a2['attempts'][:2]
    assert a['fast_initial']['l2']==again['fast_initial']['l2']==0 and a['fast_discarded']
    for r in a['attempts']:
        assert r['effective_index']==r['update_index']+1 and r['observed_through']==r['update_index']
    for k,v in m.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)


def test_short_records_and_missing_targets_are_kept_and_labels_rejected():
    m=make_model();s=signals(7)
    p,a=predict_shot(m,s,intra=True)
    assert len(p)==7 and not a['attempts']
    s=signals();s['available'][:]=False
    p,a=predict_shot(m,s,intra=True)
    assert len(p)==100 and a['accepted_updates']==0
    with pytest.raises(ValueError):predict_shot(m,dict(s,is_disrupt=True),intra=True)
