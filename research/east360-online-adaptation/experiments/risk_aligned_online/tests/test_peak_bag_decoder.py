"""An ended shot's emitted high-risk endpoint must not be omitted by training."""
import math

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
from experiments.risk_aligned_online.peak_bag_decoder import (
    BagRollingDecoder, bag_warning_loss, train_bag_pair)


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(131)
    return DecoderFfnModel(Simple23(Config(fast_site='tokens',fast_rule='normalized'),
                                   torch.zeros(23),torch.ones(23))).eval()


def shot(sid,label):
    rng=np.random.default_rng(sid)
    scores=np.zeros(230,np.float32); scores[180]=.9
    return dict(shot=sid,device='EAST',is_disrupt=label,horizon=125.,
        down=.229 if label else None,times=np.arange(230)/1000.,
        values=rng.normal(size=(230,23)).astype(np.float32),
        available=np.ones((230,23),bool),valid=np.ones(230,bool),
        _sealed_scores=scores,_bag_queries=(126,180))


def test_actual_safe_peak_receives_gradient_not_just_reference():
    logits=torch.logit(torch.tensor([.428803,.890773])).requires_grad_()
    loss=bag_warning_loss(logits,label=0,boundary=math.log(.725895/(1-.725895)))
    gradient,=torch.autograd.grad(loss,logits)
    assert gradient[0]==0 and gradient[1]>0
    assert torch.sigmoid(logits[1]-.1*gradient[1])<torch.sigmoid(logits[1])


@pytest.mark.parametrize('label,direction',[(0,1),(1,-1)])
def test_tied_maxima_distribute_gradient_without_double_weight(label,direction):
    logits=torch.tensor([1.,1.],requires_grad=True)
    gradient,=torch.autograd.grad(bag_warning_loss(logits,label=label,boundary=.7),logits)
    assert gradient[0]==gradient[1] and gradient[0]*direction>0
    single=torch.tensor([1.],requires_grad=True)
    g,=torch.autograd.grad(bag_warning_loss(single,label=label,boundary=.7),single)
    torch.testing.assert_close(gradient.sum(),g.sum())


@pytest.mark.parametrize('bad',[torch.tensor([]),torch.tensor([float('nan')]),torch.ones(2,2)])
def test_invalid_bag_is_explicit_error(bad):
    with pytest.raises(ValueError):bag_warning_loss(bad,label=0,boundary=.7)


def test_endpoints_are_reference_and_only_already_sealed_peak():
    m=model(); rule=BagRollingDecoder(m,mode='early_fixed')
    s=shot(1,True); rule.cache[(0,1)]=126
    _,ref=rule._prepare_shot(s,phase=0,device='cpu')
    prepared,_=rule._prepare_shot(s,phase=0,device='cpu')
    assert ref==126 and prepared['_bag_queries']==(126,180)
    assert np.array_equal(s['_sealed_scores'],prepared['_sealed_scores'])
    # Disrupted samples strictly later than the >30ms eligibility limit are
    # NOT made positive instances, even if their score is the absolute maximum.
    s['_sealed_scores'][205]=1.
    prepared,_=rule._prepare_shot(s,phase=0,device='cpu')
    assert prepared['_bag_queries']==(126,180)


def test_identical_reference_and_emitted_peak_is_not_duplicated():
    m=model(); rule=BagRollingDecoder(m,mode='early_fixed'); s=shot(1,False)
    rule.cache[(0,1)]=180
    prepared,_=rule._prepare_shot(s,phase=0,device='cpu')
    assert prepared['_bag_queries']==(180,)


def test_real_bag_backward_only_changes_auxiliary_head():
    m=model(); m.requires_grad_(False);m.decoder.requires_grad_(True)
    before={k:v.clone() for k,v in m.base.state_dict().items()}
    result=train_bag_pair(m,[(shot(1,False),126),(shot(2,True),126)],
        torch.optim.Adam(m.decoder.parameters(),lr=.0003),boundary=.7,mode='risk_after_update')
    assert result['risk_gradient_norm']>0 and result['decoder_delta_norm']>0
    assert all(r['queries']==[126,180] for r in result['records'])
    assert not m.memory.any()
    for k,v in m.base.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


@pytest.mark.parametrize('bad',['unsealed','ineligible','future_index','duplicate','wrong_optimizer'])
def test_training_rejects_wrong_labels_or_endpoints(bad):
    m=model();m.requires_grad_(False);m.decoder.requires_grad_(True)
    pair=[(shot(1,False),126),(shot(2,True),126)]
    optimizer=torch.optim.Adam(m.decoder.parameters())
    if bad=='unsealed':del pair[0][0]['_sealed_scores']
    elif bad=='ineligible':pair[1][0]['_bag_queries']=(126,205)
    elif bad=='future_index':pair[0][0]['_bag_queries']=(126,230)
    elif bad=='duplicate':pair[0][0]['_bag_queries']=(126,126)
    else:optimizer=torch.optim.Adam([m.memory])
    with pytest.raises(ValueError):
        train_bag_pair(m,pair,optimizer,boundary=.7,mode='risk_after_update')
