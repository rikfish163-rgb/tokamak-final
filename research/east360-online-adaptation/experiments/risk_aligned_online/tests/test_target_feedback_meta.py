import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.target_feedback_meta_20261023.learning import (
    episode_pool, block_risk, block_episode)
from experiments.risk_aligned_online.tests.test_matched_forecast_meta import fixture
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import adapt_arrived
from experiments.risk_aligned_online.target_feedback_meta_20261023.stream import predict_five
from experiments.risk_aligned_online.deepcoral_prefix_20261022.run import predict_pairs


def make_shot(disrupt=True):
    model,signal=fixture()
    times=np.arange(600)/1000.
    return model,dict(signal,device='EAST',shot=1001,times=times,down=.599,
                      is_disrupt=disrupt,valid=np.ones(600,bool),
                      y=((.599-times)<=.125).astype(float) if disrupt else np.zeros(600))


def test_pools_keep_complete_shot_roles_and_timely_targets_only():
    _,s=make_shot()
    rows=episode_pool([s],[1001])
    assert len(rows)==1 and rows[0]['shot']==1001
    for block in rows[0]['blocks']:
        ends=np.asarray(block['ends']);lead=np.round((s['down']-s['times'][ends])*1000,9)
        assert block['count']%128==0 and block['count']>=256
        assert (ends>=block['count']).all() and (ends<block['count']+128).all()
        assert ((lead>30)&(lead<=125)).all()
    with pytest.raises(ValueError): episode_pool([s],[1002])
    with pytest.raises(ValueError): episode_pool([dict(s,device='J-TEXT')],[1001])


def test_max_risk_targets_worst_safe_point_and_best_timely_point():
    z=torch.tensor([-4.,1.,-1.],requires_grad=True)
    negative=block_risk(z,False,0.,'maximum')
    negative.backward()
    assert z.grad[1]>0 and z.grad[0]==0 and z.grad[2]==0
    z.grad=None
    block_risk(z,True,0.,'maximum').backward()
    assert z.grad[1]<0 and z.grad[0]==0 and z.grad[2]==0
    with pytest.raises(ValueError):block_risk(z,True,0.,'fake')


def test_meta_risk_reaches_only_forecast_head_and_matches_deployed_fast_state():
    model,shot=make_shot();model.base.forecast_head.requires_grad_(True)
    signal={k:shot[k] for k in ('values','available')}
    before={k:v.clone() for k,v in model.state_dict().items()}
    r=block_episode(model,signal,512,list(range(512,544)),True,boundary=0.,
                    objective='maximum',device='cpu',create_graph=True)
    r['risk_after'].backward()
    assert sum(float(p.grad.norm()) for p in model.base.forecast_head.parameters())>0
    assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('base.forecast_head.'))
    fast=initial_encoder(model)
    for count in (256,384,512):fast,_=adapt_arrived(model,signal,count,fast,device='cpu')
    for k in fast:torch.testing.assert_close(fast[k],r['fast'][k],rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    changed=dict(signal,values=signal['values'].copy());changed['values'][512:]=777
    other=block_episode(model,changed,512,list(range(512,544)),True,boundary=0.,
                        objective='maximum',device='cpu',create_graph=False)
    for k in fast:torch.testing.assert_close(other['fast'][k],fast[k],rtol=0,atol=0)
    with pytest.raises(ValueError):block_episode(model,dict(signal,y=1),512,[512],True,
        boundary=0.,objective='maximum',device='cpu',create_graph=False)


def test_deduplicated_static_replay_equals_original_operator_and_resets():
    model,s=fixture()
    expected,updates=predict_pairs({'warm':model},s,device='cpu')
    actual,receipt=predict_five({'warm':model},s,device='cpu')
    np.testing.assert_array_equal(actual['static'],expected['warm_static'])
    np.testing.assert_array_equal(actual['warm_intra'],expected['warm_intra'])
    assert updates==receipt
    repeat,_=predict_five({'warm':model},s,device='cpu')
    np.testing.assert_array_equal(repeat['warm_intra'],actual['warm_intra'])
