import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.matched_forecast_meta_20261022.learning import observed_step,episode,episode_pools
from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback,initial_encoder
from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import adapt_arrived


def fixture():
    torch.set_num_threads(2);torch.manual_seed(19)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized',dim=16,heads=2),np.zeros(23),np.ones(23))
    model=PairedFeedback(base,np.ones(23));model.set_stage('frozen')
    signal=dict(values=np.random.default_rng(19).normal(size=(600,23)).astype(np.float32),
                available=np.ones((600,23),bool))
    return model,signal


def test_forward_step_matches_deployed_operator_and_cannot_see_future():
    model,s=fixture();fast=initial_encoder(model)
    a,_=observed_step(model,s,256,fast,device='cpu',create_graph=False)
    b,_=adapt_arrived(model,s,256,fast,device='cpu')
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    s['values'][256:]=999
    c,_=observed_step(model,s,256,fast,device='cpu',create_graph=False)
    for k in a:torch.testing.assert_close(a[k],c[k],rtol=0,atol=0)
    with pytest.raises(ValueError):observed_step(model,dict(s,y=1),256,fast,device='cpu',create_graph=False)


def test_risk_after_alone_reaches_forecast_head_not_frozen_classifier():
    model,s=fixture();model.base.forecast_head.requires_grad_(True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    r=episode(model,s,550,1.,device='cpu',create_graph=True)
    assert r['all_update_counts']==[256,384,512] and r['differentiated_counts']==[384,512]
    r['risk_after'].backward()
    assert sum(float(p.grad.norm()) for p in model.base.forecast_head.parameters())>0
    assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('base.forecast_head.'))
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    replay=initial_encoder(model)
    for count in (256,384,512):replay,_=adapt_arrived(model,s,count,replay,device='cpu')
    for k in replay:torch.testing.assert_close(replay[k],r['fast'][k],rtol=0,atol=0)


def test_source_episode_pool_has_training_role_and_real_label_boundaries():
    _,s=fixture()
    shot=dict(s,device='J-TEXT',shot=101,y=np.r_[np.zeros(500),np.ones(100)],valid=np.ones(600,bool))
    pools,ledger=episode_pools([shot],[101])
    assert all(e>=256 for _,ends in pools[0] for e in ends)
    assert min(pools[1][0][1])==500
    with pytest.raises(ValueError):episode_pools([dict(shot,device='EAST')],[101])
    with pytest.raises(ValueError):episode_pools([shot],[102])
