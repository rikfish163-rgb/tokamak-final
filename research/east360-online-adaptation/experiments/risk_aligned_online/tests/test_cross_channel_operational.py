import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cross_channel_operational import make_episode, episode_loss
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import make_model,signals


def source_shot(disruptive=True):
    s=signals(180);s['device']='J-TEXT'
    return dict(s,is_disrupt=disruptive,down=.18 if disruptive else None,
                valid=np.ones(180,bool),y=np.zeros(180))


def test_episode_uses_real_clock_and_shot_outcome_not_early_point_label():
    s=source_shot();e=make_episode(s,np.random.default_rng(13))
    assert e['label']==1 and not s['y'].any()
    assert e['ticks']==list(range(7,e['ticks'][-1]+1,32))
    for t,q in zip(e['ticks'],e['queries']):
        assert all(t<j<=t+32 for j in q)
        assert all(10<round((s['down']-s['times'][j])*1000,9)<=125 for j in q)
    with pytest.raises(ValueError,match='source'):
        make_episode(dict(s,device='EAST'),np.random.default_rng(13))


def test_source_segment_loss_gradient_only_prepares_decoder():
    torch.set_num_threads(2)
    m=make_model();m.decoder.requires_grad_(True)
    s=source_shot(False);e=make_episode(s,np.random.default_rng(131))
    base={k:v.clone() for k,v in m.base.state_dict().items()}
    objective,info=episode_loss(m,s,e,mode='risk',threshold=.72)
    objective.backward()
    assert torch.isfinite(objective) and info['queries']>0
    assert sum(float(p.grad.norm()) for p in m.decoder.parameters())>0
    assert all(p.grad is None for p in m.base.parameters())
    for k,v in m.base.state_dict().items():torch.testing.assert_close(v,base[k],rtol=0,atol=0)
    assert not m.memory.any()


def test_whole_shot_inner_state_matches_production_replay():
    from experiments.risk_aligned_online.cross_channel_prefix import predict_shot
    torch.set_num_threads(2)
    m=make_model();s=source_shot(False)
    e=make_episode(s,np.random.default_rng(131))
    _,info=episode_loss(m,s,e,mode='signal',threshold=.72)
    raw={k:v for k,v in s.items() if k not in {'y','valid','is_disrupt','down'}}
    _,audit=predict_shot(m,raw,intra=True)
    ref=[r for r in audit['attempts'] if r['update_index']==e['ticks'][-1]][0]
    assert info['final_fast']==ref['fast_after']
