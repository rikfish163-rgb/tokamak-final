import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.continual_episode_20261023.learning import (
    Access, episode_indices, select_pair, meta_loss, after_shot_step, predict, trim_memory,
)
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def shot(sid=1):
    rng=np.random.default_rng(sid);n=800
    return dict(shot=sid,values=rng.normal(size=(n,23)).astype('float32'),
        available=np.ones((n,23),bool),times=np.arange(n)/1000,
        y=(np.arange(n)>=675).astype('float32'),valid=np.ones(n,bool),
        is_disrupt=True,down=.8)


def test_access_requires_exact_order_seal_and_one_reveal():
    a=Access([1,2],[9]);calls=[]
    with pytest.raises(ValueError):a.begin(2)
    a.begin(1)
    with pytest.raises(ValueError):a.reveal(lambda s:calls.append(s))
    with pytest.raises(ValueError):a.seal('bad')
    a.seal('a'*64);assert a.reveal(lambda s:calls.append(s) or (True,.8))==(True,.8)
    assert calls==[1] and a.allowed=={9,1}
    with pytest.raises(ValueError):a.reveal(lambda s:calls.append(s))
    a.finish();a.begin(2)


def test_ordered_episodes_never_use_future_feedback_or_late_positive():
    s=shot();es=episode_indices(s)
    assert {e['label'] for e in es}=={0,1}
    assert len(es)<=16
    for e in es:
        assert 256<=e['count']<=e['end']<e['count']+128
        if e['label']:assert 30<round((s['down']-s['times'][e['end']])*1000,9)<=125
    a=select_pair({1:(s,es),2:(shot(2),es)},1,np.random.default_rng(131))
    assert a[0][0]['shot']==1 and a[1][0]['shot']==2
    with pytest.raises(ValueError):episode_indices({k:v for k,v in s.items() if k!='y'})


def test_meta_gradient_matches_finite_difference_and_freezes_readout():
    torch.set_num_threads(1);torch.manual_seed(131);_,m=model('increment');m=m.double()
    with torch.no_grad():m.base.forecast_head.weight.normal_(0,.02)
    s=shot();e=episode_indices(s)[-1]
    names=set(m.encoder_names())
    for k,p in m.named_parameters():p.requires_grad_(k in names)
    loss,_,_=meta_loss(m,[(s,e)],device='cpu',adapt=True)
    p=m.base.embedding.weight;g=torch.autograd.grad(loss,p)[0];idx=np.unravel_index(int(g.abs().argmax()),g.shape)
    original=float(p[idx].detach());eps=1e-5
    vals=[]
    for sign in (1,-1):
        with torch.no_grad():p[idx]=original+sign*eps
        vals.append(float(meta_loss(m,[(s,e)],device='cpu',adapt=True)[0].detach()))
    with torch.no_grad():p[idx]=original
    assert np.isclose(float(g[idx]),(vals[0]-vals[1])/(2*eps),rtol=3e-3,atol=2e-5)
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    opt=torch.optim.Adam([p for k,p in m.named_parameters() if k in names],lr=3e-5)
    receipt=after_shot_step(m,opt,[(s,e)],adapt=True,device='cpu')
    assert receipt['parameter_delta_norm']>0
    assert any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items() if k in names)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items() if k not in names)
    assert not any(p.requires_grad for p in m.parameters())


def test_stream_prefix_invariance_reset_and_no_parent_writeback():
    torch.set_num_threads(1);torch.manual_seed(44);_,m=model('increment')
    with torch.no_grad():m.base.forecast_head.weight.normal_(0,.02)
    other=copy.deepcopy(m);s=shot();signal={k:s[k][:385].copy() for k in ('values','available')}
    before={k:v.clone() for k,v in m.state_dict().items()}
    a,ar=predict(m,other,signal,device='cpu')
    changed={k:v.copy() for k,v in signal.items()};changed['values'][384:]=1000
    b,br=predict(m,other,changed,device='cpu')
    c,cr=predict(m,other,signal,device='cpu')
    assert ar==br==cr
    for arm in a:
        np.testing.assert_array_equal(a[arm][:384],b[arm][:384])
        np.testing.assert_array_equal(a[arm],c[arm])
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError):predict(m,other,dict(signal,y=np.zeros(385)),device='cpu')


def test_memory_caps_shots_not_windows_and_keeps_latest_per_outcome():
    memory={i:(dict(shot=i,is_disrupt=i%2==0),[{'label':i%2}]*i) for i in range(1,10)}
    trim_memory(memory,cap=2)
    assert list(memory)==[6,7,8,9]


def test_safe_outcome_preserves_absent_disruption_time():
    from experiments.risk_aligned_online.continual_episode_20261023.run import outcome_record
    row=outcome_record(4,(False,None),'a'*64,0)
    assert row['down'] is None and not row['is_disrupt']
    with pytest.raises(ValueError):outcome_record(4,(True,None),'a'*64,0)
