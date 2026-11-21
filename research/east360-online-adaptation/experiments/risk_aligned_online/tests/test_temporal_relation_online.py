import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.temporal_relation_20261029.model import OrderJudge,permutation
from experiments.risk_aligned_online.temporal_relation_20261029.online import step,predict


def setup():
    torch.set_num_threads(1);torch.manual_seed(41);torch.backends.mha.set_fastpath_enabled(False)
    return (Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False),
            OrderJudge().eval().requires_grad_(False))


def test_reversed_label_has_opposite_gradient_and_all_parameters_stay_frozen():
    base,net=setup();saved={k:v.clone() for k,v in base.state_dict().items()}
    x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool);p=permutation(1,np.random.default_rng(7))
    positive,ip=step(base,net,x,a,p,torch.zeros(128),'relation')
    negative,im=step(base,net,x,a,p,torch.zeros(128),'wrong_label')
    assert ip['updates']==im['updates']==1
    assert torch.allclose(positive,-negative,atol=1e-7,rtol=1e-5)
    assert 0<float(positive.norm())<=.050001
    assert all(torch.equal(v,saved[k]) for k,v in base.state_dict().items())
    assert all(p.grad is None for p in (*base.parameters(),*net.parameters()))


def test_ineligible_window_does_not_invent_supervision():
    base,net=setup();x=torch.ones(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    s=torch.randn(128)*.01;p=permutation(1,np.random.default_rng(7))
    updated,info=step(base,net,x,a,p,s,'relation')
    assert torch.equal(updated,s) and info['updates']==0 and not info['eligible']
    x=torch.randn_like(x);a[0,0,0]=False
    updated,info=step(base,net,x,a,p,s,'relation')
    assert torch.equal(updated,s) and not info['eligible']


def test_prefix_is_causal_and_temporary_state_resets_between_shots():
    base,net=setup();rng=np.random.default_rng(71)
    signal=dict(values=rng.normal(size=(288,23)).astype('float32'),available=np.ones((288,23),bool))
    first,info=predict(base,net,signal,None)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][160:]*=100
    other,_=predict(base,net,changed,None);repeat,_=predict(base,net,signal,None)
    for k in first:
        assert np.array_equal(first[k][:160],other[k][:160])
        assert np.array_equal(first[k],repeat[k])
        assert np.array_equal(first[k][:128],first['cold'][:128])
    assert all(r['updates']==2 and r['eligible']==2 and r['opportunities']==2 for r in info.values())
    assert not np.array_equal(first['relation'][128:],first['wrong_label'][128:])


def test_short_shot_is_fully_scored_even_without_an_auxiliary_pair():
    base,net=setup()
    signal=dict(values=np.ones((75,23),np.float32),available=np.ones((75,23),bool))
    scores,info=predict(base,net,signal,None)
    assert all(len(v)==75 and np.array_equal(v,scores['cold']) for v in scores.values())
    assert all(r['updates']==0 and r['opportunities']==0 for r in info.values())
