import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.latent_completion_20261028.model import Imputer
from experiments.risk_aligned_online.gradient_split_20261029.model import objective,predict,step


def setup():
    torch.set_num_threads(1);torch.manual_seed(41);torch.backends.mha.set_fastpath_enabled(False)
    return (Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False),
            Imputer().eval().requires_grad_(False))


def test_each_stop_really_removes_only_the_declared_gradient():
    p=torch.randn(2,17,64,requires_grad=True);t=torch.randn_like(p,requires_grad=True)
    v=torch.ones(2,17,dtype=torch.bool)
    for kind in ('stop_target','stop_prediction'):
        loss=objective(p,t,v,kind)
        gp,gt=torch.autograd.grad(loss,(p,t),allow_unused=True)
        assert (gt is None)==(kind=='stop_target')
        assert (gp is None)==(kind=='stop_prediction')


def test_finite_updates_and_source_frozen():
    base,net=setup();saved={k:v.clone() for k,v in base.state_dict().items()}
    x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    for kind in ('stop_target','stop_prediction'):
        state,info=step(base,net,x,a,torch.zeros(128),kind,False)
        assert info['updates']==1 and info['gradient_norm']>0
        assert 0<float(state.norm())<=1
    assert all(torch.equal(v,saved[k]) for k,v in base.state_dict().items())
    assert all(p.grad is None for p in net.parameters())


def test_prefix_sealed_future_independent_and_state_reset():
    base,net=setup();rng=np.random.default_rng(71)
    signal=dict(values=rng.normal(size=(288,23)).astype('float32'),available=np.ones((288,23),bool))
    first,info=predict(base,net,signal,None)
    changed={k:v.copy() for k,v in signal.items()};changed['values'][160:]*=100
    other,_=predict(base,net,changed,None)
    repeat,_=predict(base,net,signal,None)
    for k in first:
        assert np.array_equal(first[k][:160],other[k][:160])
        assert np.array_equal(first[k],repeat[k])
        assert np.array_equal(first[k][:128],first['cold'][:128])
    assert all(r['updates']==2 for r in info.values())
    assert not np.array_equal(first['stop_target'][128:],first['stop_target_reverse'][128:])


def test_short_shot_is_scored_without_update_or_removal():
    base,net=setup()
    signal=dict(values=np.ones((75,23),np.float32),available=np.ones((75,23),bool))
    scores,info=predict(base,net,signal,None)
    assert all(len(v)==75 and np.array_equal(v,scores['cold']) for v in scores.values())
    assert all(r['updates']==0 for r in info.values())
