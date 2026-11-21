import numpy as np
import pytest

from experiments.risk_aligned_online.diverse_support_20261023.selection import (
    describe, rank_coordinates, select, identity_folds,
)


def test_describe_masks_and_rejects_labels():
    x=np.ones((100,23));m=np.ones_like(x,dtype=bool)
    m[:,2]=False;x[:,2]=999.;s=dict(values=x,available=m)
    d=describe(s,np.zeros(23),np.ones(23))
    assert d.shape==(115,) and np.isfinite(d).all()
    assert d[2]==d[25]==d[48]==d[71]==d[94]==0
    assert d[92]==1 and d[69]==0
    with pytest.raises(ValueError):describe(dict(s,is_disrupt=True),np.zeros(23),np.ones(23))


def test_rank_ties_and_scale_invariance():
    x=np.array([[0.,7.],[0.,7.],[2.,7.],[3.,7.]])
    r=rank_coordinates(x)
    np.testing.assert_allclose(r[:,0],[1/6,1/6,2/3,1])
    assert np.all(r[:,1]==.5)
    np.testing.assert_array_equal(r,rank_coordinates(x*10+4))


def test_selection_budget_identity_order_and_coverage():
    ids=[9,3,5,7];x=np.array([[10.],[0.],[1.],[2.]])
    chosen=select(ids,x,2,'diverse',131)
    assert chosen==[5,9]
    perm=[2,0,3,1]
    assert select([ids[i] for i in perm],x[perm],2,'diverse',131)==chosen
    assert select(ids,x,2,'uniform',131)==select([ids[i] for i in perm],x[perm],2,'uniform',131)
    assert len(set(select(ids,x,4,'diverse',131)))==4
    with pytest.raises(ValueError):select(ids,x,5,'diverse',131)
    with pytest.raises(ValueError):select([9,9,5,7],x,2,'diverse',131)


def test_identity_folds_disjoint_and_complete():
    plans=identity_folds(list(range(85)))
    held=[s for p in plans for s in p['held']]
    assert len(held)==len(set(held))==85
    for p in plans:
        assert not set(p['candidate'])&set(p['held'])
        assert len(p['held']) in (21,22) and len(p['candidate']) in (63,64)


def test_single_class_budget_does_not_invent_positives():
    from types import SimpleNamespace
    from experiments.risk_aligned_online.diverse_support_20261023.learning import observed_pools,target_batch
    s=dict(shot=1,values=np.ones((200,23),np.float32),available=np.ones((200,23),bool),
           y=np.zeros(200),valid=np.ones(200,bool))
    groups=observed_pools([s])
    x,m,y,r=target_batch([s],groups,np.random.default_rng(131),SimpleNamespace(window=128),device='cpu')
    assert x.shape==(32,128,23) and not y.any() and len(r)==32
    s['y'][150:]=1
    x,m,y,r=target_batch([s],observed_pools([s]),np.random.default_rng(131),SimpleNamespace(window=128),device='cpu')
    assert int(y.sum())==16 and all((a['end']>=150)==bool(a['label']) for a in r)
