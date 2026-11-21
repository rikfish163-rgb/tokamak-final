import numpy as np
import pytest

from experiments.risk_aligned_online.preparation_factors_20261023.test_stream import select_preparation


def test_fixed_budget_and_disjoint_test_identities():
    ids=list(range(85)); held=list(range(1000,1360))
    x=np.random.default_rng(8).normal(size=(85,115)); labels={s:s<8 for s in ids}
    selected=select_preparation(ids,x,labels,held)
    assert set(selected)=={'blind','aware'}
    for values in selected.values():
        assert len(values)==len(set(values))==20
        assert set(values)<=set(ids) and not set(values)&set(held)
    assert sum(labels[s] for s in selected['aware'])==4
    for bad in ([0]+held[1:],held[:-1],held[:-1]+held[:1]):
        with pytest.raises(ValueError):select_preparation(ids,x,labels,bad)


def test_test_labels_cannot_enter_preparation_selection():
    ids=list(range(85)); held=list(range(1000,1360))
    x=np.ones((85,115));labels={s:s<8 for s in ids}
    with pytest.raises(ValueError):select_preparation(ids,x,{**labels,1000:True},held)
    with pytest.raises(ValueError):select_preparation(ids[:-1],x[:-1],{s:labels[s] for s in ids[:-1]},held)
