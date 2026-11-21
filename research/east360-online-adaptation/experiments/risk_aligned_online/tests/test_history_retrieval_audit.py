import numpy as np
import pytest

from experiments.risk_aligned_online.history_retrieval_20261023.audit import (
    choose_independently, recount, contribution_summary,
)


def test_recount_uses_shots_not_endpoint_counts_and_checks_bce():
    rows=[]
    for arm in ('nearest','initial','centroid'):
        for sid,y,n,z in [(1,0,1,-1.),(2,0,5,-3.),(3,1,2,2.)]:
            for i in range(n):
                rows.append(dict(arm=arm,shot=sid,query=i,label=y,
                    group='safe' if y==0 else 'disruptive_timely',
                    logit_before=z,logit_after=z,
                    before=float(np.logaddexp(0,z)-y*z),after=float(np.logaddexp(0,z)-y*z)))
    summary,gate=recount(rows)
    expected=(np.logaddexp(0,-1.)+np.logaddexp(0,-3.))/2
    assert summary['nearest']['groups']['safe']['before']==expected
    assert not gate
    rows[0]['before']+=.01
    with pytest.raises(AssertionError):recount(rows)


def test_separate_retrieval_preserves_class_and_unique_shot_constraints():
    keys=np.array([[1.,0.],[1.,0.],[.8,.6],[0.,1.],
                   [1.,0.],[.8,.6],[0.,1.]])
    memory=dict(keys=keys,shots=np.array([1,1,2,3,4,5,6]),labels=np.array([0,0,0,0,1,1,1]))
    selected,distances=choose_independently(memory,np.array([1.,0.]),'nearest',1)
    assert selected==[2,3,4,5]
    np.testing.assert_allclose(distances,[.2,1.,0.,.2])
    a=choose_independently(memory,np.array([1.,0.]),'centroid',1)
    b=choose_independently(memory,np.array([0.,1.]),'centroid',1)
    assert a==b


def test_gradient_contributions_account_for_mean_and_descent_sign():
    # g=(.5,0), query gradient=(-1,0); descent increases query BCE to first order.
    item=np.array([[4.,0.],[2.,0.],[-2.,0.],[-2.,0.]])
    result=contribution_summary(item,np.array([-1.,0.]))
    assert result['weighted_dots']==[-1.,-.5,.5,.5]
    assert result['total_dot']==-.5
    assert result['direction_harmful']
    with pytest.raises(ValueError):contribution_summary(item[:3],np.array([-1.,0.]))
