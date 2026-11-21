import numpy as np
import pytest

from experiments.risk_aligned_online.trajectory_rank_20261023.audit import check_history_loss, recount, check_key_recount, temporal_spread
from experiments.risk_aligned_online.trajectory_rank_20261023.learning import ARMS


def test_independent_rank_loss_recount_and_mismatch_detection():
    z=[1.,2.,3.,4.]
    value=float(np.mean([np.logaddexp(0.,z[i]-z[j]) for i in (0,1) for j in (2,3)]))
    r=dict(selected_labels=[0,0,1,1],loss_kind='pairwise',historical_logits_before=z,
           historical_logits_after=[v+5 for v in z],loss_before=value,loss_after=value)
    check_history_loss(r)
    r['loss_after']+=.01
    with pytest.raises(AssertionError):check_history_loss(r)


def test_recount_enforces_primary_controls_and_equal_shot_weights():
    rows=[]
    for arm in ARMS:
        for sid in range(20):
            y=int(sid<4);z=(2*y-1)*2.
            after=z+(2*y-1)*(.2 if arm=='trajectory_rank' else .1)
            for i in range(1 if sid==4 else 3):
                rows.append(dict(arm=arm,shot=sid,query=i,label=y,group='disruptive_timely' if y else 'safe',
                    logit_before=z,logit_after=after,before=float(np.logaddexp(0.,z)-y*z),
                    after=float(np.logaddexp(0.,after)-y*after)))
    summary,gate=recount(rows)
    assert gate and summary['trajectory_rank']['improved_disruptive_shots']==4
    rows[0]['before']=999
    with pytest.raises(AssertionError):recount(rows)


def test_independent_key_reduction_allows_roundoff_not_changed_representation():
    a=np.array([[.1,.2,.3]])
    assert check_key_recount(a,a+np.finfo(float).eps)<1e-15
    with pytest.raises(AssertionError):check_key_recount(a,a+1e-8)
    constant=np.tile(np.array([1.,0.]),3)[None]/np.sqrt(3)
    moving=np.array([[1.,0.,0.,1.,-1.,0.]])/np.sqrt(3)
    assert temporal_spread(constant)[0]==0
    assert temporal_spread(moving)[0]>0
