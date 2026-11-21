import numpy as np
import pytest

from experiments.risk_aligned_online.simple23 import Config
from experiments.risk_aligned_online.paired_feedback_20261022.sampling import (
    source_groups, balanced_pair,
)


def shots():
    def shot(sid,disrupt,y):
        return dict(shot=sid,device='J-TEXT',is_disrupt=disrupt,
            values=np.ones((12,23),'float32')*sid,available=np.ones((12,23),bool),
            y=np.asarray(y),valid=np.ones(12,bool),event_target=np.full(12,-1))
    return [shot(11,True,[0]*8+[1]*4),shot(22,False,[0]*12)]


def test_unknown_disruptive_prefix_excluded_only_in_safe_negative_arm():
    data=shots()
    a=source_groups(data,policy='balanced')
    b=source_groups(data,policy='safe_negative')
    assert {i for i,_ in a[0]}=={0,1}
    assert {i for i,_ in b[0]}=={1}
    assert all(np.array_equal(x[1],y[1]) for x,y in zip(a[1],b[1]))


def test_exact_class_balance_and_safe_negative_identity():
    data=shots();cfg=Config(window=8,patch=4,dim=16,heads=2)
    group=source_groups(data,policy='safe_negative')
    rng=np.random.default_rng(9)
    for _ in range(20):
        x,m,y,receipt=balanced_pair(data,group,rng,cfg,'cpu')
        assert y.tolist()==[0.,1.]
        assert receipt[0]['shot']==22 and receipt[1]['shot']==11
        assert x.shape==(2,8,23) and m.dtype.is_floating_point is False


def test_event_mark_does_not_change_original_binary_labels():
    data=shots();data[0]['event_target'][:8]=2
    original=data[0]['y'].copy()
    source_groups(data,policy='safe_negative')
    np.testing.assert_array_equal(data[0]['y'],original)
    with pytest.raises(ValueError): source_groups(data,policy='invented')
