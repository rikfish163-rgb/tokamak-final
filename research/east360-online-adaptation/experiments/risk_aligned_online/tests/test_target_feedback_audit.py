import numpy as np
import pytest

from experiments.risk_aligned_online.target_feedback_meta_20261023.audit import recount


def test_recount_uses_first_crossing_strict_lead_and_does_not_mutate():
    times=np.array([0.,.070,.100,.110])
    score=np.array([.1,.8,.9,.9])
    original=score.copy()
    r=recount(10,times,score,True,.100,.8)
    assert r==dict(shot=10,is_disrupt=True,alarm=True,lead_ms=30.,timely=False)
    assert recount(10,times,score,False,None,.8)['alarm']
    np.testing.assert_array_equal(score,original)
    assert not recount(10,times,np.array([0.,0.,0.,1.]),True,.100,.8)['alarm']
    with pytest.raises(AssertionError):recount(10,times,np.array([0.,np.nan,0.,0.]),False,None,.8)
