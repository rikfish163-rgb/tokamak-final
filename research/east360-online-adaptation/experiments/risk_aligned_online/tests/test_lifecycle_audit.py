import numpy as np

from experiments.risk_aligned_online.lifecycle_stream_20261025.audit import classify


def test_independent_score_uses_strict30_and_only_preterminal_predictions():
    times=np.array([.960,.970,.980,1.001])
    assert not classify(times,np.array([0.,1.,1.,1.]),True,1.,.5)['timely']
    assert classify(times,np.ones(4),True,1.,.5)['timely']
    assert not classify(times,np.array([0.,0.,0.,1.]),True,1.,.5)['alarm']
    assert classify(times,np.array([0.,0.,0.,1.]),False,None,.5)['alarm']
