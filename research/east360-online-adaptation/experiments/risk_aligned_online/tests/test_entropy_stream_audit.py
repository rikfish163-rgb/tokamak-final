import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.entropy_stream_20261023.audit import recount,stable_entropy


def test_independent_recount_uses_first_alarm_and_strict30ms():
    times=np.array([.17,.18,.2,.21]);scores=np.array([.8,.1,.1,.9])
    r=recount(1,times,scores,True,.2,.7)
    assert r['alarm'] and r['lead_ms']==30 and not r['timely']
    times[0]=.169
    assert recount(1,times,scores,True,.2,.7)['timely']
    assert recount(1,times,scores,False,0.,.7)['alarm']
    scores[:3]=0
    assert not recount(1,times,scores,True,.2,.7)['alarm']


@pytest.mark.parametrize('z',[-100.,-5.,0.,5.,100.])
def test_stable_entropy_matches_two_class_probability(z):
    scores=torch.tensor([0.,z],dtype=torch.float64)
    reference=-(scores.softmax(0)*scores.log_softmax(0)).sum()
    assert stable_entropy(z)==pytest.approx(float(reference),abs=1e-14)
