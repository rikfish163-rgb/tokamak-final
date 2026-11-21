import numpy as np

from experiments.risk_aligned_online.horizon_memory_20261023.audit import independently_score


def test_deadline_is_strict_and_first_alarm_not_peak_defines_lead():
    times=np.array([.96,.97,.98])
    thirty=independently_score(1,True,1.,times,np.array([.1,.8,.9]),.8)
    assert thirty['lead_ms']==30. and not thirty['timely']
    forty=independently_score(1,True,1.,times,np.array([.8,.9,.99]),.8)
    assert forty['lead_ms']==40. and forty['timely']
    noalarm=independently_score(2,False,None,times,np.array([.1,.2,.3]),.8)
    assert not noalarm['alarm'] and noalarm['lead_ms'] is None
    safe=independently_score(2,False,None,times,np.array([.8,.2,.3]),.8)
    assert safe['alarm'] and not safe['timely']
