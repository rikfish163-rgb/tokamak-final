import numpy as np

from experiments.risk_aligned_online.continual_episode_20261023.audit import recount


def test_recount_keeps_strict_deadline_and_safe_missing_time():
    r=recount(np.array([.069,.070,.071]),np.array([0.,.8,.9]),True,.1,.5)
    assert r['alarm'] and r['lead_ms']==30 and not r['timely']
    r=recount(np.array([.069,.070,.071]),np.array([.8,.8,.9]),True,.1,.5)
    assert r['lead_ms']==31 and r['timely']
    r=recount(np.array([0.,1.]),np.array([.6,.1]),False,None,.5)
    assert r['alarm'] and not r['timely'] and r['lead_ms'] is None
