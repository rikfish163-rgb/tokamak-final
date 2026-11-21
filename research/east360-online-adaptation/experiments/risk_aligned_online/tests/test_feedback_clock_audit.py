import json

import numpy as np
import pytest

from experiments.risk_aligned_online.feedback_clock_20261023.audit import exposure


def test_exposure_uses_real_times_and_strict_timely_boundary():
    t=np.arange(700)/1000
    r=exposure(t,.699,[256,384,512,640])
    assert r['last_timely_index']==668
    assert r['updates_in_30_125ms']==1
    assert r['last_update_age_ms']==pytest.approx(29.)
    # Nonuniform grid: never silently interpret a sample count as milliseconds.
    t=np.array([0.,.1,.2,.25,.27,.29,.3])
    r=exposure(t,.3,[2,4,6])
    assert r['last_timely_index']==3
    assert r['updates_in_30_125ms']==1
    assert r['last_update_age_ms']==pytest.approx(150.)


def test_no_timely_point_or_no_update_is_explicit():
    r=exposure(np.array([.99,1.]),1.,[])
    assert r['last_timely_index'] is None and r['last_update_age_ms'] is None
    r=exposure(np.arange(100)/1000,.099,[])
    assert r['last_timely_index']==68 and r['updates_in_30_125ms']==0
    assert r['last_update_age_ms'] is None
    with pytest.raises(ValueError):exposure(np.arange(4)/1000,.003,[0])


def test_exposure_is_strict_json_serializable():
    record=exposure(np.arange(700)/1000,.699,[256,384,512,640])
    assert json.loads(json.dumps(record,allow_nan=False))==record
    assert type(record['updates_in_30_125ms']) is int
