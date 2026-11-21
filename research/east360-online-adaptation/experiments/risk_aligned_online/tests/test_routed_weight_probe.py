import numpy as np
import pytest

from experiments.risk_aligned_online.routed_weight_probe import diagnostic_endpoint


def test_missed_disruption_uses_strict_timely_endpoint_and_matured_actual_origin():
    times=np.arange(201)/1000.
    result=diagnostic_endpoint(times,np.zeros(201),True,.2,.7)
    assert result==dict(endpoint=169,origin=100,maturity=125,effective=126)
    assert round((.2-times[result['endpoint']])*1000,9)>30


def test_safe_false_alarm_uses_first_actual_crossing_not_a_later_easy_window():
    times=np.arange(201)/1000.;scores=np.zeros(201);scores[130]=.8
    assert diagnostic_endpoint(times,scores,False,None,.7)['endpoint']==130
    scores[10]=.9
    assert diagnostic_endpoint(times,scores,False,None,.7) is None


def test_insufficient_timely_observations_are_reported_not_dropped_silently():
    assert diagnostic_endpoint(np.arange(51)/1000.,np.zeros(51),True,.05,.7) is None
    with pytest.raises(ValueError): diagnostic_endpoint(np.arange(51)/1000.,np.zeros(50),True,.05,.7)
