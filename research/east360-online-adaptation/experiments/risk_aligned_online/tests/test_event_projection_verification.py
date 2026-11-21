"""Regression checks for the event pilot's independent evidence recount."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.risk_aligned_online.verify_event_projection import (
    recount_prediction, verify_prediction_audit,
)


def valid_audit():
    return dict(intra=True, fast_discarded=True, persistent=None,
        model_parameter_change_norm=0., model_buffer_change_norm=0.,
        attempted_updates=1, accepted_updates=1, score_count=100,
        attempts=[dict(origin_index=0, maturity_index=25,
            predictions_sealed_through_index=25, effective_index=26,
            fast_at_origin={'l2': 0.}, fast_before={'l2': 0.},
            accepted=True, loss_before=1., loss_after=.9,
            fast_change_norm=.05, fast_norm_after=.05)])


def test_valid_disposable_update():
    verify_prediction_audit(valid_audit(), mode='control')


@pytest.mark.parametrize('mutation', ['same_point_update', 'persistent', 'nonzero_start', 'future_truth'])
def test_audit_rejects_contract_violations(mutation):
    a = deepcopy(valid_audit())
    if mutation == 'same_point_update':
        a['attempts'][0]['effective_index'] = 25
    elif mutation == 'persistent':
        a['persistent'] = {'accepted': True}
    elif mutation == 'nonzero_start':
        a['attempts'][0]['fast_at_origin']['l2'] = .1
    else:
        a['attempts'][0]['maturity_index'] = 24
    with pytest.raises(AssertionError):
        verify_prediction_audit(a, mode='control')


def test_strict_lead_boundary_and_mask():
    r = dict(shot=1, is_disrupt=True, down=1., times=np.array([.96, .97, .98]),
             scores=np.array([.9, .8, .7]), valid=np.array([False, True, True]))
    p = recount_prediction(r, .8)
    assert p['alarm'] and p['lead_ms'] == 30. and not p['timely']
    r['valid'][0] = True
    assert recount_prediction(r, .8)['timely']


def test_safe_shot_alarm_without_lead():
    r = dict(shot=1, is_disrupt=False, down=None, times=np.array([1.]),
             scores=np.array([.8]), valid=np.array([True]))
    assert recount_prediction(r, .8) == dict(shot=1, is_disrupt=False, alarm=True,
                                            lead_ms=None, timely=False)
