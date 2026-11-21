from copy import deepcopy

import pytest

from experiments.risk_aligned_online.tests.test_event_projection_verification import valid_audit
from experiments.risk_aligned_online.verify_event_dynamics import verify_audit


def state_audit():
    a = valid_audit()
    a['attempts'][0]['matured_objective'] = dict(mode='event', horizons=[5, 10, 15, 20, 25],
        valid_common_targets=575, state_dimensions=5, target_detached=True,
        outcome_access=False, predicted_signals_used_for_risk=False, hard_event_pseudo_labels=False)
    return a


def test_valid_state_update_and_legacy_control():
    verify_audit(state_audit(), mode='event')
    verify_audit(valid_audit(), mode='control')


@pytest.mark.parametrize('field,value', [
    ('target_detached', False), ('outcome_access', True),
    ('predicted_signals_used_for_risk', True), ('hard_event_pseudo_labels', True),
    ('horizons', [5, 10, 15, 20, 30]), ('mode', 'random'),
])
def test_wrong_or_future_state_objective_rejected(field, value):
    a = deepcopy(state_audit())
    a['attempts'][0]['matured_objective'][field] = value
    with pytest.raises(AssertionError):
        verify_audit(a, mode='event')


@pytest.mark.parametrize('mutation', ['same_time', 'no_receipt', 'persistent', 'nonzero_start'])
def test_state_update_cannot_bypass_timing_or_reset_checks(mutation):
    a = deepcopy(state_audit())
    r = a['attempts'][0]
    if mutation == 'same_time': r['effective_index'] = 25
    elif mutation == 'no_receipt': del r['matured_objective']
    elif mutation == 'persistent': a['persistent'] = {'accepted': True}
    else: r['fast_at_origin']['l2'] = 1.
    with pytest.raises(AssertionError):
        verify_audit(a, mode='event')
