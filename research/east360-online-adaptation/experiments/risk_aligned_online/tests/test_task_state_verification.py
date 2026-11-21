from copy import deepcopy

import pytest

from experiments.risk_aligned_online.tests.test_event_dynamics_verification import state_audit
from experiments.risk_aligned_online.verify_task_state import verify_audit


def task_audit(mode='risk'):
    audit = deepcopy(state_audit())
    audit['attempts'][0]['matured_objective'].update(
        mode=mode, state_dimensions=1 if mode == 'risk' else 6,
        risk_coordinate_count=1, risk_coordinate_scale=2.,
        risk_teacher_is_physical_truth=False)
    return audit


@pytest.mark.parametrize('mode', ['risk', 'event_risk', 'random_risk'])
def test_valid_task_feedback(mode):
    verify_audit(task_audit(mode), mode=mode)


@pytest.mark.parametrize('field,value', [
    ('state_dimensions', 5), ('risk_coordinate_count', 0),
    ('risk_coordinate_scale', 0.), ('risk_coordinate_scale', float('nan')),
    ('risk_teacher_is_physical_truth', True), ('target_detached', False),
    ('outcome_access', True), ('predicted_signals_used_for_risk', True),
    ('hard_event_pseudo_labels', True), ('horizons', [5, 10, 15, 20, 30]),
])
def test_task_receipt_rejects_wrong_supervision(field, value):
    audit = task_audit()
    audit['attempts'][0]['matured_objective'][field] = value
    with pytest.raises(AssertionError):
        verify_audit(audit, mode='risk')


@pytest.mark.parametrize('mutation', ['same_time', 'persistent', 'nonzero_start', 'frozen_change'])
def test_task_update_requires_causal_timing_and_reset(mutation):
    audit = task_audit()
    if mutation == 'same_time': audit['attempts'][0]['effective_index'] = 25
    elif mutation == 'persistent': audit['persistent'] = {'accepted': True}
    elif mutation == 'nonzero_start': audit['attempts'][0]['fast_at_origin']['l2'] = 1.
    else: audit['model_parameter_change_norm'] = .01
    with pytest.raises(AssertionError):
        verify_audit(audit, mode='risk')
