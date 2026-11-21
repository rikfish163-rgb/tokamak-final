import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.live_history_20261023.audit import check_guard


def receipt():
    return dict(fraction=.5,requested_step_norm=.001,accepted_step_norm=.0005,
        history_bce_change=[-.01,5e-7], history_change_from_parent=[-.1,5e-7],
        reason='history_guard_accepted',trials=[
            dict(fraction=1.,max_step_bce_increase=2e-6,max_parent_bce_increase=2e-6,accepted=False),
            dict(fraction=.5,max_step_bce_increase=5e-7,max_parent_bce_increase=5e-7,accepted=True)])


def test_acceptance_checks_every_trial_and_frozen_parent_envelope():
    r=receipt();check_guard(r)
    for key in ('history_bce_change','history_change_from_parent'):
        bad=copy.deepcopy(r);bad[key][1]=1.1e-6
        with pytest.raises(AssertionError):check_guard(bad)
    bad=copy.deepcopy(r);bad['trials'][0]['accepted']=True
    with pytest.raises(AssertionError):check_guard(bad)
    bad=copy.deepcopy(r);bad['trials'][0]['fraction']=.25
    with pytest.raises(AssertionError):check_guard(bad)
    bad=copy.deepcopy(r);bad['accepted_step_norm']=.0001
    with pytest.raises(AssertionError):check_guard(bad)


def test_rejected_proposal_must_leave_state_unchanged_and_exhaust_six_trials():
    r=receipt();r.update(fraction=0.,accepted_step_norm=0.,reason='history_guard_rejected',
                         history_bce_change=[0.,0.],history_change_from_parent=[-.1,0.],parameter_delta_norm=0.)
    r['trials']=[dict(fraction=2.**-j,max_step_bce_increase=.01,max_parent_bce_increase=.02,accepted=False) for j in range(6)]
    check_guard(r)
    r['history_bce_change'][0]=1e-8
    with pytest.raises(AssertionError):check_guard(r)
