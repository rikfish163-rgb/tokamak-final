import copy
import numpy as np
import pytest

from .macro_audit import paired_prefix


def evidence():
    event=dict(rule_file_sha256='same',history_end=32,decisions=[
        dict(action=1,end=5,effect=6),dict(action=1,end=265,effect=266)])
    return event,copy.deepcopy(event),np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)


def test_only_post_second_decision_can_differ():
    h,p,qh,qp,sh,sp=evidence()
    p['decisions'][1]['action']=-1
    qh[266:]=.9; qp[266:]=.8
    sh[266:]=.9
    result=paired_prefix(h,p,qh,qp,sh,sp,.7)
    assert result['effect_2']==266 and result['later_q_differs']
    assert result['prefix_max_abs']==0


@pytest.mark.parametrize('field',['rule','first_action','clock','q','score','crossing'])
def test_common_prefix_violation_is_not_a_valid_information_gain(field):
    h,p,qh,qp,sh,sp=evidence()
    if field=='rule': p['history_end']=33
    if field=='first_action': p['decisions'][0]['action']=-1
    if field=='clock': p['decisions'][1]['effect']=267
    if field=='q': qp[265]=.01
    if field=='score': sp[265]=.1
    if field=='crossing': sh[265]=.7; sp[265]=.7-1e-7
    with pytest.raises(ValueError): paired_prefix(h,p,qh,qp,sh,sp,.7)


def test_saturated_action_difference_is_not_actual_update_difference():
    h,p,qh,qp,sh,sp=evidence()
    p['decisions'][1]['action']=0
    assert not paired_prefix(h,p,qh,qp,sh,sp,.7)['later_q_differs']


def test_no_second_decision_means_entire_trace_must_match():
    h,p,qh,qp,sh,sp=evidence()
    h['decisions']=h['decisions'][:1]; p['decisions']=p['decisions'][:1]
    result=paired_prefix(h,p,qh,qp,sh,sp,.7)
    assert result['effect_2'] is None
    qp[-1]=.02
    with pytest.raises(ValueError): paired_prefix(h,p,qh,qp,sh,sp,.7)
