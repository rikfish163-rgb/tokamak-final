import pytest

from .evaluate import acceptance


def result(tp, fp):
    return dict(tp=tp,fp=fp,n_disrupt=180,n_safe=180,tpr=tp/180,fpr=fp/180,J=(tp-fp)/180)


def test_combined_success_cannot_replace_independent_intra_gain():
    summary = {'static':result(120,30),'intra':result(120,30),
               'inter':result(140,25),'both':result(145,24)}
    verdict = acceptance(summary)
    assert not verdict['independent_intra_gain']
    assert not verdict['all_required_gates_pass']


def test_intra_gain_outside_fpr_cap_is_not_accepted():
    summary = {'static':result(120,36),'intra':result(150,80),
               'inter':result(150,20),'both':result(155,24)}
    assert not acceptance(summary)['all_required_gates_pass']


def test_matching_denominators_required():
    summary = {a:result(120,30) for a in ('static','intra','inter','both')}
    summary['intra']['n_disrupt'] = 179
    with pytest.raises(ValueError,match='denominator'):
        acceptance(summary)


def test_j_gain_cannot_hide_lost_timely_disruptions():
    summary = {'static':result(140,40),'intra':result(130,20),
               'inter':result(140,25),'both':result(150,20)}
    verdict = acceptance(summary)
    assert not verdict['intra_timely_recall_preserved']
    assert not verdict['all_required_gates_pass']
