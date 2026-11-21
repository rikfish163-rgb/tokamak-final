import pytest

from experiments.risk_aligned_online.attention_scope_20261023.audit import contrasts


def test_restoring_static_does_not_pass_and_matching_control_not_attribution():
    baseline=dict(tp=3,fp=1,n_disrupt=4,n_safe=16,tpr=.75,fpr=.0625,J=.6875)
    rows={a:dict(baseline) for a in ('static','full','qk','value_matched','full_matched')}
    assert not contrasts(rows)['qk_user_target']
    improved=dict(baseline,tp=4,tpr=1.,J=.9375)
    rows['qk']=improved
    assert contrasts(rows)['qk_attributed_screen']
    rows['full_matched']=improved
    assert contrasts(rows)['qk_user_target'] and not contrasts(rows)['qk_attributed_screen']
    rows['static']['n_safe']=15
    with pytest.raises(ValueError):contrasts(rows)
