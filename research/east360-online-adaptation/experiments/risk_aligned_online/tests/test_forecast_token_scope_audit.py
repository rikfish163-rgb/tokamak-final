import pytest
from experiments.risk_aligned_online.forecast_token_scope_20261023.audit import contrasts


def test_user_fpr_cap_is_reported_separately_from_stricter_pareto_screen():
    names=('static','full_all','full_observation','patch_all','patch_observation')
    rows={a:dict(tp=3,fp=1,n_disrupt=4,n_safe=16,tpr=.75,fpr=.0625,J=.6875) for a in names}
    rows['full_all'].update(tp=2,tpr=.5,J=.4375)
    rows['patch_all'].update(fp=0,fpr=0,J=.75)
    rows['patch_observation'].update(tp=4,fp=2,tpr=1.,fpr=.125,J=.875)
    result=contrasts(rows)
    assert not result['vs_static']['full_all']['user_target_pass']
    assert not result['vs_static']['full_observation']['user_target_pass']
    assert result['vs_static']['patch_all']['strict_screen_pass']
    assert result['vs_static']['patch_observation']['user_target_pass']
    assert not result['vs_static']['patch_observation']['strict_screen_pass']
    assert result['readout_effect']['all']['delta_J']==.3125
    with pytest.raises(ValueError):contrasts({})
