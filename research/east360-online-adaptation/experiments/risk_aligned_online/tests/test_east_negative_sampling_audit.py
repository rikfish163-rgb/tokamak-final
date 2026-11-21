import pytest

from experiments.risk_aligned_online.negative_sampling_20261023.audit import contrasts


def test_intra_gain_cannot_be_claimed_against_a_deliberately_weaker_parent():
    rows={k:dict(tp=3,fp=1,n_disrupt=4,n_safe=16,tpr=.75,fpr=.0625,J=.6875)
          for k in ('legacy_static','legacy_intra','safe_negative_static','safe_negative_intra')}
    rows['safe_negative_static'].update(tp=2,tpr=.5,J=.4375)
    rows['safe_negative_intra'].update(tp=2,fp=0,tpr=.5,fpr=0,J=.5)
    result=contrasts(rows)
    assert result['safe_negative']['own_parent_pass']
    assert not result['safe_negative']['screen_pass']
    rows['safe_negative_intra'].update(tp=3,tpr=.75,J=.75)
    assert contrasts(rows)['safe_negative']['screen_pass']
    with pytest.raises(ValueError):contrasts({})
