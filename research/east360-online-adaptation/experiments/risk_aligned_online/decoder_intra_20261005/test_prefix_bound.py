import numpy as np
import pytest

from experiments.risk_aligned_online.decoder_intra_20261005.prefix_bound import count_prefix


def row(sid,scores,disrupt=False):
    return dict(shot=sid,scores=np.array(scores,float),is_disrupt=disrupt)


def test_first_effective_point_is_excluded_from_irrevocable_prefix():
    rows=[row(1,[.1,.8,.1]),row(2,[.1,.1,.8]),row(3,[.8,.1,.1],True)]
    result=count_prefix(rows,.7,2)
    assert result['safe_count']==2 and result['prefix_fp']==1
    assert result['fpr_lower_bound']==.5 and result['shots']==[1]
    assert result['minimum_positive_margin']==pytest.approx(.1)


def test_equality_is_alarm_and_short_shot_remains_in_denominator():
    result=count_prefix([row(1,[.7]),row(2,[.1])],.7,26)
    assert result['prefix_fp']==1 and result['minimum_positive_margin']==0
    assert result['safe_count']==2


def test_all_negative_prefix_has_no_fabricated_margin():
    result=count_prefix([row(1,[.1,.2])],.7,26)
    assert result['fpr_lower_bound']==0 and result['minimum_positive_margin'] is None


def test_missing_safe_denominator_rejected():
    with pytest.raises(ValueError):count_prefix([row(1,[.8],True)],.7,26)
