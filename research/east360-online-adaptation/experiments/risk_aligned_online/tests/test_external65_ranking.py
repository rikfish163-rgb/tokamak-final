import pytest

from ..external65_20261023.ranking import diagnostic


def fixture():
    return [dict(shot=i,is_disrupt=d,max_timely=p if d else None,max_valid=p)
            for i,(d,p) in enumerate([(True,.9),(True,.4),(False,.8),(False,.4),(False,.2)])]


def test_ties_and_caps_are_counted_exactly_without_promoting_oracle():
    result=diagnostic(fixture(),caps=(1/3,2/3))
    assert result['auc']==.75
    assert result['oracle'][0]['tp']==1 and result['oracle'][0]['fp']==0
    assert result['oracle'][1]['tp']==2 and result['oracle'][1]['fp']==2
    assert result['deployment_authorized'] is False


def test_monotone_score_change_preserves_ranking_and_oracle_counts():
    rows=fixture();modified=[dict(r,max_valid=r['max_valid']**2,
        max_timely=None if r['max_timely'] is None else r['max_timely']**2) for r in rows]
    a,b=diagnostic(rows),diagnostic(modified)
    assert a['auc']==b['auc']
    assert [(r['tp'],r['fp']) for r in a['oracle']]==[(r['tp'],r['fp']) for r in b['oracle']]
    with pytest.raises(ValueError):diagnostic(rows+rows[:1])
    with pytest.raises(ValueError):diagnostic(rows[:2])
    with pytest.raises(ValueError):diagnostic(rows,caps=(1.1,))
