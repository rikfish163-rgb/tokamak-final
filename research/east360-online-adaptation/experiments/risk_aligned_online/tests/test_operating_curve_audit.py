"""Matched false-alarm caps are retrospective diagnostics, not tuned deployment."""
import numpy as np
import pytest

from experiments.risk_aligned_online.operating_curve_audit import operational_maxima, operating_curve


def record(sid, label, scores, times=None, valid=None, down=1.):
    scores=np.asarray(scores, dtype=np.float32)
    return dict(shot=sid,is_disrupt=label,down=down if label else None,
                scores=scores,times=np.asarray(times if times is not None else [.90]*len(scores)),
                valid=np.asarray(valid if valid is not None else [True]*len(scores)))


def test_strict_thirty_ms_boundary_and_invalid_points_do_not_count():
    rows=[record(1,True,[.99,.6,.9],[.80,.969,.970],[False,True,True]),
          record(2,False,[.5,.99],valid=[True,False])]
    maxima=operational_maxima(rows)
    assert maxima['positive'].tolist()==pytest.approx([.6])
    assert maxima['negative'].tolist()==pytest.approx([.5])
    point=operating_curve(rows,.7)['fixed_source_threshold']
    assert point['tp']==point['fp']==0


def test_missing_eligible_disruption_stays_in_denominator():
    rows=[record(1,True,[.99],[.98]),record(2,False,[.2])]
    out=operating_curve(rows,.7)
    assert out['n_disrupt']==1 and out['disruptions_without_eligible_point']==[1]
    assert out['fixed_source_threshold']['tpr']==0


def test_cap_can_exclude_a_float32_safe_tie_without_excluding_a_higher_tp():
    rows=[record(1,True,[.75]),record(2,True,[.8]),
          record(3,False,[.75]),record(4,False,[.4])]
    out=operating_curve(rows,.725,caps=(0.,.5))
    zero=out['posthoc_best_at_cap']['0.00']
    assert zero['tp']==1 and zero['fp']==0 and zero['threshold']>float(np.float32(.75))
    assert out['posthoc_best_at_cap']['0.50']['tp']==2
    assert out['uses_all_stream_labels_posthoc'] and not out['deployable_threshold_selection']


def test_saturated_safe_probability_cannot_be_excluded_by_a_valid_threshold():
    out=operating_curve([record(1,True,[1.]),record(2,False,[1.])],.7,caps=(0.,))
    point=out['posthoc_best_at_cap']['0.00']
    assert point['threshold'] is None and point['decision']=='always_no_alarm'
    assert point['tp']==0 and point['fp']==0


def test_invalid_only_safe_shot_remains_safe_without_fabricated_alarm():
    rows=[record(1,True,[.8]),record(2,False,[.99],valid=[False])]
    out=operating_curve(rows,.7,caps=(0.,))
    assert out['safe_shots_without_eligible_point']==[2]
    assert out['posthoc_best_at_cap']['0.00']['tp']==1


@pytest.mark.parametrize('bad',['duplicate','single_class','scores','times','mask','down'])
def test_invalid_or_incomparable_records_are_rejected(bad):
    rows=[record(1,True,[.8]),record(2,False,[.4])]
    if bad=='duplicate':rows[1]['shot']=1
    elif bad=='single_class':rows=rows[:1]
    elif bad=='scores':rows[1]['scores'][0]=np.nan
    elif bad=='times':rows[0]['times']=np.array([.9,.8])
    elif bad=='mask':rows[1]['valid']=np.array([1])
    else:rows[0]['down']=None
    with pytest.raises(ValueError):operating_curve(rows,.7)


def test_a_lower_cap_never_has_a_better_tpr_and_inputs_are_unchanged():
    rows=[record(1,True,[.9]),record(2,True,[.65]),
          record(3,False,[.7]),record(4,False,[.5])]
    old=[r['scores'].copy() for r in rows]
    out=operating_curve(rows,.7,caps=(0.,.5,1.))
    points=out['posthoc_best_at_cap']
    assert points['0.00']['tpr']<=points['0.50']['tpr']<=points['1.00']['tpr']
    for row,before in zip(rows,old):np.testing.assert_array_equal(row['scores'],before)
