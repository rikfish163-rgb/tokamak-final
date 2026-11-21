import pytest
import numpy as np

from experiments.risk_aligned_online.temporal_feedback_diagnostic_20261023.audit import paired_summary
from experiments.risk_aligned_online.temporal_feedback_diagnostic_20261023.audit import validate_sample_metadata, validate_bins


def case(shot, raw, corrected):
    return dict(shot=shot, same_state_candidates={
        'recent': {'predicted_risk_loss_change': raw},
        'lag_bias': {'predicted_risk_loss_change': corrected}})


def test_same_state_comparison_and_equal_shot_weighting_are_distinct():
    rows=[case(1, 0., -.3), case(1, 0., -.3), case(2, 0., .2)]
    r=paired_summary(rows)
    assert r['sampled_updates']==3 and r['distinct_shots']==2
    assert r['lag_lower_predicted_loss']==2 and r['lag_higher_predicted_loss']==1
    assert r['sample_mean_difference']==pytest.approx(-.4/3)
    assert r['equal_shot_mean_difference']==pytest.approx(-.05)


def test_empty_sample_has_no_estimated_effect():
    r=paired_summary([])
    assert r['sampled_updates']==0
    assert r['equal_shot_mean_difference'] is None


@pytest.mark.parametrize('corruption',['status','alarm_already_emitted','eligible_query_points','query_stop_exclusive'])
def test_sample_inclusion_tampering_is_rejected(corruption):
    t=np.arange(400)/1000.;p=np.zeros(400);valid=np.ones(400,bool)
    row=dict(count=256,query_stop_exclusive=384,eligible_query_points=44,
             alarm_already_emitted=False,is_disrupt=True,status='measured')
    validate_sample_metadata(row,t,p,True,.330,valid,.8)
    row[corruption]={'status':'no_future_timely_or_valid_point','alarm_already_emitted':True,
                     'eligible_query_points':0,'query_stop_exclusive':400}[corruption]
    with pytest.raises(AssertionError):validate_sample_metadata(row,t,p,True,.330,valid,.8)


def test_corrupted_lead_bin_count_or_peak_is_rejected():
    from experiments.risk_aligned_online.temporal_feedback_diagnostic_20261023.run import timing_record
    t=np.arange(100)/1000.;p=np.linspace(0,1,100);v=np.ones(100,bool)
    r=timing_record(t,p,v,True,.100,.8)
    validate_bins(r,t,p,v,True,.100)
    r['lead_bin_peaks']['0_10']['observed']=-999
    with pytest.raises(AssertionError):validate_bins(r,t,p,v,True,.100)
    r=timing_record(t,p,v,True,.100,.8);r['lead_bin_peaks']['0_10']['peak']=4.
    with pytest.raises(AssertionError):validate_bins(r,t,p,v,True,.100)
