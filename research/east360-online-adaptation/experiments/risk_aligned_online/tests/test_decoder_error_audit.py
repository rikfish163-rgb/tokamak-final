"""Forensic selection must retain strict lead qualification and no-update cases."""
import numpy as np
import pytest

from experiments.risk_aligned_online.decoder_error_audit import diagnostic_anchors, preceding_updates


def shot(disruptive):
    return dict(times=np.arange(101)/1000.,valid=np.ones(101,bool),is_disrupt=disruptive,
                down=.1 if disruptive else None)


def test_first_static_alarm_and_candidate_peak_use_strict_thirty_ms_lead():
    s=shot(True);left=np.zeros(101);right=np.zeros(101)
    left[70]=.99;left[60]=.8;right[65]=.6;right[90]=.95
    out=diagnostic_anchors(s,left,right,.7)
    assert out['static']['query']==60 and out['static']['kind']=='first_timely_alarm'
    assert out['candidate']['query']==65 and out['candidate']['kind']=='eligible_peak_without_timely_alarm'


def test_safe_first_alarm_is_not_replaced_by_a_later_larger_peak():
    s=shot(False);left=np.full(101,.1);right=left.copy();right[5]=.8;right[50]=.9
    out=diagnostic_anchors(s,left,right,.7)
    assert out['candidate']['query']==5 and out['candidate']['kind']=='first_false_alarm'


def test_last_accepted_mature_updates_only_and_no_update_boundary():
    rows=[dict(origin_index=o,effective_index=o+26,accepted=o!=100) for o in (0,100,200,300)]
    assert preceding_updates(rows,326,count=2)==[200,300]
    assert preceding_updates(rows,25,count=2)==[]
    assert preceding_updates(rows,26,count=2)==[0]


@pytest.mark.parametrize('bad',['nan','shape','no_timely_point'])
def test_invalid_anchor_inputs_are_not_silently_omitted(bad):
    s=shot(True);left=np.full(101,.2);right=left.copy()
    if bad=='nan':right[0]=np.nan
    elif bad=='shape':right=right[:-1]
    else:s['valid'][:70]=False
    with pytest.raises(ValueError):diagnostic_anchors(s,left,right,.7)


def test_no_pending_or_rejected_update_is_called_a_measured_change():
    rows=[dict(origin_index=0,effective_index=26,accepted=False),
          dict(origin_index=100,effective_index=None,accepted=False)]
    assert preceding_updates(rows,200,count=2)==[]
