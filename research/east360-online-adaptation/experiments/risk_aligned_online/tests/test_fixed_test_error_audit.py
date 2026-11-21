import numpy as np
import pytest

from experiments.risk_aligned_online.fixed_test_error_audit import diagnose, paired_changes


def record(sid, scores, *, disruptive=False, down=None, start=2.):
    return dict(shot=sid, times=start+np.arange(len(scores))*.001,
                scores=np.asarray(scores, dtype=float), valid=np.ones(len(scores), bool),
                is_disrupt=disruptive, down=down)


def test_timing_is_from_record_start_and_invalid_endpoints_are_not_alarms():
    safe=record(1,[.9,.1,.8])
    safe['valid'][0]=False
    late=record(2,[.9,.1,.1],disruptive=True,down=2.030)
    timely=record(3,[.9,.1,.1],disruptive=True,down=2.030001)
    missed=record(4,[.1,.1,.1],disruptive=True,down=2.1)
    rows=diagnose([safe,late,timely,missed],.5)
    assert rows['counts']==dict(TP=1,TARDY=1,MISS=1,FP=1,TN=0)
    assert rows['shots'][0]['first_alarm_elapsed_ms']==2.
    assert rows['fp_before_elapsed_ms']['2']==0
    assert rows['fp_before_elapsed_ms']['8']==1
    assert rows['shots'][0]['fraction_eligible_endpoints_above_threshold']==.5
    assert rows['metrics']['tp']==1 and rows['metrics']['fp']==1


def test_pairing_counts_rescue_and_harm_without_hiding_late_alarms():
    before=diagnose([record(1,[.8,.1]),record(2,[.1,.1],disruptive=True,down=2.1),
                     record(3,[.8,.1],disruptive=True,down=2.1)],.5)
    after=diagnose([record(1,[.1,.1]),record(2,[.8,.1],disruptive=True,down=2.1),
                    record(3,[.1,.1],disruptive=True,down=2.1)],.5)
    pair=paired_changes(before,after)
    assert pair['rescued_fp']==1 and pair['rescued_not_timely']==1
    assert pair['new_not_timely']==1 and pair['new_fp']==0
    assert pair['net_tp_change']==0 and pair['net_fp_change']==-1
    assert len(pair['changed_shots'])==3
    with pytest.raises(ValueError,match='identities'):
        paired_changes(before,dict(after,shots=after['shots'][:-1]))


def test_post_disruption_points_cannot_rescue_a_miss():
    row=record(1,[.1,.1,.9],disruptive=True,down=2.001)
    result=diagnose([row,record(2,[.1,.1,.1])],.5)
    assert result['counts']['MISS']==1
    assert result['shots'][0]['first_alarm_elapsed_ms'] is None
