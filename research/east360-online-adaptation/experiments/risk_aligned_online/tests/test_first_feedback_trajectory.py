import numpy as np
import pytest

from experiments.risk_aligned_online.first_feedback_20261023.trajectory_diagnostic import compare


def test_full_curve_changes_do_not_imply_shot_level_improvement():
    t=np.array([1.,1.001,1.002,1.003])
    a=np.array([.1,.8,.2,.3]);b=np.array([.1,.1,.8,.99])
    r=compare(t,a,b,.7,disruptive=True,down=1.002)
    assert r['early_first_index']==1 and r['late_first_index']==2
    assert r['max_difference_index']==1 and r['threshold_disagreements']==2
    assert r['max_difference']==pytest.approx(.7)
    assert r['early_first_time']==1.001 and r['late_first_time']==1.002
    # Terminal validity removes the deliberately huge post-disruption difference.
    assert r['valid_points']==3


def test_silent_shot_reports_missing_alarm_not_an_invented_time():
    r=compare(np.array([1.,2.]),np.array([.1,.2]),np.array([.2,.1]),.7,
              disruptive=False,down=None)
    assert r['early_first_index'] is None and r['late_first_time'] is None
