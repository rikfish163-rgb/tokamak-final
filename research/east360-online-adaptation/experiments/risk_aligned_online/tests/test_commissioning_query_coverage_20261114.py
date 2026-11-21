import numpy as np

from experiments.risk_aligned_online import commissioning_query_coverage_20261114 as study


def test_disagreement_is_zero_when_both_ranked_predictors_agree():
    x = np.r_[-2.,0.,1.]
    np.testing.assert_array_equal(study.unexplained_high(x,x),np.zeros(3))


def test_support_ignores_late_observations_and_preexisting_alarm():
    t = np.array([7,15,23,31])
    z = np.zeros(36)
    z[19] = .8
    ok = study.usable(t,np.ones(4,bool),z,.5,age=5)
    np.testing.assert_array_equal(ok,[True,True,False,False])


def test_whole_window_must_already_have_landed_when_estimating_conflict():
    q = np.array([127,159,191])
    y = np.arange(141,230)
    src = study.latest_landed(q,y,offset=31)
    np.testing.assert_array_equal(src[[0,17,18,48,49,80,81]],[-1,0,0,0,1,1,2])


def test_preassigned_exploration_slots_do_not_remove_existing_acquisition_entirely():
    rows = [
        dict(shot=31,rank=41,observed_alarm=True,parent_near=.01),
        dict(shot=32,rank=42,observed_alarm=False,parent_near=.3),
    ]
    assert study.alternate(31,rows,2)==32
    assert study.alternate(31,rows,3)==31
