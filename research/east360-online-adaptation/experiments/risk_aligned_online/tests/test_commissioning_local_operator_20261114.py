import numpy as np

from experiments.risk_aligned_online import commissioning_local_operator_20261114 as trial


def test_the_operator_and_view_are_unchanged_when_only_the_future_changes():
    a = np.random.default_rng(7).normal(size=(230,2))
    m = np.ones_like(a,dtype=bool)
    q = np.array([127,159,191])
    x,y,valid = trial.past_terms(a,m,q)
    aa = a.copy()
    aa[160:] *= 9
    z,w,okay = trial.past_terms(aa,m,q[:2])
    np.testing.assert_array_equal(x[:2],z)
    np.testing.assert_array_equal(y[:2],w)
    np.testing.assert_array_equal(valid[:2],okay)


def test_joint_operator_recovers_cross_coordinate_changes_without_target_view():
    rng = np.random.default_rng(131)
    x = rng.normal(size=(2,15,2))
    actual = np.array([[.1,.8],[-.6,.1]])
    y = x@actual
    a = trial.fit_operator(x,y,ridge=1e-8)
    np.testing.assert_allclose(a,np.broadcast_to(actual,a.shape),atol=2e-7)
    diag = trial.fit_operator(x,y,ridge=1e-8,diagonal=True)
    np.testing.assert_array_equal(diag[:,[0,1],[1,0]],0)


def test_maturity_and_missing_history_keep_later_windows_out_of_support():
    q = np.array([127,159,191,223,255])
    usable = np.array([True,False,True,True,True])
    older, live = trial.earlier(q,usable)
    assert older.tolist()==[-1,0,0,2,3]
    assert live.tolist()==[False,False,True,True,True]


def test_all_unobservable_window_has_no_operator_forecast():
    got = trial.design(np.empty((0,2,2)),np.empty((0,2)))
    assert got.shape==(0,6)
