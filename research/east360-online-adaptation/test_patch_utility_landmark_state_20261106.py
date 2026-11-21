import numpy as np
import pytest

import run_patch_utility_landmark_state_20261106 as a


def test_label_is_offline_and_only_real_future_counts_with_partial():
    times = np.arange(13)/1000
    clock = np.array([0,4])
    assert a.future_label(times,clock,True,.043,25.).tolist() == [False,False]
    assert a.future_label(times,clock,True,.043,125.).tolist() == [True,False]
    assert not a.future_label(times,clock,False,None,125.).any()
    assert not a.future_label(times,clock,True,.019,125.).any()


def test_orderless_control_preserves_each_groups_observed_values():
    order = np.arange(27).reshape(1,1,9,3)[:,:, :,::-1].reshape(1,1,27)
    both = np.repeat(order,2,axis=1)
    risk = np.zeros((1,10))
    seen = np.zeros((1,8))
    x = a.feature_matrix(risk,seen,both,"sorted")
    real = a.feature_matrix(risk,seen,both,"real")
    assert np.array_equal(x[:,:18],real[:,:18])
    assert np.array_equal(x[:,18:],np.arange(27)[None])
    assert np.array_equal(real[:,18:],order[:,0])


def test_case_weight_is_equal_over_physical_shots_not_view_multiplicity():
    owner = np.array([0,0,1,2])
    keep = np.array([[True,True,True,True],[True,False,False,True]])
    label = np.array([True,True,True,False])
    w = a.class_weights(keep,label,owner)
    assert w[:,owner==0].sum() == pytest.approx(.25)
    assert w[:,owner==1].sum() == pytest.approx(.25)
    assert w[:,owner==2].sum() == pytest.approx(.5)
    assert not w[~keep].any()


def test_risk_input_rejects_an_unreached_ninth_sample():
    with pytest.raises(ValueError,match="reached"):
        a.feature_matrix(np.zeros((1,10)),np.zeros((1,9)),np.zeros((1,2,27)),"risk")
