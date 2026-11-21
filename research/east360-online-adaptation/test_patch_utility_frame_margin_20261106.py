import numpy as np
import pytest

import run_patch_utility_frame_margin_20261106 as a


def test_masked_ridge_does_not_use_unknown_or_padded_targets():
    x = np.ones((3,2))
    n = np.array([3,2,1])
    y = np.array([[1.,2.,4.],[3.,6.,np.nan],[5.,np.nan,np.nan]])
    head = a.ridge(x,y,n,np.array([.2,.3,.5]))
    pred = a.predict(head,np.ones((2,2)))
    assert pred == pytest.approx(np.array([[3.6,4.4,4.],[3.6,4.4,4.]]))


def test_no_effect_reference_is_exact_even_if_a_model_would_move_it():
    z = np.array([[1.,2.]])
    effect = np.array([[[.2,.3],[-1.,2.]]])
    assert a.compose(z,effect).tolist() == [[[1.,2.],[0.,4.]]]


def test_temporal_control_keeps_current_anchor_and_static_time():
    common = np.arange(3)[None]
    observed = np.zeros((1,3,8))
    observed[:,0] = np.arange(8)
    observed[:,1] = observed[:,0]+np.arange(8)**2
    observed[:,2] = observed[:,0]-.25
    old = np.ones((1,3,4))
    true = a.action_features(common,observed,old,"real")
    reverse = a.action_features(common,observed,old,"reverse")
    assert np.array_equal(true[:,:,:-16],reverse[:,:,:-16])
    assert np.array_equal(true[:,:,-16:-8],reverse[:,:,-16:-8][:,:,::-1])
    assert not np.array_equal(true[:,:,-8:],reverse[:,:,-8:][:,:,::-1])


def test_reindex_never_exposes_unrecorded_utility_frames():
    x = np.arange(1,49).reshape(2,24)
    t = a.align_tail(x,np.array([8,31]))
    assert np.array_equal(t[0],x[0])
    assert t[1].tolist() == [48]+[0]*23


def test_ridge_raises_when_a_horizon_has_no_observed_labels():
    with pytest.raises(ValueError,match="no training"):
        a.ridge(np.ones((2,1)),np.zeros((2,3)),np.array([1,2]),np.array([.5,.5]))


def test_nested_solver_matches_independent_weighted_solve_for_each_frame():
    x = np.array([[0.],[1.],[3.],[7.]])
    y = np.array([[1.,2.,3.],[4.,6.,np.nan],[7.,np.nan,np.nan],[8.,9.,10.]])
    lengths = np.array([3,2,1,3])
    weights = np.array([.1,.15,.3,.45])
    head = a.ridge(x,y,lengths,weights)
    center = np.average(x[:,0],weights=weights)
    scale = np.sqrt(np.average((x[:,0]-center)**2,weights=weights))
    z = np.column_stack((np.ones(len(x)),(x[:,0]-center)/scale))
    for j in range(3):
        ok = lengths>j
        w = weights[ok]/weights[ok].sum()
        zz = z[ok]
        beta = np.linalg.solve(zz.T@(w[:,None]*zz)+np.diag([0.,a.s.sel.ALPHA]),
                               zz.T@(w*y[ok,j]))
        assert a.predict(head,x)[:,j] == pytest.approx(z@beta,abs=1e-11)
