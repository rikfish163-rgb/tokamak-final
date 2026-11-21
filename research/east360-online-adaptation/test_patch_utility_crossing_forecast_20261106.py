import numpy as np
import pytest

import run_patch_utility_crossing_forecast_20261106 as x


def test_first_equality_counts_but_padded_future_does_not():
    curves = np.zeros((2, 2, 24))
    curves[:, :, 10] = 1
    curves[1, 1, 1] = .5
    first = x.first_offsets(curves, np.array([11, 5]), .5)
    assert first.tolist() == [[11, 11], [0, 2]]


def test_exposure_ends_on_first_event_or_actual_stop():
    first = np.array([[0, 9, 1], [2, 0, 8]])
    length = np.array([11, 8])
    exposed, event = x.exposure(first, length)
    assert exposed.tolist() == [[[8,3,0], [8,1,0], [1,0,0]],
                                [[2,0,0], [8,0,0], [8,0,0]]]
    assert event.tolist() == [[[0,0,0], [0,1,0], [1,0,0]],
                              [[1,0,0], [0,0,0], [1,0,0]]]


def test_paired_labels_reverse_sign_with_static_counterfactual():
    first = np.array([[0,8,9], [8,0,9]])
    labels = x.delta_target(first, 8)
    assert labels.tolist() == [[0,1,0], [0,-1,-1]]


def test_mass_is_conditional_and_conserves_residual_survival():
    h = np.array([[0.1,0.2,0.3]])
    p = x.first_mass(h)
    assert p[0,0] == pytest.approx(.1)
    assert p[0,8] == pytest.approx(.9**8*.2)
    assert p.sum() == pytest.approx(1-.9**8*.8**8*.7**8)
    assert x.first_mass(np.ones((1,3))).tolist() == [[1] + [0]*23]


def test_intercept_only_hazard_matches_at_risk_frame_frequency():
    model = x.fit_hazard(np.ones((3,2)),np.array([8,4,2]),
                         np.array([1,0,1]),np.ones(3)/3)
    assert x.hazard_predict(model,np.ones((4,2))) == pytest.approx([2/14]*4,abs=1e-7)


def test_score_compares_actions_inside_the_same_block_and_ties_get_half():
    score = np.array([[0.,1.,1.,-2.],[0.,2.,1.,2.]])
    target = np.array([[0,1,0,-1],[0,0,1,1]])
    allowed = np.ones_like(target,dtype=bool)
    allowed[:,0] = False
    acc, blocks = x.local_rank(score,target,allowed,1)
    assert acc.tolist() == [.75,.25] and blocks.tolist() == [0,1]


def test_validate_empty_or_unwitnessed_first_crossing():
    with pytest.raises(ValueError,match="outside"):
        x.exposure(np.array([[6]]),np.array([5]))


def test_crossfit_keeps_view_and_held_block_axes_distinct():
    owner = np.arange(15)
    first = np.zeros((3,15,13),dtype=np.int8)
    first[:,1::2,0] = 3
    first[...,1:4] = [3,10,20]
    features = (np.arange(13)[None,None,:,None]+np.arange(15)[None,:,None,None]/50
                +np.arange(3)[:,None,None,None]/10)
    hazard,paired,fold = x.fit_arm(features,owner,owner%5,np.ones(first.shape,bool),
                                   first,np.full(len(owner),24))
    assert hazard.shape == paired.shape == (3,15,13,3)
    assert np.isfinite(hazard).all() and np.isfinite(paired).all()
    assert len(fold) == 5 and all(len(f["hazard"])==3 for f in fold)
    for v in (0,2):
        i,j = 9,1
        head = fold[i%5]["hazard"][j]
        assert hazard[v,i,:,j] == pytest.approx(x.hazard_predict(head,features[v,i]))
