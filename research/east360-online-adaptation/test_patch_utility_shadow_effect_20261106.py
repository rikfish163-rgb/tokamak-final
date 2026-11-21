"""Dense shadow targets do not borrow actual future values as inputs."""

import numpy as np
import pytest

import run_patch_utility_shadow_effect_20261106 as e


def test_targets_refer_to_last_observed_offset_and_keep_three_future_segments():
    curve = np.zeros((1,13,32))
    curve[:,0] = np.linspace(-1,1,32)
    curve[:,1:] = curve[:,:1] + np.arange(32)[None,None,:]
    y = e.targets(curve)
    assert y.shape == (1,13,3)
    assert y[0,0].tolist() == [0,0,0]
    assert y[0,1] == pytest.approx([4.5,12.5,20.5])
    assert e.slope(curve[:,:,:8])[0,1] == pytest.approx(y[0,1])


def test_equal_physical_shot_weights_across_views_blocks_and_actions():
    owner = np.array([0,0,1])
    a = np.array([[[1,1],[1,0],[1,0]],[[1,1],[0,0],[0,0]]],dtype=bool)
    w = e.weights(a,owner)
    assert w.sum() == pytest.approx(1)
    assert w[:,owner==0].sum() == pytest.approx(.5)
    assert w[0,owner==0].sum() == pytest.approx(.25)
    assert w[0,0,0] == pytest.approx(.0625)
    assert w[:,owner==1].sum() == pytest.approx(.5)
    assert not w[~a].any()


def test_ridge_has_unpenalized_offsets():
    x = np.ones((4,3))
    y = np.broadcast_to(np.array([-.2,0.1,.3]),(4,3))
    model = e.fit(x,y,np.ones(4)/4)
    assert e.predict(model,x) == pytest.approx(y)
