"""Verify the intervention begins after the waiting prefix."""

import numpy as np
import pytest

import run_patch_utility_shadow_timing_20261106 as d


def test_wait_preserves_reached_static_and_only_changes_future():
    native = np.zeros(40)
    native[4] = 2
    raw = np.zeros((1,3,32))
    raw[:,0,0] = 2
    raw[:,1,1] = 3
    raw[:,1,2] = 5
    baseline = raw.copy()
    clock = np.array([3])
    z = d.wait(raw, 2)
    assert np.array_equal(raw,baseline)
    assert np.array_equal(z[:,:,:2],np.broadcast_to(raw[:,:1,:2],(1,3,2)))
    assert np.array_equal(z[:,:,2:],raw[:,:,2:])
    for i in (1,2):
        out = native.copy()
        out[clock[0]+1:clock[0]+33] = z[0,i]
        assert np.flatnonzero(out>=1)[0] == 4
    assert np.array_equal(d.wait(raw,0),raw)
    with pytest.raises(ValueError,match="delay"):
        d.wait(raw,33)


def test_wait_does_not_reintroduce_tail_padding_into_real_time():
    base = np.arange(32.0)[None,None,:]
    curves = np.concatenate([base,base-1],axis=1)
    y = d.wait(curves,16)
    assert np.array_equal(y[0,1,:2],base[0,0,:2])
    assert y[0,1,16] == curves[0,1,16]
