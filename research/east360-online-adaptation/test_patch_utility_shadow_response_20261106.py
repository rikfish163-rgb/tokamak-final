"""Causal boundary, symmetric action guard, and time-aligned negative control."""

import numpy as np
import pytest

import run_patch_utility_shadow_response_20261106 as r


def test_guard_rejects_both_signs_and_preserves_any_prior_alarm():
    x = np.zeros((4, 13, 8), dtype=np.float64)
    x[0,1,-1] = 0.1001
    x[1,:,3] = 0.8
    allowed = np.ones((4,13),dtype=bool)
    out = r.commit_mask(allowed,x,np.array([True,True,True,False]),
                        np.array([0.,0.,0.8,0.]),0.8)
    assert out[:,0].all()
    assert not out[0,1:3].any() and out[0,3:].all()
    assert not out[1:,1:].any()
    assert allowed.all()


def test_common_context_amplitude_and_alignment_only_change_as_promised():
    prefix = np.zeros((1,13,8))
    prefix[0,:,:] = np.array([0.,2.,3.,7.,6.,2.,5.,8.])
    prefix[0,1] += np.array([.05,-.01,.02,0.,-.06,.02,.01,.04])
    base = np.ones((1,13,5),np.float32)
    blind, amp, real, rev = [r.design(base,prefix,k,.8) for k in r.ARMS]
    assert all(x.shape==(1,13,34) for x in (blind,amp,real,rev))
    assert np.array_equal(blind[:,:,:13],real[:,:,:13])
    assert not blind[:,:,13:].any()
    assert np.array_equal(amp[:,:,:18],real[:,:,:18])
    assert np.array_equal(real[:,:,:18],rev[:,:,:18])
    assert not amp[:,:,18:].any()
    assert np.array_equal(real[:,:,18:26],rev[:,:,18:26][:,:,::-1])
    assert not np.array_equal(real[:,1,26:],rev[:,1,26:][:,::-1])
    assert not real[:,0,13:].any()


def test_interface_rejects_any_more_than_the_eight_arrived_samples():
    base = np.zeros((2,13,3))
    with pytest.raises(ValueError,match="eight"):
        r.design(base,np.ones((2,13,9)),"real",.8)


def test_guard_late_gate_is_physical_shot_unique():
    a={k:{v:dict(late_top=[]) for v in r.m.VIEWS} for k in r.ARMS}
    a["real"]["native"]["late_top"]=[12,20]
    a["reverse"]["no_MNM"]["late_top"]=[12]
    assert r.late_difference(a)==[20]
