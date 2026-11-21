"""Test causal phase, same support, and identifiable direction ranking."""

import numpy as np
import pytest

import intergroup_phase_features_20261106 as phase
import run_patch_utility_intergroup_20261106 as run


def delayed(n=96):
    rng = np.random.default_rng(83)
    a = rng.uniform(0.01, 0.12, n-1)
    b = np.r_[np.repeat(a[0], 4), a[:-4]]
    c = rng.uniform(0.01, 0.12, n-1)
    v = np.zeros((n, 23))
    for group, amp in zip(phase.CHANNELS, (a, b, c), strict=True):
        v[:, group] = np.sinh(np.r_[0, np.cumsum(amp)][:, None])
    return v, np.ones_like(v, dtype=bool)


def test_direction_and_rotation_share_exact_support():
    v, a = delayed()
    r = phase.prefix(v, a, np.zeros(23), np.ones(23), 63)
    assert r["common"] == 63
    assert r["supported"].all()
    assert r["real"][0] > .35
    assert r["real"][0] - r["shifted"][0] > .25
    assert r["within"][2::3].tolist() == [1., 1., 1.]


def test_future_and_unrelated_missingness_cannot_change_features():
    v, a = delayed()
    center, scale = np.zeros(23), np.ones(23)
    before = phase.prefix(v, a, center, scale, 63)
    v[64:] = np.nan
    a[:, [1, 2]] = False
    v[:, [1, 2]] = np.nan
    after = phase.prefix(v, a, center, scale, 63)
    for key in ("within", "real", "shifted", "supported", "common"):
        assert np.array_equal(before[key], after[key])


def test_lack_of_cross_support_is_not_imputed():
    v, a = delayed()
    a[26:32, phase.CHANNELS[2]] = False
    r = phase.prefix(v, a, np.zeros(23), np.ones(23), 63)
    assert r["common"] == 31
    assert not r["supported"].any()
    assert not r["real"].any() and not r["shifted"].any()
    assert r["within"][2] == 1
    assert r["within"][8] < 1
    assert phase.latest_longest(np.array([1,1,0,1,1],dtype=bool)) == (3,5)


def test_all_controls_share_shape_and_only_phase_changes_signed_interactions():
    source = dict(within=np.ones((2,9)), real=np.ones((2,9)),
                  shifted=np.ones((2,9))*2, supported=np.ones((2,9),bool))
    x = np.zeros((2,13,3),np.float32)
    risk, within, real, shifted = [run.add_features(x,source,mode) for mode in run.ARMS]
    assert all(y.shape == (2,13,84) for y in (risk,within,real,shifted))
    assert not risk.any()
    assert np.array_equal(real[:,:,:21],shifted[:,:,:21])
    assert np.array_equal(real[:,:,:21],within[:,:,:21])
    assert not within[:,:,21:].any()
    assert np.array_equal(real[:,1,30:],-real[:,2,30:])
    assert np.array_equal(shifted[:,:,30:],2*real[:,:,30:])
    assert not real[:,0,30:].any()


def test_rank_gate_deduplicates_physical_shots_across_views():
    d = {k: {name:dict(late_top=[]) for name in run.m.VIEWS} for k in run.ARMS}
    d["real"]["native"]["late_top"] = [42]
    d["shifted"]["no_MNM"]["late_top"] = [42]
    assert run.distinct_late(d) == []
    d["real"]["no_MNM_CIII"]["late_top"] = [18,42]
    assert run.distinct_late(d) == [18]


def test_constant_activity_has_no_fabricated_direction():
    a = np.ones((64,23),bool)
    r = phase.prefix(np.zeros(a.shape),a,np.zeros(23),np.ones(23),63)
    assert r["common"] == 63
    assert not r["supported"].any()
    with pytest.raises(ValueError,match="prefix"):
        phase.prefix(np.zeros(a.shape),a,np.zeros(23),np.ones(23),62)
