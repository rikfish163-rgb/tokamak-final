import numpy as np

from experiments.risk_aligned_online.temporal_scale_audit_20261101.core import (
    variogram,
    patch_retention,
)


def test_variogram_preserves_affine_units_without_clipping():
    x = np.arange(128, dtype=np.float64)
    m = np.ones(128, bool)
    a, n = variogram(x, m)
    b, _ = variogram(3.7 * x - 123, m)
    assert np.allclose(a, b, rtol=1e-12)
    assert n[-1] == 64
    assert a[-1] > 1  # a nonstationary ramp is not clipped to correlation bounds


def test_constant_and_insufficient_are_missing_not_zero():
    x = np.ones(128)
    a, _ = variogram(x, np.ones(128, bool))
    assert np.isnan(a).all()
    a, _ = variogram(np.arange(128), np.arange(128) < 32)
    assert np.isnan(a).all()


def test_masks_remove_invalid_pairs_without_changing_their_time_lag():
    x = np.arange(128, dtype=float)
    m = np.ones(128, bool)
    m[40:80] = False
    xx = x.copy()
    xx[~m] = np.nan
    a, counts = variogram(xx, m)
    assert counts[0] == int((m[1:] & m[:-1]).sum())
    assert np.isnan(a[-1])  # fewer than32 valid64-step pairs


def test_patch_retention_distinguishes_within_and_between_variation():
    m = np.ones(128, bool)
    between = np.repeat(np.arange(16), 8).astype(float)
    within = np.tile(np.arange(8), 16).astype(float)
    assert patch_retention(between, m) == (1.0, 16)
    assert patch_retention(within, m) == (0.0, 16)
    assert patch_retention(np.ones(128), m)[0] is None


def test_incomplete_patches_do_not_become_measured_patch_means():
    x = np.arange(128, dtype=float)
    m = np.ones(128, bool)
    m[::8] = False
    value, count = patch_retention(x, m)
    assert value is None and count == 0
