import numpy as np

from experiments.risk_aligned_online.commissioning_scale_envelope_20261110 import cue_bounds


def test_envelope_checks_interior_persistence_kink_with_locked_lag():
    # At row 32 the previous amplitude is locked at m. The function
    # -x + 2 min(x, m) peaks inside the allowed interval, not at an end.
    m = np.arcsinh(1.0)
    ratio = np.full((33, 1), m)
    lo, hi = cue_bounds(ratio, np.array([-1.0, 2.0]), [True], 32)
    np.testing.assert_allclose(lo[:32], -m)
    np.testing.assert_array_equal(lo[:32], hi[:32])
    np.testing.assert_allclose(hi[32], m)
    np.testing.assert_allclose(lo[32], min(np.arcsinh(0.25), 2 * m - np.arcsinh(4.0)))


def test_sign_change_disables_history_and_unselected_channel_stays_fixed():
    m = np.arcsinh(1.0)
    ratio = np.full((35, 2), m)
    ratio[33, 0] = -m
    lo, hi = cue_bounds(ratio, np.array([1., -2., -3., 5.]), [True, False], 1)
    # At 33, channel 0 changed sign, channel 1 is fixed and persistent.
    np.testing.assert_allclose(lo[33], 3 * m - np.arcsinh(4.))
    np.testing.assert_allclose(hi[33], 3 * m - np.arcsinh(.25))
    # At 34 both endpoints of channel 0 are free: use the correct lag
    # endpoint for the negative persistence coefficient.
    floor, ceiling = np.arcsinh([.25, 4.])
    np.testing.assert_allclose(lo[34], 3 * m - 2 * ceiling)
    np.testing.assert_allclose(hi[34], 3 * m + ceiling - 3 * floor)
