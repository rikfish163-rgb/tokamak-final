import numpy as np

from experiments.risk_aligned_online import commissioning_sxr_shape_20261114 as model


def test_whole_window_fit_is_shot_equal_and_controls_do_not_feed_on_targets():
    rng = np.random.default_rng(37)
    x, z = rng.normal(size=(8, 16)), rng.normal(size=(4, 16))
    y, w = rng.normal(size=(8, 5, 4)), rng.normal(size=(4, 5, 4))
    a = model.fit_decoders([(x, y), (z, w)])
    b = model.fit_decoders([(np.tile(x, (3, 1)), np.tile(y, (3, 1, 1))), (z, w)])
    for k in model.REGRESSION:
        for v, t in zip(a[k], b[k], strict=True):
            np.testing.assert_allclose(v, t, rtol=0, atol=5e-13)
    py = model.predict(x, a, 'all_observed')
    assert py.shape == y.shape


def test_forecast_clock_and_terminal_context_are_not_released_too_early():
    q = 127+32*np.arange(6)
    valid = np.array([True, True, False, True, True, True])
    prev = model.previous_rows(q, valid)
    assert prev.tolist() == [-1, 0, 1, -1, 3, 4]
    release = q+26+model.WAIT
    t = np.array([release[0], release[1]-1, release[1]+8, release[2]+8, release[3]-1,
                  release[4], release[4]+8])
    use, slot, fraction = model.clock(release, valid & (prev>=0), t)
    np.testing.assert_array_equal(use, [False, False, True, False, False, False, True])
    np.testing.assert_array_equal(slot[use], [2, 5])
    np.testing.assert_array_equal(fraction[use], [1., 1.])


def test_persistence_views_retain_levels_but_separate_late_retention_from_peak():
    # Shape: excursion returns in row 0; equally large excursion persists in row 1.
    # Gain in isolation cannot distinguish either row.
    y = np.zeros((2, 5, 4), dtype=np.float64)
    y[:, :, 0] = [1, 0, 2, 1, 0]
    y[0,:,1] = [4, 2, 1, 0, 0]
    y[1,:,1] = [0, 0, 1, 2, 4]
    a, b, ph = model.feedback_views(y, np.ones(4))
    np.testing.assert_array_equal(a[0], a[1])
    np.testing.assert_array_equal(b[:, 1], ph[:, 1])
    assert not np.allclose(b[:, (0,2)], ph[:, (0,2)])
    assert b[1, 0] > b[0, 0]
    np.testing.assert_allclose(b[:, 1], b[::-1, 1])
