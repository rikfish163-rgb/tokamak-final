import numpy as np

from patch_tempo_20261106 import FIRST, STRIDE, clock, gather, lag_profile


def signals(n, stretch=1.0):
    t = np.arange(n, dtype=float)[:, None]
    period = np.array([21, 24, 26, 29, 31]) * stretch
    x = np.sin(t * (2 * np.pi / period))
    return x, np.ones_like(x, dtype=bool)


def test_history_spacing_preserves_actual_last_patch_and_native_window():
    x = np.repeat(np.arange(390)[:, None], 23, axis=1).astype(np.float32)
    valid = np.ones_like(x, dtype=bool)
    end = 300
    for interval in (4, 8, 16):
        values, available = gather(x, valid, np.array([end]), interval)
        assert available.all()
        patches = values[0, :, 0].reshape(16, 8)
        np.testing.assert_array_equal(patches[-1], np.arange(end - 7, end + 1))
        np.testing.assert_array_equal(np.diff(patches[:, 0]), interval)
        np.testing.assert_array_equal(np.diff(patches, axis=1), 1)
        assert patches[0, 0] == end - 7 - 15 * interval
    native, mask = gather(x, valid, np.array([3]), 8)
    assert not mask[0, :-4].any()
    np.testing.assert_array_equal(native[0, -4:], x[:4])


def test_structure_is_scale_and_offset_invariant_on_observed_support():
    x, m = signals(128)
    m[30:40, 1] = False
    got = lag_profile(x[None], m[None])
    shifted = x * np.arange(1, 6) + 7
    shifted[~m] = np.nan
    np.testing.assert_allclose(lag_profile(shifted[None], m[None]), got, atol=1e-12)


def test_slower_synthetic_dynamics_choose_dilated_history():
    ref_x, ref_m = signals(128)
    ref = lag_profile(ref_x[None], ref_m[None])[0, 1:4]
    x, m = signals(800, stretch=2.0)
    select, delayed, support = clock(x, m, ref)
    assert np.all(support == 5)
    assert np.all(select[:FIRST] == 0)
    assert np.all(select[FIRST:] == 2)
    np.testing.assert_array_equal(select, delayed)


def test_actual_prefix_extension_cannot_change_past_choices():
    x, m = signals(900)
    ref = lag_profile(x[None, :128], m[None, :128])[0, 1:4]
    x[400:] = signals(500, stretch=2.0)[0]
    prefix = FIRST + 9 * STRIDE + 3
    true, lagged, support = clock(x, m, ref)
    t, l, s = clock(x[:prefix], m[:prefix], ref)
    np.testing.assert_array_equal(t, true[:prefix])
    np.testing.assert_array_equal(l, lagged[:prefix])
    np.testing.assert_array_equal(s, support[: len(s)])
    for anchor in range(FIRST + 4 * STRIDE, len(x), STRIDE):
        assert lagged[anchor] == true[anchor - 4 * STRIDE]


def test_two_channels_cannot_train_a_shot_local_clock():
    x, m = signals(500, stretch=2.0)
    ref = lag_profile(x[None, :128], m[None, :128])[0, 1:4]
    m[:, 2:] = False
    select, delayed, support = clock(x, m, ref)
    assert np.all(support == 2)
    assert not select.any() and not delayed.any()


def test_reference_uses_eligible_training_windows_not_fabricated_short_shots():
    from run_patch_tempo_20261106 import source_reference

    reference = source_reference()
    assert reference.shape == (3, 5)
    assert np.isfinite(reference).all()
