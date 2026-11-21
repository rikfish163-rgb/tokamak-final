"""Synthetic-only contract tests for the causal deep-miss history features."""

import numpy as np
import pytest

import deep_miss_history_features_20261106 as history


def feature(name):
    return history.FEATURE_NAMES.index(name)


def blank(length):
    return np.zeros((length, 23), dtype=np.float64), np.zeros((length, 23), dtype=bool)


def test_layout_boundaries_and_exact_nonoverlapping_windows():
    assert history.FEAT_DIM == 146
    assert len(history.FEATURE_NAMES) == len(set(history.FEATURE_NAMES)) == 146
    assert tuple(history.USED_CHANNELS) == tuple(i for i in range(23) if i not in (1, 2, 10))
    z, available = blank(1408)
    anchor = 1407
    # Each constant covers exactly one inclusive short/older block.
    for start, stop, value in ((1280, 1408, 7), (1024, 1280, 11),
                               (768, 1024, 22), (512, 768, 33),
                               (256, 512, 44)):
        z[start:stop, 0] = value
        available[start:stop, 0] = True
    result = history.build(z, available, [anchor], 42)
    assert set(result) == {"short", "real", "stale", "scrambled", "support_real", "support_stale"}
    assert result["short"].shape == (1, history.FEAT_DIM)
    for name in ("real", "stale", "scrambled"):
        assert result[name].shape == (1, 3, history.FEAT_DIM)
    for name in ("support_real", "support_stale"):
        assert result[name].shape == (1, 3) and result[name].dtype == bool
        assert result[name].all()
    mean = feature("mean:Bt")
    assert result["short"][0, mean] == 7
    np.testing.assert_array_equal(result["real"][0, :, mean], [11, 22, 33])
    np.testing.assert_array_equal(result["stale"][0, :, mean], [22, 33, 44])
    np.testing.assert_array_equal(result["real"][0, 1:], result["stale"][0, :2])

    # An older block becomes eligible only when its entire left edge exists.
    z, available = blank(1152)
    available[:, 0] = True
    anchors = [127, 382, 383, 638, 639, 894, 895, 1150, 1151]
    result = history.build(z, available, anchors, 42)
    np.testing.assert_array_equal(result["support_real"], [
        [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1],
    ])
    np.testing.assert_array_equal(result["support_stale"][-2:], [[1, 1, 0], [1, 1, 1]])
    assert not result["real"][~result["support_real"]].any()
    assert not result["stale"][~result["support_stale"]].any()


def test_short_reversal_missingness_and_lag8_joint_support():
    z, available = blank(128)
    z[:, 0] = np.arange(128)
    available[:, 0] = True
    z[:64, 3] = 4
    available[:64, 3] = True  # No second-half support.
    z[[0, 1], 4] = [2, 6]
    available[[0, 1], 4] = True  # No observations exactly eight apart.
    z[[9, 17], 5] = [3, 12]
    available[[9, 17], 5] = True
    z[~available] = np.nan
    result = history.build(z, available, [127], 1)["short"][0]
    assert result[feature("mean:Bt")] == pytest.approx(63.5)
    assert result[feature("std:Bt")] == pytest.approx(np.std(np.arange(128)))
    assert result[feature("half_delta:Bt")] == pytest.approx(64)
    assert result[feature("observed_fraction:Bt")] == 1
    assert result[feature("rms_lag8:Bt")] == pytest.approx(8)
    assert result[feature("half_delta_valid:Mir_VV")] == 0
    assert result[feature("half_delta:Mir_VV")] == 0
    assert result[feature("observed_fraction:Mir_VV")] == 0.5
    assert result[feature("rms_lag8_valid:Mir_avg_amp")] == 0
    assert result[feature("rms_lag8:Mir_avg_amp")] == 0
    assert result[feature("rms_lag8_valid:Mir_avg_fre")] == 1
    assert result[feature("rms_lag8:Mir_avg_fre")] == pytest.approx(9)
    reversed_result = history.build(z[::-1], available[::-1], [127], 1)["short"][0]
    assert reversed_result[feature("mean:Bt")] == pytest.approx(result[feature("mean:Bt")])
    assert reversed_result[feature("std:Bt")] == pytest.approx(result[feature("std:Bt")])
    assert reversed_result[feature("half_delta:Bt")] == pytest.approx(-64)
    assert reversed_result[feature("rms_lag8:Bt")] == pytest.approx(8)


def test_correlations_use_joint_support_and_mark_undefined():
    z, available = blank(128)
    # RAW23 representatives: Mir_VV=3, sxr_mean=15, xuv_ratio=20.
    z[:5, 3] = [1, 2, 4, 8, 16]
    z[:5, 15] = [1000, 2, 7, 5, -999]
    z[:5, 20] = [7, 7, 7, 7, 7]
    available[:5, 3] = True
    available[1:4, 15] = True
    available[:5, 20] = True
    out = history.build(z, available, [127], 1)["short"][0]
    expected = np.corrcoef([2, 4, 8], [2, 7, 5])[0, 1]
    assert out[feature("corr:Mir_VV|sxr_mean")] == pytest.approx(expected)
    assert out[feature("corr_valid:Mir_VV|sxr_mean")] == 1
    assert out[feature("corr:Mir_VV|xuv_ratio")] == 0
    assert out[feature("corr_valid:Mir_VV|xuv_ratio")] == 0
    available[2:4, 15] = False  # Only one jointly observed sample.
    out = history.build(z, available, [127], 1)["short"][0]
    assert out[feature("corr_valid:Mir_VV|sxr_mean")] == 0
    assert out[feature("corr:Mir_VV|sxr_mean")] == 0


def test_scramble_moves_whole_supported_blocks_and_preserves_holes():
    z, available = blank(1152)
    for start, value in ((768, 10), (512, 20), (256, 30), (0, 40)):
        z[start:start + 256, 0] = value
        available[start:start + 256, 0] = True
    full = history.build(z, available, [1151], 72)
    real, scrambled = full["real"][0], full["scrambled"][0]
    assert not np.array_equal(real, scrambled)
    assert sorted(map(tuple, real)) == sorted(map(tuple, scrambled))
    np.testing.assert_array_equal(scrambled, history.build(z, available, [1151], 72)["scrambled"][0])
    # Seed includes the shot, so it is not just one universal permutation.
    permutations = {tuple(history.build(z, available, [1151], shot)["scrambled"][0, :, feature("mean:Bt")])
                    for shot in range(20)}
    assert len(permutations) > 1
    available[512:768, 0] = False
    holes = history.build(z, available, [1151], 72)
    np.testing.assert_array_equal(holes["support_real"][0], [1, 0, 1])
    np.testing.assert_array_equal(holes["scrambled"][0, 0], holes["real"][0, 2])
    np.testing.assert_array_equal(holes["scrambled"][0, 1], holes["real"][0, 1])
    np.testing.assert_array_equal(holes["scrambled"][0, 2], holes["real"][0, 0])
    available[256:512, 0] = False
    one = history.build(z, available, [1151], 72)
    np.testing.assert_array_equal(one["real"], one["scrambled"])


def test_truncation_future_changes_and_batch_order_never_change_prefix():
    rng = np.random.default_rng(23)
    z = rng.normal(size=(1500, 23))
    available = rng.random(z.shape) > 0.22
    anchors = np.array([895, 127, 1300, 895, 639])
    batch = history.build(z, available, anchors, 52)
    for row, anchor in enumerate(anchors):
        prefix = history.build(z[:anchor + 1], available[:anchor + 1], [anchor], 52)
        for key in batch:
            np.testing.assert_array_equal(batch[key][row], prefix[key][0])
    altered = z.copy()
    altered[896:] = np.nan  # Even inconsistent future values are never read.
    earlier = history.build(altered, available, [895], 52)
    for key in batch:
        np.testing.assert_array_equal(batch[key][0], earlier[key][0])


def test_excluded_channels_never_provide_support_and_large_scales_stay_finite():
    z, available = blank(384)
    available[:, [1, 2, 10]] = True
    z[:, [1, 2, 10]] = np.inf
    out = history.build(z, available, [383], 7)
    assert not out["support_real"].any()
    assert not out["short"].any()
    assert not out["real"].any()
    available[:, 0] = True
    z[:192, 0], z[192:, 0] = -1e150, 1e150
    out = history.build(z, available, [255], 7)["short"][0]
    assert np.isfinite(out).all()
    assert out[feature("mean:Bt")] == pytest.approx(0, abs=1e135)
    assert out[feature("std:Bt")] == pytest.approx(1e150)
    assert out[feature("half_delta:Bt")] == pytest.approx(2e150)
    assert out[feature("rms_lag8:Bt")] == pytest.approx(2e150 * np.sqrt(8 / 120))
    z[:192, 0], z[192:, 0] = -1e308, 1e308
    out = history.build(z, available, [255], 7)["short"][0]
    assert np.isfinite(out).all()
    assert out[feature("half_delta:Bt")] == np.finfo(np.float64).max


def test_small_fluctuations_on_large_offset_retain_accuracy():
    z, available = blank(128)
    direction = np.where(np.arange(128) % 16 < 8, -1.0, 1.0)
    z[:, 0] = 1e12 + direction
    z[:, 3] = 1e12 + direction
    z[:, 15] = 1e12 - 3 * direction
    available[:, [0, 3, 15]] = True
    out = history.build(z, available, [127], 1)["short"][0]
    assert out[feature("std:Bt")] == pytest.approx(1.0, rel=1e-12)
    assert out[feature("rms_lag8:Bt")] == pytest.approx(2.0, rel=1e-12)
    assert out[feature("corr:Mir_VV|sxr_mean")] == pytest.approx(-1.0, abs=1e-12)


def test_correlation_remains_finite_when_raw_subtraction_would_overflow():
    z, available = blank(128)
    z[:64, 3], z[64:, 3] = -1e308, 1e308
    z[:64, 15], z[64:, 15] = 1e308, -1e308
    available[:, [3, 15]] = True
    out = history.build(z, available, [127], 1)["short"][0]
    assert np.isfinite(out).all()
    assert out[feature("corr_valid:Mir_VV|sxr_mean")] == 1
    assert out[feature("corr:Mir_VV|sxr_mean")] == pytest.approx(-1.0)


def test_random_masked_features_match_direct_window_reference():
    rng = np.random.default_rng(91)
    z = rng.normal(size=(1152, 23))
    available = rng.random(z.shape) > 0.37
    available[512:640, 3] = False
    z[~available] = np.nan
    out = history.build(z, available, [1151], 13)
    names = ("Bt", "Mir_VV", "Mir_avg_amp", "Mir_avg_fre", "Z_proxy", "dx_a",
             "dy_a", "ip", "n=1 amplitude", "ne_nG", "qa_proxy", "sxr_kurt",
             "sxr_mean", "sxr_skew", "sxr_var", "v_loop", "xuv_kurt",
             "xuv_ratio", "xuv_skew", "xuv_var")
    windows = [(1024, 1152, out["short"][0])]
    windows += [(768 - 256 * j, 1024 - 256 * j, out["real"][0, j]) for j in range(3)]
    windows += [(0, 256, out["stale"][0, 2])]
    for start, stop, actual in windows:
        expected = np.zeros(history.FEAT_DIM)
        for column, name in zip(history.USED_CHANNELS, names, strict=True):
            mask = available[start:stop, column]
            values = z[start:stop, column]
            observed = values[mask]
            expected[feature(f"observed_fraction:{name}")] = mask.mean()
            if len(observed):
                expected[feature(f"mean:{name}")] = observed.mean()
                expected[feature(f"std:{name}")] = observed.std()
            middle = len(values) // 2
            first, second = values[:middle][mask[:middle]], values[middle:][mask[middle:]]
            if len(first) and len(second):
                expected[feature(f"half_delta:{name}")] = second.mean() - first.mean()
                expected[feature(f"half_delta_valid:{name}")] = 1
            lag_mask = mask[8:] & mask[:-8]
            if lag_mask.any():
                delta = values[8:][lag_mask] - values[:-8][lag_mask]
                expected[feature(f"rms_lag8:{name}")] = np.sqrt(np.mean(delta ** 2))
                expected[feature(f"rms_lag8_valid:{name}")] = 1
        for left, right, left_column, right_column in (
            ("Mir_VV", "sxr_mean", 3, 15), ("Mir_VV", "xuv_ratio", 3, 20),
            ("sxr_mean", "xuv_ratio", 15, 20),
        ):
            joint = available[start:stop, left_column] & available[start:stop, right_column]
            left_values = z[start:stop, left_column][joint]
            right_values = z[start:stop, right_column][joint]
            if len(left_values) >= 2 and np.ptp(left_values) > 0 and np.ptp(right_values) > 0:
                expected[feature(f"corr:{left}|{right}")] = np.corrcoef(left_values, right_values)[0, 1]
                expected[feature(f"corr_valid:{left}|{right}")] = 1
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_input_validation_and_empty_batch():
    z, available = blank(128)
    empty = history.build(z, available, [], 1)
    assert empty["short"].shape == (0, history.FEAT_DIM)
    assert empty["real"].shape == (0, 3, history.FEAT_DIM)
    with pytest.raises(ValueError):
        history.build(z[:, :22], available, [127], 1)
    with pytest.raises(ValueError):
        history.build(z, available[:, :22], [127], 1)
    with pytest.raises((ValueError, TypeError)):
        history.build(z, available.astype(int), [127], 1)
    for bad in ([126], [128], [-1], [127.0], [[127]]):
        with pytest.raises((ValueError, TypeError)):
            history.build(z, available, bad, 1)
    z[0, 0] = np.nan
    available[0, 0] = True
    with pytest.raises(ValueError):
        history.build(z, available, [127], 1)
