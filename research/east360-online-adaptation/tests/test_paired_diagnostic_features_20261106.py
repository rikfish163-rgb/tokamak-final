"""Synthetic CPU-only contract tests for paired diagnostic features."""

import numpy as np
import pytest

import paired_diagnostic_features_20261106 as paired


def inputs(length):
    values = np.zeros((length, 23), dtype=np.float64)
    available = np.ones_like(values, dtype=bool)
    center = np.zeros(23, dtype=np.float64)
    scale = np.ones(23, dtype=np.float64)
    return values, available, center, scale


def test_constants_and_output_geometry_include_exact_end():
    assert paired.BLOCK == 32
    assert paired.GROUPS == (
        ("mir", (3, 4, 5)),
        ("sxr", (14, 15, 16, 17)),
        ("xuv", (19, 20, 21, 22)),
    )
    assert paired.CHANNELS == (3, 4, 5, 14, 15, 16, 17, 19, 20, 21, 22)
    assert paired.GROUP_INDEX == {
        channel: group
        for group, (_, channels) in enumerate(paired.GROUPS)
        for channel in channels
    }
    result = paired.schedule(*inputs(128))
    assert set(result) == {"origin", "maturity", "delta", "valid", "support"}
    np.testing.assert_array_equal(result["maturity"], [63, 95, 127])
    np.testing.assert_array_equal(result["origin"], [31, 63, 95])
    assert result["origin"].dtype == result["maturity"].dtype == np.int64
    assert result["delta"].shape == result["valid"].shape == (3, 11)
    assert result["delta"].dtype == np.float32
    assert result["valid"].dtype == result["support"].dtype == np.bool_
    assert result["support"].shape == (3,)
    assert result["valid"].all() and result["support"].all()
    assert not result["delta"].any()


def test_adjacent_blocks_and_true_median_in_native_coordinates():
    values, available, center, scale = inputs(96)
    center[3], scale[3] = 10, 2
    # Median of the transformed values is 1, not asinh(median(raw)).
    values[:16, 3] = 10
    values[16:32, 3] = 10 + 2 * np.sinh(2)
    values[32:64, 3] = 10 + 2 * np.sinh(3)
    values[64:, 3] = 10 + 2 * np.sinh(-1)
    result = paired.schedule(values, available, center, scale)
    np.testing.assert_allclose(result["delta"][:, 0], [2, -4], atol=1e-6)
    assert result["valid"][:, 0].all()


def test_oracle_for_each_channel_and_block_with_unaligned_missingness():
    rng = np.random.default_rng(123)
    values = rng.normal(size=(160, 23))
    available = rng.random(values.shape) > 0.17
    center = np.linspace(-1, 1, 23)
    scale = np.linspace(0.4, 2, 23)
    result = paired.schedule(values, available, center, scale)
    for row, maturity in enumerate(result["maturity"]):
        for col, channel in enumerate(paired.CHANNELS):
            blocks = (slice(maturity - 63, maturity - 31),
                      slice(maturity - 31, maturity + 1))
            observed = [values[block, channel][available[block, channel]]
                        for block in blocks]
            valid = all(len(block) >= 16 for block in observed)
            assert result["valid"][row, col] == valid
            if valid:
                medians = [np.median(np.arcsinh((block - center[channel]) / scale[channel]))
                           for block in observed]
                np.testing.assert_allclose(result["delta"][row, col], medians[1] - medians[0],
                                           rtol=1e-6, atol=1e-7)
            else:
                assert result["delta"][row, col] == 0


def test_prefix_at_each_maturity_and_changed_suffix_are_independent():
    rng = np.random.default_rng(456)
    values = rng.normal(size=(161, 23))
    available = rng.random(values.shape) > 0.2
    center = rng.normal(size=23)
    scale = rng.uniform(0.25, 3, size=23)
    whole = paired.schedule(values, available, center, scale)
    for row, maturity in enumerate(whole["maturity"]):
        prefix = paired.schedule(values[:maturity + 1], available[:maturity + 1],
                                 center, scale)
        changed_values = values.copy()
        changed_mask = available.copy()
        changed_values[maturity + 1:] = rng.normal(1000, 50, changed_values[maturity + 1:].shape)
        changed_mask[maturity + 1:] = ~changed_mask[maturity + 1:]
        changed = paired.schedule(changed_values, changed_mask, center, scale)
        for key in whole:
            np.testing.assert_array_equal(prefix[key], whole[key][:row + 1])
            np.testing.assert_array_equal(changed[key][:row + 1], whole[key][:row + 1])


def test_swapping_blocks_reverses_sign_and_preserves_support():
    rng = np.random.default_rng(789)
    values = rng.normal(size=(64, 23))
    available = rng.random(values.shape) > 0.25
    center = rng.normal(size=23)
    scale = rng.uniform(0.5, 2, size=23)
    normal = paired.schedule(values, available, center, scale)
    swapped = paired.schedule(np.concatenate((values[32:], values[:32])),
                              np.concatenate((available[32:], available[:32])),
                              center, scale)
    np.testing.assert_array_equal(swapped["delta"], -normal["delta"])
    np.testing.assert_array_equal(swapped["valid"], normal["valid"])
    np.testing.assert_array_equal(swapped["support"], normal["support"])


def test_missingness_threshold_group_support_and_diagnostics_when_unsupported():
    values, available, center, scale = inputs(64)
    available[:] = False
    values[32:] = np.sinh(1)
    available[:15, 3] = available[32:, 3] = True  # 15 in A: invalid.
    available[:16, 4] = available[32:48, 4] = True  # Exactly 16: valid.
    available[:32, 14] = available[32:47, 14] = True  # 15 in B: invalid.
    available[:16, 15] = available[32:48, 15] = True
    available[:15, 19] = available[32:48, 19] = True
    values[~available] = np.nan
    supported = paired.schedule(values, available, center, scale)
    assert supported["support"].tolist() == [True]
    assert np.flatnonzero(supported["valid"][0]).tolist() == [1, 4]
    np.testing.assert_allclose(supported["delta"][0, [1, 4]], 1)
    assert not supported["delta"][~supported["valid"]].any()

    available[47, 15] = False
    unsupported = paired.schedule(values, available, center, scale)
    assert unsupported["support"].tolist() == [False]
    assert np.flatnonzero(unsupported["valid"][0]).tolist() == [1]
    np.testing.assert_allclose(unsupported["delta"][0, 1], 1)
    assert not unsupported["delta"][~unsupported["valid"]].any()


def test_group_support_requires_same_channel_in_both_blocks():
    values, available, center, scale = inputs(64)
    available[:] = False
    available[:32, 3] = True
    available[32:, 4] = True  # MIR has observations in both halves, but no pair.
    available[:, 14] = True
    result = paired.schedule(values, available, center, scale)
    assert np.flatnonzero(result["valid"][0]).tolist() == [3]
    assert result["support"].tolist() == [False]


def test_masked_and_excluded_nonfinite_values_are_ignored():
    values, available, center, scale = inputs(64)
    values[:, 0] = np.inf
    values[:, 1] = np.nan
    center[0] = np.nan
    scale[0], scale[1] = 0, np.inf
    values[0, 3], values[1, 14] = np.nan, -np.inf
    available[0, 3] = available[1, 14] = False
    result = paired.schedule(values, available, center, scale)
    assert result["valid"].all() and result["support"].all()
    assert np.isfinite(result["delta"]).all() and not result["delta"].any()


def test_read_only_inputs_are_supported_and_not_modified():
    values, available, center, scale = inputs(64)
    values[0, 3] = np.nan
    available[0, 3] = False
    arrays = (values, available, center, scale)
    snapshots = tuple(array.copy() for array in arrays)
    for array in arrays:
        array.setflags(write=False)
    result = paired.schedule(*arrays)
    assert result["support"].tolist() == [True]
    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)


def test_extreme_finite_values_do_not_overflow_native_coordinates():
    values, available, center, scale = inputs(64)
    largest = np.finfo(np.float64).max
    smallest = np.nextafter(0.0, 1.0)
    values[:32, 3], values[32:, 3] = -largest, largest
    center[3], scale[3] = largest, smallest
    result = paired.schedule(values, available, center, scale)
    expected = (np.log(np.longdouble(4)) + np.log(np.longdouble(largest))
                - np.log(np.longdouble(smallest)))
    assert result["valid"][0, 0] and np.isfinite(result["delta"][0, 0])
    np.testing.assert_allclose(result["delta"][0, 0], expected, rtol=1e-6)


@pytest.mark.parametrize("length", [0, 1, 31, 62, 63])
def test_short_inputs_return_typed_empty_arrays(length):
    result = paired.schedule(*inputs(length))
    for key in ("origin", "maturity"):
        assert result[key].shape == (0,) and result[key].dtype == np.int64
    assert result["delta"].shape == (0, 11) and result["delta"].dtype == np.float32
    assert result["valid"].shape == (0, 11) and result["valid"].dtype == np.bool_
    assert result["support"].shape == (0,) and result["support"].dtype == np.bool_


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("row", [0, 64])
def test_observed_selected_nonfinite_values_fail_fast(bad, row):
    values, available, center, scale = inputs(65)
    values[row, 3] = bad
    with pytest.raises(ValueError, match="finite"):
        paired.schedule(values, available, center, scale)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_selected_scale_fails_fast_even_for_short_input(bad):
    values, available, center, scale = inputs(1)
    scale[20] = bad
    with pytest.raises(ValueError, match="scale"):
        paired.schedule(values, available, center, scale)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_selected_center_fails_fast(bad):
    values, available, center, scale = inputs(64)
    center[3] = bad
    with pytest.raises(ValueError, match="center"):
        paired.schedule(values, available, center, scale)


@pytest.mark.parametrize("case", ["values_ndim", "channels", "mask_shape", "mask_type",
                                  "center_shape", "scale_shape"])
def test_malformed_inputs_fail_fast(case):
    values, available, center, scale = inputs(64)
    if case == "values_ndim":
        values = values[0]
    elif case == "channels":
        values, available, center, scale = (values[:, :22], available[:, :22],
                                            center[:22], scale[:22])
    elif case == "mask_shape":
        available = available[:-1]
    elif case == "mask_type":
        available = available.astype(np.int64)
    elif case == "center_shape":
        center = center[:-1]
    else:
        scale = scale.reshape(1, -1)
    with pytest.raises(ValueError):
        paired.schedule(values, available, center, scale)
