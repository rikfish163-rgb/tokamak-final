import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_transition_data_20261110 import (
    inputs,
    reverse_valid,
    targets,
)


def test_inputs_column_order_counts_clipping_and_future_independence():
    values = np.array([[6, -4], [5, -2], [12, 0], [0, 1], [10, 2]])
    z = np.zeros((96, 2), dtype=np.float64)
    z[:80] = np.repeat(values, 16, axis=0)
    mask = np.ones_like(z, dtype=bool)
    mask[23:32, 1] = False
    mask[40:48, 1] = False
    mask[72:80, 1] = False
    z[~mask] = np.nan

    context = np.tile([6.0, -7.0], (96, 1))
    context_mask = np.ones_like(context, dtype=bool)
    context_mask[64:80, 1] = False
    context_mask[64:72, 1] = True
    context[64:72, 1] = -5
    context[~context_mask] = np.nan

    actual = inputs(z, mask, context, context_mask, np.array([79]))
    expected = np.array(
        [[4, -4, -4, -4, -4, 0, 2, -2, -4, -1, 0, 0,
          1, 1, 1, 7 / 16, 1, 1 / 2, 1, 1, 1, 1 / 2, 4, 2]],
        dtype=np.float32,
    )
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)

    z[80:] = -999
    mask[80:] = False
    context[80:] = 999
    context_mask[80:] = False
    np.testing.assert_array_equal(
        inputs(z, mask, context, context_mask, np.array([79])), expected
    )


def test_inputs_prefix_padding_and_eight_point_threshold():
    z = np.full((32, 1), 3.0)
    mask = np.ones_like(z, dtype=bool)
    context = np.column_stack([np.arange(32, dtype=float), np.full(32, np.nan)])
    context_mask = np.column_stack([np.ones(32, dtype=bool), np.zeros(32, dtype=bool)])

    actual = inputs(z, mask, context, context_mask, np.array([6, 7, 23]))
    expected = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7 / 16, 0],
         [3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 3],
         [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 1, 3]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_inputs_invalid_recent_bin_zeros_differences_but_preserves_fractions():
    z = np.ones((80, 1))
    mask = np.ones_like(z, dtype=bool)
    mask[71:80, 0] = False
    z[~mask] = np.nan
    context = np.full((80, 2), np.nan)
    context_mask = np.zeros_like(context, dtype=bool)

    actual = inputs(z, mask, context, context_mask, np.array([79]))
    np.testing.assert_array_equal(actual[0, :7], np.zeros(7))
    np.testing.assert_array_equal(actual[0, 7:12], [1, 1, 1, 1, 7 / 16])
    assert actual[0, 12] == 0


def test_targets_directional_inclusive_boundaries_and_minimum_slope():
    z = np.zeros((57, 4), dtype=np.float64)
    z[:16, 1] = 1
    z[16:32, 0] = 1
    z[16:32, 2] = 0.05
    z[16:32, 3] = np.nextafter(0.05, 0.0)
    for group, (positive, negative) in enumerate(
        zip([0.75, 0.75001, 1, 1.24999, 1.25],
            [0.25, 0.24999, 0, -0.24999, -0.25])
    ):
        z[32 + 5 * group:37 + 5 * group] = [positive, negative, 0.05, 0]

    labels, valid = targets(z, np.ones_like(z, dtype=bool), np.array([31]))
    assert labels.dtype == np.int64
    assert valid.dtype == np.bool_
    assert labels.shape == valid.shape == (1, 5, 4)
    np.testing.assert_array_equal(labels[0, :, 0], [0, 1, 1, 1, 2])
    np.testing.assert_array_equal(labels[0, :, 1], [0, 1, 1, 1, 2])
    np.testing.assert_array_equal(labels[0, :, 2:], np.ones((5, 2)))
    np.testing.assert_array_equal(valid[0, :, :3], np.ones((5, 3), dtype=bool))
    assert not valid[0, :, 3].any()


def test_targets_eight_history_and_three_future_samples_with_missing_nans():
    z = np.zeros((57, 4))
    z[16:32] = 1
    z[32:] = 1.5
    mask = np.ones_like(z, dtype=bool)
    mask[8:16, 0] = False
    mask[24:32, 0] = False
    mask[35:37, 0] = False
    mask[0:9, 1] = False
    mask[16:25, 2] = False
    mask[34:37, 3] = False
    z[~mask] = np.nan

    labels, valid = targets(z, mask, np.array([31]))
    expected_valid = np.array(
        [[True, False, False, False],
         [True, False, False, True],
         [True, False, False, True],
         [True, False, False, True],
         [True, False, False, True]]
    )
    np.testing.assert_array_equal(valid[0], expected_valid)
    np.testing.assert_array_equal(labels[0][valid[0]], 2)
    np.testing.assert_array_equal(labels[0][~valid[0]], 1)


def test_targets_only_reads_through_the_twenty_fifth_future_point():
    z = np.zeros((80, 1))
    z[16:32] = 1
    z[32:57] = 0.5
    mask = np.ones_like(z, dtype=bool)
    expected_labels, expected_valid = targets(z, mask, np.array([31]))

    z[57:] = 1e6
    mask[57:] = False
    actual_labels, actual_valid = targets(z, mask, np.array([31]))
    np.testing.assert_array_equal(expected_labels, np.zeros((1, 5, 1)))
    assert expected_valid.all()
    np.testing.assert_array_equal(actual_labels, expected_labels)
    np.testing.assert_array_equal(actual_valid, expected_valid)


def test_reverse_valid_reverses_only_valid_times_per_sample_and_channel():
    labels = np.array(
        [[[0, 2], [1, 0], [2, 1], [1, 2], [0, 0]],
         [[2, 1], [1, 2], [0, 0], [2, 1], [1, 2]]],
        dtype=np.int64,
    )
    valid = np.array(
        [[[1, 1], [0, 1], [1, 0], [1, 0], [0, 1]],
         [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0]]],
        dtype=bool,
    )
    expected = np.array(
        [[[1, 0], [1, 0], [2, 1], [0, 2], [0, 2]],
         [[1, 1], [2, 2], [0, 0], [1, 1], [2, 2]]],
        dtype=np.int64,
    )
    original_labels = labels.copy()
    original_valid = valid.copy()

    actual = reverse_valid(labels, valid)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(valid, original_valid)
    np.testing.assert_array_equal(actual[~valid], labels[~valid])
    for batch in range(labels.shape[0]):
        for channel in range(labels.shape[2]):
            selected = valid[batch, :, channel]
            np.testing.assert_array_equal(
                actual[batch, selected, channel], labels[batch, selected, channel][::-1]
            )


def test_empty_batches_preserve_shapes_and_dtypes():
    z = np.empty((0, 3), dtype=np.float32)
    mask = np.empty((0, 3), dtype=bool)
    context = np.empty((0, 2), dtype=np.float32)
    context_mask = np.empty((0, 2), dtype=bool)
    indices = np.empty(0, dtype=np.int64)

    x = inputs(z, mask, context, context_mask, indices)
    labels, valid = targets(z, mask, indices)
    reversed_labels = reverse_valid(labels, valid)
    assert x.shape == (0, 35) and x.dtype == np.float32
    assert labels.shape == valid.shape == reversed_labels.shape == (0, 5, 3)
    assert labels.dtype == reversed_labels.dtype == np.int64
    assert valid.dtype == np.bool_


@pytest.mark.parametrize("endpoints", [np.array([-1]), np.array([57]), np.array([1.5])])
def test_inputs_rejects_invalid_endpoints(endpoints):
    z = np.zeros((57, 1))
    with pytest.raises((ValueError, TypeError)):
        inputs(z, np.ones_like(z, dtype=bool), np.zeros((57, 2)),
               np.ones((57, 2), dtype=bool), endpoints)


@pytest.mark.parametrize("origins", [np.array([30]), np.array([32]), np.array([31.0])])
def test_targets_rejects_invalid_origins(origins):
    z = np.zeros((57, 1))
    with pytest.raises((ValueError, TypeError)):
        targets(z, np.ones_like(z, dtype=bool), origins)


def test_interfaces_reject_mismatched_masks():
    z = np.zeros((57, 1))
    with pytest.raises((ValueError, TypeError)):
        inputs(z, np.ones((57, 2), dtype=bool), np.zeros((57, 2)),
               np.ones((57, 2), dtype=bool), np.array([31]))
    with pytest.raises((ValueError, TypeError)):
        targets(z, np.ones((56, 1), dtype=bool), np.array([31]))
    with pytest.raises((ValueError, TypeError)):
        reverse_valid(np.zeros((1, 5, 1), dtype=np.int64), np.ones((1, 4, 1), dtype=bool))
