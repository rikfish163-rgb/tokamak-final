"""Synthetic CPU-only tests for the common-support history kernel."""

import numpy as np
import pytest

import common_support_history_features_20261106 as common
import deep_miss_history_features_20261106 as v1


def blank(length):
    return np.zeros((length, 23), dtype=np.float64), np.zeros((length, 23), dtype=bool)


def historical_window(anchor, block, slot):
    stop = anchor - 127 - slot * block
    return stop - block, stop


@pytest.mark.parametrize("block", [16, 17, 63, 128, 256])
def test_support_complete_length_boundaries_unsorted_and_empty(block):
    length = 128 + 4 * block + 5
    _, mask = blank(length)
    mask[:, 0] = True
    anchors = np.array([
        127 + 4 * block, 127, 127 + 3 * block - 1, 127 + block,
        127 + 2 * block, 127 + 4 * block - 1, 127 + 4 * block,
    ])
    expected = np.array([
        [anchor - 127 >= (slot + 1) * block for slot in range(4)]
        for anchor in anchors
    ])
    result = common.support(mask, anchors, block)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)
    empty = common.support(mask, [], block)
    assert empty.shape == (0, 4) and empty.dtype == np.bool_


@pytest.mark.parametrize("block", [16, 17, 63])
def test_support_each_half_open_window_has_strict_edges_and_no_overlap(block):
    anchor = 129 + 4 * block
    _, mask = blank(anchor + 1)
    for slot in range(4):
        start, stop = historical_window(anchor, block, slot)
        for position in (start, stop - 1):
            mask[:] = False
            mask[position, 0] = True
            expected = np.zeros(4, dtype=bool)
            expected[slot] = True
            np.testing.assert_array_equal(common.support(mask, [anchor], block)[0], expected)
        for position in (start - 1, stop):
            mask[:] = False
            mask[position, 0] = True
            assert not common.support(mask, [anchor], block)[0, slot]


@pytest.mark.parametrize("block", [16, 17, 64, 256])
def test_build_exact_windows_slot_mapping_and_nonidentity_scramble(block):
    anchor = 129 + 4 * block
    z, mask = blank(anchor + 1)
    z[:, 0] = np.arange(anchor + 1)
    short_start = anchor - 127
    mask[short_start:anchor + 1, 0] = True
    for slot in range(4):
        start, stop = historical_window(anchor, block, slot)
        mask[start:stop, 0] = True

    result = common.build(z, mask, [anchor], shot=72, block=block)
    assert set(result) == {"short", "real", "stale", "scrambled", "support"}
    assert result["support"].shape == (1, 4)
    assert result["support"].dtype == np.bool_ and result["support"].all()
    assert result["short"].shape == (1, 146)
    assert result["short"].dtype == np.float64
    for arm in ("real", "stale", "scrambled"):
        assert result[arm].shape == (1, 3, 146)
        assert result[arm].dtype == np.float64

    expected_short, _ = v1._summarize(z, mask, short_start, anchor + 1)
    blocks = np.stack([
        v1._summarize(z, mask, *historical_window(anchor, block, slot))[0]
        for slot in range(4)
    ])
    np.testing.assert_array_equal(result["short"][0], expected_short)
    np.testing.assert_array_equal(result["real"][0], blocks[:3])
    np.testing.assert_array_equal(result["stale"][0], blocks[1:])
    permutation = v1._permutation(72, anchor, 3)
    assert permutation != (0, 1, 2)
    np.testing.assert_array_equal(result["scrambled"][0], blocks[list(permutation)])
    for left, right in (("real", "stale"), ("real", "scrambled"), ("stale", "scrambled")):
        assert not np.array_equal(result[left], result[right])
    repeated = common.build(z, mask, [anchor], shot=72, block=block)
    for key in result:
        np.testing.assert_array_equal(result[key], repeated[key])


def test_frozen_v1_equivalence_at_256_on_common_rows_with_missing_channels():
    rng = np.random.default_rng(401)
    z = rng.normal(size=(1400, 23))
    mask = rng.random(z.shape) > 0.42
    mask[:, 19:23] = False  # A diagnostic group is entirely unavailable.
    z[~mask] = np.nan
    anchors = [1399, 1151, 127, 1150, 1151, 1279]
    result = common.build(z, mask, anchors, "shot-27", 256)
    old = v1.build(z, mask, anchors, "shot-27")
    np.testing.assert_array_equal(result["support"][:, :3], old["support_real"])
    np.testing.assert_array_equal(result["support"][:, 1:], old["support_stale"])
    eligible = result["support"].all(axis=1)
    assert eligible.tolist() == [True, True, False, False, True, True]
    for key in ("short", "real", "stale", "scrambled"):
        np.testing.assert_array_equal(result[key][eligible], old[key][eligible])
        assert not result[key][~eligible].any()


@pytest.mark.parametrize("block", [16, 17, 63, 128])
def test_random_masked_overlapping_windows_match_independent_v1_references(block):
    rng = np.random.default_rng(block)
    length = 128 + 5 * block + 10
    z = rng.normal(size=(length, 23))
    mask = rng.random(z.shape) > 0.8
    mask[:, 19:23] = False
    z[~mask] = np.nan
    anchors = [length - 1, length - 1 - block, 127, 127 + 4 * block,
               127 + 3 * block, length - 1, 127 + 4 * block + 1]
    result = common.build(z, mask, anchors, "random", block)
    for row, anchor in enumerate(anchors):
        windows = [historical_window(anchor, block, slot) for slot in range(4)]
        expected_support = [
            start >= 0 and mask[start:stop][:, v1.USED_CHANNELS].any()
            for start, stop in windows
        ]
        np.testing.assert_array_equal(result["support"][row], expected_support)
        if not all(expected_support):
            for key in ("short", "real", "stale", "scrambled"):
                assert not result[key][row].any()
            continue
        expected_short = v1._summarize(z, mask, anchor - 127, anchor + 1)[0]
        blocks = np.stack([v1._summarize(z, mask, start, stop)[0] for start, stop in windows])
        permutation = v1._permutation("random", anchor, 3)
        np.testing.assert_array_equal(result["short"][row], expected_short)
        np.testing.assert_array_equal(result["real"][row], blocks[:3])
        np.testing.assert_array_equal(result["stale"][row], blocks[1:])
        np.testing.assert_array_equal(result["scrambled"][row], blocks[list(permutation)])


def test_any_used_observation_is_sufficient_but_excluded_channels_do_not_support():
    block = 16
    anchor = 127 + 4 * block
    z, mask = blank(anchor + 1)
    mask[:, [1, 2, 10]] = True
    z[:, [1, 2, 10]] = np.inf
    for slot, column in enumerate(v1.USED_CHANNELS[:4]):
        start, _ = historical_window(anchor, block, slot)
        mask[start, column] = True
        z[start, column] = slot + 1
    result = common.build(z, mask, [anchor], 8, block)
    assert result["support"].all()
    assert not result["short"].any()  # No used channel is observed in the short window.
    start, stop = historical_window(anchor, block, 2)
    mask[start:stop, v1.USED_CHANNELS] = False
    result = common.build(z, mask, [anchor], 8, block)
    np.testing.assert_array_equal(result["support"][0], [True, True, False, True])
    for key in ("short", "real", "stale", "scrambled"):
        assert not result[key].any()


def test_unsupported_rows_are_zero_without_inspecting_nonfinite_values():
    block = 32
    anchor = 127 + 4 * block
    z, mask = blank(anchor + 1)
    mask[:, 0] = True
    mask[0:block, 0] = False  # Only the oldest block is unsupported at this anchor.
    z[:, 0] = np.nan
    result = common.build(z, mask, [127, anchor], 2, block)
    assert not result["support"].all(axis=1).any()
    for key in ("short", "real", "stale", "scrambled"):
        assert not result[key].any()


@pytest.mark.parametrize("window", ["short", 0, 1, 2, 3])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_common_row_rejects_nonfinite_observed_used_coordinate(window, bad_value):
    block = 17
    anchor = 127 + 4 * block
    z, mask = blank(anchor + 1)
    mask[:, 0] = True
    position = anchor if window == "short" else historical_window(anchor, block, window)[0]
    z[position, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        common.build(z, mask, [anchor], 9, block)
    mask[position, 0] = False
    if window != "short":
        assert common.support(mask, [anchor], block).all()
    result = common.build(z, mask, [anchor], 9, block)
    for key in ("short", "real", "stale", "scrambled"):
        assert np.isfinite(result[key]).all()


def test_future_changes_prefix_truncation_and_batch_order_cannot_affect_any_row():
    rng = np.random.default_rng(41)
    block = 32
    z = rng.normal(size=(600, 23))
    mask = rng.random(z.shape) > 0.3
    original_z, original_mask = z.copy(), mask.copy()
    anchors = [400, 127, 255, 599, 400, 254, 317]
    batch = common.build(z, mask, anchors, 19, block)
    np.testing.assert_array_equal(batch["support"], common.support(mask, anchors, block))
    for row, anchor in enumerate(anchors):
        changed_z, changed_mask = z.copy(), mask.copy()
        changed_z[anchor + 1:] = np.nan
        changed_mask[anchor + 1:] = ~changed_mask[anchor + 1:]
        future_changed = common.build(changed_z, changed_mask, [anchor], 19, block)
        prefix = common.build(z[:anchor + 1], mask[:anchor + 1], [anchor], 19, block)
        for key in batch:
            np.testing.assert_array_equal(batch[key][row], future_changed[key][0])
            np.testing.assert_array_equal(batch[key][row], prefix[key][0])
    np.testing.assert_array_equal(z, original_z)
    np.testing.assert_array_equal(mask, original_mask)


def test_empty_batch_and_very_large_block():
    z, mask = blank(0)
    result = common.build(z, mask, [], 3, np.int64(16))
    assert result["short"].shape == (0, 146)
    assert result["support"].shape == (0, 4)
    for key in ("real", "stale", "scrambled"):
        assert result[key].shape == (0, 3, 146)
    z, mask = blank(128)
    mask[:, 0] = True
    result = common.build(z, mask, [127], 3, 2**80)
    assert not result["support"].any()
    for key in ("short", "real", "stale", "scrambled"):
        assert not result[key].any()


@pytest.mark.parametrize("bad_block", [15, 0, -16, 16.0, "16", True, None])
def test_invalid_block_rejected_in_both_entry_points(bad_block):
    z, mask = blank(128)
    with pytest.raises((ValueError, TypeError)):
        common.support(mask, [], bad_block)
    with pytest.raises((ValueError, TypeError)):
        common.build(z, mask, [], 1, bad_block)


@pytest.mark.parametrize("bad_anchors", [[126], [128], [-1], [127.0], [[127]], [True]])
def test_invalid_anchors_rejected_in_both_entry_points(bad_anchors):
    z, mask = blank(128)
    with pytest.raises((ValueError, TypeError)):
        common.support(mask, bad_anchors, 16)
    with pytest.raises((ValueError, TypeError)):
        common.build(z, mask, bad_anchors, 1, 16)


def test_invalid_shapes_and_dtypes_rejected_in_both_entry_points():
    z, mask = blank(128)
    with pytest.raises(ValueError, match="same shape"):
        common.build(z, mask[:-1], [], 1, 16)
    for bad_mask in (mask[:, :22], mask.astype(int), mask[0], np.zeros((128, 23, 1), dtype=bool)):
        with pytest.raises((ValueError, TypeError)):
            common.support(bad_mask, [], 16)
        with pytest.raises((ValueError, TypeError)):
            common.build(z, bad_mask, [], 1, 16)
    for bad_z in (z[:, :22], z[0], z.astype(complex), z.astype(str), z.astype(bool)):
        with pytest.raises((ValueError, TypeError)):
            common.build(bad_z, mask, [], 1, 16)
