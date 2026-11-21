import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_pending_view_20261114 import pending_views


COLUMNS = (14, 15, 16, 17, 19, 20, 21, 22)
DELAY = 8


def _windows(ends, width=16):
    """Synthetic already-masked windows on the clock's physical time grid."""
    ends = np.asarray(ends, dtype=np.int64)
    times = ends[:, None] - np.arange(width - 1, -1, -1)
    values = (times[..., None] * 100 + np.arange(23)).astype(np.float32)
    arrived = np.broadcast_to((times >= 0)[..., None], values.shape).copy()
    arrived[:, -DELAY:, COLUMNS] = False
    values[~arrived] = 0
    return values, arrived


def _center():
    return np.arange(23, dtype=np.float32) + 0.25


def test_carry_and_center_use_the_same_mask_and_exact_last_arrived_anchor():
    ends = np.array([30, 50])
    values, arrived = _windows(ends)
    # An older value exists but the exact anchor for this channel is absent.
    arrived[0, 7, COLUMNS[1]] = False
    values[0, 7, COLUMNS[1]] = 0
    # The encoder also treats nonfinite anchors as unavailable.
    values[1, 7, COLUMNS[2]] = np.nan
    # Even an already-observed tail element must remain untouched.
    arrived[0, -3, COLUMNS[3]] = True
    values[0, -3, COLUMNS[3]] = 1234
    before_values, before_arrived = values.copy(), arrived.copy()

    carry, expected_mask, center_only = pending_views(
        values, arrived, ends, COLUMNS, DELAY, _center()
    )

    new = np.zeros_like(arrived)
    new[:, -8:, COLUMNS] = True
    new[0, -8:, COLUMNS[1]] = False
    new[1, -8:, COLUMNS[2]] = False
    new[0, -3, COLUMNS[3]] = False
    np.testing.assert_array_equal(expected_mask, before_arrived | new)
    np.testing.assert_array_equal(carry[~new], before_values[~new])
    np.testing.assert_array_equal(center_only[~new], before_values[~new])
    for row in range(len(ends)):
        for column in COLUMNS:
            selected = new[row, :, column]
            np.testing.assert_array_equal(
                carry[row, selected, column],
                np.full(selected.sum(), before_values[row, 7, column]),
            )
            np.testing.assert_array_equal(
                center_only[row, selected, column],
                np.full(selected.sum(), _center()[column]),
            )
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(arrived, before_arrived)
    assert carry.dtype == center_only.dtype == values.dtype
    assert expected_mask.dtype == np.bool_
    assert not np.shares_memory(carry, values)
    assert not np.shares_memory(center_only, values)
    assert not np.shares_memory(expected_mask, arrived)


def test_shot_start_requires_the_anchor_to_exist_at_physical_time_zero():
    ends = np.array([0, 7, 8, 9])
    values, arrived = _windows(ends)
    carry, mask, control = pending_views(values, arrived, ends, COLUMNS, DELAY, _center())

    # No anchor exists before t=8, even though some pending slots are in-shot.
    np.testing.assert_array_equal(carry[:2], values[:2])
    np.testing.assert_array_equal(mask[:2], arrived[:2])
    np.testing.assert_array_equal(control[:2], values[:2])
    # At t=8 the anchor is physical sample 0; at t=9 it is sample 1.
    for row, anchor_time in ((2, 0), (3, 1)):
        for column in COLUMNS:
            np.testing.assert_array_equal(carry[row, -8:, column], anchor_time * 100 + column)
            np.testing.assert_array_equal(control[row, -8:, column], _center()[column])
            assert mask[row, -8:, column].all()
    # Padding before the shot is never altered, including its original mask.
    for row, end in enumerate(ends):
        padding = max(0, values.shape[1] - 1 - end)
        np.testing.assert_array_equal(carry[row, :padding], values[row, :padding])
        np.testing.assert_array_equal(mask[row, :padding], arrived[row, :padding])
        np.testing.assert_array_equal(control[row, :padding], values[row, :padding])


@pytest.mark.parametrize("width", [1, 8])
def test_anchor_outside_window_never_falls_back_to_an_older_or_other_row(width):
    ends = np.array([20, 30])
    values, arrived = _windows(ends, width=width)
    result = pending_views(values, arrived, ends, COLUMNS, DELAY, _center())
    for actual, expected in zip(result, (values, arrived, values), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_anchor_at_first_window_position_is_still_usable():
    values, arrived = _windows([8], width=9)
    carry, mask, control = pending_views(values, arrived, np.array([8]), COLUMNS, DELAY, _center())
    for column in COLUMNS:
        np.testing.assert_array_equal(carry[0, 1:, column], values[0, 0, column])
        np.testing.assert_array_equal(control[0, 1:, column], _center()[column])
        assert mask[0, 1:, column].all()


def test_zero_delay_is_an_unchanged_independent_view():
    ends = np.array([0, 30])
    values, arrived = _windows(ends)
    result = pending_views(values, arrived, ends, COLUMNS, 0, _center())
    for actual, expected in zip(result, (values, arrived, values), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)


def test_existing_clock_never_exposes_pending_quality_or_future_to_views():
    from experiments.risk_aligned_online import commissioning_observation_clock_20261114 as clock

    assert tuple(clock.RADIATION) == COLUMNS
    t = 31
    data = np.arange(50 * 23, dtype=np.float32).reshape(50, 23)
    signal = {"values": data, "available": np.ones_like(data, dtype=bool)}
    changed = {key: value.copy() for key, value in signal.items()}
    # Concealed values and quality are changed, as is the whole future suffix.
    changed["values"][np.ix_(np.arange(t - 7, t + 1), COLUMNS)] = 1e7
    changed["available"][np.ix_(np.arange(t - 7, t + 1), COLUMNS)] = False
    changed["values"][t + 1 :] = -1e7
    changed["available"][t + 1 :] = False
    ends = np.array([t])
    original = clock.windows(signal, ends, 16, DELAY)
    forged = clock.windows(changed, ends, 16, DELAY)
    for a, b in zip(original, forged, strict=True):
        np.testing.assert_array_equal(a, b)
    carry, mask, control = pending_views(*original, ends, COLUMNS, DELAY, _center())
    for column in COLUMNS:
        np.testing.assert_array_equal(carry[0, -8:, column], signal["values"][t - 8, column])
        np.testing.assert_array_equal(control[0, -8:, column], _center()[column])
        assert mask[0, -8:, column].all()
    for a, b in zip(
        (carry, mask, control),
        pending_views(*forged, ends, COLUMNS, DELAY, _center()),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "change",
    [
        lambda v, m, e, c, d, z: (v[:, :, :22], m, e, c, d, z),
        lambda v, m, e, c, d, z: (v, m.astype(np.int8), e, c, d, z),
        lambda v, m, e, c, d, z: (v, m, e[:0], c, d, z),
        lambda v, m, e, c, d, z: (v, m, np.array([-1]), c, d, z),
        lambda v, m, e, c, d, z: (v, m, e, (23,), d, z),
        lambda v, m, e, c, d, z: (v, m, e, c, -1, z),
        lambda v, m, e, c, d, z: (v, m, e, c, d, z[:22]),
    ],
)
def test_rejects_malformed_array_contracts(change):
    values, arrived = _windows([20])
    with pytest.raises((ValueError, TypeError)):
        pending_views(*change(values, arrived, np.array([20]), COLUMNS, DELAY, _center()))
