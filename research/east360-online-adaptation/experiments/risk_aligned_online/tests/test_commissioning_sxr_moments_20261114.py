import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_sxr_moments_20261114 as sxr


COLS = [15, 17, 16, 14]


def synthetic_series(n):
    t = np.arange(n, dtype=np.float64)
    values = np.full((n, 23), np.nan)
    skew = (t % 7 - 3) / 4
    values[:, COLS] = np.column_stack((
        np.exp(.01 * t), np.exp(-.004 * t), skew, skew**2 - 1.5 + .001 * t,
    ))
    available = np.zeros((n, 23), dtype=bool)
    available[:, COLS] = True
    return values, available


def test_coordinate_order_skew_sign_and_gain_invariance():
    values, available = synthetic_series(2)
    values[:, COLS] = [[4, 9, -1, 3], [4, 9, 1, 3]]
    coords, valid = sxr.coordinates(values, available)
    u, w = np.log(4.), .5 * np.log(9.)
    expected = np.array([[(u + w) / np.sqrt(2), (w - u) / np.sqrt(2), -1, np.log1p(4)],
                         [(u + w) / np.sqrt(2), (w - u) / np.sqrt(2), 1, np.log1p(4)]])
    np.testing.assert_allclose(coords, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(valid, [True, True])
    assert coords.dtype == np.float64
    assert valid.dtype == np.bool_

    for gain in [0.03125, 3.7, 1e4]:
        scaled = values.copy()
        scaled[:, 15] *= gain
        scaled[:, 17] *= gain**2
        actual, scaled_valid = sxr.coordinates(scaled, available)
        np.testing.assert_array_equal(scaled_valid, valid)
        np.testing.assert_allclose(actual[:, 0] - coords[:, 0], np.sqrt(2) * np.log(gain),
                                   atol=2e-15, rtol=0)
        np.testing.assert_allclose(actual[:, 1:], coords[:, 1:], atol=2e-15, rtol=0)


@pytest.mark.parametrize('col', COLS)
def test_joint_availability_and_finiteness_are_required(col):
    values, available = synthetic_series(5)
    available[1, col] = False
    values[2, col] = np.nan
    values[3, col] = np.inf
    values[4, col] = -np.inf
    coords, valid = sxr.coordinates(values, available)
    np.testing.assert_array_equal(valid, [True, False, False, False, False])
    np.testing.assert_array_equal(coords[1:], np.zeros((4, 4)))
    assert np.isfinite(coords).all()


@pytest.mark.parametrize('col', [15, 17])
@pytest.mark.parametrize('bad', [0., -0.1])
def test_nonpositive_mean_or_variance_is_invalid(col, bad):
    values, available = synthetic_series(2)
    values[1, col] = bad
    coords, valid = sxr.coordinates(values, available)
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_array_equal(coords[1], np.zeros(4))


def test_gap_threshold_only_clips_narrow_negative_roundoff():
    values, available = synthetic_series(4)
    values[:, 16] = 0
    boundary = -2 - 1e-5
    values[:, 14] = [-1.75, -2., np.nextafter(boundary, np.inf),
                     np.nextafter(boundary, -np.inf)]
    coords, valid = sxr.coordinates(values, available)
    np.testing.assert_array_equal(valid, [True, True, True, False])
    np.testing.assert_allclose(coords[:3, 3], [np.log1p(.25), 0, 0], atol=0, rtol=1e-15)
    np.testing.assert_array_equal(coords[3], np.zeros(4))


def test_packet_uses_two_masked_windows_and_five_ordered_future_bins():
    values, available = synthetic_series(80)
    available[:8, 16] = False
    available[16:24, 14] = False
    values[:8, 15] = 1e50
    values[16:24, 17] = 1e50
    context, targets, issue, mature = sxr.packet(values, available, np.array([31]))
    coords, observed = sxr.coordinates(values, available)
    recent, previous = coords[24:32].mean(axis=0), coords[8:16].mean(axis=0)
    expected_targets = np.stack([coords[32 + 5*i:37 + 5*i].mean(axis=0) - recent
                                 for i in range(5)])
    np.testing.assert_array_equal(observed[:8], False)
    np.testing.assert_allclose(context[0], np.r_[recent, recent - previous], rtol=0, atol=1e-15)
    np.testing.assert_allclose(targets[0], expected_targets, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(issue, [True])
    np.testing.assert_array_equal(mature, [True])
    assert context.dtype == targets.dtype == np.float64
    assert issue.dtype == mature.dtype == np.bool_


def test_history_threshold_and_start_of_series_are_not_padded():
    values, available = synthetic_series(80)
    context, target, issue, mature = sxr.packet(values, available, np.array([22, 23, 31]))
    np.testing.assert_array_equal(issue, [False, True, True])
    np.testing.assert_array_equal(mature, [False, True, True])
    np.testing.assert_array_equal(context[0], np.zeros(8))
    np.testing.assert_array_equal(target[0], np.zeros((5, 4)))

    available[:9, 15] = False
    context, target, issue, mature = sxr.packet(values, available, np.array([31]))
    np.testing.assert_array_equal(issue, [False])
    np.testing.assert_array_equal(mature, [False])
    np.testing.assert_array_equal(context, np.zeros((1, 8)))
    np.testing.assert_array_equal(target, np.zeros((1, 5, 4)))


@pytest.mark.parametrize('future_offset', range(1, 26))
def test_each_future_point_is_required_for_a_mature_target(future_offset):
    values, available = synthetic_series(80)
    available[31 + future_offset, COLS[future_offset % 4]] = False
    context, target, issue, mature = sxr.packet(values, available, np.array([31]))
    np.testing.assert_array_equal(issue, [True])
    np.testing.assert_array_equal(mature, [False])
    assert np.any(context)
    np.testing.assert_array_equal(target, np.zeros((1, 5, 4)))


def test_maturity_waits_for_q_plus_26_but_uses_only_q_plus_1_to_25():
    values, available = synthetic_series(58)
    q = np.array([31])
    short = sxr.packet(values[:57], available[:57], q)  # last row is q+25
    full = sxr.packet(values, available, q)  # q+26 exists
    np.testing.assert_array_equal(short[0], full[0])
    np.testing.assert_array_equal(short[2], [True])
    np.testing.assert_array_equal(short[3], [False])
    np.testing.assert_array_equal(short[1], np.zeros((1, 5, 4)))
    np.testing.assert_array_equal(full[3], [True])

    extended = sxr.packet(*synthetic_series(80), q)
    for complete, later in zip(full, extended, strict=True):
        np.testing.assert_array_equal(complete, later)

    values[57, COLS] = np.nan
    available[57, COLS] = False
    changed = sxr.packet(values, available, q)
    for before, after in zip(full, changed, strict=True):
        np.testing.assert_array_equal(before, after)


def test_issue_context_is_strictly_prefix_causal_despite_future_corruption():
    values, available = synthetic_series(100)
    q = np.array([31, 42])
    full = sxr.packet(values, available, q)
    prefix = sxr.packet(values[:43], available[:43], q)
    np.testing.assert_array_equal(full[0], prefix[0])
    np.testing.assert_array_equal(full[2], prefix[2])
    np.testing.assert_array_equal(prefix[3], [False, False])
    np.testing.assert_array_equal(prefix[1], np.zeros((2, 5, 4)))

    values[43:, COLS] = np.nan
    available[43:, COLS] = False
    corrupt = sxr.packet(values, available, q)
    np.testing.assert_array_equal(full[0], corrupt[0])
    np.testing.assert_array_equal(full[2], corrupt[2])
    np.testing.assert_array_equal(corrupt[3], [False, False])
    np.testing.assert_array_equal(corrupt[1], np.zeros((2, 5, 4)))


def test_gain_shifts_only_issue_level_g_and_cancels_from_targets():
    values, available = synthetic_series(100)
    q = np.array([31, 41, 73])
    baseline = sxr.packet(values, available, q)
    gain = 13.
    values[:, 15] *= gain
    values[:, 17] *= gain**2
    scaled = sxr.packet(values, available, q)
    np.testing.assert_array_equal(scaled[2], baseline[2])
    np.testing.assert_array_equal(scaled[3], baseline[3])
    np.testing.assert_allclose(scaled[0][:, 0] - baseline[0][:, 0], np.sqrt(2)*np.log(gain),
                               atol=1e-15, rtol=0)
    np.testing.assert_allclose(scaled[0][:, 1:], baseline[0][:, 1:], atol=1e-15, rtol=0)
    np.testing.assert_allclose(scaled[1], baseline[1], atol=1e-15, rtol=0)


def test_empty_queries_have_empty_well_typed_results():
    values, available = synthetic_series(0)
    coords, valid = sxr.coordinates(values, available)
    assert coords.shape == (0, 4) and valid.shape == (0,)
    context, target, issue, mature = sxr.packet(values, available, np.array([], dtype=int))
    assert context.shape == (0, 8) and target.shape == (0, 5, 4)
    assert issue.shape == mature.shape == (0,)


def test_query_order_and_repeated_issue_indices_are_preserved():
    values, available = synthetic_series(80)
    multiple = sxr.packet(values, available, np.array([42, 31, 42]))
    for i, q in enumerate([42, 31, 42]):
        single = sxr.packet(values, available, np.array([q]))
        for many, one in zip(multiple, single, strict=True):
            np.testing.assert_array_equal(many[i], one[0])


def test_invalid_input_shapes_masks_and_query_indices_are_rejected():
    values, available = synthetic_series(60)
    with pytest.raises(ValueError):
        sxr.coordinates(values[:, :22], available[:, :22])
    with pytest.raises(ValueError):
        sxr.coordinates(values, available.astype(int))
    with pytest.raises(ValueError):
        sxr.coordinates(values, available[:-1])
    with pytest.raises(ValueError):
        sxr.packet(values, available, np.array([[31]]))
    with pytest.raises(TypeError):
        sxr.packet(values, available, np.array([31.5]))
    for q in [-1, len(values)]:
        with pytest.raises(ValueError):
            sxr.packet(values, available, np.array([q]))
