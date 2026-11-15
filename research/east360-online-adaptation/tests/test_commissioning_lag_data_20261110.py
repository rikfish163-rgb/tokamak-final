import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_lag_data_20261110 import (
    GROUPS,
    LAGS,
    PHASES,
    feedback,
    signatures,
)


def integrate(increments):
    increments = np.asarray(increments, dtype=np.float64)
    return np.vstack((np.zeros((1, increments.shape[1])), np.cumsum(increments, axis=0)))


def scalar_signature(z, mask, ends, phases, groups):
    """A deliberately scalar statement of the window and edge definitions."""
    edges = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))
             if groups[i] != groups[j]]
    values = np.zeros((len(ends), len(LAGS) * len(edges)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    for row, end in enumerate(ends):
        if end < 25:
            continue
        units = []
        for channel, group in enumerate(groups):
            window = z[end - 25:end + 1, channel]
            if not np.all(mask[end - 25:end + 1, channel]):
                units.append(None)
                continue
            increments = np.diff(window)
            increments -= np.mean(increments)
            norm = np.sqrt(sum(float(x) ** 2 for x in increments))
            units.append(np.roll(increments / norm, phases[group]) if norm >= 1e-4 else None)
        for lag_index, lag in enumerate(LAGS):
            for edge_index, (i, j) in enumerate(edges):
                if units[i] is None or units[j] is None:
                    continue
                forward = sum(units[i][t] * units[j][(t + lag) % 25] for t in range(25))
                reverse = sum(units[j][t] * units[i][(t + lag) % 25] for t in range(25))
                column = lag_index * len(edges) + edge_index
                values[row, column] = 0.5 * (forward - reverse)
                valid[row, column] = True
    return values, valid


def test_contract_constants_are_fixed():
    assert GROUPS == (0, 0, 0, 1, 1, 3, 2, 2, 2, 2)
    assert LAGS == (4, 8)
    assert PHASES == ((0, 0, 0, 0), (0, 6, 12, 18), (0, -6, -12, -18))


def test_known_direction_lag_major_order_and_roll_sign():
    increments = np.zeros((25, 3))
    increments[2, 0] = increments[6, 1] = increments[10, 2] = 1
    z = integrate(increments)
    mask = np.ones_like(z, dtype=bool)

    values, valid = signatures(z, mask, [25], groups=(0, 1, 2))
    np.testing.assert_allclose(values, [[25 / 48, 0, 25 / 48, 0, 25 / 48, 0]], atol=1e-7)
    assert valid.all()
    assert values.dtype == np.float32 and valid.dtype == bool

    shifted, shifted_valid = signatures(z, mask, [25], phases=(0, 4, 0, 0), groups=(0, 1, 2))
    np.testing.assert_allclose(shifted[0, [0, 3]], [0, 25 / 48], atol=1e-7)
    np.testing.assert_array_equal(shifted_valid, valid)

    reversed_values, _ = signatures(z[:, [1, 0]], mask[:, [1, 0]], [25], groups=(1, 0))
    np.testing.assert_allclose(reversed_values, [[-25 / 48, 0]], atol=1e-7)


@pytest.mark.parametrize('phases', PHASES)
def test_matches_scalar_definition_with_default_groups_and_missing_values(phases):
    rng = np.random.default_rng(120)
    z = rng.normal(size=(65, 10)).cumsum(axis=0)
    mask = np.ones_like(z, dtype=bool)
    mask[29, 2] = False
    z[29, 2] = np.nan
    z[:, 5] = 2.0
    ends = [59, 24, 31, 31, -1, 25, 64]

    actual = signatures(z, mask, ends, phases=phases)
    expected = scalar_signature(z, mask, ends, phases, GROUPS)
    assert actual[0].shape == (len(ends), 70)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual[0][~actual[1]], 0)
    np.testing.assert_array_equal(actual[0][2], actual[0][3])


def test_group_roll_is_applied_to_the_whole_increment_sequence_without_changing_availability():
    rng = np.random.default_rng(17)
    increments = rng.normal(size=(25, 4)) + np.array([1.0, -0.5, 0.1, 0.4])
    z = integrate(increments)
    mask = np.ones_like(z, dtype=bool)
    mask[7, 2] = False
    groups = (0, 0, 1, 3)
    phases = (3, -5, 12, 9)
    rolled = np.column_stack([np.roll(increments[:, c], phases[group])
                              for c, group in enumerate(groups)])

    phased = signatures(z, mask, [25], phases=phases, groups=groups)
    physically_rolled = signatures(integrate(rolled), mask, [25], groups=groups)
    original_valid = signatures(z, mask, [25], groups=groups)[1]
    common_offset = signatures(z, mask, [25], phases=tuple(p + 7 for p in phases), groups=groups)
    np.testing.assert_array_equal(phased[1], original_valid)
    np.testing.assert_array_equal(phased[1], physically_rolled[1])
    np.testing.assert_allclose(phased[0], physically_rolled[0], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(common_offset[1], phased[1])
    np.testing.assert_allclose(common_offset[0], phased[0], rtol=1e-6, atol=1e-7)


def test_window_edges_norm_threshold_and_missing_values():
    amplitude = 1e-4 / np.sqrt(24 / 25)
    increments = np.zeros((25, 4))
    increments[3, 0] = amplitude * 1.01
    increments[7, 1] = amplitude * 1.01
    increments[8, 2] = amplitude * 1.01
    increments[9, 3] = amplitude * 0.99
    z = integrate(increments)
    mask = np.ones_like(z, dtype=bool)
    mask[0, 2] = False
    values, valid = signatures(z, mask, [24, 25], groups=(0, 1, 2, 3))
    np.testing.assert_array_equal(valid[0], False)
    np.testing.assert_array_equal(values[0], 0)
    # Edges are 01,02,03,12,13,23 for each lag.
    np.testing.assert_array_equal(valid[1], [True, False, False, False, False, False] * 2)
    np.testing.assert_array_equal(values[~valid], 0)

    mask[0, 2] = True
    mask[25, 1] = False
    _, end_missing = signatures(z, mask, [25], groups=(0, 1, 2, 3))
    np.testing.assert_array_equal(end_missing[0], [False, True, False, False, False, False] * 2)


def test_feedback_uses_same_valid_edges_and_joint_budget():
    rng = np.random.default_rng(723)
    z = rng.normal(size=(90, 10)).cumsum(axis=0)
    mask = np.ones_like(z, dtype=bool)
    mask[40, 2] = False
    mask[70, :] = False
    ends = [24, 25, 40, 56, 70, 89, 56]
    real, null, eligible = feedback(z, mask, ends)
    r, rv = signatures(z, mask, ends, phases=PHASES[0])
    p, pv = signatures(z, mask, ends, phases=PHASES[1])
    n, nv = signatures(z, mask, ends, phases=PHASES[2])
    np.testing.assert_array_equal(rv, pv)
    np.testing.assert_array_equal(rv, nv)
    expected_real = r - (p + n) / 2
    expected_null = p - n
    real_norm = np.linalg.norm(expected_real, axis=1)
    null_norm = np.linalg.norm(expected_null, axis=1)
    expected_eligible = rv.any(axis=1) & (real_norm >= 1e-4) & (null_norm >= 1e-4)
    assert expected_eligible.any() and (~expected_eligible).any()
    np.testing.assert_array_equal(eligible, expected_eligible)
    np.testing.assert_allclose(real[eligible], expected_real[eligible] / real_norm[eligible, None], atol=1e-7)
    np.testing.assert_allclose(null[eligible], expected_null[eligible] / null_norm[eligible, None], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(real[eligible], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(null[eligible], axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(real[~eligible], 0)
    np.testing.assert_array_equal(null[~eligible], 0)
    np.testing.assert_array_equal(real[~rv], 0)
    np.testing.assert_array_equal(null[~rv], 0)
    assert real.dtype == np.float32 and null.dtype == np.float32 and eligible.dtype == bool


@pytest.mark.parametrize('second', ['same', 'quadrature'])
def test_feedback_rejects_both_when_either_delta_is_degenerate(second):
    t = np.arange(25)
    x = np.sin(2 * np.pi * t / 25)
    y = x if second == 'same' else np.cos(2 * np.pi * t / 25)
    z = integrate(np.column_stack((x, y)))
    mask = np.ones_like(z, dtype=bool)
    r, _ = signatures(z, mask, [25], phases=PHASES[0], groups=(0, 1))
    p, _ = signatures(z, mask, [25], phases=PHASES[1], groups=(0, 1))
    n, _ = signatures(z, mask, [25], phases=PHASES[2], groups=(0, 1))
    real_norm = np.linalg.norm(r - (p + n) / 2)
    null_norm = np.linalg.norm(p - n)
    assert (real_norm < 1e-4 and null_norm >= 1e-4) if second == 'same' else (
        real_norm >= 1e-4 and null_norm < 1e-4)
    real, null, eligible = feedback(z, mask, [25], groups=(0, 1))
    np.testing.assert_array_equal(eligible, [False])
    np.testing.assert_array_equal(real, 0)
    np.testing.assert_array_equal(null, 0)


@pytest.mark.parametrize('small_delta', ['real', 'null'])
@pytest.mark.parametrize('factor, expected', [(0.98, False), (1.02, True)])
def test_feedback_joint_norm_threshold_on_both_sides(small_delta, factor, expected):
    omega = 2 * np.pi / 25
    lag_norm = np.linalg.norm([np.sin(omega * lag) for lag in (4, 8)])
    if small_delta == 'real':
        scale = (1 - np.cos(6 * omega)) * lag_norm
        theta = np.arcsin(factor * 1e-4 / scale)
    else:
        scale = 2 * np.sin(6 * omega) * lag_norm
        theta = np.pi / 2 - np.arcsin(factor * 1e-4 / scale)
    t = np.arange(25)
    z = integrate(np.column_stack((np.sin(omega * t), np.sin(omega * t + theta))))
    mask = np.ones_like(z, dtype=bool)

    real, null, eligible = feedback(z, mask, [25], groups=(0, 1))
    np.testing.assert_array_equal(eligible, [expected])
    if expected:
        np.testing.assert_allclose([np.linalg.norm(real), np.linalg.norm(null)], [1, 1], atol=1e-6)
    else:
        np.testing.assert_array_equal(real, 0)
        np.testing.assert_array_equal(null, 0)


@pytest.mark.parametrize('groups', [(3, 0), (2, 2, 0, 3, 1), (1,), ()])
def test_custom_group_ids_and_empty_channel_set_match_scalar_definition(groups):
    rng = np.random.default_rng(50 + len(groups))
    z = rng.normal(size=(40, len(groups))).cumsum(axis=0)
    mask = np.ones_like(z, dtype=bool)
    phases = (2, -7, 4, 11)
    ends = [39, 25, 24]
    actual = signatures(z, mask, ends, phases=phases, groups=groups)
    expected = scalar_signature(z, mask, ends, phases, groups)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-7)
    if len(set(groups)) < 2:
        real, null, eligible = feedback(z, mask, ends, groups=groups)
        assert real.shape == null.shape == (len(ends), 0)
        np.testing.assert_array_equal(eligible, False)


def test_full_input_and_each_causal_prefix_agree_for_all_phases_and_feedback():
    rng = np.random.default_rng(918)
    z = rng.normal(size=(86, 10)).cumsum(axis=0)
    mask = rng.random(z.shape) > 0.01
    ends = [55, 25, 41, 55, 24, 72]
    for phase in PHASES:
        full = signatures(z, mask, ends, phases=phase)
        for row, end in enumerate(ends):
            prefix = signatures(z[:end + 1], mask[:end + 1], [end], phases=phase)
            np.testing.assert_array_equal(full[1][row], prefix[1][0])
            np.testing.assert_allclose(full[0][row], prefix[0][0], rtol=0, atol=1e-7)

    full = feedback(z, mask, ends)
    for row, end in enumerate(ends):
        prefix = feedback(z[:end + 1], mask[:end + 1], [end])
        for a, b in zip(full, prefix):
            np.testing.assert_allclose(a[row], b[0], rtol=0, atol=1e-7)


def test_empty_arrays_no_edges_and_short_record_shapes():
    z = np.zeros((10, 10))
    mask = np.ones_like(z, dtype=bool)
    values, valid = signatures(z, mask, [])
    assert values.shape == valid.shape == (0, 70)
    real, null, eligible = feedback(z, mask, [])
    assert real.shape == null.shape == (0, 70) and eligible.shape == (0,)
    values, valid = signatures(z, mask, [0, 9])
    assert values.shape == valid.shape == (2, 70)
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(valid, False)

    z = np.zeros((30, 2))
    mask = np.ones_like(z, dtype=bool)
    values, valid = signatures(z, mask, [25, 29], groups=(1, 1))
    real, null, eligible = feedback(z, mask, [25, 29], groups=(1, 1))
    assert values.shape == valid.shape == real.shape == null.shape == (2, 0)
    np.testing.assert_array_equal(eligible, False)


def test_chunked_calls_preserve_duplicate_and_unsorted_end_order():
    rng = np.random.default_rng(34)
    z = rng.normal(size=(85, 4)).cumsum(axis=0)
    mask = rng.random(z.shape) > 0.015
    original_z, original_mask = z.copy(), mask.copy()
    ends = rng.integers(20, 85, size=777)
    groups = (0, 2, 1, 3)
    for call in (lambda e: signatures(z, mask, e, phases=PHASES[1], groups=groups),
                 lambda e: feedback(z, mask, e, groups=groups)):
        together = call(ends)
        parts = [call(part) for part in np.array_split(ends, 7)]
        for index, value in enumerate(together):
            np.testing.assert_allclose(value, np.concatenate([part[index] for part in parts]), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(z, original_z)
    np.testing.assert_array_equal(mask, original_mask)
