"""Synthetic-only checks for the independent scalar phase-feature reference."""

import unittest

import numpy as np

from verify_intergroup_phase_features_20261106 import (
    _latest_longest_run,
    _rotate_right,
    reference,
)


GROUPS = ((3, 4, 5), (14, 15, 16, 17), (19, 20, 21, 22))
MULTIPLIERS = ((1.0, -2.0, 3.0), (-0.5, 1.0, -1.5, 2.0), (1.0, -1.0, 2.0, -2.0))
FACTORS = np.array([np.sqrt(np.mean(np.square(x))) for x in MULTIPLIERS])
PAIRS = ((0, 1), (0, 2), (1, 2))
LAGS = (4, 8, 16)
SHIFTS = (7, 11, 13)


def synthetic_activity(n_increments=63, seed=398):
    rng = np.random.default_rng(seed)
    activity = rng.uniform(0.004, 0.025, (3, n_increments))
    # B follows A by four increments without using a circular pairing.
    activity[1, 4:] = activity[0, :-4]
    return activity


def make_trajectory(activity):
    """Create raw values whose normalized signed increments are known exactly."""
    n_increments = activity.shape[1]
    center = np.linspace(-0.7, 0.9, 23, dtype=np.float64)
    scale = np.linspace(0.4, 1.7, 23, dtype=np.float64)
    normalized = np.zeros((n_increments + 1, 23), dtype=np.float64)
    for group, channels in enumerate(GROUPS):
        for channel, multiplier in zip(channels, MULTIPLIERS[group]):
            normalized[1:, channel] = np.cumsum(activity[group] * multiplier)
    values = center + scale * np.sinh(normalized)
    available = np.ones(values.shape, dtype=bool)
    return values, available, center, scale


def expected_phase(activity, start, length):
    """Independent NumPy/corrcoef oracle for supported synthetic examples."""
    real, shifted = [], []
    for (a, b), shift in zip(PAIRS, SHIFTS):
        for lag in LAGS:
            x = activity[a, start + 16 : start + length - 16]
            plus = activity[b, start + 16 + lag : start + length - 16 + lag]
            minus = activity[b, start + 16 - lag : start + length - 16 - lag]
            real.append((np.corrcoef(x, plus)[0, 1] - np.corrcoef(x, minus)[0, 1]) / 2)
            shifted.append(
                (
                    np.corrcoef(x, np.roll(plus, shift))[0, 1]
                    - np.corrcoef(x, np.roll(minus, shift))[0, 1]
                )
                / 2
            )
    return np.array(real), np.array(shifted)


class GuardedSamples:
    """Reject out-of-window accesses and, for values, unavailable channels."""

    def __init__(self, array, first, last, observed=None):
        self.array = array
        self.first = first
        self.last = last
        self.observed = observed
        self.reads = []

    def __getitem__(self, index):
        sample, channel = index
        if not self.first <= sample <= self.last:
            raise AssertionError(f"out-of-window read: {index}")
        if channel not in {c for group in GROUPS for c in group}:
            raise AssertionError(f"unselected-channel read: {index}")
        if self.observed is not None and not self.observed[index]:
            raise AssertionError(f"unavailable-value read: {index}")
        self.reads.append(index)
        return self.array[index]


class ReferenceTests(unittest.TestCase):
    def test_known_asymmetric_lag_and_all_pair_lag_positions(self):
        activity = synthetic_activity()
        values, available, center, scale = make_trajectory(activity)
        result = reference(values, available, center, scale, 63)
        expected_real, expected_shifted = expected_phase(activity * FACTORS[:, None], 0, 63)

        self.assertEqual(set(result), {"within", "real", "shifted", "supported", "common"})
        for name in ("within", "real", "shifted"):
            self.assertEqual(result[name].shape, (9,))
            self.assertEqual(result[name].dtype, np.dtype(np.float64))
        self.assertEqual(result["supported"].shape, (9,))
        self.assertEqual(result["supported"].dtype, np.dtype(bool))
        self.assertIs(type(result["common"]), int)
        self.assertEqual(result["common"], 63)
        self.assertTrue(np.all(result["supported"]))
        self.assertGreater(result["real"][0], 0.3)
        np.testing.assert_allclose(result["real"], expected_real, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(result["shifted"], expected_shifted, rtol=2e-11, atol=2e-11)

    def test_within_group_masks_normalization_and_missing_values(self):
        activity = synthetic_activity()
        values, available, center, scale = make_trajectory(activity)
        available[10, 3] = False  # removes increments 9 and 10 from group 0
        available[0, 14] = False  # removes increment 0 from group 1
        available[63, 17] = False  # removes increment 62 from group 1
        available[:, 21] = False  # no group 2 increment is actually observed
        values[~available] = np.nan

        result = reference(values, available, center, scale, 63)
        expected = []
        for group, excluded in enumerate(((9, 10), (0, 62), tuple(range(63)))):
            observed = np.delete(activity[group] * FACTORS[group], excluded)
            if observed.size:
                expected.extend((np.log1p(np.sqrt(np.mean(observed**2))), np.log1p(np.std(observed)), observed.size / 63))
            else:
                expected.extend((0.0, 0.0, 0.0))
        np.testing.assert_allclose(result["within"], expected, rtol=2e-11, atol=2e-11)
        self.assertEqual(result["common"], 0)
        np.testing.assert_array_equal(result["real"], np.zeros(9))
        np.testing.assert_array_equal(result["shifted"], np.zeros(9))
        self.assertFalse(np.any(result["supported"]))

    def test_fixed_right_rotation_preserves_both_exact_multisets(self):
        plus = [float(i * 3) for i in range(31)]
        minus = [float(i * 7 + 1) for i in range(31)]
        plus_before, minus_before = plus[:], minus[:]
        for shift in SHIFTS:
            rotated_plus = _rotate_right(plus, shift)
            rotated_minus = _rotate_right(minus, shift)
            self.assertEqual(rotated_plus, list(np.roll(plus, shift)))
            self.assertEqual(rotated_minus, list(np.roll(minus, shift)))
            self.assertEqual(sorted(rotated_plus), sorted(plus))
            self.assertEqual(sorted(rotated_minus), sorted(minus))
        self.assertEqual(plus, plus_before)
        self.assertEqual(minus, minus_before)

    def test_latest_tied_run_selected_even_though_below_phase_minimum(self):
        common_mask = [True] * 30 + [False] * 3 + [True] * 30
        self.assertEqual(_latest_longest_run(common_mask), (33, 30))
        self.assertEqual(_latest_longest_run([False] * 63), (0, 0))
        self.assertEqual(_latest_longest_run([True, True, False]), (0, 2))
        activity = synthetic_activity()
        values, available, center, scale = make_trajectory(activity)
        available[31:33, 3] = False
        result = reference(values, available, center, scale, 63)
        self.assertEqual(result["common"], 30)
        self.assertAlmostEqual(result["within"][2], 60 / 63)
        self.assertFalse(np.any(result["supported"]))
        np.testing.assert_array_equal(result["real"], np.zeros(9))
        np.testing.assert_array_equal(result["shifted"], np.zeros(9))

    def test_present_zero_activity_and_disjoint_group_runs_are_distinct(self):
        values, available, center, scale = make_trajectory(np.zeros((3, 63)))
        zero_activity = reference(values, available, center, scale, 63)
        np.testing.assert_array_equal(zero_activity["within"], [0, 0, 1] * 3)
        self.assertEqual(zero_activity["common"], 63)
        self.assertFalse(np.any(zero_activity["supported"]))
        np.testing.assert_array_equal(zero_activity["real"], np.zeros(9))
        np.testing.assert_array_equal(zero_activity["shifted"], np.zeros(9))

        values, available, center, scale = make_trajectory(synthetic_activity())
        available[32:, 3] = False
        available[:32, 14] = False
        disjoint = reference(values, available, center, scale, 63)
        np.testing.assert_allclose(disjoint["within"][[2, 5, 8]], [31 / 63, 31 / 63, 1])
        self.assertEqual(disjoint["common"], 0)
        self.assertFalse(np.any(disjoint["supported"]))
        np.testing.assert_array_equal(disjoint["real"], np.zeros(9))
        np.testing.assert_array_equal(disjoint["shifted"], np.zeros(9))

    def test_exact_49_increment_noncentral_run_and_48_threshold(self):
        activity = synthetic_activity()
        values, available, center, scale = make_trajectory(activity)
        available[:7, 3] = False
        available[57:, 3] = False  # observed samples 7..56 => increments 7..55
        at_threshold = reference(values, available, center, scale, 63)
        expected_real, expected_shifted = expected_phase(activity * FACTORS[:, None], 7, 49)
        self.assertEqual(at_threshold["common"], 49)
        self.assertTrue(np.all(at_threshold["supported"]))
        np.testing.assert_allclose(at_threshold["real"], expected_real, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(at_threshold["shifted"], expected_shifted, rtol=2e-11, atol=2e-11)

        available[7, 3] = False
        below_threshold = reference(values, available, center, scale, 63)
        self.assertEqual(below_threshold["common"], 48)
        self.assertFalse(np.any(below_threshold["supported"]))
        np.testing.assert_array_equal(below_threshold["real"], np.zeros(9))
        np.testing.assert_array_equal(below_threshold["shifted"], np.zeros(9))

    def test_each_zero_norm_makes_only_affected_correlations_unsupported(self):
        for zero_role in ("x", "plus", "minus"):
            with self.subTest(zero_role=zero_role):
                activity = synthetic_activity()
                if zero_role == "x":
                    activity[0, :] = 0.01
                else:
                    activity[1, :] = 0.01
                    if zero_role == "plus":
                        activity[1, 12:20] = np.linspace(0.004, 0.025, 8)
                    else:
                        activity[1, 43:51] = np.linspace(0.004, 0.025, 8)
                values, available, center, scale = make_trajectory(activity)
                result = reference(values, available, center, scale, 63)
                self.assertEqual(result["common"], 63)
                self.assertFalse(result["supported"][0])
                self.assertEqual(result["real"][0], 0.0)
                self.assertEqual(result["shifted"][0], 0.0)
                self.assertTrue(np.any(result["supported"]))

    def test_near_constant_norm_on_each_side_of_cutoff(self):
        for magnitude, expected in ((1e-14, False), (1e-10, True)):
            with self.subTest(magnitude=magnitude):
                activity = synthetic_activity()
                activity[0] = 0.01 + magnitude * np.where(np.arange(63) % 2, 1.0, -1.0)
                values, available, center, scale = make_trajectory(activity)
                result = reference(values, available, center, scale, 63)
                self.assertEqual(bool(result["supported"][0]), expected)
                if not expected:
                    self.assertEqual(result["real"][0], 0.0)
                    self.assertEqual(result["shifted"][0], 0.0)

    def test_future_truncation_last_64_boundary_and_input_immutability(self):
        activity = synthetic_activity(n_increments=100)
        values, available, center, scale = make_trajectory(activity)
        m = 80
        original_values = values.copy()
        original_available = available.copy()
        original_center, original_scale = center.copy(), scale.copy()
        full = reference(values, available, center, scale, m)
        truncated = reference(values[: m + 1], available[: m + 1], center, scale, m)
        expected_real, expected_shifted = expected_phase(activity * FACTORS[:, None], m - 63, 63)
        np.testing.assert_allclose(full["real"], expected_real, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(full["shifted"], expected_shifted, rtol=2e-11, atol=2e-11)

        changed_values = values.copy()
        changed_available = available.copy()
        changed_values[: m - 63, :] = np.nan
        changed_values[m + 1 :, :] = np.nan
        changed_available[: m - 63, :] = False
        changed_available[m + 1 :, :] = False
        # Unselected channels within the window also cannot affect results.
        changed_values[m - 63 : m + 1, 0] = np.nan
        changed_available[m - 63 : m + 1, 0] = False
        changed = reference(changed_values, changed_available, center, scale, m)
        for other in (truncated, changed):
            self.assertEqual(full["common"], other["common"])
            for name in ("within", "real", "shifted", "supported"):
                np.testing.assert_array_equal(full[name], other[name])
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(available, original_available)
        np.testing.assert_array_equal(center, original_center)
        np.testing.assert_array_equal(scale, original_scale)

    def test_read_guard_enforces_prefix_selected_channels_and_observation(self):
        activity = synthetic_activity(n_increments=100)
        values, available, center, scale = make_trajectory(activity)
        m = 80
        available[40, 3] = False
        available[75, 20] = False
        values[~available] = np.nan
        expected = reference(values, available, center, scale, m)
        guarded_values = GuardedSamples(values, m - 63, m, available)
        guarded_available = GuardedSamples(available, m - 63, m)
        actual = reference(guarded_values, guarded_available, center, scale, m)
        self.assertEqual(actual["common"], expected["common"])
        for name in ("within", "real", "shifted", "supported"):
            np.testing.assert_array_equal(actual[name], expected[name])
        accessed_samples = [sample for sample, _ in guarded_values.reads]
        self.assertEqual(min(accessed_samples), m - 63)
        self.assertEqual(max(accessed_samples), m)

    def test_randomized_masks_against_independent_synthetic_oracle(self):
        rng = np.random.default_rng(763)
        for trial in range(32):
            with self.subTest(trial=trial):
                activity = synthetic_activity(seed=1000 + trial)
                values, available, center, scale = make_trajectory(activity)
                if trial % 4 == 0:
                    first_missing = int(rng.integers(0, 8))
                    last_missing = int(rng.integers(0, 8))
                    available[:first_missing, 3] = False
                    if last_missing:
                        available[-last_missing:, 20] = False
                elif trial % 4 == 1:
                    for _ in range(int(rng.integers(1, 4))):
                        group = int(rng.integers(0, 3))
                        channel = int(rng.choice(GROUPS[group]))
                        available[int(rng.integers(0, 64)), channel] = False
                elif trial % 4 == 2:
                    available[:, int(rng.choice(GROUPS[trial % 3]))] = False
                else:
                    available[int(rng.integers(0, 6)) :: 10, 15] = False
                values[~available] = np.nan
                result = reference(values, available, center, scale, 63)

                expected_within = []
                group_valid = []
                for group, channels in enumerate(GROUPS):
                    selected = available[:, list(channels)]
                    valid = np.all(selected[:-1] & selected[1:], axis=1)
                    group_valid.append(valid)
                    observed = (activity[group] * FACTORS[group])[valid]
                    if observed.size:
                        expected_within.extend(
                            (np.log1p(np.sqrt(np.mean(observed**2))), np.log1p(np.std(observed)), observed.size / 63)
                        )
                    else:
                        expected_within.extend((0.0, 0.0, 0.0))
                np.testing.assert_allclose(result["within"], expected_within, rtol=2e-11, atol=2e-11)

                common_mask = np.all(group_valid, axis=0)
                edges = np.diff(np.concatenate(([False], common_mask, [False])).astype(np.int8))
                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)
                if starts.size:
                    lengths = ends - starts
                    longest = int(np.max(lengths))
                    start = int(starts[np.flatnonzero(lengths == longest)[-1]])
                else:
                    longest, start = 0, 0
                self.assertEqual(result["common"], longest)
                if longest >= 49:
                    expected_real, expected_shifted = expected_phase(activity * FACTORS[:, None], start, longest)
                    self.assertTrue(np.all(result["supported"]))
                    np.testing.assert_allclose(result["real"], expected_real, rtol=2e-11, atol=2e-11)
                    np.testing.assert_allclose(result["shifted"], expected_shifted, rtol=2e-11, atol=2e-11)
                else:
                    self.assertFalse(np.any(result["supported"]))
                    np.testing.assert_array_equal(result["real"], np.zeros(9))
                    np.testing.assert_array_equal(result["shifted"], np.zeros(9))


if __name__ == "__main__":
    unittest.main()
