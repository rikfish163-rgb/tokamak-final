"""CPU-only contract tests for descriptive, already-arrived patch changes."""

import json
import math
import unittest

import numpy as np

from experiments.risk_aligned_online.level_change_audit_20261031.observations import (
    at_endpoint,
    patch_diagnostics,
)


class PatchDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.lower = np.full(23, -1.0)
        self.upper = np.full(23, 1.0)

    def test_constant_outside_source_has_high_level_but_zero_arrived_change(self):
        means = np.full((6, 23), 3.0)
        diag = patch_diagnostics(means, np.ones_like(means, dtype=bool),
                                 self.lower, self.upper)

        np.testing.assert_array_equal(diag["observed_count"], [23] * 6)
        np.testing.assert_array_equal(diag["outside_count"], [23] * 6)
        np.testing.assert_array_equal(diag["outside_fraction"], [1.0] * 6)
        self.assertTrue(np.isnan(diag["increment_rms"][0]))
        np.testing.assert_array_equal(diag["increment_rms"][1:], [0.0] * 5)
        np.testing.assert_array_equal(diag["feedback_pairs"], [0, 23, 46, 69, 92, 92])
        np.testing.assert_array_equal(diag["cumulative_squared_change"], [0.0] * 6)
        np.testing.assert_array_equal(diag["cumulative_feedback_pairs"],
                                      [0, 23, 69, 138, 230, 322])
        np.testing.assert_array_equal(diag["prefix_exactly_constant"],
                                      [False, True, True, True, True, True])

    def test_change_rms_uses_all_available_pairs_up_to_four_lags(self):
        means = np.full((6, 23), np.nan)
        means[:, 0] = [0, 1, 3, 3, 6, 10]
        means[:, 1] = [2, 2, 4, 5, 5, 8]
        complete = np.isfinite(means)
        diag = patch_diagnostics(means, complete, self.lower, self.upper)

        np.testing.assert_array_equal(diag["feedback_pairs"], [0, 2, 4, 6, 8, 8])
        self.assertAlmostEqual(diag["increment_rms"][1], math.sqrt(1 / 2))
        self.assertAlmostEqual(diag["increment_rms"][2], math.sqrt(21 / 4))
        self.assertAlmostEqual(diag["increment_rms"][5], math.sqrt(265 / 8))
        self.assertGreater(diag["increment_rms"][5], 1.0)  # RMS is not a fraction.
        np.testing.assert_array_equal(diag["cumulative_squared_change"],
                                      [0, 1, 22, 54, 152, 417])
        np.testing.assert_array_equal(diag["cumulative_feedback_pairs"],
                                      [0, 2, 6, 12, 20, 28])
        np.testing.assert_array_equal(diag["prefix_exactly_constant"], [False] * 6)

    def test_masked_nonfinite_values_are_zeroed_before_subtraction_and_excluded(self):
        means = np.full((3, 23), np.nan)
        means[0, :2] = [2, 10]
        means[1, 1:3] = [13, 100]
        means[2, [0, 2]] = [5, 104]
        means[0, 2] = np.inf  # An unavailable value is not an observation.
        complete = np.zeros_like(means, dtype=bool)
        complete[0, :2] = True
        complete[1, 1:3] = True
        complete[2, [0, 2]] = True
        original = means.copy()
        original_mask = complete.copy()

        with np.errstate(invalid="raise", over="raise"):
            diag = patch_diagnostics(means, complete, self.lower, self.upper)

        np.testing.assert_array_equal(diag["feedback_pairs"], [0, 1, 2])
        self.assertEqual(diag["increment_rms"][1], 3.0)
        self.assertAlmostEqual(diag["increment_rms"][2], math.sqrt(25 / 2))
        np.testing.assert_array_equal(diag["cumulative_squared_change"], [0, 9, 34])
        np.testing.assert_array_equal(diag["cumulative_feedback_pairs"], [0, 1, 3])
        np.testing.assert_array_equal(means, original)
        np.testing.assert_array_equal(complete, original_mask)

    def test_missing_or_disjoint_feedback_never_claims_a_constant_prefix(self):
        means = np.full((3, 23), np.nan)
        means[0, 0] = 1
        means[2, 1] = 1
        complete = np.isfinite(means)
        diag = patch_diagnostics(means, complete, self.lower, self.upper)

        np.testing.assert_array_equal(diag["observed_count"], [1, 0, 1])
        np.testing.assert_array_equal(diag["outside_count"], [0, 0, 0])
        self.assertTrue(np.isnan(diag["outside_fraction"][1]))
        self.assertTrue(np.isnan(diag["increment_rms"]).all())
        np.testing.assert_array_equal(diag["feedback_pairs"], [0, 0, 0])
        np.testing.assert_array_equal(diag["prefix_exactly_constant"], [False] * 3)
        for end in (8, 16, 24):
            row = at_endpoint(diag, end)
            self.assertFalse(row["prefix_exactly_constant"])
            self.assertIsNone(row["increment_rms"])
            json.dumps(row, allow_nan=False)

    def test_boundaries_are_inclusive_and_fraction_uses_observed_channels(self):
        means = np.full((1, 23), np.nan)
        means[0, :5] = [-2, -1, 0, 1, 2]
        diag = patch_diagnostics(means, np.isfinite(means), self.lower, self.upper)
        self.assertEqual(diag["observed_count"][0], 5)
        self.assertEqual(diag["outside_count"][0], 2)
        self.assertEqual(diag["outside_fraction"][0], 0.4)

    def test_no_future_patch_can_change_any_earlier_result(self):
        rng = np.random.default_rng(912)
        means = rng.normal(size=(10, 23))
        complete = rng.random((10, 23)) > 0.3
        first = patch_diagnostics(means, complete, self.lower, self.upper)
        changed_means = means.copy()
        changed_complete = complete.copy()
        changed_means[5:] = rng.normal(100, 25, size=(5, 23))
        changed_complete[5:] = ~changed_complete[5:]
        second = patch_diagnostics(changed_means, changed_complete,
                                   self.lower, self.upper)

        for key in first:
            with self.subTest(key=key):
                np.testing.assert_equal(first[key][:5], second[key][:5])
        self.assertEqual(at_endpoint(first, 40), at_endpoint(second, 40))

    def test_randomized_results_match_independent_scalar_reference(self):
        rng = np.random.default_rng(31031)
        lower = np.linspace(-1.5, 1.5, 23)
        upper = lower + 1.0
        for patches in range(1, 11):
            for _ in range(8):
                means = rng.normal(size=(patches, 23))
                complete = rng.random((patches, 23)) > rng.uniform(0.1, 0.9)
                means[~complete] = np.nan
                diag = patch_diagnostics(means, complete, lower, upper)
                cumulative_squared, cumulative_pairs = 0.0, 0
                for j in range(patches):
                    observed = [c for c in range(23) if complete[j, c]]
                    outside = sum(means[j, c] < lower[c] or means[j, c] > upper[c]
                                  for c in observed)
                    differences = [means[j, c] - means[j - h, c]
                                   for h in range(1, min(4, j) + 1)
                                   for c in range(23)
                                   if complete[j, c] and complete[j - h, c]]
                    squared = math.fsum(float(d) ** 2 for d in differences)
                    cumulative_squared += squared
                    cumulative_pairs += len(differences)
                    self.assertEqual(diag["observed_count"][j], len(observed))
                    self.assertEqual(diag["outside_count"][j], outside)
                    self.assertEqual(diag["feedback_pairs"][j], len(differences))
                    self.assertEqual(diag["cumulative_feedback_pairs"][j], cumulative_pairs)
                    self.assertAlmostEqual(float(diag["cumulative_squared_change"][j]),
                                           cumulative_squared, places=11)
                    self.assertEqual(bool(diag["prefix_exactly_constant"][j]),
                                     cumulative_pairs > 0 and cumulative_squared == 0)
                    row = at_endpoint(diag, 8 * (j + 1))
                    if observed:
                        self.assertAlmostEqual(row["outside_fraction"], outside / len(observed))
                        self.assertTrue(0 <= row["outside_fraction"] <= 1)
                    else:
                        self.assertIsNone(row["outside_fraction"])
                    if differences:
                        self.assertAlmostEqual(row["increment_rms"],
                                               math.sqrt(squared / len(differences)))
                    else:
                        self.assertIsNone(row["increment_rms"])
                    json.dumps(row, allow_nan=False)

    def test_constant_prefix_flag_does_not_recover_after_observed_change(self):
        means = np.vstack([np.zeros(23), np.zeros(23), np.ones(23), np.ones(23)])
        diag = patch_diagnostics(means, np.ones_like(means, dtype=bool),
                                 self.lower, self.upper)
        np.testing.assert_array_equal(diag["prefix_exactly_constant"],
                                      [False, True, False, False])

    def test_missing_current_feedback_retains_earlier_constant_evidence(self):
        means = np.zeros((3, 23))
        complete = np.ones((3, 23), dtype=bool)
        complete[2] = False
        diag = patch_diagnostics(means, complete, self.lower, self.upper)
        row = at_endpoint(diag, 24)
        self.assertEqual(row["feedback_pairs"], 0)
        self.assertIsNone(row["increment_rms"])
        self.assertEqual(row["cumulative_feedback_pairs"], 23)
        self.assertTrue(row["prefix_exactly_constant"])

    def test_rms_does_not_round_to_zero_when_squared_sum_is_subnormal(self):
        with np.errstate(under="ignore"):
            min_positive = np.nextafter(np.longdouble(0), np.longdouble(1))
            means = np.zeros((2, 23), dtype=np.longdouble)
            means[1, 0] = np.sqrt(min_positive)
            diag = patch_diagnostics(means, np.ones_like(means, dtype=bool),
                                     self.lower, self.upper)
            expected = np.sqrt(diag["cumulative_squared_change"][1]) / np.sqrt(
                np.longdouble(23)
            )
        self.assertGreater(diag["cumulative_squared_change"][1], 0)
        self.assertGreater(diag["increment_rms"][1], 0)
        self.assertEqual(diag["increment_rms"][1], expected)
        self.assertFalse(diag["prefix_exactly_constant"][1])

    def test_unrepresentable_squared_change_raises_instead_of_false_constancy(self):
        means = np.zeros((2, 23), dtype=np.longdouble)
        complete = np.ones((2, 23), dtype=bool)
        means[1, 0] = np.finfo(np.longdouble).tiny
        with self.assertRaisesRegex(ValueError, "too small"):
            patch_diagnostics(means, complete, self.lower, self.upper)
        means[0, 0] = -np.finfo(np.longdouble).max
        means[1, 0] = np.finfo(np.longdouble).max
        with self.assertRaisesRegex(ValueError, "numeric range"):
            patch_diagnostics(means, complete, self.lower, self.upper)

    def test_integer_inputs_preserve_a_unit_change_beyond_float64_precision(self):
        if np.finfo(np.longdouble).nmant < 63:
            return  # Extended precision is not available on every platform.
        means = np.zeros((2, 23), dtype=np.int64)
        means[0, 0] = 2**53
        means[1, 0] = 2**53 + 1
        complete = np.zeros_like(means, dtype=bool)
        complete[:, 0] = True
        diag = patch_diagnostics(means, complete, self.lower, self.upper)
        self.assertEqual(diag["increment_rms"][1], 1)
        self.assertEqual(diag["cumulative_squared_change"][1], 1)
        self.assertFalse(diag["prefix_exactly_constant"][1])

    def test_empty_patch_sequence(self):
        diag = patch_diagnostics(np.empty((0, 23)), np.empty((0, 23), dtype=bool),
                                 self.lower, self.upper)
        for value in diag.values():
            self.assertEqual(value.shape, (0,))
        self.assertFalse(at_endpoint(diag, 7)["patch_available"])
        with self.assertRaises(ValueError):
            at_endpoint(diag, 8)

    def test_bad_shapes_dtypes_and_nonfinite_observations_or_bounds(self):
        means = np.zeros((2, 23))
        complete = np.ones((2, 23), dtype=bool)
        invalid_cases = [
            (np.zeros((2, 22)), complete, self.lower, self.upper, ValueError),
            (np.zeros(23), complete, self.lower, self.upper, ValueError),
            (means, np.ones((1, 23), dtype=bool), self.lower, self.upper, ValueError),
            (means, complete.astype(int), self.lower, self.upper, TypeError),
            (means.astype(bool), complete, self.lower, self.upper, TypeError),
            (means.astype(str), complete, self.lower, self.upper, TypeError),
            (means.astype(complex), complete, self.lower, self.upper, TypeError),
            (means, complete, np.zeros(22), self.upper, ValueError),
            (means, complete, self.lower, np.zeros((23, 1)), ValueError),
            (means, complete, self.lower.astype(str), self.upper, TypeError),
            (means, complete, self.lower, self.upper.astype(complex), TypeError),
            (means, complete, np.full(23, np.nan), self.upper, ValueError),
            (means, complete, self.lower, np.full(23, np.inf), ValueError),
            (means, complete, np.ones(23), np.zeros(23), ValueError),
        ]
        for args in invalid_cases:
            with self.subTest(shapes=[np.shape(arg) for arg in args[:4]], error=args[4]):
                with self.assertRaises(args[4]):
                    patch_diagnostics(*args[:4])
        for invalid_value in (np.nan, np.inf, -np.inf):
            bad = means.copy()
            bad[1, 2] = invalid_value
            with self.subTest(available_value=invalid_value), self.assertRaises(ValueError):
                patch_diagnostics(bad, complete, self.lower, self.upper)

    def test_endpoint_mapping_json_and_explicit_missing_state(self):
        means = np.vstack([np.zeros(23), np.full(23, 2.0)])
        diag = patch_diagnostics(means, np.ones_like(means, dtype=bool),
                                 self.lower, self.upper)
        for end in (0, 7):
            row = at_endpoint(diag, end)
            self.assertFalse(row["patch_available"])
            self.assertIsNone(row["patch_index"])
            self.assertIsNone(row["outside_fraction"])
            self.assertIsNone(row["increment_rms"])
            json.dumps(row, allow_nan=False)
        for end, expected in ((8, 0), (15, 0), (16, 1), (23, 1)):
            row = at_endpoint(diag, end)
            self.assertTrue(row["patch_available"])
            self.assertEqual(row["patch_index"], expected)
            self.assertEqual(row["outside_fraction"], float(expected))
            self.assertEqual(row["feedback_pairs"], 23 * expected)
            self.assertEqual(row["increment_rms"], 2.0 if expected else None)
            json.dumps(row, allow_nan=False)
        self.assertEqual(at_endpoint(diag, np.int64(16))["patch_index"], 1)
        for end in (-9, -1, 24, 100):
            with self.subTest(end=end), self.assertRaises(ValueError):
                at_endpoint(diag, end)
        for end in (True, 8.0, "8"):
            with self.subTest(end=end), self.assertRaises(TypeError):
                at_endpoint(diag, end)

    def test_endpoint_does_not_hide_unexpected_nonfinite_diagnostics(self):
        means = np.zeros((2, 23))
        original = patch_diagnostics(means, np.ones_like(means, dtype=bool),
                                     self.lower, self.upper)
        for key, index, invalid_value in (
            ("outside_fraction", 0, np.nan),
            ("outside_fraction", 0, np.inf),
            ("increment_rms", 1, np.nan),
            ("increment_rms", 1, np.inf),
            ("increment_rms", 0, np.inf),
            ("cumulative_squared_change", 1, np.inf),
            ("outside_fraction", 0, -0.1),
            ("outside_fraction", 0, 1.1),
        ):
            diag = {name: values.copy() for name, values in original.items()}
            diag[key][index] = invalid_value
            with self.subTest(key=key, invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    at_endpoint(diag, (index + 1) * 8)

    def test_endpoint_rejects_nonzero_change_that_float_json_would_round_to_zero(self):
        tiny = np.longdouble("1e-200")
        if tiny * tiny == 0 or float(tiny * tiny) != 0:
            return  # This platform has no extra precision for this example.
        means = np.zeros((2, 23), dtype=np.longdouble)
        complete = np.zeros((2, 23), dtype=bool)
        complete[:, 0] = True
        means[1, 0] = tiny
        diag = patch_diagnostics(means, complete, self.lower, self.upper)
        self.assertGreater(diag["cumulative_squared_change"][1], 0)
        self.assertFalse(diag["prefix_exactly_constant"][1])
        with self.assertRaises(ValueError):
            at_endpoint(diag, 16)


if __name__ == "__main__":
    unittest.main()
