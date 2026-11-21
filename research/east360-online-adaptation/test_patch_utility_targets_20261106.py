"""CPU-only synthetic checks, including an independent exhaustive oracle."""

import itertools
import math
import unittest

import numpy as np

import patch_utility_targets_20261106 as subject


def menu(native, clock, actions=3):
    """Make arbitrary finite padding and copy native only on real blocks."""
    native = np.asarray(native)
    curves = np.full((len(clock), actions, 32), 1234.0)
    for b, m in enumerate(clock):
        size = min(32, len(native) - m - 1)
        curves[b, :, :size] = native[m + 1:m + 1 + size]
    return curves, np.ones((len(clock), actions), dtype=bool)


def reference_score(times, values, disruptive, down, tau):
    first = None
    for i, value in enumerate(values):
        if value >= tau and (not disruptive or (down - times[i]) * 1000 >= -1e-6):
            first = i
            break
    lead = round(float((down - times[first]) * 1000), 9) if disruptive and first is not None else None
    return dict(first=first, lead_ms=lead, timely=lead is not None and lead > 10,
                false_alarm=not disruptive and first is not None)


def reference_utility(times, values, disruptive, down, horizon, tau, baseline):
    # This deliberately operates on the entire independently replayed shot.
    c = []
    running = -math.inf
    for value in values:
        running = max(running, value)
        x = (running - tau) / 0.05
        c.append(1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x)))
    if not disruptive:
        return 1 - c[-1]
    if baseline["timely"]:
        k = baseline["first"]
        selected = c[max(0, k - 8):k + 1]
        return sum(selected) / len(selected)
    indices = [i for i, t in enumerate(times)
               if 10 < round(float((down - t) * 1000), 9) <= horizon + 1e-6]
    if not indices:
        return 0.0
    return sum(c[i] for i in indices) / len(indices) - (c[indices[0] - 1] if indices[0] else 0.0)


def reference_replay(native, clock, curves, choices):
    result = np.array(native, dtype=float, copy=True)
    for b, m in enumerate(clock):
        size = min(32, len(native) - m - 1)
        result[m + 1:m + 1 + size] = curves[b, choices[b], :size]
    return result


class ReplayTests(unittest.TestCase):
    def test_replay_only_changes_post_release_and_ignores_padding(self):
        native = np.arange(42, dtype=float) / 100
        clock = np.array([34, 2])  # Input order need not be chronological.
        curves, allowed = menu(native, clock)
        curves[0, 1, :7] = 2
        curves[1, 2, :] = 3
        snapshots = [x.copy() for x in (native, clock, curves, allowed)]
        actual = subject.replay(native, clock, curves, allowed, [1, 2])
        expected = native.copy()
        expected[3:35] = 3
        expected[35:42] = 2
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[2], native[2])  # release index is not edited
        for x, before in zip((native, clock, curves, allowed), snapshots):
            np.testing.assert_array_equal(x, before)

    def test_identity_and_empty_menu(self):
        native = np.array([0.1, 0.2, 0.3])
        curves, allowed = menu(native, [1, 2])
        np.testing.assert_array_equal(subject.replay(native, [1, 2], curves, allowed, [0, 0]), native)
        np.testing.assert_array_equal(
            subject.replay(native, [], np.empty((0, 2, 32)), np.empty((0, 2), dtype=bool), []), native)

    def test_rejects_overlap_unavailable_nonidentity_and_bad_shapes(self):
        native = np.zeros(80)
        curves, allowed = menu(native, [2, 33])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 33], curves, allowed, [0, 0])
        curves, allowed = menu(native, [2, 34])
        allowed[0, 1] = False
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [1, 0])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [-1, 0])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [3, 0])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [0.0, 0.0])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed.astype(int), [0, 0])
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves[:, :, :31], allowed, [0, 0])
        allowed[1, 0] = False
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [0, 0])
        allowed[1, 0] = True
        curves[1, 0, 0] = 1
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [0, 0])
        curves[1, 0, 0] = 0
        curves[0, 2, -1] = np.nan
        with self.assertRaises(ValueError):
            subject.replay(native, [2, 34], curves, allowed, [0, 0])

    def test_invalid_clock(self):
        native = np.zeros(4)
        curves, allowed = menu(native, [1])
        for clock in ([-1], [4], [1.0], [True]):
            with self.subTest(clock=clock), self.assertRaises(ValueError):
                subject.replay(native, clock, curves, allowed, [0])

    def test_mixed_numeric_identity_is_exact_and_prefix_is_not_rounded(self):
        native = np.array([0, 2**53 + 1, 0], dtype=np.int64)
        curves = np.zeros((1, 2, 32), dtype=np.float64)
        allowed = np.ones((1, 2), dtype=bool)
        curves[0, 0, 0] = float(2**53)  # Not the native value, even if NumPy coerces it.
        with self.assertRaises(ValueError):
            subject.replay(native, [0], curves, allowed, [0])
        native = np.array([2**53 + 1, 0, 0], dtype=np.int64)
        curves[0, 0, :2] = 0
        curves[0, 1, 0] = .5
        result = subject.replay(native, [0], curves, allowed, [1])
        self.assertEqual(int(result[0]), 2**53 + 1)
        self.assertEqual(result[1], .5)


class ScoreTests(unittest.TestCase):
    def test_strict_rounded_timing_and_preterminal_tolerance(self):
        for lead, timely in [(10.0, False), (10.0000000004, False),
                             (10.0000000006, True), (10.01, True)]:
            with self.subTest(lead=lead):
                result = subject.score([-lead / 1000], [0.5], True, 0.0, 0.5)
                self.assertEqual(result["first"], 0)
                self.assertEqual(result["timely"], timely)
                self.assertEqual(result["lead_ms"], round(lead, 9))
        inside = subject.score([1e-9 - 1e-16], [0.5], True, 0.0, 0.5)
        outside = subject.score([1e-9 + 1e-16], [0.5], True, 0.0, 0.5)
        self.assertEqual((inside["first"], inside["lead_ms"]), (0, -0.000001))
        self.assertIsNone(outside["first"])
        self.assertTrue(subject.score([1e-9 + 1e-16], [0.5], False, None, 0.5)["false_alarm"])

    def test_first_hit_and_safe_down_is_ignored(self):
        times = np.array([0.0, 0.01, 0.02])
        curve = np.array([0.1, 0.5, 0.9])
        self.assertEqual(subject.score(times, curve, True, 0.021, 0.5),
                         dict(first=1, lead_ms=11.0, timely=True, false_alarm=False))
        self.assertEqual(subject.score(times, curve, False, float("nan"), 0.5),
                         dict(first=1, lead_ms=None, timely=False, false_alarm=True))
        self.assertEqual(subject.score(times, [0, 0, 0], False, None, 0.5),
                         dict(first=None, lead_ms=None, timely=False, false_alarm=False))

    def test_bad_inputs_are_rejected(self):
        for args in [([], [], True, 0, .5), ([0, 0], [0, 0], True, 0, .5),
                     ([0], [float('nan')], True, 0, .5), ([0], [0], 1, 0, .5),
                     ([0], [0], True, None, .5), ([0], [0], False, None, float('inf'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                subject.score(*args)

    def test_large_integer_scores_do_not_round_up_to_threshold(self):
        result = subject.score([0], np.array([2**54 - 1], dtype=np.int64),
                               False, None, float(2**54))
        self.assertIsNone(result["first"])


class OutcomeTests(unittest.TestCase):
    def test_first_passage_outside_soft_horizon_still_counts(self):
        times = np.arange(210) / 1000
        native = np.zeros(len(times))
        native[195] = 0.8
        curves, allowed = menu(native, [10])
        curves[0, 1, 9] = 0.8  # first at 20, lead 180 ms, beyond horizon
        result = subject.outcomes(native, times, [10], curves, allowed, tau=0.5,
                                  disruptive=True, down=0.2, horizon=30)
        self.assertFalse(result["baseline"]["timely"])
        self.assertEqual(result["hard"][0, 1], 1)
        self.assertTrue(result["oracle_timely"])
        self.assertFalse(result["oracle_no_false_alarm"])
        self.assertIn(20, result["reachable_first"])

    def test_single_edit_targets_do_not_predict_joint_selection(self):
        native = np.zeros(70)
        native[[10, 50]] = 0.8
        curves, allowed = menu(native, [0, 32], 2)
        curves[0, 1, 9] = 0
        curves[1, 1, 17] = 0
        result = subject.outcomes(native, np.arange(70) / 1000, [0, 32], curves, allowed,
                                  tau=0.5, disruptive=False, down=None, horizon=25)
        np.testing.assert_array_equal(result["hard"], np.zeros((2, 2)))
        self.assertEqual(result["reachable_first"], [10, 50, 70])
        self.assertTrue(result["oracle_no_false_alarm"])
        self.assertFalse(result["oracle_timely"])
        self.assertFalse(subject.score(np.arange(70) / 1000,
                         subject.replay(native, [0, 32], curves, allowed, [1, 1]),
                         False, None, 0.5)["false_alarm"])

    def test_prefix_and_gap_hits_are_obligatory(self):
        native = np.zeros(90)
        native[36] = .8
        curves, allowed = menu(native, [0, 40], 2)
        curves[0, 1, 9] = .8
        curves[1, 1, 9] = .8
        kwargs = dict(tau=.5, disruptive=False, down=None, horizon=25)
        result = subject.outcomes(native, np.arange(90) / 1000, [0, 40], curves, allowed, **kwargs)
        self.assertEqual(result["reachable_first"], [10, 36])
        self.assertFalse(result["oracle_no_false_alarm"])
        native[0] = .9
        result = subject.outcomes(native, np.arange(90) / 1000, [0, 40], curves, allowed, **kwargs)
        self.assertEqual(result["reachable_first"], [0])
        self.assertFalse(result["oracle_no_false_alarm"])

    def test_unavailable_neutral_and_empty_canonical(self):
        native = np.zeros(40)
        curves, allowed = menu(native, [0])
        curves[0, 1, 1] = .8
        allowed[0, 1] = False
        result = subject.outcomes(native, np.arange(40) / 1000, [0], curves, allowed,
                                  tau=.5, disruptive=True, down=.05, horizon=10)
        np.testing.assert_array_equal(result["hard"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["soft"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["target"], np.zeros((1, 3)))
        self.assertEqual(result["reachable_first"], [40])

    def test_input_checks_and_extreme_finite_values(self):
        native = np.array([-1e308, -1e308, 1e308])
        curves, allowed = menu(native, [0], 2)
        curves[0, 1, :2] = [-1e308, -1e308]
        result = subject.outcomes(native, [0, .001, .002], [0], curves, allowed,
                                  tau=0, disruptive=False, down=None, horizon=25)
        self.assertTrue(np.isfinite(result["soft"]).all())
        self.assertEqual(result["hard"][0, 1], 1)
        with self.assertRaises(ValueError):
            subject.outcomes(native, [0, .001], [0], curves, allowed,
                             tau=0, disruptive=False, down=None, horizon=25)
        with self.assertRaises(ValueError):
            subject.outcomes(native, [0, .001, .002], [0], curves, allowed,
                             tau=0, disruptive=False, down=None, horizon=float('nan'))

    def test_large_finite_leads_keep_canonical_coordinates(self):
        times = np.array([-1e305, -5e304, 0.0])
        native = np.array([0.0, 0.0, .8])
        curves, allowed = menu(native, [0], 2)
        curves[0, 1, 0] = .5
        with np.errstate(all="raise"):
            result = subject.outcomes(native, times, [0], curves, allowed,
                                      tau=.5, disruptive=True, down=0, horizon=1e308)
        self.assertGreater(result["soft"][0, 1], 0)
        self.assertTrue(result["oracle_timely"])

    def test_canonical_upper_cutoff_includes_one_microsecond_ms_tolerance(self):
        times = np.array([-0.025000001001, -0.025000001, -0.011, 0.0])
        native = np.zeros(4)
        curves, allowed = menu(native, [0], 2)
        curves[0, 1, 0] = .5
        result = subject.outcomes(native, times, [0], curves, allowed,
                                  tau=.5, disruptive=True, down=0, horizon=25)
        expected = .5 - 1 / (1 + math.exp(10))
        self.assertAlmostEqual(result["soft"][0, 1], expected, places=12)

    def test_arbitrary_finite_extended_precision_padding_and_actions(self):
        if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
            self.skipTest("platform long double has no wider finite range")
        huge = np.longdouble("1e400")
        native = np.zeros(3, dtype=np.longdouble)
        curves = np.full((1, 2, 32), huge, dtype=np.longdouble)
        curves[0, 0, :2] = native[1:]
        curves[0, 1, :2] = [huge, -huge]
        allowed = np.ones((1, 2), dtype=bool)
        edited = subject.replay(native, [0], curves, allowed, [1])
        self.assertEqual(edited[1], huge)
        result = subject.outcomes(native, [0, .001, .002], [0], curves, allowed,
                                  tau=.5, disruptive=False, down=None, horizon=25)
        self.assertEqual(result["hard"][0, 1], -1)
        self.assertEqual(result["reachable_first"], [1, 3])

    def test_empty_menu_and_postterminal_hits(self):
        native = np.array([0., .8])
        curves = np.empty((0, 2, 32))
        allowed = np.empty((0, 2), dtype=bool)
        safe = subject.outcomes(native, [0, .01], [], curves, allowed,
                                tau=.5, disruptive=False, down=None, horizon=25)
        self.assertEqual(safe["reachable_first"], [1])
        self.assertFalse(safe["oracle_no_false_alarm"])
        disruptive = subject.outcomes(native, [0, .01], [], curves, allowed,
                                      tau=.5, disruptive=True, down=0, horizon=25)
        self.assertEqual(disruptive["reachable_first"], [2])
        self.assertFalse(disruptive["oracle_timely"])

    def test_randomized_single_edits_and_exact_oracle_against_enumeration(self):
        rng = np.random.default_rng(20261106)
        for case in range(140):
            n = int(rng.choice([18, 42, 75, 110]))
            clock = np.array([m for m in [0, 35, 70] if m < n], dtype=int)
            rng.shuffle(clock)
            times = np.arange(n) / 1000
            native = rng.choice([.1, .3, .49, .5, .7], n, p=[.35, .3, .25, .04, .06])
            curves, allowed = menu(native, clock)
            for b, m in enumerate(clock):
                size = min(32, n - m - 1)
                curves[b, 1:, :size] = rng.choice([.1, .3, .49, .5, .7], (2, size),
                                                   p=[.35, .3, .25, .04, .06])
            allowed[:, 1:] = rng.random((len(clock), 2)) > .25
            disruptive = bool(rng.integers(2))
            down = float(rng.choice([.012, .035, .065, .100])) if disruptive else None
            horizon = float(rng.choice([5, 10, 20, 40, 100]))
            tau = .5
            kwargs = dict(tau=tau, disruptive=disruptive, down=down, horizon=horizon)
            result = subject.outcomes(native, times, clock, curves, allowed, **kwargs)
            baseline = reference_score(times, native, disruptive, down, tau)
            self.assertEqual(result["baseline"], baseline, f"case {case}")
            base_u = reference_utility(times, native, disruptive, down, horizon, tau, baseline)
            expected_hard = np.zeros_like(allowed, dtype=float)
            expected_soft = np.zeros_like(allowed, dtype=float)
            for b, a in np.argwhere(allowed):
                choices = np.zeros(len(clock), dtype=int)
                choices[b] = a
                edited = reference_replay(native, clock, curves, choices)
                after = reference_score(times, edited, disruptive, down, tau)
                expected_hard[b, a] = ((int(after["timely"]) - int(baseline["timely"]))
                                       if disruptive else
                                       (int(baseline["false_alarm"]) - int(after["false_alarm"])))
                expected_soft[b, a] = (reference_utility(times, edited, disruptive, down, horizon,
                                                           tau, baseline) - base_u)
            np.testing.assert_array_equal(result["hard"], expected_hard, err_msg=f"case {case}")
            np.testing.assert_allclose(result["soft"], expected_soft, rtol=0, atol=3e-15,
                                       err_msg=f"case {case}")
            np.testing.assert_allclose(result["target"], .75 * expected_hard + .25 * expected_soft,
                                       rtol=0, atol=3e-15, err_msg=f"case {case}")
            all_outcomes = [reference_score(times, reference_replay(native, clock, curves, picks),
                                            disruptive, down, tau)
                            for picks in itertools.product(*(np.flatnonzero(row) for row in allowed))]
            firsts = sorted({n if row["first"] is None else row["first"] for row in all_outcomes})
            self.assertEqual(result["reachable_first"], firsts, f"case {case}")
            self.assertEqual(result["oracle_timely"], any(row["timely"] for row in all_outcomes))
            self.assertEqual(result["oracle_no_false_alarm"],
                             not disruptive and any(not row["false_alarm"] for row in all_outcomes))


if __name__ == "__main__":
    unittest.main()
