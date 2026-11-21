"""Independent splice and exhaustive-choice tests for the action utilities."""

from itertools import product

import numpy as np
import pytest

from layer0_action_utility_20261106 import alarm_envelope, block_credit


def _first_hit(times, curve, disruptive, down, threshold):
    for i, (time, prob) in enumerate(zip(times, curve)):
        if disruptive and (down - time) * 1000 < -1e-6:
            continue
        if prob >= threshold:
            return i
    return len(times)


def _utilities(times, curve, disruptive, down, threshold):
    first = _first_hit(times, curve, disruptive, down, threshold)
    if not disruptive:
        return (-int(first < len(times)),) * 2
    if first == len(times):
        return (0, 0)
    lead = round((down - times[first]) * 1000, 9)
    return int(lead > 30), int(30 < lead <= 3000)


def _brute_credit(times, native, adapted, disruptive, down, threshold, first, stride):
    anchors = list(range(first, len(times), stride))
    columns = [[] for _ in range(6)]
    native_u = _utilities(times, native, disruptive, down, threshold)
    adapted_u = _utilities(times, adapted, disruptive, down, threshold)
    for start in anchors:
        end = min(start + stride, len(times))
        before = np.r_[adapted[:start], native[start:]]
        after = np.r_[adapted[:end], native[end:]]
        isolated = native.copy()
        isolated[start:end] = adapted[start:end]
        leaveout = adapted.copy()
        leaveout[start:end] = native[start:end]
        comparisons = (
            (_utilities(times, after, disruptive, down, threshold),
             _utilities(times, before, disruptive, down, threshold)),
            (_utilities(times, isolated, disruptive, down, threshold), native_u),
            (adapted_u, _utilities(times, leaveout, disruptive, down, threshold)),
        )
        for j, (later, earlier) in enumerate(comparisons):
            for metric in range(2):
                columns[2 * j + metric].append(later[metric] - earlier[metric])
    names = (
        "strict_telescoping", "capped_telescoping", "strict_isolated",
        "capped_isolated", "strict_leaveout", "capped_leaveout",
    )
    answer = {name: np.asarray(col, dtype=int) for name, col in zip(names, columns)}
    answer["anchor_indices"] = np.asarray(anchors, dtype=int)
    answer["native_first"] = _first_hit(times, native, disruptive, down, threshold)
    answer["adapted_first"] = _first_hit(times, adapted, disruptive, down, threshold)
    return answer


def _brute_envelope(times, probs, disruptive, down, threshold, first, stride):
    anchors = list(range(first, len(times), stride))
    outcomes = set()
    for choice in product(range(len(probs)), repeat=len(anchors)):
        curve = probs[0].copy()
        for start, arm in zip(anchors, choice):
            curve[start:min(start + stride, len(times))] = probs[
                arm, start:min(start + stride, len(times))
            ]
        outcomes.add(_first_hit(times, curve, disruptive, down, threshold))
    return np.asarray(sorted(outcomes), dtype=int)


def _assert_credit(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert isinstance(actual[key], np.ndarray), key
            assert np.issubdtype(actual[key].dtype, np.integer), key
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert isinstance(actual[key], (int, np.integer)), key
            assert actual[key] == value, key


def test_randomized_credit_matches_independent_splices():
    rng = np.random.default_rng(20261106)
    for case in range(800):
        n = int(rng.integers(0, 25))
        times = np.cumsum(rng.uniform(0.001, 0.7, size=n)) - 3.0
        native = rng.choice([0.0, 0.49, 0.5, 0.7, 1.0], size=n)
        adapted = rng.choice([0.0, 0.49, 0.5, 0.7, 1.0], size=n)
        first = int(rng.integers(0, n + 4))
        stride = int(rng.integers(1, 8))
        adapted[:first] = native[:first]
        disruptive = bool(case % 2)
        down = float(rng.uniform(-3.0, 5.0)) if disruptive else None
        kwargs = dict(disruptive=disruptive, down=down, threshold=0.5,
                      first=first, stride=stride)
        actual = block_credit(times, native, adapted, **kwargs)
        expected = _brute_credit(times, native, adapted, **kwargs)
        _assert_credit(actual, expected)
        native_u = _utilities(times, native, disruptive, down, 0.5)
        adapted_u = _utilities(times, adapted, disruptive, down, 0.5)
        assert actual["strict_telescoping"].sum() == adapted_u[0] - native_u[0]
        assert actual["capped_telescoping"].sum() == adapted_u[1] - native_u[1]


def test_credit_exhaustive_binary_curves_and_partial_blocks():
    times = np.array([-4.0, -2.0, -0.03, 0.001])
    for bits in product((0.0, 1.0), repeat=8):
        native = np.asarray(bits[:4])
        adapted = np.asarray(bits[4:]).copy()
        for first in (0, 1):
            adapted[:first] = native[:first]
            for disruptive, down in ((False, None), (True, 0.0)):
                for stride in (1, 3):
                    kwargs = dict(disruptive=disruptive, down=down, threshold=0.5,
                                  first=first, stride=stride)
                    _assert_credit(block_credit(times, native, adapted, **kwargs),
                                   _brute_credit(times, native, adapted, **kwargs))


def test_randomized_envelope_matches_exhaustive_arm_choices():
    rng = np.random.default_rng(20261107)
    for case in range(300):
        n = int(rng.integers(0, 11))
        times = np.cumsum(rng.uniform(0.001, 0.5, size=n)) - 1.5
        arms = int(rng.integers(1, 5))
        probs = rng.choice([0.0, 0.49, 0.5, 1.0], size=(arms, n))
        first = int(rng.integers(0, n + 3))
        stride = int(rng.integers(2, 6))
        probs[:, :first] = probs[0, :first]
        disruptive = bool(case % 2)
        down = float(rng.uniform(-1.5, 3.0)) if disruptive else None
        kwargs = dict(disruptive=disruptive, down=down, threshold=0.5,
                      first=first, stride=stride)
        actual = alarm_envelope(times, probs, **kwargs)
        assert isinstance(actual, np.ndarray)
        assert np.issubdtype(actual.dtype, np.integer)
        np.testing.assert_array_equal(actual, _brute_envelope(times, probs, **kwargs))


def test_envelope_exhaustive_three_arm_binary_curves():
    times = np.array([-4.0, -2.0, 1.0])
    for bits in product((0.0, 1.0), repeat=9):
        probs = np.asarray(bits).reshape(3, 3)
        for disruptive, down in ((False, None), (True, 0.0)):
            for stride in (1, 2):
                kwargs = dict(disruptive=disruptive, down=down, threshold=0.5,
                              first=0, stride=stride)
                np.testing.assert_array_equal(
                    alarm_envelope(times, probs, **kwargs),
                    _brute_envelope(times, probs, **kwargs),
                )


@pytest.mark.parametrize(
    ("lead_ms", "strict", "capped"),
    [
        (30.0, 0, 0),
        (30.0000000004, 0, 0),
        (30.0000000006, 1, 1),
        (3000.0, 1, 1),
        (3000.0000000004, 1, 1),
        (3000.0000000006, 1, 0),
    ],
)
def test_disruptive_rounded_lead_boundaries(lead_ms, strict, capped):
    result = block_credit([-lead_ms / 1000], [0], [0.5], disruptive=True,
                          down=0.0, threshold=0.5, first=0, stride=1)
    assert result["native_first"] == 1
    assert result["adapted_first"] == 0
    for variant in ("telescoping", "isolated", "leaveout"):
        np.testing.assert_array_equal(result[f"strict_{variant}"], [strict])
        np.testing.assert_array_equal(result[f"capped_{variant}"], [capped])


def test_first_early_hit_is_not_replaced_by_later_timely_hit():
    result = block_credit([-4.0, -2.0], [0, 0], [1, 1], disruptive=True,
                          down=0.0, threshold=0.5, first=0, stride=1)
    np.testing.assert_array_equal(result["strict_telescoping"], [1, 0])
    np.testing.assert_array_equal(result["capped_telescoping"], [0, 0])
    np.testing.assert_array_equal(result["capped_isolated"], [0, 1])
    np.testing.assert_array_equal(result["capped_leaveout"], [-1, 0])


def test_repeated_safe_crossings_require_suppressing_the_last_one():
    result = block_credit([0, 1, 2], [1, 1, 1], [0, 0, 0], disruptive=False,
                          down=None, threshold=0.5, first=0, stride=1)
    for metric in ("strict", "capped"):
        np.testing.assert_array_equal(result[f"{metric}_telescoping"], [0, 0, 1])
        np.testing.assert_array_equal(result[f"{metric}_isolated"], [0, 0, 0])
        np.testing.assert_array_equal(result[f"{metric}_leaveout"], [1, 1, 1])
    assert (result["native_first"], result["adapted_first"]) == (0, 3)


@pytest.mark.parametrize("disruptive, down", [(False, None), (True, 4.0)])
def test_shared_prewrite_hit_makes_actions_irrelevant(disruptive, down):
    times = [0, 1, 2, 3]
    native = [0.5, 0, 1, 1]
    adapted = [0.5, 0, 0, 0]
    kwargs = dict(disruptive=disruptive, down=down, threshold=0.5, first=2, stride=1)
    result = block_credit(times, native, adapted, **kwargs)
    for key in result:
        if key not in ("anchor_indices", "native_first", "adapted_first"):
            np.testing.assert_array_equal(result[key], [0, 0])
    assert result["native_first"] == result["adapted_first"] == 0
    np.testing.assert_array_equal(alarm_envelope(times, [native, adapted], **kwargs), [0])


def test_eligible_endpoint_tolerance_and_post_disruption_exclusion():
    times = [-0.001, 0.0, 0.5e-9, 2e-9]
    kwargs = dict(disruptive=True, down=0.0, threshold=0.5, first=0, stride=1)
    native = [0, 0, 1, 1]
    adapted = [0, 0, 0, 1]
    result = block_credit(times, native, adapted, **kwargs)
    assert (result["native_first"], result["adapted_first"]) == (2, 4)
    np.testing.assert_array_equal(alarm_envelope(times, [native, adapted], **kwargs), [2, 4])


@pytest.mark.parametrize("n", [0, 1, 5])
def test_no_action_defaults_and_silence(n):
    times = np.arange(n, dtype=float)
    curve = np.zeros(n)
    result = block_credit(times, curve, curve, disruptive=False, down=None, threshold=0.5)
    assert result["anchor_indices"].size == 0
    assert result["native_first"] == result["adapted_first"] == n
    for key in result:
        if key.endswith(("telescoping", "isolated", "leaveout")):
            assert result[key].size == 0
    np.testing.assert_array_equal(
        alarm_envelope(times, np.stack([curve, curve]), disruptive=False,
                       down=None, threshold=0.5), [n]
    )


def test_default_anchors_include_the_final_partial_block():
    times = np.arange(200, dtype=float)
    native = np.zeros(200)
    adapted = native.copy()
    adapted[-1] = 0.5
    kwargs = dict(disruptive=False, down=None, threshold=0.5)
    result = block_credit(times, native, adapted, **kwargs)
    np.testing.assert_array_equal(result["anchor_indices"], [135, 167, 199])
    np.testing.assert_array_equal(result["strict_telescoping"], [0, 0, -1])
    assert (result["native_first"], result["adapted_first"]) == (200, 199)
    np.testing.assert_array_equal(alarm_envelope(times, [native, adapted], **kwargs),
                                  [199, 200])


def test_large_integer_block_parameters_and_readonly_inputs():
    times = np.arange(3, dtype=float)
    native = np.array([0.0, 0.0, 0.0])
    adapted = np.array([0.0, 1.0, 0.0])
    arms = np.stack([native, adapted])
    for array in (times, native, adapted, arms):
        array.setflags(write=False)
    kwargs = dict(disruptive=False, down=None, threshold=0.5,
                  first=0, stride=10**100)
    result = block_credit(times, native, adapted, **kwargs)
    np.testing.assert_array_equal(result["anchor_indices"], [0])
    np.testing.assert_array_equal(result["strict_telescoping"], [-1])
    np.testing.assert_array_equal(alarm_envelope(times, arms, **kwargs), [1, 3])
    kwargs.update(first=10**100, stride=1)
    result = block_credit(times, native, native, **kwargs)
    assert result["anchor_indices"].size == 0
    np.testing.assert_array_equal(alarm_envelope(times, [native], **kwargs), [3])


def test_large_synthetic_case_has_one_result_per_block():
    n = 100_003
    times = np.arange(n, dtype=float) / 1000
    native = np.zeros(n)
    adapted = np.zeros(n)
    native[80_000] = 1
    adapted[90_000] = 1
    kwargs = dict(disruptive=False, down=None, threshold=0.5)
    result = block_credit(times, native, adapted, **kwargs)
    assert len(result["anchor_indices"]) == len(range(135, n, 32))
    assert result["strict_telescoping"].sum() == 0
    np.testing.assert_array_equal(alarm_envelope(times, [native, adapted], **kwargs),
                                  [80_000, 90_000, n])


def test_envelope_with_different_earliest_hits_and_optional_silence():
    times = np.arange(6, dtype=float)
    probs = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    kwargs = dict(disruptive=False, down=None, threshold=0.5, first=2, stride=2)
    np.testing.assert_array_equal(alarm_envelope(times, probs, **kwargs), [2, 3, 5, 6])
    # If every arm must fire in the first block, later hits and silence are impossible.
    probs[2, 2] = 1
    np.testing.assert_array_equal(alarm_envelope(times, probs, **kwargs), [2, 3])


@pytest.mark.parametrize(
    ("times", "native", "adapted", "changes"),
    [
        ([[0, 1]], [0, 0], [0, 0], {}),
        ([0, 1], [[0, 0]], [0, 0], {}),
        ([0, 1], [0, 0], [0], {}),
        ([0, 0], [0, 0], [0, 0], {}),
        ([1, 0], [0, 0], [0, 0], {}),
        ([0, np.nan], [0, 0], [0, 0], {}),
        ([0, np.inf], [0, 0], [0, 0], {}),
        ([0, 1], [0, np.nan], [0, 0], {}),
        ([0, 1], [0, 0], [0, np.inf], {}),
        ([0, 1], [0, 0], [0, 0.5 + 1j], {}),
        ([0, 1], [-0.01, 0], [0, 0], {}),
        ([0, 1], [0, 0], [0, 1.01], {}),
        ([0, 1], [0, 0], [1, 0], {"first": 1}),
        ([0, 1], [0, 0], [0, 0], {"threshold": -0.01}),
        ([0, 1], [0, 0], [0, 0], {"threshold": 1.01}),
        ([0, 1], [0, 0], [0, 0], {"threshold": np.nan}),
        ([0, 1], [0, 0], [0, 0], {"threshold": np.inf}),
        ([0, 1], [0, 0], [0, 0], {"threshold": [0.5]}),
        ([0, 1], [0, 0], [0, 0], {"disruptive": 1}),
        ([0, 1], [0, 0], [0, 0], {"down": 1.0}),
        ([0, 1], [0, 0], [0, 0], {"disruptive": True, "down": None}),
        ([0, 1], [0, 0], [0, 0], {"disruptive": True, "down": np.nan}),
        ([0, 1], [0, 0], [0, 0], {"disruptive": True, "down": np.inf}),
        ([0, 1], [0, 0], [0, 0], {"disruptive": True, "down": [1.0]}),
        ([0, 1], [0, 0], [0, 0], {"first": -1}),
        ([0, 1], [0, 0], [0, 0], {"first": 0.0}),
        ([0, 1], [0, 0], [0, 0], {"first": True}),
        ([0, 1], [0, 0], [0, 0], {"stride": 0}),
        ([0, 1], [0, 0], [0, 0], {"stride": -1}),
        ([0, 1], [0, 0], [0, 0], {"stride": 1.0}),
        ([0, 1], [0, 0], [0, 0], {"stride": False}),
    ],
)
def test_block_credit_rejects_bad_inputs(times, native, adapted, changes):
    kwargs = dict(disruptive=False, down=None, threshold=0.5, first=0, stride=1)
    kwargs.update(changes)
    with pytest.raises((ValueError, TypeError)):
        block_credit(times, native, adapted, **kwargs)


@pytest.mark.parametrize(
    ("times", "probs", "changes"),
    [
        ([0, 1], [0, 0], {}),
        ([0, 1], [], {}),
        ([0, 1], np.empty((0, 2)), {}),
        ([0, 1], [[0], [0]], {}),
        ([0, 1], [[0, 0], [0, 1, 1]], {}),
        ([0, 1], [[0, 0], [1, 0]], {"first": 1}),
        ([0, 1], [[0, 0], [0, np.nan]], {}),
        ([0, 1], [[0, 0], [0, -0.1]], {}),
        ([0, 0], [[0, 0]], {}),
        ([0, 1], [[0, 0]], {"threshold": np.nan}),
        ([0, 1], [[0, 0]], {"disruptive": True, "down": None}),
        ([0, 1], [[0, 0]], {"down": 2.0}),
        ([0, 1], [[0, 0]], {"first": -1}),
        ([0, 1], [[0, 0]], {"stride": 0}),
    ],
)
def test_envelope_rejects_bad_inputs(times, probs, changes):
    kwargs = dict(disruptive=False, down=None, threshold=0.5, first=0, stride=1)
    kwargs.update(changes)
    with pytest.raises((ValueError, TypeError)):
        alarm_envelope(times, probs, **kwargs)


def test_extreme_finite_times_must_not_silently_overflow_lead():
    with pytest.raises(ValueError):
        block_credit([-1e308], [0], [1], disruptive=True, down=1e308,
                     threshold=0.5, first=0, stride=1)
    with pytest.raises(ValueError):
        alarm_envelope([-1e308], [[0], [1]], disruptive=True, down=1e308,
                       threshold=0.5, first=0, stride=1)
