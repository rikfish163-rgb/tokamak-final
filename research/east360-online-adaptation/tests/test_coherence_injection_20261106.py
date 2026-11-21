"""CPU-only contracts for synthetic injection and paired logit scoring."""

import math

import numpy as np
import pytest

import coherence_injection_20261106 as subject


GROUPS = {
    "mir": (3, 4, 5),
    "sxr": (14, 15, 16, 17),
    "xuv": (19, 20, 21, 22),
}
ARMS = ("static", "single", "coherent", "stale", "wrong_family", "scalar", "oracle")


def test_fixed_protocol_constants():
    assert subject.GROUPS == tuple(GROUPS.items())
    assert subject.CASES == ("clean", "mir", "sxr", "xuv", "common", "staggered")
    assert (subject.BLOCK, subject.ONSET, subject.OFF, subject.N, subject.AMP) == (
        32, 256, 512, 640, 0.60
    )
    assert subject.ARMS == ARMS


@pytest.mark.parametrize("case", ["clean", "mir", "sxr", "xuv", "common", "staggered"])
@pytest.mark.parametrize("shot, sign", [(42, 1), (43, -1)])
def test_injection_exact_values_times_sign_and_norm(case, shot, sign):
    mask = np.ones((640, 23), dtype=bool)
    mask_before = mask.copy()
    result = subject.injection(mask, shot, case)

    expected = np.zeros((640, 23), dtype=np.float32)
    for index, (name, columns) in enumerate(GROUPS.items()):
        if case in (name, "common", "staggered"):
            start = 256 + (32 * index if case == "staggered" else 0)
            expected[start:512, columns] = sign * 0.60
    assert isinstance(result, np.ndarray)
    assert result.shape == (640, 23)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mask, mask_before)
    assert np.linalg.norm(result.astype(np.float64)) == pytest.approx(
        float(np.float32(0.60)) * math.sqrt(np.count_nonzero(expected))
    )


@pytest.mark.parametrize("case", ["mir", "sxr", "xuv", "common", "staggered"])
def test_any_missing_point_in_any_designated_region_rejects_whole_injection(case):
    selected = [case] if case in GROUPS else list(GROUPS)
    for name in selected:
        group_index = list(GROUPS).index(name)
        start = 256 + (group_index * 32 if case == "staggered" else 0)
        for time in (start, start + 1, 511):
            for channel in GROUPS[name]:
                mask = np.ones((640, 23), dtype=np.float32)
                mask[time, channel] = 0
                assert subject.injection(mask, 2, case) is None


def test_missing_values_outside_designated_regions_are_never_filled():
    mask = np.zeros((640, 23), dtype=bool)
    for index, columns in enumerate(GROUPS.values()):
        mask[256 + 32 * index:512, columns] = True
    result = subject.injection(mask, np.int64(12), "staggered")
    assert result is not None
    assert not np.any(result[~mask])
    assert np.count_nonzero(result) == sum(
        (512 - (256 + 32 * i)) * len(columns)
        for i, columns in enumerate(GROUPS.values())
    )
    assert subject.injection(mask, 12, "common") is None
    np.testing.assert_array_equal(subject.injection(np.zeros_like(mask), 12, "clean"), 0)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.ones((639, 23)),
        np.ones((640, 22)),
        np.ones((23,)),
        np.full((640, 23), 0.5),
        np.full((640, 23), np.nan),
        np.full((640, 23), np.inf),
        np.full((640, 23), "1"),
        np.full((640, 23), 1 + 0j),
    ],
)
def test_injection_rejects_invalid_mask_schema(bad_mask):
    with pytest.raises((TypeError, ValueError)):
        subject.injection(bad_mask, 2, "mir")


@pytest.mark.parametrize("shot", [1.0, "1", True, None])
def test_injection_rejects_invalid_shot_schema(shot):
    with pytest.raises((TypeError, ValueError)):
        subject.injection(np.ones((640, 23)), shot, "mir")


@pytest.mark.parametrize("case", ["MIR", "other", "", None, 0])
def test_unknown_cases_rejected_by_both_functions(case):
    with pytest.raises((TypeError, ValueError)):
        subject.injection(np.ones((640, 23)), 0, case)
    times = np.arange(640)
    with pytest.raises((TypeError, ValueError)):
        subject.score_case(times, np.zeros(640), {arm: np.zeros(640) for arm in ARMS}, case)


def curves_for(times):
    static = np.full(len(times), 15.0)
    offsets = {
        "static": 0.0,
        "single": 3.0,
        "coherent": -3.0,
        "stale": 1.0,
        "wrong_family": -4.0,
        "scalar": 2.0,
        "oracle": -5.0,
    }
    curves = {arm: static + offset for arm, offset in offsets.items()}
    curves["scalar"][times % 32 == 0] += 20.0
    return curves


@pytest.mark.parametrize("case", ["mir", "sxr", "xuv"])
def test_single_group_scoring_uses_clean_target_and_static_movement(case):
    times = np.arange(640)
    clean = np.full(640, 10.0)
    curves = curves_for(times)
    result = subject.score_case(times, clean, curves, case)

    assert set(result) == {"mature", "nonanchor", "pre", "recovery", "counts"}
    assert result["counts"] == {"mature": 192, "nonanchor": 186, "pre": 256, "recovery": 96}
    for key in ("mature", "nonanchor", "pre", "recovery"):
        assert set(result[key]) == set(ARMS)
    assert result["mature"]["static"] == pytest.approx(5)
    assert result["mature"]["single"] == pytest.approx(8)
    assert result["mature"]["coherent"] == pytest.approx(2)
    assert result["mature"]["oracle"] == pytest.approx(0)
    assert result["mature"]["scalar"] == pytest.approx(7 + 20 * 6 / 192)
    assert result["nonanchor"]["scalar"] == pytest.approx(7)
    assert result["pre"]["static"] == 0
    assert result["pre"]["single"] == pytest.approx(3)
    assert result["pre"]["scalar"] == pytest.approx(2 + 20 * 8 / 256)
    assert result["recovery"]["coherent"] == pytest.approx(3)
    assert result["recovery"]["scalar"] == pytest.approx(2 + 20 * 3 / 96)


@pytest.mark.parametrize("case", ["clean", "common", "staggered"])
def test_control_scoring_preserves_same_scenario_static_not_original_clean(case):
    times = np.arange(640)
    curves = curves_for(times)
    result = subject.score_case(times, np.full(640, -1000.0), curves, case)
    assert result["mature"]["static"] == 0
    assert result["mature"]["single"] == pytest.approx(3)
    assert result["mature"]["coherent"] == pytest.approx(3)
    assert result["mature"]["oracle"] == pytest.approx(5)
    assert result["mature"]["scalar"] == pytest.approx(2 + 20 * 6 / 192)
    assert result["nonanchor"]["scalar"] == pytest.approx(2)
    assert result["pre"]["single"] == pytest.approx(3)


def test_subselected_absolute_clock_not_array_index_controls_windows_and_anchors():
    times = np.array([0, 17, 255, 256, 319, 320, 321, 351, 384, 447, 510, 511, 512, 543, 544, 600, 639])
    curves = curves_for(times)
    result = subject.score_case(times, np.full(times.size, 10.0), curves, "mir")
    assert result["counts"] == {"mature": 7, "nonanchor": 5, "pre": 3, "recovery": 3}
    assert result["mature"]["scalar"] == pytest.approx(7 + 40 / 7)
    assert result["nonanchor"]["scalar"] == pytest.approx(7)
    assert result["pre"]["scalar"] == pytest.approx(2 + 20 / 3)
    assert result["recovery"]["scalar"] == pytest.approx(2 + 20 / 3)


def test_absent_optional_windows_are_none_with_zero_counts():
    times = np.array([320, 321, 479, 511])
    result = subject.score_case(times, np.zeros(times.size), curves_for(times), "mir")
    assert result["counts"] == {"mature": 4, "nonanchor": 3, "pre": 0, "recovery": 0}
    assert all(value is None for value in result["pre"].values())
    assert all(value is None for value in result["recovery"].values())


def test_anchor_only_subselection_does_not_fabricate_nonanchor_score():
    times = np.arange(0, 640, 32)
    result = subject.score_case(times, np.zeros(times.size), curves_for(times), "mir")
    assert result["counts"]["mature"] == 6
    assert result["counts"]["nonanchor"] == 0
    assert all(value is None for value in result["nonanchor"].values())


@pytest.mark.parametrize(
    "times",
    [
        np.array([], dtype=int),
        np.arange(640, dtype=float),
        np.array([False, True]),
        np.array([[0, 320, 511, 639]]),
        np.array([0, 320, 320, 511]),
        np.array([0, 400, 320, 511]),
        np.array([0, 400, 320, 511], dtype=np.uint64),
        np.array([-1, 320, 511]),
        np.array([0, 320, 511, 640]),
        np.array([321, 400, 511, 639]),
        np.array([0, 320, 510]),
        np.array([0, 512, 639]),
    ],
)
def test_scoring_rejects_invalid_or_insufficient_clock(times):
    length = times.size
    with pytest.raises((TypeError, ValueError)):
        subject.score_case(times, np.zeros(length), {arm: np.zeros(length) for arm in ARMS}, "mir")


@pytest.mark.parametrize("problem", ["missing", "extra", "not_dict", "short", "two_dimensional", "nan", "inf", "complex", "string", "bool"])
def test_scoring_rejects_invalid_curve_schema(problem):
    times = np.arange(640)
    curves = {arm: np.zeros(640) for arm in ARMS}
    if problem == "missing":
        del curves["oracle"]
    elif problem == "extra":
        curves["unexpected"] = np.zeros(640)
    elif problem == "not_dict":
        curves = list(curves.values())
    elif problem == "short":
        curves["coherent"] = np.zeros(639)
    elif problem == "two_dimensional":
        curves["coherent"] = np.zeros((640, 1))
    elif problem == "nan":
        curves["coherent"][0] = np.nan
    elif problem == "inf":
        curves["coherent"][-1] = np.inf
    elif problem == "complex":
        curves["coherent"] = np.zeros(640, dtype=complex)
    elif problem == "string":
        curves["coherent"] = np.full(640, "0")
    else:
        curves["coherent"] = np.zeros(640, dtype=bool)
    with pytest.raises((TypeError, ValueError)):
        subject.score_case(times, np.zeros(640), curves, "mir")


@pytest.mark.parametrize("clean", [np.zeros(639), np.zeros((640, 1)), np.full(640, np.nan), np.full(640, np.inf), np.zeros(640, dtype=complex), np.full(640, "0"), np.zeros(640, dtype=bool)])
def test_scoring_rejects_invalid_clean_curve_even_for_preservation(clean):
    times = np.arange(640)
    with pytest.raises((TypeError, ValueError)):
        subject.score_case(times, clean, curves_for(times), "common")


@pytest.mark.parametrize("masked_input", ["mask", "times", "clean", "curve"])
def test_masked_array_schema_cannot_silently_lose_missingness(masked_input):
    times = np.arange(640)
    clean = np.zeros(640)
    curves = curves_for(times)
    mask = np.ones((640, 23))
    if masked_input == "mask":
        mask = np.ma.array(mask, mask=False)
        mask.mask[256, 3] = True
        with pytest.raises((TypeError, ValueError)):
            subject.injection(mask, 0, "mir")
        return
    if masked_input == "times":
        times = np.ma.array(times, mask=False)
        times.mask[320] = True
    elif masked_input == "clean":
        clean = np.ma.array(clean, mask=False)
        clean.mask[320] = True
    else:
        curves["coherent"] = np.ma.array(curves["coherent"], mask=False)
        curves["coherent"].mask[320] = True
    with pytest.raises((TypeError, ValueError)):
        subject.score_case(times, clean, curves, "mir")


@pytest.mark.parametrize("case", ["mir", "common", "clean"])
def test_paired_scores_are_shift_invariant_and_do_not_mutate_inputs(case):
    times = np.arange(640)
    clean = np.linspace(-2.0, 2.0, 640)
    curves = curves_for(times)
    old_clean = clean.copy()
    old_curves = {arm: curve.copy() for arm, curve in curves.items()}
    result = subject.score_case(times, clean, curves, case)
    shifted = subject.score_case(
        times, clean + 12.5, {arm: curve + 12.5 for arm, curve in curves.items()}, case
    )
    assert result["counts"] == shifted["counts"]
    for window in ("mature", "nonanchor", "pre", "recovery"):
        for arm in ARMS:
            assert result[window][arm] == pytest.approx(shifted[window][arm])
    np.testing.assert_array_equal(clean, old_clean)
    for arm in ARMS:
        np.testing.assert_array_equal(curves[arm], old_curves[arm])


def test_finite_large_logits_do_not_overflow_during_mean_accumulation():
    times = np.arange(640)
    curves = {arm: np.zeros(640) for arm in ARMS}
    curves["single"][:] = 1e308
    result = subject.score_case(times, np.zeros(640), curves, "mir")
    assert result["mature"]["single"] == pytest.approx(1e308)


def test_unrepresentable_logit_distance_is_rejected_instead_of_reporting_infinity():
    times = np.arange(640)
    clean = np.full(640, -1e308)
    curves = {arm: np.full(640, 1e308) for arm in ARMS}
    with pytest.raises(ValueError, match="not representable"):
        subject.score_case(times, clean, curves, "mir")
