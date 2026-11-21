import numpy as np
import pytest

import verify_peak_contrast_capacity_20261106 as verify


def test_independent_recount_separates_score_only_new_crossing_from_shape():
    native = np.full((1, 2, 32), verify.run.TAU - 0.5)
    z = np.broadcast_to(native[:, :, None, None], (1, 2, 3, 2, 32)).copy()
    shifts = np.zeros((1, 2, 3, 2))
    z[:, 0, :, 0] += 0.6
    shifts[:, 0, :, 0] = 0.6
    z[:, 0, :, 1, 3] += 0.6
    z[:, 1, :, 1, 31] += 0.6  # Outside matched offsets, still a new FP.
    positive = np.arange(32)[None] < 8
    protect = np.zeros_like(positive)
    counts = verify.own_counts(native, z, shifts, positive, protect, np.array([False]))
    assert counts["raw_reachable_by_seed"] == [1, 1, 1]
    assert counts["shape_reachable_by_seed"] == [1, 1, 1]
    assert counts["feasible_any_arm_by_seed"] == [0, 0, 0]
    assert counts["raw_new_safe_cross"] == [1, 1, 1]


def test_match_recount_includes_masks_clock_and_high_floor():
    common = dict(
        age_ms=63.0, anchor=verify.run.HIGH + 0.05, trend=0.12, valid=[True] * 11
    )
    d, s = dict(common, shot=123), dict(common, shot=456)
    verify.check_match(d, s, 0.0, high=True)
    with pytest.raises(ValueError, match="predeclared"):
        verify.check_match(d, dict(s, age_ms=130.0), 0.0, high=False)
    with pytest.raises(ValueError, match="predeclared"):
        verify.check_match(d, dict(s, anchor=verify.run.HIGH - 0.05), 0.16, high=True)
