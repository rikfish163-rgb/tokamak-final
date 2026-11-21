import numpy as np

from confirmation_recovery_headroom_20261105 import classify_miss


def test_subthreshold_change_and_largest_uplift_at_wrong_time_are_distinct():
    native = np.array([0.65, 0.2, 0.1])
    timely = np.array([True, True, False])
    small = classify_miss(native, np.array([0.68, 0.21, 0.99]), native, timely, 0.7)
    off_peak = classify_miss(native, np.array([0.65, 0.69, 0.99]), native, timely, 0.7)
    assert small["category"] == "unreached_uplift_smaller_than_gap"
    assert off_peak["category"] == "unreached_uplift_not_at_crossing"


def test_eligible_proposal_rejection_and_recovery_are_distinct():
    native, active = np.array([0.6, 0.2]), np.array([0.75, 0.75])
    timely = np.array([True, False])
    rejected = classify_miss(native, active, np.array([0.6, 0.75]), timely, 0.7)
    accepted = classify_miss(native, active, active, timely, 0.7)
    assert rejected["category"] == "reached_but_not_confirmed"
    assert accepted["category"] == "confirmed_recovery"
    assert (
        classify_miss(native, active, active, timely & False, 0.7)["category"]
        == "no_eligible_time"
    )
