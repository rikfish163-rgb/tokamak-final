import numpy as np

from experiments.risk_aligned_online.commissioning_ordinal_cue_transfer_20261111 import (
    compare, eligible_endpoints,
)


def test_progression_subtracts_the_matched_safe_change_and_reports_shot_weighting():
    rows = dict(
        intra=np.array([[0., 3.], [0., 1.], [0., 2.]]),
        lag=np.array([[0., 1.], [0., 4.], [0., 2.]]),
        shot=np.array([11, 12, 20]),
    )
    matched = np.array([0, 0, 1]), np.array([2, 2, 2]), None, None
    pred = dict(latest=rows['intra'], previous=rows['lag'])
    result = compare(rows, matched, np.array([True, True, True]), pred)
    assert result['pairs'] == 3 and result['disruptive_shots'] == 2
    assert result['safe_donors'] == 1
    assert result['views']['latest']['progression_wins'] == 2
    assert result['views']['previous']['progression_wins'] == 1
    assert result['views']['latest']['shot_mean_progression_win_rate'] == .5
    assert result['latest_vs_previous']['pair_win_flips'] == [2, 1]


def test_pair_requires_feedback_mature_at_far_endpoint_and_at_least_two_groups():
    # First origin has only one usable family, so no feedback until 159+26.
    observed = np.array([[True, False, False], [True, False, True]])
    track = dict(origins=np.array([127, 159]), observed=observed,
                 matchable=np.array([True, True, True, True, True, False]))
    ix = np.array([[184, 216], [185, 217], [186, 218]])
    np.testing.assert_array_equal(eligible_endpoints(ix, track, np.array([0, 0, 1])),
                                  [False, True, False])
