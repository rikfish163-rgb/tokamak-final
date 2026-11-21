import numpy as np

from experiments.risk_aligned_online.commissioning_gap_return_20261112 import (
    examples,
    gap_landmarks,
    lag_matched_deltas,
    matured_offsets,
    return_targets,
    SPECTRUM,
)


def test_gap_landmark_needs_a_known_positive_anchor_and_contiguous_zeros():
    zero = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1], bool)
    positive = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], bool)
    start, issue = gap_landmarks(zero, positive, age=1, minimum=0)
    np.testing.assert_array_equal(start, [2, 7])
    np.testing.assert_array_equal(issue, [3, 8])


def test_complete_horizon_and_unknown_exit_are_not_the_same():
    zero = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
    positive = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], bool)
    event, known = return_targets(zero, positive, [1, 5, 9, 13], horizon=2)
    np.testing.assert_array_equal(event, [3, -1, -1, -1])
    np.testing.assert_array_equal(known, [True, False, True, False])


def test_prefix_features_and_issued_landmarks_ignore_future_replacement():
    n = 128
    x = np.tile(np.linspace(.5, 2., n, dtype=np.float32)[:, None], (1, 23))
    x[70:84, SPECTRUM] = 0.
    x[91:103, SPECTRUM] = 0.
    available = np.ones(x.shape, bool)
    a = examples(x, available, age=8)
    b = examples(x[:83], available[:83], age=8)
    assert a['event'][0] == 84 and b['event'][0] == -1
    assert a['known'][0] and not b['known'][0]
    np.testing.assert_array_equal(a['issue'][a['issue'] < 83], b['issue'])
    for key in ('history', 'recent', 'stale'):
        np.testing.assert_array_equal(a[key][a['issue'] < 83], b[key])


def test_local_update_uses_only_returns_strictly_before_issue():
    issue = np.array([4, 9, 10, 20])
    event = np.array([9, -1, 13, -1])
    truth = np.array([[1., -1.], [np.nan, np.nan], [2., -2.], [np.nan, np.nan]])
    a = matured_offsets(issue, event, truth, np.zeros_like(truth))
    np.testing.assert_allclose(a, [[0, 0], [0, 0], [.2, -.2], [.5, -.5]])


def test_matched_continuous_pair_is_same_lag_in_already_observed_positive_run():
    pos = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], bool)
    log = np.column_stack((np.arange(len(pos)), 2*np.arange(len(pos))))
    matched = lag_matched_deltas(pos, log, np.array([5, 8]), np.array([7, 11]))
    np.testing.assert_allclose(matched, [[3., 6.], [4., 8.]])
