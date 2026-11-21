import numpy as np

from run_first_passage_target_replay_20261106 import best, first_utility


def test_safe_score_does_not_accumulate_identical_near_threshold_points():
    z = np.full((1, 20), -0.01)
    np.testing.assert_allclose(
        first_utility(z, 0.0, "safe", np.arange(20)),
        first_utility(z[:, :1], 0.0, "safe", np.array([0])),
    )


def test_recovery_rewards_earlier_within_label_window_not_out_of_band():
    z = np.array([[-1, -1, 0.3, -1], [-1, -1, -1, 0.3], [-1, 0.3, -1, -1]])
    a = first_utility(z, 0.0, "recover", np.array([2, 3]))
    assert a[0] > a[1] > a[2]
    assert a[2] == 0.0


def test_protection_distinguishes_delay_when_peak_same():
    z = np.array([[-1, 0.3, -1], [-1, -1, 0.3]])
    a = first_utility(z, 0.0, "protect", np.array([0, 1]))
    assert a[0] > a[1]
    assert best(np.array([0.0, 0.0])) is None
    assert best(np.array([0.0, 1.0, 1.0])) == 1
