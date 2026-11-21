import numpy as np

import run_patch_utility_pivotal_support_20261106 as p


def trial(native, disruptive=True, down=.060):
    native = np.asarray(native, dtype=float)
    times = np.arange(len(native)) / 1000
    return p.payoff(native, times, disruptive, down, 25., np.array([0]), .5)


def test_timely_crossing_outside_label_window_can_be_pivotal():
    reward, valid, band, prior, full = trial(np.zeros(48))
    assert reward[0].tolist() == [1] * 24
    assert valid.all() and not prior[0] and full[0]
    assert not band.any()


def test_suffix_alarm_that_already_satisfies_objective_zeroes_the_payoff():
    z = np.zeros(48)
    z[-1] = 1
    assert not trial(z)[0].any()
    assert not trial(z, False, None)[0].any()


def test_danger_band_uses_strict_deadline_and_the_observed_clock():
    reward, _, band, _, _ = trial(np.zeros(36), down=.027)
    assert np.flatnonzero(reward[0]).tolist() == list(range(8))
    assert np.array_equal(reward > 0, band)


def test_safe_first_alarm_and_already_reported_prefix_are_distinct():
    z = np.zeros(36)
    z[8] = 1
    reward, _, band, prior, _ = trial(z, False, None)
    assert prior[0] and not reward.any() and not band.any()
    z[8] = 0
    assert trial(z, False, None)[0][0].tolist() == [-1] * 24


def test_partial_block_retains_only_actual_samples():
    reward, valid, _, _, full = trial(np.zeros(10), False, None)
    assert valid.sum() == 1 and not full[0]
    assert reward[0, 0] == -1 and not reward[0, 1:].any()


def test_first_crossing_delta_not_later_or_padded_hit():
    scores = np.zeros((1, 3, 3))
    scores[0, 0] = [0., 1., 1.]
    scores[0, 1] = [1., 1., 1.]
    scores[0, 2] = [0., 0., 1.]
    hard, first = p.effect(scores, np.array([[1, 0, 0]]),
                           np.array([[True, True, False]]), .5)
    assert hard.tolist() == [[0, 1, 0]]
    assert first.tolist() == [[1, 0, -1]]
