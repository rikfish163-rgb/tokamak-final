import numpy as np

from experiments.risk_aligned_online import commissioning_local_dynamics_20261114 as trial


def test_block_boundaries_cannot_change_the_issued_prediction():
    q = np.array([0, 32, 64, 96])
    x = np.ones((4, 2))
    y = np.ones((4, trial.motion.K, 2))
    ready = np.array([True, True, True, True])
    a = trial.direct(q, x, y, ready)
    y[1:] = -25.
    b = trial.direct(q, x, y, ready)
    np.testing.assert_array_equal(a[:, :2], b[:, :2])
    assert not a[:, 0].any()
    assert np.abs(a[:, 2] - b[:, 2]).max() > .1


def test_cross_diagnostic_fit_can_learn_a_response_in_a_different_channel():
    q = np.arange(0, 32 * 8, 32)
    x = np.column_stack((np.linspace(.6, 1.4, len(q)), np.zeros(len(q))))
    y = np.zeros((len(q), trial.motion.K, 2))
    y[:, :, 1] = x[:, 0, None]
    diagonal, cross, recent = trial.direct(q, x, y, np.ones(len(q), dtype=bool))
    assert not diagonal.any()
    assert (cross[-1, :, 1] > .5).all() and (recent[-1, :, 1] > .5).all()


def test_projection_is_direct_on_whole_waveform_not_an_autoregressive_target():
    knots = np.arange(2 * trial.motion.K * 2, dtype=float).reshape(2, trial.motion.K, 2)
    y = trial.waveform(knots)
    np.testing.assert_allclose(trial.fit_knots(y), knots, atol=1e-11)
