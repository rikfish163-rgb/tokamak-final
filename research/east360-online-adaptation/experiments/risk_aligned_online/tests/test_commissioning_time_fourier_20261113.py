import numpy as np

from experiments.risk_aligned_online import commissioning_time_fourier_20261113 as probe
from experiments.risk_aligned_online import io23


def test_trailing_features_are_prefix_invariant_and_detect_known_bands():
    n = 210
    t = np.arange(n)
    x = np.full((n, 23), 17.)
    x[:, probe.COLS[0]] = 4. + np.sin(2 * np.pi * 2 * t / 128)
    x[:, probe.COLS[1]] = -8. + np.sin(2 * np.pi * 24 * t / 128)
    x[:, probe.COLS[2]] = t / 16
    mask = np.ones_like(x, dtype=bool)
    mask[128, probe.COLS[3]] = False
    signal = dict(values=x, available=mask)
    ends = np.array([126, 127, 128, 150, 179, 209])
    scale = np.ones(23)
    features, observed = probe.features(signal, scale, ends)
    sub, avail = probe.features({k: v[:151] for k, v in signal.items()}, scale, ends[:4])
    np.testing.assert_array_equal(features[:, :4], sub)
    np.testing.assert_array_equal(observed[:4], avail)
    assert not observed[0].any()
    assert observed[1].all()
    assert not observed[2, 3] and observed[2, 4]
    assert not features[:, 2, 3].any()
    assert features[1, 1, 0].argmax() == 0
    assert features[1, 1, 1].argmax() == 2
    assert features[0, 1, 2, 0] > 0
    np.testing.assert_allclose(features[:, 1, 4:], 0., atol=1e-10)
    assert tuple(io23.RAW23[c] for c in probe.COLS) == probe.CHANNELS


def test_shot_balance_and_equal_positive_negative_mass_within_disrupted_shot():
    labels = [np.array([False, True, True]), np.ones(21, bool), np.zeros(2, bool),
              np.zeros(17, bool), np.zeros(9, bool)]
    weights = probe.shot_weights(labels, [True, True, False, False, False])
    np.testing.assert_allclose([row.sum() for row in weights], [.25, .25, 1/6, 1/6, 1/6])
    assert weights[0][0] == weights[0][1:].sum() == .125


def test_missing_inputs_do_not_shift_original_scores():
    x = np.zeros((2, 3, len(probe.CHANNELS), 3))
    known = np.zeros(x.shape[1:3], bool)
    known[1:, 0] = True
    x[0, 1:, 0, 0] = [0., 2.]
    fit = dict(center=np.ones((2, len(probe.CHANNELS), 3)),
               scale=np.ones((2, len(probe.CHANNELS), 3)),
               coef=np.ones((2, len(probe.CHANNELS), 3)))
    original = np.array([.5, .5, .5], np.float32)
    score, parts = probe.apply_extra(original, x, known, fit)
    np.testing.assert_array_equal(score[:, 0], .5)
    assert score[0, 1] < score[0, 2] < .5
    assert not parts[:, :, 1:].any()
