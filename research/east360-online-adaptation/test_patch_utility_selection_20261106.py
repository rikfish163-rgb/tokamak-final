import numpy as np

from run_patch_utility_selection_20261106 import calibrate, features


def test_calibration_rejects_new_fp_even_when_net_fp_would_fall():
    names = {
        0.0: dict(recovered=[4], lost=[], fp_added=[8], fp_removed=[6, 7]),
        0.01: dict(recovered=[4], lost=[], fp_added=[], fp_removed=[]),
        0.02: dict(recovered=[4], lost=[], fp_added=[], fp_removed=[]),
        np.inf: dict(recovered=[], lost=[], fp_added=[], fp_removed=[]),
    }
    best, details = calibrate(names.__getitem__, tuple(names))
    assert best == 0.02 and len(details) == 4


def test_waveform_features_change_candidate_alignment_without_changing_controls():
    n, k = 2, 13
    state = np.zeros((n, k, 64), np.float32)
    state[:, 1, 0] = 0.2
    state[:, 2, 0] = -0.2
    g = np.zeros((n, 2, 9, 64), np.float32)
    g[:, 0, :, 0] = 1.0
    g[:, 1, :, 0] = -1.0
    args = dict(
        risk=np.zeros((n, 10), np.float32),
        effect=np.zeros((n, k, 5), np.float32),
        ordered=np.zeros((n, 2, 27), np.float32),
        states=state,
        g=g,
    )
    true = features(args, "true")
    rev = features(args, "reverse")
    null = features(args, "risk")
    assert true.shape == rev.shape == null.shape
    np.testing.assert_allclose(true + rev, 2 * null)
    assert not np.allclose(true[:, 1], rev[:, 1])
