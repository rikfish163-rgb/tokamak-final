import numpy as np

from audit_layer0_arrival_motion_fit_20261106 import errors, promotion


def test_shape_scores_mask_both_ends_of_each_increment():
    p = np.zeros((1, 2, 25, 23))
    y = np.zeros((1, 25, 23))
    m = np.zeros_like(y, dtype=bool)
    m[0, [0, 5, 6], 0] = True
    y[0, 5, 0] = 1.0
    p[0, 1, 5, 0] = 1.0
    p[:, :, 1, 0] = 100
    values, supported = errors(p, y, m)
    assert supported.tolist() == [True]
    np.testing.assert_allclose(values[0, 0], [1 / 6, 0.5, 1 / 3])
    np.testing.assert_array_equal(values[0, 1], [0, 0, 0])


def test_context_must_beat_local_on_shape_and_combined():
    scores = {
        k: dict(level=0.1, shape=0.1, combined=0.1) for k in ("hold", "linear", "local")
    }
    scores["joint"] = dict(level=0.01, shape=0.11, combined=0.06)
    assert not promotion(scores)
    scores["joint"]["shape"] = 0.09
    assert promotion(scores)
