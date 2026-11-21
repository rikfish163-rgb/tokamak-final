import numpy as np

from run_deep_miss_history_20261106 import (
    evaluation_anchors,
    group_weights,
    source_training_anchors,
)


def test_training_schedule_keeps_frozen_label_semantics_and_causal_support():
    y = np.zeros(310, dtype=np.float32)
    y[221:247] = 1
    a = np.ones(310, dtype=bool)
    a[245:] = False
    q = source_training_anchors(y, a)
    np.testing.assert_array_equal(q, [127, 191, 221, 225, 229, 233, 237, 241, 244])
    assert (q >= 127).all() and a[q].all()


def test_three_source_strata_equalize_shots_without_counting_windows_as_shots():
    sid = np.array([1, 1, 1, 2, 4, 4, 1, 2])
    group = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    w = group_weights(sid, group)
    for k, mass in enumerate((0.4, 0.4, 0.2)):
        np.testing.assert_allclose(w[group == k].sum(), mass)
    np.testing.assert_allclose(w[:3].sum(), w[3])
    np.testing.assert_allclose(w[-2], w[-1])


def test_held_schedule_has_no_labels_or_anchor_after_observation():
    np.testing.assert_array_equal(
        evaluation_anchors(210, 32, 170), [127, 159, 170, 191]
    )
    np.testing.assert_array_equal(evaluation_anchors(210, 32, 19), [127, 159, 191])
    np.testing.assert_array_equal(evaluation_anchors(126, 8, None), [])
