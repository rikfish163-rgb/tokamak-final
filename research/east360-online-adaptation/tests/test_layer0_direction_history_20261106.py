import numpy as np

from layer0_direction_history_20261106 import support_counts


def test_polarity_examples_count_per_shot_and_union_per_anchor():
    rows = [
        (10, np.array([1, 1, 0, 0]), np.array([0, 1, -1, 0])),
        (20, np.array([-1, 0]), np.array([0, 0])),
        (30, np.array([0, 0]), np.array([0, 0])),
    ]
    report = support_counts(rows)
    assert report["benefit"]["shots"] == [10] and report["benefit"]["anchors"] == 2
    assert report["harm"]["shots"] == [10, 20] and report["harm"]["anchors"] == 2
    assert report["different_sign_value"]["anchors"] == 3


def test_opposite_directions_can_help_and_harm_at_same_anchor():
    report = support_counts([(4, np.array([1, 0]), np.array([-1, 0]))])
    assert report["benefit_and_harm_same_anchor"]["shots"] == [4]
