import numpy as np

from run_adjacent_patch_capacity_20261106 import adjusted_objective, gate


def test_equal_anchor_objective_subtracts_only_the_optimized_target_score():
    z = np.arange(7.0, dtype=np.float64)
    j = np.arange(14.0, dtype=np.float64).reshape(7, 2)
    score, jac = adjusted_objective(z, j, native_anchor=0.5)
    mask = np.array([0, 1, 2, 4, 5, 6])
    np.testing.assert_array_equal(score[mask], z[mask])
    np.testing.assert_array_equal(jac[mask], j[mask])
    np.testing.assert_array_equal(jac[3], j[3] - j[2])
    assert score[3] == 1.5 and z[3] == 3


def test_gate_rejects_equal_score_or_other_representation_rescues():
    hit = dict(
        motion=[1, 2, 3],
        motion_reverse=[2],
        level=[1],
        level_reverse=[],
        motion_shape_only=[1, 2, 3],
        motion_scalar=[3],
    )
    p = dict(
        windows=270,
        common_rank_support=270,
        motion_vs_level=dict(mean=0, resample_range=[-0.1, 0.1]),
        motion_vs_scalar=dict(mean=0, resample_range=[-0.1, 0.1]),
    )
    score = dict(positive=p, protect=dict(harms=[0] * 7), safe=dict(harms=[0] * 7))
    assert not gate(score, hit, [])["promote"]
    hit["motion"] += [4]
    hit["motion_shape_only"] += [4]
    assert gate(score, hit, [])["unique_motion"] == [4]
    assert not gate(score, hit, [4])["promote"]
