import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_spectral_support_20261112 import (
    active_slots,
    spectral_counterfactual,
)


def test_half_shift_keeps_paired_values_and_gaps():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    support = torch.ones_like(x, dtype=torch.bool)
    support[0, 2, 0] = False
    support[0, 6, 1] = False
    moved, joint, eligible = spectral_counterfactual(x, support)
    observed = [0, 1, 3, 4, 5, 7]
    torch.testing.assert_close(moved[0, observed], x[0, [4, 5, 7, 0, 1, 3]])
    torch.testing.assert_close(moved[~joint], x[~joint])
    assert joint[0].nonzero().flatten().tolist() == observed
    assert eligible.tolist() == [True]


def test_no_contrast_when_insufficient_joint_support_or_constant():
    x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]], [[1., 2.]] * 3])
    support = torch.tensor([[[True, False], [True, True], [False, True]],
                            [[True, True]] * 3])
    moved, _, eligible = spectral_counterfactual(x, support)
    torch.testing.assert_close(moved, x)
    assert eligible.tolist() == [False, False]


def test_cannot_act_before_feedback_and_off_block_expires_previous_action():
    ends = np.array([127, 152, 153, 184, 185, 216, 217])
    points, origins = active_slots(np.array([127, 159, 191]), np.array([True, False, True]), ends)
    np.testing.assert_array_equal(points, [2, 3, 6])
    np.testing.assert_array_equal(origins, [0, 0, 2])
