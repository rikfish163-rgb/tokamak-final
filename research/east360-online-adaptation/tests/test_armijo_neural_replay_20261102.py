import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.armijo_feedback_20261102.neural_replay import (
    reference_state,
)


def test_reused_fixed_state_uses_causal_archived_full_not_current_meta():
    archive = {
        "full_real_after": np.stack(
            [np.full((4, 64), i + 1, np.float32) for i in range(3)]
        )
    }
    for q, expected in ((0, 0), (159, 0), (160, 1), (287, 1), (288, 2), (416, 3)):
        state = reference_state("fixed_full", q, archive, "cpu")
        assert torch.equal(state, torch.full((4, 64), float(expected)))
    assert not reference_state("cold", 416, archive, "cpu").any()


def test_missing_state_and_unknown_reused_arm_do_not_fabricate_zero():
    with pytest.raises(ValueError):
        reference_state(
            "fixed_full",
            416,
            {"full_real_after": np.zeros((2, 4, 64), np.float32)},
            "cpu",
        )
    with pytest.raises(ValueError):
        reference_state("meta_real", 160, {}, "cpu")
