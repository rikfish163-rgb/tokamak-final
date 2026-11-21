import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_split_learning_20261111 import (
    feature_view, direct_targets, early_phase, paired_loss,
)


def test_direct_targets_keep_first_eight_when_record_ends_before_full_horizon():
    z = np.broadcast_to(np.arange(137, dtype=np.float32)[:, None], (137, 10)).copy()
    mask = np.ones_like(z, dtype=bool)
    mask[130, 1] = False
    target, seen = direct_targets(z, mask, np.array([127]))
    assert target.shape == (1, 25, 10) and seen.shape == target.shape
    assert seen[0, :8, 0].all() and not seen[0, 9:, :].any()
    assert not seen[0, 2, 1] and target[0, 2, 1] == 0.
    assert target[0, 0, 0] == 8.5


def test_zero_fast_state_is_exact_frozen_view_and_prefix_has_original_warmup():
    pre = torch.linspace(-1, 1, 72).reshape(3, 24)
    key = torch.ones(3, 4)
    state = torch.zeros(3, 20)
    mapper = torch.ones(4, 20, 24)
    torch.testing.assert_close(feature_view(pre, key, state, mapper), pre.tanh(), rtol=0, atol=0)
    np.testing.assert_array_equal(early_phase(np.array([127, 135, 136, 152, 153, 167, 168])),
                                  [False, False, True, True, False, False, True])


def test_paired_loss_rewards_latest_feedback_only_if_it_improves_separation():
    fixed = torch.zeros(1, 2)
    delayed = torch.tensor([[.1, .1]])
    helpful = torch.tensor([[.4, -.1]])
    global_up = torch.tensor([[.4, .4]])
    assert paired_loss(helpful, delayed, fixed) < paired_loss(global_up, delayed, fixed)
