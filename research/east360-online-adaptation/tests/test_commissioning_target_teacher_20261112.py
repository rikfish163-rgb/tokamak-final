import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online import commissioning_target_teacher_20261112 as target
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_bank_rejects_same_or_future_rank_and_needs_both_outcomes():
    ctx = np.ones((1, 1))
    pos = dict(rank=21, positive=True, context=ctx, feedback=ctx, desired=ctx)
    neg = dict(rank=22, positive=False, context=ctx, feedback=-ctx, desired=-ctx)
    assert target.fit_bank([pos], 22) is None
    with pytest.raises(ValueError, match='previous'):
        target.fit_bank([pos, neg], 22)
    fitted = target.fit_bank([pos, neg], 23)
    inter, raw, both, lagged = target.forecast_directions(ctx, ctx, -ctx, fitted)
    np.testing.assert_array_equal(inter, [[0.]])
    assert raw[0, 0] == 1 and both[0, 0] > 0 and lagged[0, 0] < 0


def test_target_anchors_require_observed_update_and_use_strict_timing():
    t = np.arange(250)*.001
    risk = np.arange(250)*.001
    dose = np.ones_like(t)
    dose[:100] = 0.
    endpoints = target.teaching_points(t, risk, dose, True, .200, timely=False)
    assert len(endpoints) == 4 and endpoints.min()>=100 and endpoints.max() == 169
    assert len(target.teaching_points(t, risk, dose, True, .200, timely=True))==0
    np.testing.assert_array_equal(target.teaching_points(t, risk, dose, False, None, timely=False),
                                  [137, 175, 212, 249])


def test_all_target_directions_match_the_actual_archived_budget_in_real_encoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        base = Simple23(Config(dim=16), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
        x = torch.randn(2, 128, 23)
        mask = torch.ones_like(x, dtype=torch.bool)
        width = len(group.lag.CHANNELS)
        transform = group.layer.patcher(torch.randn(2, width, width)*.1, torch.randn(2, width)*.02)
        moves = torch.randn(4, 2, 15, len(group.TRUSTED))
        requested = torch.tensor([.017, .029])
        view, actual, error = group.forward_hidden(base, x, mask, transform, writes=moves,
            handoff=(moves.flip(2), torch.tensor([.25, .75])), target_norm=requested)
        torch.testing.assert_close(actual, requested, atol=1e-6, rtol=0.)
        assert error<1e-5 and view.shape==(5,2,16)
