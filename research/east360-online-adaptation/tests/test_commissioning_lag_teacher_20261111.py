import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.commissioning_lag_mixer_20261111 import ORDERED
from experiments.risk_aligned_online.commissioning_lag_teacher_20261111 import (
    GROUPS, allocate, direction_loss, grouped_error, match_shots, weights,
)


def test_grouping_sums_to_untaught_loss_and_uniform_weights_are_identity():
    p = torch.linspace(-.7, .8, 4 * 2 * len(ORDERED[0])).reshape(4, -1).requires_grad_()
    y = p.detach().flip(1)
    original = F.smooth_l1_loss(p, y, beta=.2, reduction='none').mean(1)
    g = grouped_error(p, y)
    torch.testing.assert_close(g.sum(1), original)
    w = weights(torch.zeros(GROUPS, 4), torch.randn(4, 4))
    torch.testing.assert_close(w, torch.ones_like(w), rtol=0., atol=0.)
    actual, = torch.autograd.grad((g * w).sum(), p, retain_graph=True)
    reference, = torch.autograd.grad(original.sum(), p)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-8)


def test_weighted_update_changes_direction_and_not_norm():
    g = torch.zeros(2, GROUPS, 10, 10)
    g[:, 0, 0, 3] = 1.
    g[:, 1, 0, 4] = 1.
    prior = allocate(g, torch.ones(2, GROUPS))
    raw = torch.zeros(GROUPS, 4)
    raw[0, 0] = 3.
    context = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
    current = allocate(g, weights(raw, context))
    torch.testing.assert_close(current.norm(dim=(1, 2)), prior.norm(dim=(1, 2)))
    assert current[0, 0, 3].abs() > prior[0, 0, 3].abs()
    assert current[1, 0, 3].abs() < prior[1, 0, 3].abs()


def test_match_uses_distinct_shots_and_rejects_bad_risk_or_operating_overlap():
    positive = [{'shot': 1, 'match': [0., 0., 0., 0.]},
                {'shot': 1, 'match': [.05, 0., 0., 0.]},
                {'shot': 2, 'match': [.2, .06, 0., 0.]}]
    safe = [{'shot': 3, 'match': [0., 0., 0., 0.]},
            {'shot': 4, 'match': [.2, .05, 0., 0.]},
            {'shot': 5, 'match': [2., 0., 0., 0.]},
            {'shot': 6, 'match': [.2, 2., 0., 0.]}]
    pairs = match_shots(positive, safe)
    assert {(positive[i]['shot'], safe[j]['shot']) for i, j in pairs} == {(1, 3), (2, 4)}
    assert len(pairs) == 2
    assert match_shots(positive, safe[-2:]).shape == (0, 2)


def test_teacher_rewards_correct_direction_and_penalizes_false_crossing():
    m = torch.as_tensor(np.array([[-.05, -.05]], np.float32))
    good = torch.tensor([[.1, -.08]])
    assert direction_loss(good, m) < direction_loss(-good, m)
    assert direction_loss(good, m) < direction_loss(torch.tensor([[.1, .08]]), m)
