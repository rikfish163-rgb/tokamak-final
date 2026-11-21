import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_cross_diagnostic_20261113 as cross
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_one_group_update_does_not_consume_the_held_measurements():
    torch.manual_seed(11)
    z = torch.randn(2, 60, requires_grad=True)
    matrix = torch.randn(40, 60)
    pred = (z @ matrix.T).reshape(2, 5, 8)
    current = torch.randn_like(pred)
    earlier = torch.randn_like(pred)
    weight = torch.ones(2, 4)
    first = cross.directions(pred, z, current, earlier, weight)
    current[..., cross.MEMBERS == 1] += 10
    second = cross.directions((z @ matrix.T).reshape_as(pred), z, current, earlier, weight)
    torch.testing.assert_close(first[cross.ARMS.index('mag_to_position_now')],
                               second[cross.ARMS.index('mag_to_position_now')])
    assert not torch.allclose(first[0], second[0])
    assert not torch.allclose(first[cross.ARMS.index('position_to_mag_now')],
                              second[cross.ARMS.index('position_to_mag_now')])


def test_disjoint_issue_uses_the_original_grid_and_complete_second_target():
    q = np.array([127, 159, 191, 255, 287, 319])
    eligible = np.ones(len(q), bool)
    i, j = cross.follow_window(q, np.arange(len(q)), eligible)
    np.testing.assert_array_equal(i, [0, 2, 3])
    np.testing.assert_array_equal(j, [2, 3, 5])
    eligible[2] = False
    i, j = cross.follow_window(q, np.arange(len(q)), eligible)
    np.testing.assert_array_equal(i, [2, 3])
    np.testing.assert_array_equal(j, [3, 5])


def test_outcome_timing_is_only_for_analysis_and_stays_strict():
    clock = np.arange(300)/1000
    action = np.array([120, 124, 125, 200, 220])
    travel = np.array([.01, .01, .01, .01, .01])
    np.testing.assert_array_equal(cross.analysis_windows(action, clock, down=.250, horizon=125.,
        deadline=30., first_alarm_s=None, dose=travel), [False, False, True, True, False])
    np.testing.assert_array_equal(cross.analysis_windows(action, clock, down=None, horizon=None,
        deadline=30., first_alarm_s=.200, dose=travel), [True, True, True, True, False])


def test_masked_loss_scores_only_the_held_group_and_balances_its_channels():
    ref = torch.zeros(2, 5, 8)
    pred = ref.clone()
    pred[..., cross.MEMBERS == 2] = .1
    loss = cross.group_loss(pred[None], ref)[0]
    torch.testing.assert_close(loss[:, [0,1,3]], torch.zeros((2,3)))
    torch.testing.assert_close(loss[:, 2], torch.full((2,), .025))
    torch.testing.assert_close(cross.relative_gain(loss, loss), torch.zeros_like(loss))


def test_direction_is_a_finite_equal_dose_first_block_intervention():
    torch.manual_seed(131)
    group = cross.group
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2, 16, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    state = torch.zeros(2, 3, 4, requires_grad=True)
    def tr(v):
        return v+.2  # Positive inherited radius; same internal write site.

    h = group.forward_hidden(base, x, m, tr, state=state)
    pred = (h @ torch.randn(8, 40)).reshape(2, 5, 8)
    g = cross.directions(pred, state, torch.randn_like(pred), torch.randn_like(pred), torch.ones(2, 4))
    hidden, dose, err = group.forward_hidden(base, x, m, tr, writes=g)
    assert (dose>0).all() and float(err)<1e-5
    torch.testing.assert_close(hidden[0], h)
    assert not torch.allclose(hidden[1], hidden[2])
    assert all(v.grad is None for v in base.parameters())
