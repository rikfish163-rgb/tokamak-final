import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_phase_innovation_20261111 import (
    innovation_losses, past_bias, phase_path,
)


def test_phase_retains_approach_to_same_current_and_slope():
    up = np.repeat([40., 60., 80., 80.], 32)
    down = np.repeat([120., 100., 80., 80.], 32)
    a, av = phase_path(up, np.ones(len(up), bool), np.array([127]))
    b, bv = phase_path(down, np.ones(len(down), bool), np.array([127]))
    assert av[0] and bv[0]
    assert a[0, 2] == b[0, 2] == 0.
    assert a[0, 0] > 0 > b[0, 0]
    assert a[0, -1] == 0 > b[0, -1]


def test_phase_matches_truncation_and_ignores_unobserved_values():
    ip = np.r_[np.linspace(200., 80., 180), np.full(60, 500.)]
    mask = np.ones(len(ip), bool)
    mask[1:9] = False
    ip[~mask] = 1e8
    q = np.array([127, 159, 179])
    full = phase_path(ip, mask, q)
    short = phase_path(ip[:180], mask[:180], q)
    for a, b in zip(full, short, strict=True):
        np.testing.assert_array_equal(a, b)
    mask[:32] = False
    feat, valid = phase_path(ip, mask, q)
    assert not valid[0]
    np.testing.assert_array_equal(feat[0], np.zeros(4))


def test_bias_can_change_donors_without_changing_observation_budget():
    q = np.array([127, 159, 191, 223, 255])
    current = np.zeros((len(q), 3), np.float32)
    phase = np.array([[1.], [1.], [-1.], [-1.], [1.]], np.float32)
    e = np.array([[.2], [.2], [-.2], [-.2], [99.]], np.float32)
    bias, donors = past_bias(q, current, phase, np.ones(len(q), bool), e)
    np.testing.assert_array_equal(donors[-1], [[3, 2], [1, 0]])
    np.testing.assert_allclose(bias[:, -1, 0], [-.1, .1])
    changed, _ = past_bias(q, current, phase, np.ones(len(q), bool), e+np.array([[0], [0], [0], [0], [123.]]))
    np.testing.assert_allclose(changed[:, -1], bias[:, -1])


def test_reference_never_selects_unmatured_or_unusable_prior():
    q = np.array([127, 153, 175, 179, 210])
    x = np.zeros((len(q), 3), np.float32)
    usable = np.array([True, True, True, True, False])
    bias, donors = past_bias(q, x, x, usable, np.ones((len(q), 2), np.float32))
    # At 175, 153's feedback becomes usable only at 179. Equality at 179
    # permits that block but still excludes 175's own unrevealed target.
    assert (donors[:3] == -1).all()
    np.testing.assert_array_equal(donors[3], [[1, 0], [1, 0]])
    assert (donors[4] == -1).all()
    assert np.count_nonzero(bias[:, :3]) == np.count_nonzero(bias[:, 4]) == 0


def test_feedback_backward_uses_preissued_reference_and_current_target():
    from experiments.risk_aligned_online import commissioning_lag_teacher_20261111 as teacher

    d = teacher.MEMBERS.shape[0]
    p = torch.zeros(1, d, requires_grad=True)
    observed, shifted = torch.full_like(p, .15), torch.full_like(p, -.15)
    bias = torch.stack((torch.zeros_like(p), torch.full_like(p, .10)))
    losses = innovation_losses(p, observed, shifted, bias)
    assert len(losses) == 3
    grads = [torch.autograd.grad(v.sum(), p, retain_graph=True)[0] for v in losses]
    assert (grads[0] < grads[1]).all() and (grads[1] < 0).all() and (grads[2] > 0).all()
