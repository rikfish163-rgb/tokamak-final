import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_persistence_20261111 as persistent
from experiments.risk_aligned_online import commissioning_recent_visibility_20261111 as recent


def readiness(length):
    q = np.arange(127, length, persistent.STRIDE)
    data = np.zeros((len(q), 10), np.float32)
    if len(q) > 1:
        data[1, :2] = [1.4, -1.1]
    if len(q) > 2:
        data[2, :2] = [1.4, 1.1]
    return data, np.ones_like(data, bool), q, np.zeros((len(q), 3)), np.ones(len(q), bool), length


def test_previous_direction_uses_same_release_even_if_previous_gate_was_inactive():
    data = readiness(250)
    quiet = np.zeros(10, np.float32), np.ones(10, np.float32)
    state = persistent.replay(*data, quiet)[0]['current_equal_norm']
    old = recent.previous_direction(data, quiet, state)
    begin = persistent.FIRST_EFFECT
    assert not old[:begin].any()
    assert state[begin, 1] > 0. and old[begin, 1] < 0.
    np.testing.assert_array_equal(old[begin], old[begin+31])
    prefix = readiness(begin+2)
    subset = persistent.replay(*prefix, quiet)[0]['current_equal_norm']
    np.testing.assert_array_equal(recent.previous_direction(prefix, quiet, subset), old[:begin+2])


def test_visible_write_keeps_budget_and_signed_direction_but_not_saturated_component():
    h = torch.tensor([[0., 4.], [0., 0.], [800., 799.]])
    d = torch.tensor([[.5, -.5], [0., 0.], [.01, .01]])
    a = recent.visible_direction(h, d)
    torch.testing.assert_close(a.norm(dim=1), d.norm(dim=1))
    assert a[0, 0] > .69 and -.01 < a[0, 1] < 0.
    assert a[2, 1] > a[2, 0] > 0.
    assert not a[1].any()


def test_projection_control_keeps_exact_support_and_norm():
    own = torch.tensor([[3., 4.], [0., 0.]])
    other = torch.tensor([[-2., 0.], [1., 0.]])
    torch.testing.assert_close(recent.same_radius(own, other), torch.tensor([[-5., 0.], [0., 0.]]))


def test_geometry_encloses_feasible_shifts_and_preserving_sign_is_more_restrictive():
    h = torch.tensor([[2.5, -1.2, .7], [3., -3., 0.], [1., 0., 0.]], dtype=torch.float64)
    d = torch.tensor([[.9, -.4, .4], [-.7, .7, 0.], [0., 0., 0.]], dtype=torch.float64)
    beta = torch.tensor([1., -1., .4], dtype=torch.float64)
    b = recent.head_geometry(h, d, beta)
    rho = d.norm(dim=1)
    rng = torch.Generator().manual_seed(831)
    x = torch.randn((600, 3, 3), generator=rng, dtype=h.dtype)
    x *= rho[None, :, None]/x.norm(dim=2, keepdim=True)
    gain = ((h+x).tanh()-h.tanh()) @ beta
    assert (gain.max(dim=0).values <= b['tight']+1e-12).all()
    assert (b['feasible'] <= b['tight']+1e-12).all()
    assert (b['tight'] <= rho*beta.norm()+1e-12).all()
    assert (b['same_sign_upper'] <= b['tight']+1e-12).all()
    assert b['same_sign_upper'][1] == 0. and b['tight'][1] > 0.
    assert all(x[2] == 0. for x in b.values())
