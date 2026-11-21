import torch

from paired_anchor_neutral_20261106 import neutralize


def test_finite_anchor_neutrality_is_stronger_than_linear_projection():
    u = torch.tensor([[0.2, 0.3], [-0.2, -0.3]], dtype=torch.float64)
    g = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=u.dtype)
    z0 = torch.tensor([0.1, 0.2], dtype=u.dtype)

    def anchor(ix, state):
        return z0[ix] + state[:, 0] + 0.7 * state[:, 1].square()

    action, valid, residual, angle = neutralize(u, g, z0, anchor)
    assert valid.all() and (angle < 0).all()
    torch.testing.assert_close(action.norm(dim=1), u.norm(dim=1))
    torch.testing.assert_close(anchor(torch.arange(2), action), z0, atol=1e-4, rtol=0)
    assert (residual.abs() <= 1e-4).all()
    assert (action[:, 1] * u[:, 1] > 0).all()


def test_no_root_or_no_tangent_does_not_invent_an_update():
    u = torch.tensor([[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]])
    g = torch.tensor([[1.0, 0.0]]).repeat(3, 1)

    def anchor(_ix, state):
        return state[:, 0] + 4.0 * state.square().sum(1)

    action, valid, residual, angle = neutralize(u, g, u[:, 0] * 0, anchor)
    assert not valid.any() and not action.any()
    assert not residual.any() and not angle.any()


def test_exact_zero_anchor_delta_preserves_projected_direction():
    u = torch.tensor([[0.12, -0.16]])
    g = torch.tensor([[1.0, 0.0]])

    def anchor(_ix, state):
        return state[:, 0]

    action, valid, residual, angle = neutralize(u, g, u[:, 0] * 0, anchor)
    assert valid.all() and not angle.any() and not residual.any()
    torch.testing.assert_close(action, torch.tensor([[0.0, -0.2]]))
