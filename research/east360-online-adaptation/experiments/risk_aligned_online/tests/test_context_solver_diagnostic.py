import torch

from experiments.risk_aligned_online.trajectory_keys_20261028.solver_diagnostic import (
    ball_least_squares, causal_local_context,
)


def test_ball_solution_unconstrained_and_boundary_optimum():
    a = torch.eye(8, dtype=torch.float64)[None].repeat(2, 1, 1)
    y = torch.stack((torch.ones(8) * .1, torch.ones(8) * 2)).double()
    w = torch.ones_like(y)
    c = ball_least_squares(a, y, w)
    torch.testing.assert_close(c[0], y[0], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(c[1], y[1] / y[1].norm(), atol=1e-9, rtol=1e-9)
    assert (c.norm(dim=-1) <= 1 + 1e-10).all()


def test_missing_targets_do_not_create_information_or_nan():
    a = torch.randn(3, 12, 8, dtype=torch.float64)
    y = torch.randn(3, 12, dtype=torch.float64); w = torch.ones_like(y)
    y[0] = float('nan'); a[0] = float('nan'); w[0] = 0
    c = ball_least_squares(a, y, w)
    assert not c[0].any() and torch.isfinite(c).all()
    assert (((a[1:] @ c[1:, :, None]).squeeze(-1) - y[1:]).square().sum(1)
            <= y[1:].square().sum(1) + 1e-9).all()


def test_causal_local_solve_prefix_invariance_and_initial_zero():
    torch.manual_seed(1)
    x = torch.randn(18, 23, dtype=torch.float64)
    valid = torch.rand_like(x) > .1
    basis = torch.randn(18, 4, 23, 8, dtype=torch.float64)
    weights = torch.ones(4, 23, dtype=torch.float64)
    full = causal_local_context(x, valid, basis, weights)
    prefix = causal_local_context(x[:9], valid[:9], basis[:9], weights)
    torch.testing.assert_close(full[:9], prefix, rtol=1e-9, atol=1e-9)
    assert not full[0].any()
    assert (full.norm(dim=-1) <= 1 + 1e-9).all()
