import torch

from experiments.risk_aligned_online.risk_bridge_forecaster_20261102.core import augment


def test_zero_table_preserves_forecast_and_has_a_trainable_derivative():
    forecast = torch.randn(2, 25, 23, dtype=torch.float64)
    z = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(25, 23, dtype=torch.float64, requires_grad=True)
    actual = augment(forecast, z, b, 0.1, 0.4)
    assert torch.equal(actual, forecast)
    (gradient,) = torch.autograd.grad(actual.sum(), b)
    torch.testing.assert_close(gradient, torch.full_like(b, (0.5 + 1.5 - 0.2) / 0.4))


def test_added_forecast_vjp_is_the_fixed_risk_axis_times_error_projection():
    torch.manual_seed(131)
    w = torch.randn(4, dtype=torch.float64, requires_grad=True)
    axis = torch.randn(4, dtype=torch.float64)
    matrix = torch.randn(575, 4, dtype=torch.float64)
    b = torch.randn(25, 23, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 25, 23, dtype=torch.float64)
    base = (matrix @ w).reshape(1, 25, 23)
    z = (axis @ w + 0.2).reshape(1)
    actual = augment(base, z, b, 0.2, 2.0)
    (gradient,) = torch.autograd.grad((actual * v).sum(), w, create_graph=True)
    expected = matrix.T @ v.flatten() + axis * (b * v[0]).sum() / 2
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    (meta,) = torch.autograd.grad(gradient.square().sum(), b)
    assert torch.isfinite(meta).all() and meta.norm() > 0
