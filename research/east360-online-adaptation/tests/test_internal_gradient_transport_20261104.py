import numpy as np
import torch

from internal_gradient_transport_20261104 import (
    fit_transport,
    predict_transport,
    apply_direction,
)


def test_fit_is_warm_only_and_hold_and_reverse_use_same_mapping():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 5))
    y = rng.normal(size=(9, 4))
    weights = np.full(9, 1 / 9)
    model = fit_transport(x, y, weights)
    real = predict_transport(model, x)
    held = x.copy()
    held[:, -2:] = 0
    rev = x.copy()
    rev[:, -2:] *= -1
    assert real.shape == (9, 4)
    assert np.isfinite(real).all()
    assert not np.array_equal(real, predict_transport(model, held))
    assert not np.array_equal(real, predict_transport(model, rev))


def test_actual_backward_step_is_fixed_norm_and_does_not_mutate_teacher():
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    saved = parameter.detach().clone()
    direction = np.array([3.0, 4.0])
    count = apply_direction((parameter,), direction)
    assert count == 2
    delta = parameter.detach() - saved
    torch.testing.assert_close(delta, torch.tensor([-0.03, -0.04]))
    assert np.array_equal(direction, [3.0, 4.0])
