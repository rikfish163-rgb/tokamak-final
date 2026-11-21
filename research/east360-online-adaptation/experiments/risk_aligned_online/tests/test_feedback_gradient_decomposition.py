import numpy as np
import torch

from experiments.risk_aligned_online.feedback_gradient_20261023.geometry import (
    losses, vector_geometry, lead_ms,
)
from experiments.risk_aligned_online.tests.test_increment_forecast import model


def test_flat_arrived_values_have_zero_innovation_gradient():
    _, m = model('increment')
    x = torch.randn(2, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    truth = x[:, -1:].expand(2, 25, 23).clone()
    prediction = torch.randn_like(truth, requires_grad=True)
    parts = losses(m, prediction, x, available, truth, torch.ones_like(truth, dtype=torch.bool))
    grad = torch.autograd.grad(parts['change'], prediction)[0]
    assert not grad.any()
    assert parts['change'].item() == 0.


def test_quadratic_region_gradient_difference_is_observed_change_not_prediction_bias():
    _, m = model('increment')
    x = torch.zeros(1, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    truth = torch.full((1, 25, 23), float(torch.sinh(torch.tensor(.2))))
    mask = torch.ones_like(truth, dtype=torch.bool)
    gradients = []
    for bias in (.1, .5):
        prediction = torch.full_like(truth, bias, requires_grad=True)
        parts = losses(m, prediction, x, available, truth, mask)
        gradients.append(torch.autograd.grad(parts['change'], prediction)[0])
    expected = torch.full_like(truth, -.2 / truth.numel())
    for g in gradients:
        torch.testing.assert_close(g, expected)


def test_unavailable_targets_and_origins_do_not_contribute():
    _, m = model('increment')
    x = torch.zeros(1, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    available[:, -1, 0] = False
    truth = torch.ones(1, 25, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, :, 1] = False
    truth[:, :, 2] = float('nan')
    prediction = torch.zeros_like(truth, requires_grad=True)
    parts = losses(m, prediction, x, available, truth, mask)
    g = torch.autograd.grad(parts['change'], prediction)[0]
    assert not g[:, :, :3].any()
    assert torch.isfinite(g).all()
    assert g[:, :, 3:].abs().sum() > 0


def test_geometry_and_zero_vector_are_explicit():
    a = (torch.tensor([3., 4.]),)
    b = (torch.tensor([-3., -4.]),)
    g = vector_geometry(a, b)
    assert g == dict(norm=5., reference_norm=5., dot=-25., cosine=-1.)
    z = vector_geometry((torch.zeros(2),), b)
    assert z['cosine'] is None and z['norm'] == 0.


def test_timely_boundary_matches_official_rounding_not_float_noise():
    from experiments.risk_aligned_online.simple_train import metrics
    times = np.array([4.970, 4.969, 4.971, 4.9700000000000002])
    leads = lead_ms(times, 5.)
    np.testing.assert_array_equal(leads, [30.,31.,29.,30.])
    for t,value in zip(times,leads):
        result=metrics([dict(shot=1,times=np.array([t]),scores=np.array([1.]),
                            down=5.,is_disrupt=True)],.5,lead=30.)
        assert result['predictions'][0]['lead_ms']==value
        assert result['tp']==int(value>30.)
