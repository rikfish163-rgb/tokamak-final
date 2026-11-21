import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_innovation_east_20261111 as fit


def test_direct_forecast_fit_never_uses_unobserved_coordinates():
    rng = np.random.default_rng(131)
    h = rng.normal(size=(64, 4))
    d = rng.normal(size=(5, 4))
    b = rng.normal(size=5)
    y = h @ d.T + b
    have = rng.random(y.shape) > .25
    missing = np.where(have, y, np.nan)
    changed = np.where(have, y, -1e9)
    a, c = fit.fit_decoder(h, missing, have, ridge=1e-8)
    ae, ce = fit.fit_decoder(h, changed, have, ridge=1e-8)
    np.testing.assert_allclose(a, d, atol=1e-6)
    np.testing.assert_allclose(c, b, atol=1e-6)
    np.testing.assert_array_equal(a, ae)
    np.testing.assert_array_equal(c, ce)


def test_metric_starts_at_identity_and_remains_positive_and_bounded():
    a = torch.tensor([4., -5., 3., 2.])
    b = torch.tensor([0., 14., 7., 10.])
    torch.testing.assert_close(fit.bounded_metric(a, a), torch.eye(4), atol=0, rtol=0)
    p = fit.bounded_metric(a, b)
    torch.testing.assert_close(p, p.T, atol=0, rtol=0)
    eig = torch.linalg.eigvalsh(p)
    assert eig.min() >= .1 - 1e-6
    assert eig.max() <= 1.9 + 1e-6
    for v in (a, b):
        assert v @ p @ v > 0


def test_risk_alignment_uses_difference_of_differences_and_not_a_common_shift():
    positive = torch.tensor([.3, .2])
    safe = torch.tensor([.1, -.4])
    before = fit.separation_loss(positive, safe)
    torch.testing.assert_close(before, fit.separation_loss(positive+5., safe+5.))
    assert fit.separation_loss(positive+.1, safe) < before
    assert fit.separation_loss(positive, safe+.1) > before
