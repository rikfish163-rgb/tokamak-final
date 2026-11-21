from unittest.mock import patch
import torch

from experiments.risk_aligned_online.centered_fast_window_20261023.model import CenteredLayer
from experiments.risk_aligned_online.curvature_fast_window_20261023.model import CurvatureLayer


def test_constant_observed_coordinates_do_not_update():
    layer = CenteredLayer(8, rank=4)
    h = torch.ones(2, 6, 8); valid = torch.ones(2, 6, dtype=torch.bool)
    fast, info = layer.fast_weights(h, valid, mode='on')
    torch.testing.assert_close(fast, layer.origin.expand(2, -1, -1), atol=1e-7, rtol=0)
    assert torch.all(info['loss_after'] <= info['loss_before'] + 1e-7)


def test_centering_removes_auxiliary_level_not_measured_order():
    torch.manual_seed(7); layer = CenteredLayer(8, rank=4)
    h = torch.randn(2, 6, 8); valid = torch.ones(2, 6, dtype=torch.bool)
    q, k, v = layer.coordinates(h)
    with patch.object(layer, 'coordinates', return_value=(q, k, v)):
        expected, _ = layer.fast_weights(h, valid, mode='on')
        reversed_w, _ = layer.fast_weights(h, valid, mode='reverse')
    with patch.object(layer, 'coordinates', return_value=(q, k + 2., v - 3.)):
        actual, _ = layer.fast_weights(h, valid, mode='on')
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
    assert not torch.allclose(expected, reversed_w)


def test_gradient_equals_autograd_and_loss_descends():
    torch.manual_seed(13); layer = CenteredLayer(8, rank=4)
    h = torch.randn(2, 6, 8); valid = torch.tensor([[False, True, True, True, True, True], [False]*4+[True, True]])
    _, k, v = layer.coordinates(h); observed = valid.clone(); observed[:, -1] = False
    n = observed.sum(-1).clamp_min(1)
    kc = (k - (k * observed[..., None]).sum(1)[:, None] / n[:, None, None]) * observed[..., None]
    vc = (v - (v * observed[..., None]).sum(1)[:, None] / n[:, None, None]) * observed[..., None]
    w = layer.origin.detach().expand(2, -1, -1).clone().requires_grad_(True)
    loss = ((kc @ w - vc).square().sum((1, 2)) / (2*n)).sum()
    g, = torch.autograd.grad(loss, w)
    trace = kc.square().sum((1, 2)) / n
    expected = w - (layer.eta/(1+trace))[:, None, None]*g*(n>=2)[:, None, None]
    actual, info = layer.fast_weights(h, valid, mode='on')
    torch.testing.assert_close(actual, expected)
    assert torch.all(info['loss_after'] <= info['loss_before'] + 1e-7)
    assert info['updated'].tolist() == [True, False]


def test_off_forward_and_backward_exactly_match_prior():
    torch.manual_seed(5); old = CurvatureLayer(8, rank=4); new = CenteredLayer(8, rank=4)
    new.load_state_dict(old.state_dict()); h = torch.randn(2, 6, 8); valid = torch.ones(2, 6, dtype=torch.bool)
    for model in (old, new):
        with torch.no_grad(): model.output.weight.fill_(.1)
    a, _ = old(h, valid, mode='off'); b, _ = new(h, valid, mode='off')
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().sum().backward(); b.square().sum().backward()
    for p, q in zip(old.parameters(), new.parameters()):
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
