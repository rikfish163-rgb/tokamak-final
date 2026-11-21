import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from .model import ClsModel
from .source import causal_logit, enable_decoder_only
from .stream import predict_shot
from .ridge import solve_ridge


@pytest.mark.parametrize('use_task', [False, True])
def test_replayed_source_peak_matches_stream_and_decoder_gradient(use_task):
    old, shot, _ = case(); model = ClsModel(old.base).double()
    enable_decoder_only(model)
    shot = {k: v[:33].copy() if isinstance(v, np.ndarray) else v for k, v in shot.items()}
    scores, _ = predict_shot(model, shot, use_task=use_task, offline_source=True)
    z, row = causal_logit(model, shot, 32, use_task=use_task, create_graph=True)
    assert float(z.detach().sigmoid()) == pytest.approx(float(scores[32]), abs=2e-7)
    grad, = torch.autograd.grad(z, model.decoder.weight)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    j = int(grad.abs().argmax()); index = np.unravel_index(j, grad.shape)
    original = float(model.decoder.weight[index].detach()); values = []
    for sign in (1, -1):
        with torch.no_grad(): model.decoder.weight[index] = original + sign * 1e-5
        values.append(float(causal_logit(model, shot, 32, use_task=use_task)[0].detach()))
    assert float(grad[index]) == pytest.approx((values[0] - values[1]) / 2e-5, rel=.004, abs=2e-7)
    assert row['differentiated_updates'] == 3
    assert not any(p.requires_grad for p in model.base.parameters())


def test_masked_ridge_matches_explicit_mean_objective():
    torch.manual_seed(33)
    x = torch.cat((torch.randn(30, 3, dtype=torch.float64), torch.ones(30, 1, dtype=torch.float64)), dim=1)
    y = torch.randn(30, 2, dtype=torch.float64)
    m = torch.ones_like(y, dtype=torch.bool); m[:10, 1] = False
    gram = torch.einsum('bo,bi,bj->oij', m.double(), x, x)
    rhs = torch.einsum('bo,bi->oi', y.masked_fill(~m, 0.), x)
    coef, count = solve_ridge(gram, rhs, m.sum(0), penalty=.001)
    for k in range(2):
        use = m[:, k]; xx = x[use]; yy = y[use, k]
        diagonal = torch.diag(torch.tensor([.001, .001, .001, 0.], dtype=x.dtype))
        ref = torch.linalg.solve(xx.T @ xx / use.sum() + diagonal, xx.T @ yy / use.sum())
        torch.testing.assert_close(coef[k], ref)
    assert count == []


def test_all_missing_ridge_output_is_declared_untrained():
    coef, empty = solve_ridge(torch.zeros(1, 3, 3).double(), torch.zeros(1, 3).double(),
                              torch.zeros(1, dtype=torch.long), penalty=.001)
    assert not coef.any() and empty == [0]
