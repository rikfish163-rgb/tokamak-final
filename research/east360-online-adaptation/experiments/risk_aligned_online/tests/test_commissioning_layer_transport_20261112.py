import torch

from experiments.risk_aligned_online.commissioning_layer_transport_20261112 import (
    add_to_patches, edge_vectors, hidden_pass, match_dose, project_edges,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_adjacent_projection_has_zero_sum_and_respects_missing_patch():
    x = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [0., 9.], [0., 2.]]])
    valid = torch.tensor([[True, True, True, False, True]])
    edge = edge_vectors(x, valid)
    v = torch.tensor([[[2., 5.], [1., 2.], [0., -1.], [7., 9.], [0., 2.]]])
    actual = project_edges(v, edge)
    torch.testing.assert_close(actual, torch.tensor([[[1., 0.], [0., 0.], [-1., 0.], [0., 0.], [0., 0.]]]))
    torch.testing.assert_close(project_edges(actual, edge), actual)


def test_dose_matches_usable_rows_and_does_not_invent_direction():
    x = torch.tensor([[[1., 0.], [-1., 0.]], [[0., 0.], [0., 0.]]])
    y = match_dose(x, torch.tensor([.7, .9]))
    torch.testing.assert_close(y.flatten(1).norm(dim=1), torch.tensor([.7, 0.]))
    torch.testing.assert_close(y.sum(1), torch.zeros((2, 2)))


def test_patch_only_write_cannot_modify_classifier_token_at_injection():
    x = torch.arange(24.).reshape(2, 4, 3)
    delta = torch.ones_like(x[:, :-1])
    y = add_to_patches(x, delta)
    torch.testing.assert_close(y[:, :-1], x[:, :-1]+1.)
    assert torch.equal(x[:, -1], y[:, -1])


def test_layer_hooks_measure_two_real_gradients_and_final_patches_disconnect():
    torch.manual_seed(131)
    cfg = Config(window=8, patch=2, dim=8, heads=2, layers=2, horizon=3)
    model = Simple23(cfg, torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2, 8, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    sites, score = hidden_pass(model, x, m, differentiable=True)
    a, b = torch.autograd.grad(score.sum(), sites)
    assert a[:, :-1].norm().item() > 0
    assert b[:, :-1].norm().item() > 0
    with torch.no_grad():
        base = model(x, m)['logit']
        hook = model.encoder.register_forward_hook(lambda _layer, _args, out: add_to_patches(out, 4.*out[:, :-1]))
        try:
            torch.testing.assert_close(model(x, m)['logit'], base, atol=1e-6, rtol=0.)
        finally:
            hook.remove()
