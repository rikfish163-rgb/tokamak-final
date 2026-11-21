import math

import pytest
import torch

from experiments.risk_aligned_online.mask_repair_20261104 import geometry


def _old_step(state, gradient):
    rows = []
    for old, grad in zip(state, gradient, strict=True):
        norm = float(grad.double().norm())
        candidate = old.double().clone()
        if norm >= 1e-8:
            candidate -= .025 * grad.double() / norm
            radius = float(candidate.norm())
            if radius > .25:
                candidate *= .25 * (1 - 4 * torch.finfo(old.dtype).eps) / radius
        rows.append(candidate.to(old.dtype))
    return torch.stack(rows)


def test_context_uses_first_last_valid_tokens_and_real_frames_only():
    embedding = torch.arange(2 * 16 * 64, dtype=torch.float32).reshape(2, 16, 64)
    mask = torch.zeros(2, 128, 23, dtype=torch.bool)
    indices = [1, 4, 7, 9, 12, 15]
    for i in indices:
        mask[0, i * 8:(i + 1) * 8, 0] = True
    mask[0, 8:12, 1] = True  # CIII: 4 / 48 actual frames
    mask[0, 32:40, 2] = True  # MNM: 8 / 48 actual frames
    mask[1, 9 * 8:12 * 8, 0] = True  # only three valid tokens
    mask[1, 9 * 8:12 * 8, 1] = True
    before_e, before_a = embedding.clone(), mask.clone()
    context = geometry.context_from_embedding(embedding, mask)
    assert context.shape == (2, 130)
    torch.testing.assert_close(context[0, :64], embedding[0, indices].mean(0))
    expected_difference = embedding[0, indices[-4:]].mean(0) - embedding[0, indices[:4]].mean(0)
    torch.testing.assert_close(context[0, 64:128], expected_difference)
    torch.testing.assert_close(context[1, 64:128], torch.zeros(64), atol=0, rtol=0)
    torch.testing.assert_close(context[0, -2:], torch.tensor([4 / 48, 8 / 48]))
    torch.testing.assert_close(context[1, -2:], torch.tensor([1., 0.]))
    assert torch.equal(embedding, before_e) and torch.equal(mask, before_a)


def test_context_rejects_empty_prefix_and_invalid_inputs():
    embedding, mask = torch.zeros(1, 16, 64), torch.zeros(1, 128, 23, dtype=torch.bool)
    with pytest.raises(ValueError, match="observed|valid|padding"):
        geometry.context_from_embedding(embedding, mask)
    mask[:, -1, 0] = True
    for invalid_e, invalid_a in ((embedding[:, :-1], mask), (embedding, mask.float()),
                                  (embedding.double(), mask[:, :-1]),
                                  (torch.full_like(embedding, float("nan")), mask)):
        with pytest.raises((ValueError, TypeError)):
            geometry.context_from_embedding(invalid_e, invalid_a)
    with pytest.raises((ValueError, TypeError)):
        geometry.context_from_embedding(embedding.to("meta"), mask)
    with pytest.raises((ValueError, TypeError)):
        geometry.context_from_embedding(embedding.to("meta"), mask.to("meta"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_init_is_exact_identity_and_matches_old_update(dtype):
    torch.manual_seed(12)
    model = geometry.ContextGeometry().to(dtype=dtype)
    context = torch.randn(4, 130, dtype=dtype)
    state = torch.zeros(4, 16, dtype=dtype)
    state[1, 0], state[2, 0] = .24, -.249
    gradient = torch.randn(4, 16, dtype=dtype)
    gradient[1, 0] = -40  # force projection
    gradient[3] = 0
    gradient[3, 0] = 1e-10  # old 1e-8 no-op threshold
    state_before, grad_before, context_before = state.clone(), gradient.clone(), context.clone()
    out, matrix = model.propose(state, gradient, context)
    assert torch.equal(matrix, torch.eye(16, dtype=dtype).expand(4, -1, -1))
    torch.testing.assert_close(out, _old_step(state, gradient), atol=0, rtol=0)
    assert torch.equal(state, state_before) and torch.equal(gradient, grad_before)
    assert torch.equal(context, context_before)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float64, 1e-12)])
def test_spd_symmetric_and_exp_two_spectral_bound(dtype, tolerance):
    torch.manual_seed(7)
    model = geometry.ContextGeometry().to(dtype=dtype)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 3)
        model.net[-1].bias.normal_(0, 3)
    matrix = model(torch.randn(8, 130, dtype=dtype))
    assert torch.equal(matrix, matrix.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(matrix)
    assert (eig > 0).all()
    assert eig.min() >= math.exp(-2) - tolerance
    assert eig.max() <= math.exp(2) + tolerance


def test_matrix_is_symmetric_exponential_of_bounded_upper_triangle():
    torch.manual_seed(19)
    model = geometry.ContextGeometry().double()
    with torch.no_grad():
        model.net[-1].weight.normal_(0, .3)
        model.net[-1].bias.normal_(0, .3)
    context = torch.randn(2, 130, dtype=torch.float64)
    entries = .125 * torch.tanh(model.net(context))
    rows, cols = torch.triu_indices(16, 16)
    upper = torch.zeros(2, 16, 16, dtype=torch.float64)
    upper[:, rows, cols] = entries
    symmetric = upper + upper.transpose(-1, -2) - torch.diag_embed(upper.diagonal(dim1=-2, dim2=-1))
    torch.testing.assert_close(model(context), torch.linalg.matrix_exp(symmetric), atol=1e-14, rtol=1e-14)


def test_zero_gradient_is_exactly_noop_and_does_not_nan():
    model = geometry.ContextGeometry()
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 1)
    state = torch.randn(5, 16) * .01
    context = torch.randn(5, 130)
    out, matrix = model.propose(state, torch.zeros_like(state), context)
    assert torch.equal(out, state)
    assert torch.isfinite(out).all() and torch.isfinite(matrix).all()
    out.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_symmetric_slots_receive_gradient_at_identity():
    torch.manual_seed(13)
    model = geometry.ContextGeometry().double()
    context = torch.randn(2, 130, dtype=torch.float64, requires_grad=True)
    matrix = model(context)
    matrix.sum().backward()
    bias_gradient = model.net[-1].bias.grad
    assert bias_gradient.shape == (136,)
    assert torch.isfinite(bias_gradient).all() and (bias_gradient > 0).all()
    i, j = torch.triu_indices(16, 16)
    expected = torch.where(i == j, .25, .5).double()  # two samples; diagonal counted once
    torch.testing.assert_close(bias_gradient, expected, atol=1e-14, rtol=0)
    assert model.net[-1].weight.grad is not None and model.net[-1].weight.grad.abs().sum() > 0
    # A zero last layer necessarily makes dM/dcontext zero at initialization.
    assert torch.count_nonzero(context.grad) == 0
    model.zero_grad(set_to_none=True)
    context.grad = None
    with torch.no_grad():
        model.net[-1].weight.normal_(0, .03)
    model(context).square().sum().backward()
    assert torch.isfinite(context.grad).all() and context.grad.abs().sum() > 0


def test_propose_trains_preconditioner_from_gradient_direction():
    torch.manual_seed(9)
    model = geometry.ContextGeometry().double()
    state = torch.zeros(2, 16, dtype=torch.float64)
    gradient = torch.randn_like(state)
    target = torch.randn_like(state)
    context = torch.randn(2, 130, dtype=torch.float64)
    proposed, _ = model.propose(state, gradient, context)
    (proposed * target).sum().backward()
    assert torch.isfinite(model.net[-1].weight.grad).all()
    assert model.net[-1].weight.grad.abs().sum() > 0


def test_update_direction_is_only_preconditioned_reconstruction_gradient():
    torch.manual_seed(10)
    model = geometry.ContextGeometry().double()
    with torch.no_grad():
        model.net[-1].weight.normal_(0, .3)
    state = torch.zeros(3, 16, dtype=torch.float64)
    gradient = torch.randn_like(state)
    context = torch.randn(3, 130, dtype=torch.float64)
    proposed, matrix = model.propose(state, gradient, context)
    direction = (matrix @ gradient.unsqueeze(-1)).squeeze(-1)
    expected = -.025 * direction / direction.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(proposed, expected, atol=1e-15, rtol=1e-14)


def test_independent_batches_and_projection_radius():
    torch.manual_seed(4)
    model = geometry.ContextGeometry().double()
    with torch.no_grad():
        model.net[-1].weight.normal_(0, .1)
    state = torch.randn(4, 16, dtype=torch.float64)
    state = state / state.norm(dim=-1, keepdim=True) * .249
    gradient = torch.randn_like(state)
    context = torch.randn(4, 130, dtype=torch.float64)
    output, matrices = model.propose(state, gradient, context)
    assert (output.norm(dim=-1) <= .25).all()
    for i in range(4):
        single, matrix = model.propose(state[i:i + 1], gradient[i:i + 1], context[i:i + 1])
        torch.testing.assert_close(single[0], output[i], atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(matrix[0], matrices[i], atol=1e-14, rtol=1e-13)


def test_float32_cast_cannot_cross_radius_from_inside():
    torch.manual_seed(23)
    vectors = torch.randn(64, 16, dtype=torch.float64)
    unit = vectors[7:8] / vectors[7:8].norm(dim=-1, keepdim=True)
    state, gradient = (.225 * unit).float(), (-unit).float()
    candidate = state.double() - .025 * gradient.double() / gradient.double().norm(dim=-1, keepdim=True)
    assert candidate.norm() <= .25 and candidate.float().double().norm() > .25
    model = geometry.ContextGeometry()
    proposed, _ = model.propose(state, gradient, torch.zeros(1, 130))
    assert proposed.double().norm() <= .25


def test_strict_shape_dtype_finite_and_radius_checks():
    model = geometry.ContextGeometry()
    s, g, c = torch.zeros(2, 16), torch.ones(2, 16), torch.ones(2, 130)
    invalid = [
        (s[:, :-1], g, c), (s, g[:1], c), (s, g, c[:, :-1]),
        (s.double(), g, c), (s, g.to("meta"), c),
        (s.to("meta"), g.to("meta"), c.to("meta")),
        (torch.full_like(s, float("nan")), g, c),
        (s, torch.full_like(g, float("inf")), c),
        (s, g, torch.full_like(c, float("nan"))),
        (torch.ones_like(s), g, c),
        (torch.empty(0, 16), torch.empty(0, 16), torch.empty(0, 130)),
    ]
    for args in invalid:
        with pytest.raises((TypeError, ValueError, FloatingPointError)):
            model.propose(*args)
    with torch.no_grad():
        model.net[1].weight[0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        model.propose(s, g, c)
