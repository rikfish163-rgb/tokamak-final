import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_history_span_20261114 as span
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_history_spans_end_at_32_and_preserve_the_same_recipient_support():
    x = torch.arange(260, dtype=torch.float32).reshape(-1, 1)
    m = torch.ones_like(x, dtype=torch.bool)
    ends = np.array([179, 201])
    old, donor = span.span_coordinates(ends, device=x.device)
    assert (old[:, -1] == torch.tensor(ends)-32).all()
    assert (donor[:, :, -1] == old[:, None, -1]).all()
    assert (donor[:, 0, 0] == torch.tensor(ends)-95).all()
    assert (donor[:, 1, 0] == torch.tensor(ends)-175).all()
    m[81] = False
    d = span.span_changes(x, m, ends)
    torch.testing.assert_close(d.sum(dim=2), torch.zeros(2, 2, 1), atol=2e-4, rtol=0.)
    assert (d[0, :, 81-(179-127)] == 0.).all()    # do not replace a missing recipient
    assert not torch.allclose(d[:, 0], d[:, 1])
    x[202:] = 1e7
    torch.testing.assert_close(d, span.span_changes(x, m, ends))


def test_current_and_earlier_choices_use_distinct_mature_windows_only():
    q = np.array([127, 159, 191, 223, 255, 287])
    ok = np.array([False, False, True, True, True, True])
    fits = np.array([[2, 1], [2, 1], [2, 1], [1, 2], [2, 1], [1, 2]])
    wrong = fits[:, ::-1]
    choices, shared = span.choices_from_losses(q, ok, np.ones(len(q), bool), fits, wrong)
    np.testing.assert_array_equal(shared, [False, False, False, True, True, True])
    # Dials are short weight: a smaller short loss sets 1; smaller long sets 0.
    np.testing.assert_array_equal(choices[:, 4], [1., 0., .5, 0., 1., 1.])
    np.testing.assert_array_equal(choices[:, 5], [1., 0., .5, 1., 0., 0.])
    later = q[3]+span.lag.H+1+span.WAIT
    slot, used = span.action_clock(q, shared, np.array([later-1, later, later+1]))
    np.testing.assert_array_equal(used, [False, False, True])
    assert slot[-1] == 4


def test_zero_budget_and_early_tokens_return_the_frozen_graph():
    torch.manual_seed(5)
    model = Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    ends = np.array([199, 211])
    values = torch.randn(232, 23)
    mask = torch.ones_like(values, dtype=torch.bool)
    x, m = (torch.stack([v[e-127:e+1] for e in ends]) for v in (values, mask))
    h = span.normalized_history(model, dict(values=values.numpy(), available=mask.numpy()))
    deltas = span.history_basis(model, *h, ends)
    assert tuple(deltas.shape) == (2, 2, 16, 64)
    torch.testing.assert_close(deltas[:, :, -4:], torch.zeros(2, 2, 4, 64))
    dial = torch.tensor([[1., 0.], [0., 1.]])
    def identity(z):
        return z

    state, dose, cap, error = span.forward_scale(model, x, m, identity, deltas, dial, torch.tensor([0., 0.]))
    torch.testing.assert_close(state[0], state[1])
    torch.testing.assert_close(state[0], state[2])
    assert not dose.any() and not cap.any() and error == 0.
    # A real embedding-input perturbation creates a real nonzero budget.
    def displaced(p):
        d = torch.zeros_like(p).unflatten(-1, (8, 46))
        d[..., 0] = .2
        return p+d.flatten(-2)
    state, dose, cap, error = span.forward_scale(model, x, m, displaced, deltas, dial)
    assert (dose > 0.).all() and (cap > 0.).all() and error < 1e-5
    torch.testing.assert_close(dose, cap)
    assert not torch.allclose(state[0], state[1])
    assert all(x.grad is None for x in model.parameters())
