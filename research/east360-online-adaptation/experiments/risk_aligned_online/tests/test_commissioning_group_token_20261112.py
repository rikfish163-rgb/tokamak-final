import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_separate_observed_group_edges_and_exclude_untrusted_spectral_channels():
    c, excluded = group.TRUSTED[0][0], group.EXCLUDED[0]
    x = torch.zeros(1, 4, 2, 46)
    x[..., 23:] = 1.
    x[:, 1:, :, c] = 1.
    x[:, 2, :, 23+c] = 0.
    x[:, 1:, :, excluded] = 99.
    w = torch.zeros(2, 2, 46)
    w[0, :, c] = 2.
    w[1, :, excluded] = 1.
    edges = group.group_edges(x.flatten(2), w.flatten(1))
    assert edges.shape == (1, 3, 4, 2)
    torch.testing.assert_close(edges[0, 0, 0], torch.tensor([1., 0.]))
    assert edges.count_nonzero().item() == 1
    move = group.exchange(torch.ones((1, 3, 4)), edges)
    torch.testing.assert_close(move[0], torch.tensor([[-1., 0.], [1., 0.], [0., 0.], [0., 0.]]))
    torch.testing.assert_close(move.sum(1), torch.zeros((1, 2)))


def test_exchange_does_not_mix_scored_endpoints_in_a_batch():
    edges = torch.randn(2, 3, 4, 7)
    state = torch.randn(2, 3, 4)
    reference = group.exchange(state, edges)
    edges[0] = 0.
    torch.testing.assert_close(group.exchange(state, edges)[1], reference[1])


def test_wrong_feedback_keeps_time_bin_and_magnitude_not_diagnostic_identity():
    y = torch.arange(20.).reshape(1, 5, 4)
    torch.testing.assert_close(group.sham_error(y).sort(dim=-1).values, y)
    assert not torch.any(group.sham_error(y) == y)
    state = group.feedback_write(y, 15)
    torch.testing.assert_close(state, y.repeat_interleave(3, dim=1))


def test_radius_has_finite_gradients_and_ignores_excluded_diagnostics():
    columns = len(group.lag.CHANNELS)
    x = torch.zeros(2, 25, columns, requires_grad=True)
    x2 = x.detach().clone()
    for c in group.EXCLUDED:
        x2[..., group.lag.CHANNELS.index(c)] = 99.
    center = torch.zeros(2, columns)
    scale = torch.ones_like(center)
    a = group.radii(x, center, scale)
    b = group.radii(x2, center, scale)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(torch.autograd.grad(a.sum(), x)[0]).all()


def test_exact_dose_matches_all_directions_or_skips_all():
    torch.manual_seed(131)
    x = torch.randn(3, 2, 4, 2)
    x[0, 1] = 0.
    a = group.matched_writes(x, torch.tensor([.7, .9]))
    torch.testing.assert_close(a.flatten(2).norm(dim=-1), torch.tensor([[.7, 0.]]).expand(3, -1))


def test_real_first_block_forward_has_gradients_for_distinct_diagnostic_losses():
    torch.manual_seed(131)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2, 16, 23)
    m = torch.ones_like(x, dtype=torch.bool)
    state = torch.zeros(2, 3, 4, requires_grad=True)
    h = group.forward_hidden(base, x, m, lambda z: z, state=state)
    loss_a, loss_b = h[:, 0].sum(), h[:, 1].sum()
    a = torch.autograd.grad(loss_a, state, retain_graph=True)[0]
    b = torch.autograd.grad(loss_b, state)[0]
    assert a.norm() > 0. and b.norm() > 0.
    assert not torch.allclose(a, b)
    assert all(p.grad is None for p in base.parameters())


def test_source_origin_cannot_read_its_own_feedback_and_shot_state_is_local():
    q = np.array([159, 191, 255])
    from experiments.risk_aligned_online.commissioning_lag_mixer_20261111 import delayed_slots

    np.testing.assert_array_equal(delayed_slots(q, q)[0], [0, 1, 2])
    np.testing.assert_array_equal(delayed_slots(q, np.array([184, 185, 216, 217]))[0], [0, 1, 1, 2])
    np.testing.assert_array_equal(delayed_slots(np.empty(0, dtype=int), q)[0], [0, 0, 0])


def test_transport_preserves_issued_error_but_uses_the_released_jacobian():
    old = torch.zeros((1, 4), requires_grad=True)
    new = torch.zeros_like(old, requires_grad=True)
    issued = 2.*old
    released = new*torch.tensor([-1., 3., 1., 2.])
    pred = issued[:, None]
    error = torch.tensor([[[.1, -.1, .3, -.3]]])
    a, b, wrong = group.release_backprop(pred, issued, old, released, new, error)
    signed = (error[:, 0]/group.BETA).clamp(-1., 1.)
    torch.testing.assert_close(a, 2.*signed)
    torch.testing.assert_close(b, signed*torch.tensor([-1., 3., 1., 2.]))
    torch.testing.assert_close(wrong, signed.roll(1, -1)*torch.tensor([-1., 3., 1., 2.]))


def test_real_transport_uses_only_the_issue_and_legal_release_windows(monkeypatch):
    torch.manual_seed(131)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    values = np.random.default_rng(131).normal(size=(220, 23)).astype(np.float32)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    n = len(group.lag.CHANNELS)
    q = np.array([159, 191])
    early, release = [], []
    grab = group.lag._windows
    def observe(base, signal, indices):
        (early if not early else release).extend(indices)
        return grab(base, signal, indices)
    monkeypatch.setattr(group.lag, '_windows', observe)
    # The archived lag patcher expects 128 real samples; identity here makes
    # this test use a smaller Transformer without replacing its forward.
    monkeypatch.setattr(group.layer, 'patcher', lambda *_: lambda raw: raw)
    phase, bias = torch.zeros(3, n, n), torch.zeros(3, n)
    fields = torch.zeros(2, n), torch.zeros(2, n), torch.ones(2, n), torch.ones(n)
    weights = torch.randn(group.lag.H*n, base.cfg.dim)
    decoder = torch.zeros(base.cfg.dim), torch.ones(base.cfg.dim), weights, torch.zeros(len(weights))
    states = group._estimate(base, signal, q, np.array([True, False]), phase, bias,
                            torch.zeros(2, group.lag.H, n), fields, torch.zeros(2, 5, 4),
                            decoder, transport=True)
    assert early == [159] and release == [185]
    assert states[:, [0, 2]].count_nonzero() == 0
    assert (states[:, 1].flatten(1).norm(dim=1) > 0.).all()


def test_handoff_transitions_between_previous_and_new_at_equal_bounded_dose():
    a = torch.tensor([[[[1., 0.]], [[1., 0.]], [[1., 0.]]],
                      [[[0., 1.]], [[0., 1.]], [[0., 1.]]]])
    b = torch.tensor([[[[0., 1.]], [[0., 1.]], [[0., 1.]]],
                      [[[0., -1.]], [[0., -1.]], [[0., -1.]]]])
    radii = torch.full((3,), .8)
    v = group.blend_writes(a, b, radii, torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(v[:, 0], .8*b[:, 0])
    torch.testing.assert_close(v[:, 2], .8*a[:, 2])
    # If one control cancels during a handoff, no arm gets a larger dose.
    torch.testing.assert_close(v[:, 1], torch.zeros(2, 1, 2))
    assert v.flatten(2).norm(dim=2).max() <= .8


def test_handoff_clock_cannot_use_a_later_feedback_state():
    q = np.array([159, 191])
    endpoints = np.array([184, 185, 189, 193, 216, 217])
    slots, _ = group.lag.delayed_slots(q, endpoints)
    np.testing.assert_array_equal(slots, [0, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(group.handoff_fraction(q[slots[1:]-1]+group.lag.H+1, endpoints[1:], 8),
                                  [0., .5, 1., 1., 0.])
