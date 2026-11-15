import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_carry_20261112 as carry
from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_coefficients_keep_prior_direction_only_within_the_shot():
    g = torch.tensor([[[[0., 0.]], [[6., 0.]], [[0., 9.]], [[90., 90.]], [[0., 9.]]]])
    eligible = np.array([True, True, False, True])
    u = carry.carried_directions(g, eligible)
    torch.testing.assert_close(u[0, 2, 0], torch.tensor([.25, .5]))
    torch.testing.assert_close(u[0, 3, 0], torch.zeros(2))
    torch.testing.assert_close(u[0, 4, 0], torch.tensor([.0625, .625]))
    torch.testing.assert_close(carry.carried_directions(g[:, :3], eligible[:2]), u[:, :3])
    fresh = g[:, 3:].clone()
    fresh[:, 0] = 0.
    torch.testing.assert_close(carry.carried_directions(fresh, eligible[-1:])[0, 1, 0], torch.tensor([0., .5]))


def test_carry_does_not_add_writable_endpoints_or_change_the_token_dose(monkeypatch):
    torch.manual_seed(104)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    signal = dict(values=np.random.default_rng(104).normal(size=(85, 23)).astype(np.float32),
                  available=np.ones((85, 23), dtype=bool))
    q = np.array([15, 47])
    eligible = np.array([True, True])
    monkeypatch.setattr(group.layer, 'patcher', lambda *_: lambda raw: raw*.9)
    phase = correction = torch.zeros(len(q)+1)
    ends = np.arange(len(signal['values']))
    x, mask = group.lag._windows(base, signal, ends)
    with torch.no_grad():
        h = group.forward_hidden(base, x, mask, group.layer.patcher(None, None),
                                 state=torch.zeros(len(x), 3, 4))
    parent = np.full(len(ends), .5, dtype=np.float32)
    g = torch.randn(2, len(q)+1, 3, 4)
    g[:, 0] = 0.
    budget = np.zeros(len(ends), np.float32)
    budget[[42, 73, 74, 77]] = .02
    directions = torch.cat((g, carry.carried_directions(g, eligible)))
    curves, _, error, dose_error = carry.render(
        base, signal, q, h, torch.zeros(len(ends)), torch.randn(base.cfg.dim),
        phase, correction, directions, parent, budget)
    np.testing.assert_array_equal(curves[:, budget==0.], np.broadcast_to(parent[budget==0.], (4, 81)))
    np.testing.assert_allclose(curves[:2, 42], curves[2:, 42], rtol=0., atol=1e-6)
    assert np.any(np.abs(curves[:2, 77]-curves[2:, 77]) > 1e-5)
    assert error < 1e-6 and dose_error < 1e-6


def test_own_forecast_is_issued_before_its_target_and_uses_an_already_released_state(monkeypatch):
    torch.manual_seed(71)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    q = np.array([15, 47, 79, 111, 143])
    width = len(group.lag.CHANNELS)
    values = np.random.default_rng(71).normal(size=(170, 23)).astype(np.float32)
    monkeypatch.setattr(group.layer, 'patcher', lambda *_: lambda raw: raw*.9)
    fields = (torch.zeros(len(q), width), torch.zeros(len(q), width),
              torch.ones(len(q), width), torch.ones(width))
    prior = torch.zeros(len(q), group.lag.H, width)
    decoder = (torch.zeros(8), torch.ones(8), torch.randn(group.lag.H*width, 8)*.4,
               torch.zeros(group.lag.H*width))
    weights = torch.ones(2, 4)
    phase = correction = torch.zeros(len(q)+1)
    budget = np.zeros(len(values), np.float32)
    budget[q[4]] = .04

    def replay(x, dose):
        waveform = torch.from_numpy(x[q[:, None]+np.arange(1, 26)][:, :, group.lag.CHANNELS])
        signal = dict(values=x, available=np.ones_like(x, dtype=bool))
        return carry.own_feedback_directions(base, signal, q, np.ones(len(q), bool), phase, correction,
                                             prior, fields, waveform, decoder, weights, dose)

    changed = values.copy()
    changed[q[3]+1:] *= -2.
    g, issued = replay(values, budget)
    other, issued_other = replay(changed, budget)
    zero, issued_zero = replay(values, np.zeros_like(budget))
    torch.testing.assert_close(issued[:, :4], issued_other[:, :4])
    torch.testing.assert_close(g[:, :4], other[:, :4])
    assert not torch.allclose(g[0, 4], other[0, 4])
    torch.testing.assert_close(g[:, :5], zero[:, :5])
    assert not torch.allclose(issued[:, 4], issued_zero[:, 4])
    assert not torch.allclose(g[0, 5], zero[0, 5])
    assert all(p.grad is None for p in base.parameters())
