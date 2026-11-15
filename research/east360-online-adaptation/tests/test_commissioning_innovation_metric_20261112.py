import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online import commissioning_innovation_metric_20261112 as metric
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_linear_gradient_transform_can_leave_input_direction_but_has_no_offset():
    raw = torch.tensor([[[2., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
    factors = torch.tensor([[[0.], [0.], [1.], [0.]], [[1.], [0.], [0.], [0.]]])
    actual = group.gradient_transform(raw, factors)
    torch.testing.assert_close(actual[0], torch.tensor([[2., 0.], [2., 0.]]))
    torch.testing.assert_close(actual[1], raw[1], rtol=0., atol=0.)
    torch.testing.assert_close(group.gradient_transform(-raw[:1], factors), -actual[:1])


def test_gradient_transform_is_local_to_each_release_and_control():
    generator = torch.Generator().manual_seed(13)
    raw = torch.randn(2, 5, 3, 4, generator=generator)
    factors = torch.randn(2, 12, 2, generator=generator)
    full = group.gradient_transform(raw, factors)
    torch.testing.assert_close(group.gradient_transform(raw[:, :3], factors), full[:, :3])
    torch.testing.assert_close(group.gradient_transform(raw[:1], factors), full[:1])
    torch.testing.assert_close(group.gradient_transform(raw, torch.zeros_like(factors)), raw)


def test_source_peaks_require_mature_stable_feedback_and_strictly_timely_lead():
    signal = np.zeros(221)
    valid = np.zeros(len(signal), dtype=bool)
    valid[[157, 160, 165, 166, 167, 180, 193, 198, 204]] = True
    lead = np.zeros(len(signal))
    lead[valid] = [50., 50., 50., 10., 11., 12., 50., 50., 9.]
    release = np.array([100, 132, 164]) + group.lag.H + 1
    eligible = np.array([False, True, True])
    chosen = metric.source_peaks(signal, release, eligible, valid, 8, lead_ms=lead)
    np.testing.assert_array_equal(chosen, [167, 180, 198])
    np.testing.assert_array_equal(metric.source_peaks(signal, release, eligible, valid, 8),
                                  [166, 180, 198, 204])


def test_surrogate_rewards_timely_peak_but_penalizes_any_safe_peak():
    margins = torch.tensor([-1., .2, -.7, .5], requires_grad=True)
    owner = torch.tensor([0, 0, 1, 1])
    peaks = metric.shot_max(margins, owner, 2)
    loss = metric.shot_loss(peaks, torch.tensor([True, False]), 0.)
    (slope,) = torch.autograd.grad(loss, margins)
    assert slope[1] < 0 < slope[3]
    assert slope[[0, 2]].abs().sum() == 0.


def test_finite_source_objective_backpropagates_through_real_frozen_encoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        base = Simple23(Config(dim=16), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
        x = torch.randn(2, 128, 23)
        m = torch.ones_like(x, dtype=torch.bool)
        width = len(group.lag.CHANNELS)
        phase = torch.randn(2, width, width)*.1
        correction = torch.randn(2, width)*.02
        g = torch.randn(2, 15, len(group.TRUSTED))
        item = (x, m, phase, correction, torch.zeros(2), g, g.flip(1))
        rows = [(item, dict(endpoints=[300, 330], positive=True))]
        factors = torch.randn(2, 15*len(group.TRUSTED), 4).mul(.1).requires_grad_()
        peaks, positive, dose, error = metric.batch_views(base, rows, factors, controls=True)
        loss = metric.shot_loss(peaks[3], positive, 0.)
        (slope,) = torch.autograd.grad(loss, factors)
        assert torch.isfinite(slope).all() and slope.abs().sum()>0.
        assert (dose>0.).all() and error<1e-5
        assert all(p.grad is None for p in base.parameters())
