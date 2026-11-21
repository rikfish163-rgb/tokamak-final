import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_group_token_20261112 as group
from experiments.risk_aligned_online import commissioning_signed_handoff_20261112 as signed
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_signed_trajectory_distinguishes_reflections_without_using_excluded_channels():
    center = torch.zeros(1, 10)
    scale = torch.ones_like(center)
    data = torch.zeros(1, 25, 10)
    data[0, 5:10, group.USE[0]] = -2.
    data[0, 20:, group.USE[-1]] = 3.
    torch.testing.assert_close(group.radii(data, center, scale), group.radii(-data, center, scale))
    signed = group.signed_levels(data, center, scale)
    torch.testing.assert_close(signed, -group.signed_levels(-data, center, scale))
    assert signed[0, 1, 0] < 0. < signed[0, 4, -1]
    data[:, :, :2] = 1.e6
    torch.testing.assert_close(group.signed_levels(data, center, scale), signed)


def test_uniform_level_and_dynamics_weights_preserve_force_and_time_reversal():
    force = torch.linspace(-1., 1., 80).reshape(2, 5, 8)
    torch.testing.assert_close(group.shape_force(force, torch.ones(2, 4)), force)
    weights = torch.arange(8.).reshape(2, 4)
    a = group.shape_force(force, weights)
    b = group.shape_force(force.flip(1), weights)
    torch.testing.assert_close(a.flip(1), b)
    torch.testing.assert_close(a.mean(1), force.mean(1)*weights[0, group.SIGNED_GROUPS])


def test_teaching_keeps_constant_error_out_of_temporal_component():
    force = torch.randn(2, 1, 8).expand(-1, 5, -1)
    w = torch.tensor([[0., 0., 0., 0.], [2., 2., 2., 2.]])
    torch.testing.assert_close(group.shape_force(force, w), torch.zeros_like(force), atol=1.e-6, rtol=0.)


def test_history_teacher_rewards_risk_separation_with_same_real_write_norm():
    directions = torch.zeros(2, 2, 8, 2)
    directions[:, :, 0] = torch.tensor([1., 0.])
    directions[:, :, 1] = torch.tensor([-1.4, 1.])
    g = torch.tensor([[[1., 0.], [-1., 0.]]]).expand(2, -1, -1)
    gram = directions @ directions.transpose(-1, -2)
    slope = (directions @ g[..., None]).squeeze(-1)
    radius = torch.ones_like(slope[..., 0])
    before = signed.predicted_changes(gram, slope, radius, torch.ones(2, 4))
    learned = signed.fit_weights(gram, slope, radius)
    after = signed.predicted_changes(gram, slope, radius, learned)
    assert (after[:, 0]>before[:, 0]).all() and (after[:, 1]<before[:, 1]).all()
    assert (learned >= .25).all() and (learned <= 1.75).all()


def test_signed_replay_directions_use_observed_release_and_no_future_label(monkeypatch):
    torch.manual_seed(131)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    values = np.random.default_rng(131).normal(size=(220, 23)).astype(np.float32)
    signal = dict(values=values, available=np.ones_like(values, dtype=bool))
    used = []
    real = group.lag._windows
    def windows(base, signal, indices):
        used.extend(indices)
        return real(base, signal, indices)
    monkeypatch.setattr(group.lag, '_windows', windows)
    monkeypatch.setattr(group.layer, 'patcher', lambda *_: lambda raw: raw)
    n, h = len(group.lag.CHANNELS), group.lag.H
    fields = torch.zeros(2, n), torch.zeros(2, n), torch.ones(2, n), torch.ones(n)
    decoder = torch.zeros(8), torch.ones(8), torch.randn(h*n, 8), torch.zeros(h*n)
    args = (base, signal, np.array([159, 191]), np.array([True, False]),
            torch.zeros(3, n, n), torch.zeros(3, n), torch.zeros(2, h, n), fields, torch.randn(2, h, n),
            decoder, torch.ones(2, 4))
    states = group._estimate_signed(*args)
    assert used == [159, 185]
    assert states[:, [0, 2]].count_nonzero() == 0
    torch.testing.assert_close(states[1], states[3])
    torch.testing.assert_close(states[2], states[4])
    assert not torch.allclose(states[1], states[2])
    components = group._estimate_signed(*args, split=True)
    torch.testing.assert_close(components[1], states[3])
    torch.testing.assert_close(components[2]+components[3], components[1])


def test_level_dynamics_ablation_recombines_and_keeps_dimension():
    f = torch.linspace(-1., 1., 80).reshape(2, 5, 8).roll(2, dims=1)
    w = torch.linspace(.4, 1.6, 8).reshape(2, 4)
    persistent, varying = group.force_components(f, w)
    assert persistent.shape == varying.shape == f.shape
    torch.testing.assert_close(varying.mean(1), torch.zeros_like(f[:, 0]))
    torch.testing.assert_close(persistent+varying, group.shape_force(f, w))
    torch.testing.assert_close(persistent, group.force_components(f.flip(1), w)[0])


def test_abs_per_diagnostic_force_keeps_signed_feature_dimensions_but_not_polarity():
    a = torch.linspace(-2., 2., 40).reshape(1, 5, 8)
    y = (a+0.25).flip(1)
    w = torch.ones(2, 4)
    pred, force = group.absolute_component(a, y, w)
    pred_mirror, force_mirror = group.absolute_component(-a, -y, w)
    assert pred.shape == force.shape == a.shape
    torch.testing.assert_close(pred, pred_mirror)
    torch.testing.assert_close(force, force_mirror)


def test_repeated_force_reference_uses_only_windows_mature_at_the_issue():
    q = np.array([127, 159, 191, 223, 255, 287])
    force = torch.arange(1., 7.).reshape(-1, 1, 1)
    good = np.array([True, True, False, True, True, True])
    expected, residual, previous, use = group.past_force_parts(q, force, good)
    torch.testing.assert_close(expected[:, 0, 0], torch.tensor([0., .5, 1., 1., 1.75, 2.4]))
    np.testing.assert_array_equal(use, [False, False, False, False, True, True])
    torch.testing.assert_close(previous[4], residual[3])
    torch.testing.assert_close(previous[5], residual[4])
    changed = force.clone()
    changed[4:] = -999.
    p, _, lagged, _ = group.past_force_parts(q, changed, good)
    torch.testing.assert_close(p[4], expected[4])
    torch.testing.assert_close(lagged[4], previous[4])
    torch.testing.assert_close(p[3], expected[3])


def test_repeated_force_reference_expires_on_clock_not_number_of_valid_blocks():
    q = np.array([127, 159, 191, 127+12*group.lag.STRIDE])
    expected, _, _, use = group.past_force_parts(q, torch.ones(4, 1, 1), np.ones(4, dtype=bool))
    assert expected[-1].item() == 0.
    assert not use.any()


def test_history_replay_directions_are_prefix_invariant_across_gradient_batches(monkeypatch):
    torch.manual_seed(131)
    base = Simple23(Config(window=16, patch=4, dim=8, heads=2, layers=2, horizon=3),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    rng = np.random.default_rng(131)
    values = rng.normal(size=(360, 23)).astype(np.float32)
    m = np.ones_like(values, dtype=bool)
    monkeypatch.setattr(group.layer, 'patcher', lambda *_: lambda x: x)
    monkeypatch.setattr(group, 'BATCH', 2)
    q = 159+np.arange(6)*group.lag.STRIDE
    n, h = len(group.lag.CHANNELS), group.lag.H
    target = torch.randn(6, h, n)
    decoder = torch.zeros(8), torch.ones(8), torch.randn(h*n, 8), torch.zeros(h*n)

    def replay(k):
        end = q[k-1]+h+2
        signal = dict(values=values[:end], available=m[:end])
        fields = torch.zeros(k, n), torch.zeros(k, n), torch.ones(k, n), torch.ones(n)
        return group._estimate_innovation(base, signal, q[:k], np.ones(k, dtype=bool),
            torch.zeros(k+1, n, n), torch.zeros(k+1, n), torch.zeros(k, h, n), fields, target[:k],
            decoder, torch.ones(2, 4))

    full, active = replay(6)
    head, first = replay(5)
    np.testing.assert_array_equal(active, [False, False, False, True, True, True])
    np.testing.assert_array_equal(first, active[:5])
    torch.testing.assert_close(head, full[:, :6], atol=5e-6, rtol=1e-5)
    assert full[:, :4].count_nonzero() == 0 and full[:, 4:].count_nonzero() > 0
