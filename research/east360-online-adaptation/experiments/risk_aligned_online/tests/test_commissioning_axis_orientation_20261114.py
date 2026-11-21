import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_axis_orientation_20261114 as model


def test_only_observed_lane_motion_enters_embedding_shift():
    rng = torch.Generator().manual_seed(31)
    raw = torch.randn(2, 16, 8, 46, generator=rng, dtype=torch.float64)
    weight = torch.randn(6, 8, 46, generator=rng, dtype=torch.float64)
    col = model.axis.TARGET_INDEX
    observed = torch.rand(2, 16, 8, generator=rng) > .25
    raw[..., 23+col] = observed
    actual = model.lane_basis(raw.flatten(-2), weight.flatten(-2))
    x = raw[..., col]
    center = x.masked_fill(~observed, 0.).sum(1)/observed.sum(1).clamp_min(1)
    rewritten = raw.clone()
    rewritten[..., col] += (x-center[:, None]).masked_fill(~observed, 0.)
    expected = (rewritten-raw).flatten(-2) @ weight.flatten(-2).T
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-14)
    torch.testing.assert_close(actual.sum(1), torch.zeros_like(actual[:, 0]), rtol=0, atol=5e-14)
    raw[..., 23+col] = 0.
    assert torch.count_nonzero(model.lane_basis(raw.flatten(-2), weight.flatten(-2))) == 0


def test_mature_signed_fit_detects_reflection_without_seeing_earlier_or_missing_obs():
    q = 127 + 32*np.arange(8)
    forecast = np.tile([-1., -.5, 0., .5, 1.], (len(q), 1))
    valid = np.ones(len(q), dtype=bool)
    now, qualified = model.rolling_alignment(q, forecast, -forecast, valid)
    assert not qualified[:4].any()  # Earlier control needs a previous four-block state.
    np.testing.assert_allclose(now[0, qualified], -1.)
    np.testing.assert_allclose(now[2, qualified], -1.)
    assert (now[1, qualified]>0.).all()
    release = q[4]+model.axis.RELEASE_OFFSET+model.WAIT
    t = np.arange(release-2, release+model.PATCH+1)
    coef = model.handoff(now, qualified, q, t)
    assert not coef[:, :3].any()
    np.testing.assert_allclose(coef[[0, 2], -1], -1.)
    np.testing.assert_allclose(coef[[0, 2], 3], -1./model.PATCH)
    valid[3] = False
    _, cut = model.rolling_alignment(q, forecast, -forecast, valid)
    assert not cut.any()


def test_adaptive_token_norm_matches_true_fixed_sign_and_controls():
    basis = torch.zeros((4, 16, 3), dtype=torch.float64)
    basis[:3, 0, 0] = torch.tensor([.25, 2., .25])
    rho = torch.tensor([[.50, -.90, -.95, .90],
                        [-.5,  .10,   0.,  .80],
                        [ .5, -.10,  .5, -.80]], dtype=basis.dtype)
    cap = torch.tensor([.40, .30, .10, 1.], dtype=basis.dtype)
    change, dose, gain = model.aligned_write(basis, rho, cap)
    torch.testing.assert_close(dose, torch.tensor([.25, .30, 0., 0.], dtype=basis.dtype))
    torch.testing.assert_close(change.flatten(2).norm(dim=2), dose.expand(5, -1))
    assert torch.all(change[-2, :2, 0, 0] < 0.)
    assert torch.all(change[-1, :2, 0, 0] > 0.)
    torch.testing.assert_close(gain, torch.tensor([2., .85, 1., 1.], dtype=basis.dtype))


def test_release_ramp_does_not_retain_update_after_an_invalid_current_block():
    q = 127 + 32*np.arange(8)
    available = np.arange(len(q)) >= 4
    available[6] = False
    rho = np.full((3, len(q)), .5)
    rel = q+model.axis.RELEASE_OFFSET+model.WAIT
    t = np.array([rel[4]-1, rel[4], rel[4]+8, rel[6]-1, rel[6], rel[7], rel[7]+8])
    got = model.handoff(rho, available, q, t)
    np.testing.assert_allclose(got, np.tile([0., 0., .5, .5, 0., 0., .5], (3, 1)))
