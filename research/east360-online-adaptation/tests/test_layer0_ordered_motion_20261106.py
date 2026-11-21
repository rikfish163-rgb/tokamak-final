import numpy as np
import pytest
import torch

import layer0_ordered_motion_20261106 as ordered


def block():
    level = torch.arange(25, dtype=torch.float64)[None, :, None] ** 2
    z = level.expand(2, 25, 23).clone()
    return z, torch.ones_like(z, dtype=torch.bool)


def test_four_point_half_difference_and_exact_reversal():
    z, m = block()
    m[:, 4, 4] = False
    z[~m] = float("nan")
    y, keep = ordered.block_motion(z, m)
    rev, rm = ordered.block_motion(z.flip(1), m.flip(1))
    shifted, sm = ordered.block_motion(z + 32, m)
    assert y.shape == (2, 10, 23)
    assert torch.equal(keep, rm) and torch.equal(keep, sm)
    assert torch.equal(y, -rev)
    assert torch.equal(y, shifted)
    # Linear trend of the five-point differences has symmetric half-sums.
    assert torch.equal(y[:, :, 0], torch.full((2, 10), 120.0, dtype=z.dtype))
    assert not keep[:, 4, 4].any()
    assert not y[:, 4, 4].any()


def test_common_observation_support_excludes_absent_endpoints_and_categories():
    z, m = block()
    origin = torch.ones((2, 23), dtype=torch.bool)
    origin[0, 3] = False
    past = torch.ones((2, 10, 23), dtype=torch.bool)
    past[1, 0, 3] = False
    y, keep = ordered.target(z, m, origin, past, torch.zeros(23), torch.ones(23))
    assert not keep[:, :, [2, 10]].any()
    assert not keep[0, :, 3].any() and not keep[1, 0, 3]
    assert keep[1, 1, 3]
    assert not y[~keep].any()


def test_history_is_from_exact_last_fifty_points_and_has_no_missing_payload():
    x = torch.arange(128, dtype=torch.float64)[None, :, None].expand(2, 128, 23).clone()
    m = torch.ones_like(x, dtype=torch.bool)
    m[:, -1, 1] = False
    x[~m] = float("nan")
    x_changed = x.clone()
    x_changed[:, :-50] = 1e6
    args = (m, torch.zeros(23), torch.ones(23))
    result = ordered.history(x, *args)
    assert all(
        torch.equal(a, b) for a, b in zip(result, ordered.history(x_changed, *args))
    )
    local, prev, mask, end = result
    assert local.shape == (2, 23, 49)
    assert not mask[:, :, 1].any() and not end[:, 1].any()
    assert not local[:, 1].any() and not prev[:, :, 1].any()
    assert torch.isfinite(local).all() and torch.isfinite(prev).all()


def test_readout_is_odd_and_matches_stale_rms_direction_without_inventing_one():
    head = ordered.OrderedHead(torch.linspace(1, 2, 128))
    with torch.no_grad():
        head.context.weight.fill_(0.1)
    delta = torch.randn(4, 128, generator=torch.Generator().manual_seed(3))
    stale = delta.roll(1, 1) * 3
    stale[0] = 0
    delta[1] = 0
    matched = head.match_stale(delta, stale)
    assert not matched[:2].any()
    rms = head.scale
    torch.testing.assert_close(
        (matched[2:] / rms).norm(dim=1), (delta[2:] / rms).norm(dim=1)
    )
    assert torch.equal(head.context_motion(-delta), -head.context_motion(delta))


def test_loss_counts_and_gradients_do_not_use_missing_targets():
    pred = torch.tensor([2.0, 20.0, 0.0, -20.0]).view(2, 2, 1).requires_grad_()
    y = torch.tensor([0.0, float("nan"), float("nan"), float("inf")]).view_as(pred)
    m = torch.tensor([True, False, False, False]).view_as(pred)
    err, n = ordered.masked_loss(pred, y, m)
    torch.testing.assert_close(err, torch.tensor([1.5, 0.0]))
    assert torch.equal(n, torch.tensor([1, 0]))
    err.sum().backward()
    torch.testing.assert_close(pred.grad.flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        ordered.masked_loss(pred, y, torch.ones_like(m))


def test_exact_pair_windows_ignore_future_and_share_source_masks():
    v = np.broadcast_to(np.arange(220, dtype=np.float32)[:, None], (220, 23)).copy()
    a = np.ones_like(v, dtype=bool)
    origins = np.array([177, 190])
    windows, mask = ordered.pair_windows(v, a, origins, source=True)
    assert windows.shape == (2, 3, 128, 23)
    np.testing.assert_array_equal(
        windows[:, :, -1, 0], [[177, 152, 127], [190, 165, 140]]
    )
    assert not mask[..., 2].any()
    assert mask[0, ..., 1].all() and not mask[1, ..., 1].any()
    v[178:] = -1000.0
    edited, _ = ordered.pair_windows(v, a, origins, source=True)
    np.testing.assert_array_equal(edited[0], windows[0])
    with pytest.raises(ValueError, match="complete"):
        ordered.pair_windows(v, a, np.array([176]), source=True)


def test_frozen_context_uses_two_true_windows_and_a_third_for_the_lag(monkeypatch):
    x = torch.zeros((2, 3, 128, 23))
    x[0, :, -1, 0] = torch.tensor([7.0, 5.0, 1.0])
    x[1, :, -1, 0] = torch.tensor([20.0, 3.0, -4.0])
    m = torch.ones_like(x, dtype=torch.bool)

    def fake_features(base, x, a, action=None):
        v = x[:, -1, 0]
        if action is not None:
            v = v * action[:, 0]
        return v[:, None].expand(-1, 128), None

    monkeypatch.setattr(ordered.motion, "features", fake_features)
    action = torch.ones((2, 128), requires_grad=True)
    now, late = ordered.context_pair(None, x, m, action)
    torch.testing.assert_close(now[:, 0], torch.tensor([2.0, 17.0]))
    torch.testing.assert_close(late[:, 0], torch.tensor([4.0, 7.0]))
    now[:, 0].sum().backward()
    torch.testing.assert_close(action.grad[:, 0], torch.tensor([2.0, 17.0]))
