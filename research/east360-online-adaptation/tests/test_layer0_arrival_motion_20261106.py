"""CPU-only synthetic checks for the bounded arrival-motion primitives."""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.simple23 import Config, Simple23
import latent_trajectory_adapt_20261105 as latent
import layer0_arrival_motion_20261106 as motion
import trajectory_policy_20261105 as policy


def normalization(dtype=torch.float64):
    return torch.linspace(-1, 1, 23, dtype=dtype), torch.linspace(0.5, 2, 23, dtype=dtype)


def signal_from_coordinates(z, center, scale):
    return center + scale * z.sinh()


def test_origin_endpoint_local_and_anchored_slope_with_sparse_masks():
    center, scale = normalization()
    j = torch.arange(-24, 1, dtype=center.dtype)
    z = torch.zeros(2, 128, 23, dtype=center.dtype)
    z[:, -25:, 0] = 0.4 + 0.03 * j
    z[:, -25:, 1] = -0.2 - 0.02 * j
    z[:, -25:, 3] = 0.5  # A lone endpoint has zero slope.
    values = signal_from_coordinates(z, center, scale)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, -20:-8:2, 0] = False
    mask[0, -25:-1, 3] = False
    mask[1, -1, 1] = False
    mask[1, -25:-1, 4] = False
    values[~mask] = float("nan")

    hold, linear, local, origin_valid = motion.origin_terms(values, mask, center, scale)
    assert hold.shape == linear.shape == (2, 25, 23)
    assert local.shape == (2, 23, 24)
    assert origin_valid.dtype == torch.bool and origin_valid.shape == (2, 23)
    assert torch.equal(origin_valid, mask[:, -1])
    torch.testing.assert_close(hold[:, :, 0], torch.full((2, 25), 0.4, dtype=center.dtype))
    expected = 0.4 + 0.03 * torch.arange(1, 26, dtype=center.dtype)
    torch.testing.assert_close(linear[:, :, 0], expected.expand(2, -1))
    assert torch.equal(hold[1, :, 1], torch.zeros(25, dtype=center.dtype))
    assert torch.equal(linear[1, :, 1], torch.zeros(25, dtype=center.dtype))
    assert torch.equal(local[1, 1], torch.zeros(24, dtype=center.dtype))
    assert torch.equal(local[0, 3], torch.zeros(24, dtype=center.dtype))
    torch.testing.assert_close(linear[0, :, 3], hold[0, :, 3])
    expected_local = 0.03 * j[:-1]
    expected_local = expected_local.masked_fill(~mask[0, -25:-1, 0], 0)
    torch.testing.assert_close(local[0, 0], expected_local)
    assert all(torch.isfinite(t).all() for t in (hold, linear, local))


def test_origin_slope_is_least_squares_through_endpoint_not_free_intercept():
    center, scale = normalization()
    z = torch.zeros(1, 128, 23, dtype=center.dtype)
    mask = torch.zeros_like(z, dtype=torch.bool)
    z[0, -1, 0], z[0, -2, 0], z[0, -4, 0] = 3, 2, -3
    mask[0, [-1, -2, -4], 0] = True
    hold, linear, local, _ = motion.origin_terms(signal_from_coordinates(z, center, scale), mask, center, scale)
    # ((-1)*(2-3) + (-3)*(-3-3)) / ((-1)^2 + (-3)^2) = 1.9.
    torch.testing.assert_close(linear[0, :, 0], 3 + 1.9 * torch.arange(1, 26, dtype=center.dtype))
    torch.testing.assert_close(hold[0, 0, 0], torch.tensor(3., dtype=center.dtype))
    torch.testing.assert_close(local[0, 0, [-1, -3]], torch.tensor([-1., -6.], dtype=center.dtype))


def test_origin_ignores_everything_before_past_25_and_is_prefix_causal():
    center, scale = normalization()
    g = torch.Generator().manual_seed(36)
    stream = torch.randn(170, 23, generator=g, dtype=center.dtype)
    available = torch.rand(170, 23, generator=g) > 0.2
    windows, masks = policy.window_views(stream, available)
    changed = stream.clone()
    changed[141:] = 10000
    altered, _ = policy.window_views(changed, available)
    first = motion.origin_terms(windows[140:141], masks[140:141], center, scale)
    second = motion.origin_terms(altered[140:141], masks[140:141], center, scale)
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    prefix = windows[140:141].clone()
    prefix[:, :-25] = float("nan")  # Irrelevant even when its mask is true.
    cropped = motion.origin_terms(prefix, masks[140:141], center, scale)
    assert all(torch.equal(x, y) for x, y in zip(first, cropped))


def test_target_masks_exclusions_origin_and_missing_nonfinite():
    center, scale = normalization()
    z = torch.arange(25 * 23, dtype=center.dtype).reshape(1, 25, 23) / 1000
    raw = signal_from_coordinates(z, center, scale)
    truth_mask = torch.ones_like(raw, dtype=torch.bool)
    truth_mask[0, 0, 3] = False
    raw[0, 0, 3] = float("nan")
    raw[0, 1, 4] = float("inf")
    truth_mask[0, 1, 4] = False
    origin = torch.ones(1, 23, dtype=torch.bool)
    origin[0, 5] = False
    target, valid = motion.target_terms(raw, truth_mask, origin, center, scale)
    expected = truth_mask & origin[:, None, :]
    expected[:, :, [2, 10]] = False
    assert target.shape == valid.shape == (1, 25, 23) and valid.dtype == torch.bool
    assert torch.equal(valid, expected)
    torch.testing.assert_close(target[valid], z[valid])
    assert torch.isfinite(target).all()
    assert valid[0, 0, 0] and valid[0, 24, 22]


def test_loss_is_per_row_with_exact_lag_five_pair_mask_and_explicit_support():
    pred = torch.zeros(4, 25, 23, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.zeros_like(pred, dtype=torch.bool)
    valid[0, [0, 5, 6, 10], 0] = True  # (0,5) and (5,10), not (0,6).
    valid[1, [0, 4], 0] = True       # Level but no shape support.
    valid[2, [1, 6], 1] = True
    with torch.no_grad():
        pred[0, [0, 5, 6, 10], 0] = torch.tensor([0., 1., 2., 3.], dtype=pred.dtype)
        pred[1, [0, 4], 0] = 1
        pred[2, [1, 6], 1] = torch.tensor([2., -1.], dtype=pred.dtype)
        pred[~valid] = float("nan")
    target[~valid] = float("nan")
    losses = motion.row_loss(pred, target, valid)
    assert set(losses) == {"level", "shape", "combined", "supported", "counts_level", "counts_shape"}
    assert all(value.shape == (4,) for value in losses.values())
    assert losses["counts_level"].tolist() == [4, 2, 2, 0]
    assert losses["counts_shape"].tolist() == [2, 0, 1, 0]
    assert losses["supported"].tolist() == [True, False, True, False]
    assert losses["supported"].dtype == torch.bool
    expected_level = F.smooth_l1_loss(torch.tensor([0., 1., 2., 3.], dtype=pred.dtype), torch.zeros(4, dtype=pred.dtype))
    expected_shape = F.smooth_l1_loss(torch.tensor([1., 2.], dtype=pred.dtype), torch.zeros(2, dtype=pred.dtype))
    torch.testing.assert_close(losses["level"][0], expected_level)
    torch.testing.assert_close(losses["shape"][0], expected_shape)
    torch.testing.assert_close(losses["combined"][0], (expected_level + expected_shape) / 2)
    assert losses["level"][1] == 0.5 and losses["shape"][1] == 0
    assert losses["combined"][1] == losses["combined"][3] == 0
    assert all(torch.isfinite(losses[k]).all() for k in ("level", "shape", "combined"))
    losses["combined"].sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[~valid].count_nonzero() == 0


def test_linear_synthetic_baseline_beats_hold_in_supported_rows():
    center, scale = normalization()
    j = torch.arange(-127, 26, dtype=center.dtype)
    z = (0.2 + j * 0.015)[None, :, None].expand(2, -1, 23).clone()
    raw = signal_from_coordinates(z, center, scale)
    mask = torch.ones_like(raw, dtype=torch.bool)
    hold, linear, _, origin = motion.origin_terms(raw[:, :128], mask[:, :128], center, scale)
    target, valid = motion.target_terms(raw[:, 128:], mask[:, 128:], origin, center, scale)
    baseline = motion.row_loss(hold, target, valid)
    predicted = motion.row_loss(linear, target, valid)
    assert baseline["supported"].all() and predicted["supported"].all()
    assert baseline["counts_level"].tolist() == [25 * 21, 25 * 21]
    assert baseline["counts_shape"].tolist() == [20 * 21, 20 * 21]
    assert (predicted["combined"] < baseline["combined"]).all()
    assert predicted["combined"].max() < 1e-25


@pytest.fixture
def base_and_input():
    torch.manual_seed(431)
    base = Simple23(Config(), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(2, 128, 23)
    a = torch.rand(2, 128, 23) > 0.15
    return base, x, a


def hook_ids(base):
    return {name: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for name, m in base.named_modules()}


def test_features_direct_and_real_routed_path_match_and_have_action_gradients(base_and_input):
    base, x, a = base_and_input
    before = {k: v.clone() for k, v in base.state_dict().items()}
    initial_hooks = hook_ids(base)
    captured = []
    handle = base.forecast_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        native = base(x, a)["forecast"]
    finally:
        handle.remove()
    hidden = captured[0].reshape(2, 17, 64)
    expected_ctx = torch.cat((hidden[:, 15], hidden[:, -1]), dim=1)
    ctx, forecast = motion.features(base, x, a)
    torch.testing.assert_close(ctx, expected_ctx, rtol=0, atol=0)
    torch.testing.assert_close(forecast, native, rtol=0, atol=0)
    assert ctx.shape == (2, 128) and forecast.shape == (2, 25, 23)

    action = (torch.randn(2, 128) * 0.03).requires_grad_()
    with patch.object(policy, "routed_pair", wraps=policy.routed_pair) as routed, \
         patch.object(latent, "routes_all", wraps=latent.routes_all) as routes:
        action_ctx, action_forecast = motion.features(base, x, a, action)
    assert routed.call_count == routes.call_count == 1
    args = routed.call_args.args
    assert args[:3] == (base, x, a)
    assert torch.equal(args[3], latent.routes_all(a))
    assert torch.equal(args[4][:, :128], action)
    assert torch.equal(args[4][:, 128:], torch.zeros_like(action))
    weighted_context = (action_ctx * torch.linspace(-1, 1, 128)).sum()
    grad_context = torch.autograd.grad(weighted_context, action, retain_graph=True)[0]
    grad_forecast = torch.autograd.grad(action_forecast.square().sum(), action)[0]
    assert torch.isfinite(grad_context).all() and grad_context.abs().sum() > 0
    assert torch.isfinite(grad_forecast).all() and grad_forecast.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())
    assert hook_ids(base) == initial_hooks
    zero_ctx, zero_forecast = motion.features(base, x, a, torch.zeros_like(action))
    # Hooks disable a fused Transformer path, introducing float32 roundoff.
    torch.testing.assert_close(zero_ctx, ctx, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(zero_forecast, forecast, rtol=1e-5, atol=2e-6)


def test_features_return_the_base_forecast_after_model_specific_conversion():
    torch.manual_seed(412)
    base = Simple23(Config(input_view="window"), torch.ones(23), torch.full((23,), 2.)).eval().requires_grad_(False)
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    action = torch.randn(1, 128, requires_grad=True) / 100
    reference = []
    handle = base.register_forward_hook(lambda _m, _args, out: reference.append(out["forecast"]))
    try:
        policy.routed_pair(base, x, a, latent.routes_all(a), torch.cat((action, torch.zeros_like(action)), dim=1))
    finally:
        handle.remove()
    _, forecast = motion.features(base, x, a, action)
    torch.testing.assert_close(forecast, reference[0], rtol=0, atol=0)
    gradient, = torch.autograd.grad(forecast.square().sum(), action)
    assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all()


def test_features_action_autograd_matches_a_real_routed_finite_difference():
    torch.manual_seed(95)
    base = Simple23(Config(), torch.zeros(23), torch.ones(23)).double().eval().requires_grad_(False)
    x = torch.randn(1, 128, 23, dtype=torch.float64)
    a = torch.rand(1, 128, 23) > 0.2
    action = (torch.randn(1, 128, dtype=x.dtype) * 0.02).requires_grad_()
    direction = torch.randn_like(action)
    direction = direction / direction.norm()
    coefficients = torch.linspace(-1, 1, 128, dtype=x.dtype)

    def objective(value):
        ctx, forecast = motion.features(base, x, a, value)
        return (ctx * coefficients).sum() + forecast.square().mean()

    (gradient,) = torch.autograd.grad(objective(action), action)
    with torch.no_grad():
        epsilon = 1e-5
        finite_difference = (objective(action + epsilon * direction) - objective(action - epsilon * direction)) / (2 * epsilon)
    assert gradient.norm() > 0
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("with_action", [False, True])
def test_features_removes_only_its_own_hooks_on_exception(base_and_input, with_action):
    base, x, a = base_and_input
    existing_head = base.forecast_head.register_forward_pre_hook(lambda *_: None)
    existing_norm = base.encoder.layers[0].norm1.register_forward_hook(lambda *_: None)
    def fail(*_):
        raise RuntimeError("synthetic failure")
    bad = base.forecast_head.register_forward_hook(fail)
    before = hook_ids(base)
    try:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            motion.features(base, x, a, torch.zeros(2, 128) if with_action else None)
        assert hook_ids(base) == before
    finally:
        bad.remove()
        existing_norm.remove()
        existing_head.remove()


def test_motion_head_zero_initialization_independent_parameters_and_freezing():
    mean, scale = torch.arange(128, dtype=torch.float32), torch.full((128,), 2.0)
    head = motion.MotionHead(mean, scale)
    assert set(dict(head.named_parameters())) == {"local_weight", "context.weight"}
    assert set(dict(head.named_buffers())) == {"mean", "scale"}
    assert head.local_weight.shape == (23, 24, 25)
    assert head.context.weight.shape == (25 * 23, 128) and head.context.bias is None
    assert head.local_weight.count_nonzero() == head.context.weight.count_nonzero() == 0
    hold, local, ctx = torch.randn(2, 25, 23), torch.randn(2, 23, 24), torch.randn(2, 128)
    torch.testing.assert_close(head(hold, local, ctx), hold, rtol=0, atol=0)
    with torch.no_grad():
        head.local_weight[4, 5, 6] = 3
        head.context.weight[6 * 23 + 4, 0] = 2
    local_result = head.local(hold, local)
    correction = head.context_correction(ctx)
    torch.testing.assert_close(local_result[:, 6, 4], hold[:, 6, 4] + 3 * local[:, 4, 5])
    torch.testing.assert_close(correction[:, 6, 4], 2 * (ctx[:, 0] - mean[0]) / scale[0])
    assert (local_result - hold).count_nonzero() == 2 and correction.count_nonzero() == 2
    torch.testing.assert_close(head(hold, local, ctx), local_result + correction)
    head.local_weight.requires_grad_(False)
    head(hold, local, ctx).sum().backward()
    assert head.local_weight.grad is None and head.context.weight.grad is not None


def test_motion_head_synthetic_two_phase_fit_freezes_local_parameters():
    torch.manual_seed(171)
    head = motion.MotionHead(torch.zeros(128), torch.ones(128))
    hold = torch.zeros(8, 25, 23)
    local = torch.zeros(8, 23, 24)
    local[:, 0, 0] = torch.linspace(-1, 1, 8)
    ctx = torch.zeros(8, 128)
    ctx[:, 0] = torch.tensor([-1., 1., 1., -1., -1., 1., 1., -1.])
    valid = torch.zeros_like(hold, dtype=torch.bool)
    valid[:, :, 0] = True
    target = torch.zeros_like(hold)
    target[:, :, 0] = local[:, 0, 0, None] * torch.linspace(0.2, 1, 25)
    head.context.requires_grad_(False)
    optimizer = torch.optim.Adam([head.local_weight], lr=0.05)
    initial = motion.row_loss(head.local(hold, local), target, valid)["combined"].mean().item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        motion.row_loss(head.local(hold, local), target, valid)["combined"].mean().backward()
        optimizer.step()
    fitted = motion.row_loss(head.local(hold, local), target, valid)["combined"].mean().item()
    assert fitted < initial / 20

    head.local_weight.requires_grad_(False)
    head.local_weight.grad = None
    head.context.requires_grad_(True)
    local_before = head.local_weight.detach().clone()
    target[:, :, 0] += 0.4 * ctx[:, 0, None]
    optimizer = torch.optim.Adam(head.context.parameters(), lr=0.05)
    initial = motion.row_loss(head(hold, local, ctx), target, valid)["combined"].mean().item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        motion.row_loss(head(hold, local, ctx), target, valid)["combined"].mean().backward()
        optimizer.step()
    fitted = motion.row_loss(head(hold, local, ctx), target, valid)["combined"].mean().item()
    assert fitted < initial / 20
    assert torch.equal(head.local_weight, local_before) and head.local_weight.grad is None


@pytest.mark.parametrize("bad_scale", [torch.zeros(128), -torch.ones(128), torch.full((128,), float("nan"))])
def test_motion_head_rejects_invalid_context_scale(bad_scale):
    with pytest.raises(ValueError):
        motion.MotionHead(torch.zeros(128), bad_scale)


def test_rank_mask_and_observed_nonfinite_contracts(base_and_input):
    center, scale = normalization()
    x = torch.zeros(1, 128, 23, dtype=center.dtype)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError):
        motion.origin_terms(x[0], mask[0], center, scale)
    with pytest.raises(ValueError):
        motion.origin_terms(x, mask.float(), center, scale)
    with pytest.raises(ValueError):
        motion.origin_terms(x, mask, center, torch.zeros_like(scale))
    invalid = x.clone()
    invalid[0, -1, 0] = float("nan")
    with pytest.raises(ValueError):
        motion.origin_terms(invalid, mask, center, scale)
    truth = torch.zeros(1, 25, 23, dtype=center.dtype)
    truth[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        motion.target_terms(truth, torch.ones_like(truth, dtype=torch.bool), mask[:, -1], center, scale)
    with pytest.raises(ValueError):
        motion.row_loss(truth, truth, torch.ones_like(truth))
    base, values, available = base_and_input
    with pytest.raises(ValueError):
        motion.features(base, values, available, torch.zeros(2, 127))
    with pytest.raises(ValueError):
        motion.MotionHead(torch.zeros(127), torch.ones(128))
    with pytest.raises(ValueError):
        motion.MotionHead(torch.full((128,), float("inf")), torch.ones(128))
