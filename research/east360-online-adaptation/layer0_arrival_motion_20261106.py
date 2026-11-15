"""Causal arrival-motion features and explicitly supported forecast losses.

The caller owns training, freezing, and paired true/flip mask construction.
Rows without both level and lagged-shape observations are explicitly marked
unsupported; their zero combined loss is not a replacement training target.
"""

import torch
from torch import nn
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy


H, W, WIDTH, C, LAG = 25, 128, 128, 23, 5


def _check_values(values, mask, length):
    if (
        not isinstance(values, torch.Tensor)
        or not values.is_floating_point()
        or values.ndim != 3
        or values.shape[1:] != (length, C)
        or not isinstance(mask, torch.Tensor)
        or mask.shape != values.shape
        or mask.dtype != torch.bool
        or mask.device != values.device
    ):
        raise ValueError(f"floating [B,{length},{C}] values and matching boolean mask required")


def _normalization(center, scale, reference, width):
    center = torch.as_tensor(center, dtype=reference.dtype, device=reference.device)
    scale = torch.as_tensor(scale, dtype=reference.dtype, device=reference.device)
    if (
        center.shape != (width,)
        or scale.shape != (width,)
        or not torch.isfinite(center).all()
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("finite normalization and strictly positive scale required")
    return center, scale


def _coordinates(values, valid, center, scale):
    if not torch.isfinite(values).masked_fill(~valid, True).all():
        raise ValueError("measurements marked valid must be finite")
    clean = torch.where(valid, values, center)
    return torch.asinh((clean - center) / scale)


def origin_terms(values, mask, center, scale):
    """Return hold, anchored linear trend, local deviations, endpoint mask.

    Only the final 25 samples are used; all earlier prefix values are ignored.
    An invalid endpoint yields zero hold, trend, and local deviations even if
    earlier samples were measured. Invalid samples may contain non-finite data.
    """
    _check_values(values, mask, W)
    center, scale = _normalization(center, scale, values, C)
    recent, available = values[:, -H:], mask[:, -H:]
    z = _coordinates(recent, available, center, scale)
    origin_valid = available[:, -1]
    endpoint = z[:, -1]
    paired = available[:, :-1] & origin_valid[:, None, :]
    delta = (z[:, :-1] - endpoint[:, None, :]).masked_fill(~paired, 0)
    j = torch.arange(-(H - 1), 0, dtype=values.dtype, device=values.device)[None, :, None]
    denominator = (j.square() * paired).sum(dim=1)
    slope = (j * delta).sum(dim=1) / denominator.clamp_min(1)
    hold = endpoint[:, None, :].expand(-1, H, -1)
    future = torch.arange(1, H + 1, dtype=values.dtype, device=values.device)[None, :, None]
    linear = hold + future * slope[:, None, :]
    return hold, linear, delta.transpose(1, 2), origin_valid


def target_terms(raw_truth, truth_mask, origin_valid, center, scale):
    """Normalize only supported future measurements in source coordinates.

    MNM (2) and mode number (10) are excluded from all target support.
    """
    _check_values(raw_truth, truth_mask, H)
    if (
        not isinstance(origin_valid, torch.Tensor)
        or origin_valid.shape != (len(raw_truth), C)
        or origin_valid.dtype != torch.bool
        or origin_valid.device != raw_truth.device
    ):
        raise ValueError("one boolean endpoint mask per channel required")
    center, scale = _normalization(center, scale, raw_truth, C)
    channels = torch.ones(C, dtype=torch.bool, device=raw_truth.device)
    channels[[2, 10]] = False
    valid = truth_mask & origin_valid[:, None, :] & channels
    return _coordinates(raw_truth, valid, center, scale), valid


def row_loss(pred, target, valid):
    """Per-window smooth-L1 losses and their exact level/lag-5 support.

    Both lagged endpoints must be valid. ``supported`` requires nonzero
    counts for both terms; unsupported combined values are zero and must not
    be treated as a fallback observation by downstream training or reporting.
    """
    _check_values(pred, valid, H)
    if not isinstance(target, torch.Tensor) or target.shape != pred.shape:
        raise ValueError("prediction and target shapes must match")
    if target.dtype != pred.dtype or target.device != pred.device:
        raise ValueError("prediction and target dtype/device must match")
    if not torch.isfinite(pred).masked_fill(~valid, True).all() or not torch.isfinite(target).masked_fill(~valid, True).all():
        raise ValueError("supported prediction and target values must be finite")
    pred = torch.where(valid, pred, 0.0)
    target = torch.where(valid, target, 0.0)
    level_points = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    counts_level = valid.sum(dim=(1, 2))
    level = level_points.masked_fill(~valid, 0.0).sum(dim=(1, 2)) / counts_level.clamp_min(1)
    paired = valid[:, LAG:] & valid[:, :-LAG]
    pred_delta = pred[:, LAG:] - pred[:, :-LAG]
    target_delta = target[:, LAG:] - target[:, :-LAG]
    shape_points = F.smooth_l1_loss(pred_delta, target_delta, reduction="none", beta=1.0)
    counts_shape = paired.sum(dim=(1, 2))
    shape = shape_points.masked_fill(~paired, 0.0).sum(dim=(1, 2)) / counts_shape.clamp_min(1)
    supported = (counts_level > 0) & (counts_shape > 0)
    combined = torch.where(supported, (level + shape) / 2, 0.0)
    return dict(level=level, shape=shape, combined=combined, supported=supported,
                counts_level=counts_level, counts_shape=counts_shape)


def features(base, x, a, action=None):
    """Return last-patch/CLS context and the exact native forecast.

    ``base`` must already be frozen. With an action the existing routed path
    owns the first-layer intervention, and gradients remain connected to it.
    All temporary hooks are removed, including when the forward pass raises.
    """
    _check_values(x, a, W)
    if any(p.requires_grad for p in base.parameters()):
        raise ValueError("base parameters must be frozen")
    if action is not None and (
        not isinstance(action, torch.Tensor)
        or action.shape != (len(x), WIDTH)
        or action.dtype != x.dtype
        or action.device != x.device
    ):
        raise ValueError("action must be [B,128] with the input dtype and device")
    inputs, outputs, handles = [], [], []

    def capture_input(_module, args):
        inputs.append(args[0])

    def capture_output(_module, _args, result):
        outputs.append(result["forecast"])

    try:
        handles.append(base.forecast_head.register_forward_pre_hook(capture_input))
        handles.append(base.register_forward_hook(capture_output))
        if action is None:
            base(x, a)
        else:
            delta = torch.cat((action, torch.zeros_like(action)), dim=1)
            policy.routed_pair(base, x, a, latent.routes_all(a), delta)
    finally:
        for handle in handles:
            handle.remove()
    if len(inputs) != 1 or inputs[0].shape != (len(x), 17 * 64):
        raise RuntimeError("forecast head must see exactly one [B,17*64] token tensor")
    if len(outputs) != 1 or outputs[0].shape != (len(x), H, C):
        raise RuntimeError("base must return exactly one [B,25,23] native forecast")
    tokens = inputs[0].reshape(len(x), 17, 64)
    return torch.cat((tokens[:, 15], tokens[:, -1]), dim=1), outputs[0]


class MotionHead(nn.Module):
    """Independent zero-initialized local and standardized context corrections."""

    def __init__(self, mean, scale):
        super().__init__()
        mean = torch.as_tensor(mean)
        if not mean.is_floating_point():
            mean = mean.float()
        mean, scale = _normalization(mean, scale, mean, WIDTH)
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.local_weight = nn.Parameter(mean.new_zeros(C, H - 1, H))
        self.context = nn.Linear(WIDTH, H * C, bias=False, device=mean.device, dtype=mean.dtype)
        nn.init.zeros_(self.context.weight)

    def local(self, hold, local):
        if hold.ndim != 3 or hold.shape[1:] != (H, C) or local.shape != (len(hold), C, H - 1):
            raise ValueError("hold [B,25,23] and local [B,23,24] required")
        return hold + torch.einsum("bck,ckh->bhc", local, self.local_weight)

    def context_correction(self, ctx):
        if ctx.ndim != 2 or ctx.shape[1] != WIDTH:
            raise ValueError("context [B,128] required")
        return self.context((ctx - self.mean) / self.scale).reshape(len(ctx), H, C)

    def forward(self, hold, local, ctx):
        return self.local(hold, local) + self.context_correction(ctx)
