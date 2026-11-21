"""Observed-pair loss into temporary first-FFN parameters, never risk patches."""

import torch
from torch.nn import functional as F

from ..lora_stream_20261102.core import cap_shift, finish


def data_tokens(pack, state):
    if state.shape not in ((4, 64), (len(pack["phi"]), 4, 64)):
        raise ValueError("global or independent per-episode first-FFN4x64 states")
    shift, rms = cap_shift(pack["phi"] @ state, pack["valid"][:, :16])
    return pack["residual"] + (pack["ffn"] + shift), rms


def representation(pack, state):
    data, rms = data_tokens(pack, state)
    valid = pack["valid"][:, :16]
    count = valid.sum(1)
    mean = data.masked_fill(~valid[:, :, None], 0).sum(1) / count.clamp_min(1)[:, None]
    indices = torch.arange(16, device=data.device)[None].expand(len(data), -1)
    last_index = indices.masked_fill(~valid, -1).max(1).values.clamp_min(0)
    last = data[torch.arange(len(data), device=data.device), last_index]
    value = torch.cat((mean, last), 1).masked_fill((count == 0)[:, None], 0)
    return value, count > 0, rms


def risk(model, pack, state):
    data, rms = data_tokens(pack, state)
    h = torch.cat((data, pack["h"][:, -1:]), 1)
    return finish(model, h, pack["valid"], forecast=False)["logit"], rms


def order_step(origin, mature, teacher, std, *, reverse=False):
    """One loss-gradient step per episode, with no label or acceptance oracle.

    At zero state the normalized direction is algebraically independent of
    softplus confidence. Log that limitation; do not call this error-dependent
    step sizing or use same-pair loss reduction as evidence of risk benefit.
    """
    n = len(origin["phi"])
    if teacher.shape != (128,) or std.shape != (128,) or not (std > 0).all():
        raise ValueError("fixed128 teacher and positive frozen TRAIN scales required")
    with torch.enable_grad():
        state = origin["phi"].new_zeros(n, 4, 64).requires_grad_(True)
        a, va, _ = representation(origin, state)
        b, vb, _ = representation(mature, state)
        eligible = va & vb
        margin = (((b - a) / std) * teacher).sum(1)
        before = F.softplus(margin if reverse else -margin)
        gradient = torch.autograd.grad((before * eligible).sum(), state)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite temporal-order gradient")
    norm = gradient.double().flatten(1).norm(dim=1)
    denominator = norm.clamp_min(torch.finfo(torch.float64).tiny)
    after = (-0.1 * gradient.double() / denominator[:, None, None]).to(state.dtype)
    after = after.masked_fill((~eligible | (norm == 0))[:, None, None], 0).detach()
    with torch.no_grad():
        a1, _, ra = representation(origin, after)
        b1, _, rb = representation(mature, after)
        margin1 = (((b1 - a1) / std) * teacher).sum(1)
        loss1 = F.softplus(margin1 if reverse else -margin1)
    return after, dict(
        eligible=eligible.cpu().tolist(),
        gradient_norm=norm.cpu().tolist(),
        state_norm=after.double().flatten(1).norm(dim=1).cpu().tolist(),
        margin_before=margin.detach().cpu().tolist(),
        margin_after=margin1.cpu().tolist(),
        loss_before=before.detach().cpu().tolist(),
        loss_after=loss1.cpu().tolist(),
        max_pair_shift_rms=torch.maximum(ra, rb).cpu().tolist(),
        zero_gradient=(norm == 0).cpu().tolist(),
    )
