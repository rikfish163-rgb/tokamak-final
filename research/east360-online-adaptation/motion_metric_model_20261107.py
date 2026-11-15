"""Frozen native source with a batched, shot-local rank4 first-FFN update."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.risk_aligned_online.lora_stream_20261102.core import cap_shift


@torch.no_grad()
def basis_for(base):
    _, _, vh = torch.linalg.svd(base.encoder.layers[0].linear2.weight.double().cpu(), full_matrices=False)
    basis = vh[:4].clone()
    pivot = basis.abs().argmax(1)
    basis *= basis[torch.arange(4), pivot].sign()[:, None]
    return basis.to(base.center)


@torch.no_grad()
def encode_windows(base, x, available, basis):
    if base.training or any(p.requires_grad for p in base.parameters()):
        raise ValueError("qualified source must remain frozen and eval")
    captured = {}
    layer = base.encoder.layers[0]
    def residual(_m, args):
        captured["residual"] = args[0][:, :16].detach()
    def ffn(_m, args, output):
        captured["ffn"] = output[:, :16].detach()
        captured["phi"] = (args[0][:, :16] @ basis.T / 2).detach()
    def final(_m, _args, output):
        captured["h"] = output.detach()
    def risk_state(_m, args):
        captured["risk_hidden"] = args[0].detach()
    handles = [layer.norm2.register_forward_pre_hook(residual),
               layer.linear2.register_forward_hook(ffn), layer.register_forward_hook(final),
               base.risk_head.register_forward_pre_hook(risk_state)]
    try:
        native = base(x, available)
    finally:
        for handle in handles:
            handle.remove()
    valid = (available & torch.isfinite(x)).reshape(len(x), 16, -1).any(-1)
    captured["valid"] = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
    captured["native"] = native["logit"].detach()
    return captured


def forward_state(base, pack, state):
    if state.shape != (len(pack["h"]), 4, 64):
        raise ValueError("one temporary4x64 FFN parameter matrix per window required")
    shift, rms = cap_shift(pack["phi"] @ state, pack["valid"][:, :16])
    data = pack["residual"] + (pack["ffn"] + shift)
    sequence = torch.cat((data, pack["h"][:, -1:]), 1)
    with sdpa_kernel(SDPBackend.MATH):
        sequence = base.encoder.layers[1](sequence, src_key_padding_mask=~pack["valid"])
    gain, bias = base.initial_fast(len(sequence)).chunk(2, -1)
    sequence = base.norm(sequence) * (1 + .1 * gain[:, None].tanh()) + .1 * bias[:, None].tanh()
    return dict(logit=base.risk_head(sequence[:, -1]).squeeze(-1),
                forecast=base.forecast_head(sequence.masked_fill(~pack["valid"][:, :, None], 0).flatten(1))
                             .reshape(len(sequence), 25, 23), rms=rms)


@torch.no_grad()
def motion_target(base, x, available, future, future_available):
    valid = future_available & available[:, -1:, :]
    valid = valid.clone()
    valid[:, :, [2, 10]] = False
    previous = torch.asinh((torch.where(available[:, -1:], x[:, -1:], base.center) - base.center) / base.scale)
    arrived = torch.asinh((torch.where(future_available, future, base.center) - base.center) / base.scale)
    return (arrived - previous).masked_fill(~valid, 0), valid


def scramble(motion, mask, *, axis):
    """Cyclically reassign observed entries without changing the validity mask."""
    if axis not in ("time", "channel"):
        raise ValueError("declared time or channel control required")
    value, observed = (v.transpose(1, 2) if axis == "time" else v for v in (motion, mask))
    shift = 7 if axis == "time" else 1
    size = value.shape[-1]
    order = torch.argsort((~observed).to(torch.int8), dim=-1, stable=True)
    sorted_value = torch.gather(value, -1, order)
    count = observed.sum(-1, keepdim=True)
    rank = torch.arange(size, device=motion.device)
    origin = (rank + shift) % count.clamp_min(1)
    changed = torch.gather(sorted_value, -1, origin)
    result = torch.zeros_like(value).scatter(-1, order, changed).masked_fill(~observed, 0)
    return result.transpose(1, 2) if axis == "time" else result
