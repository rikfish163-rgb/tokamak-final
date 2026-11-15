"""Source-native first-FFN low-rank data path; source tensors stay unchanged."""

import math

import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from ..dual_history_ttt_20261101.model import _check_model, encode_views, latent_forward


@torch.no_grad()
def source_basis(model):
    _check_model(model)
    weight = model.base.encoder.layers[0].linear2.weight.detach().cpu().double()
    if weight.shape != (64, 128):
        raise ValueError("source first FFN64x128 output matrix required")
    _, _, vh = torch.linalg.svd(weight, full_matrices=False)
    a = vh[:4].clone()
    pivot = a.abs().argmax(1)
    a *= a[torch.arange(4), pivot].sign()[:, None]
    return a.float().to(model.base.center.device)


@torch.no_grad()
def encode(model, signal, ends, basis):
    _check_model(model)
    if basis.shape != (4, 128) or not torch.isfinite(basis).all():
        raise ValueError("finite source-native4x128 basis required")
    captured = []

    def capture(_module, args):
        captured.append(args[0].detach())

    hook = model.base.encoder.layers[0].linear2.register_forward_pre_hook(capture)
    try:
        pack = encode_views(model, signal, ends)
    finally:
        hook.remove()
    if len(captured) != 3:
        raise RuntimeError("short/long/snapshot must each expose one source FFN input")
    phi = captured[0][:, :16] @ basis.T / math.sqrt(4)
    phi = phi.masked_fill(~pack["valid"][:, :16, None], 0)
    return dict(pack, phi=phi)


def cached_risk(model, pack, delta):
    if delta.shape != (len(pack["h"]), 16, 64) or not torch.isfinite(delta).all():
        raise ValueError("one finite data-token shift per endpoint required")
    delta = delta.masked_fill(~pack["valid"][:, :16, None], 0)
    h = torch.cat((pack["h"][:, :16] + delta, pack["h"][:, -1:]), 1)
    native = {k: pack[k] for k in ("valid", "long", "snapshot")}
    native["h"] = h
    return latent_forward(model, native, h.new_zeros(16, 2))["logit"]


def direct_lora(model, signal, ends, basis, state, enabled):
    """Actual layer-output intervention, not a correction to the risk score.

    The source layer's linear2 output receives its low-rank delta before the
    residual addition. Cache replay adds after that addition; rounding order
    can differ, so witnesses must be certified through this direct path.
    """
    _check_model(model)
    if (
        basis.shape != (4, 128)
        or state.shape != (4, 64)
        or not torch.isfinite(basis).all()
        or not torch.isfinite(state).all()
    ):
        raise ValueError("finite rank4 source basis and temporary map required")
    x, available = batch(signal, ends, model.base.center.device)
    if (
        enabled.shape != (len(ends),)
        or enabled.dtype != torch.bool
        or enabled.device != x.device
    ):
        raise ValueError("explicit mutable-endpoint mask required")
    valid_data = (available & torch.isfinite(x)).reshape(len(x), 16, -1).any(-1)
    mutable = valid_data & enabled[:, None]

    def insert(_module, args, output):
        phi = args[0][:, :16] @ basis.T / math.sqrt(4)
        delta = (phi @ state).masked_fill(~mutable[:, :, None], 0)
        return output + torch.cat((delta, output.new_zeros(len(output), 1, 64)), 1)

    hook = model.base.encoder.layers[0].linear2.register_forward_hook(insert)
    try:
        h, valid = model.tokens(x, available)
    finally:
        hook.remove()
    logit = model.risk(h, valid, h.new_zeros(len(h), 92))
    if not torch.isfinite(logit).all():
        raise FloatingPointError("nonfinite direct first-FFN adaptation")
    return dict(logit=logit, h=h, valid=valid)
