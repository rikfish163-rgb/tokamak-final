"""Padding-aware bounded embedding repair for the frozen source Simple23.

The repair is an internal representation, not a measured or restored signal.
Only the supplied prefix and its true availability participate; this module
does not read data, labels, or metadata, nor does it own the source model.
"""

import torch
from torch import nn

from ..native_token_probe_20261104.core import _check_source


class TokenRepair(nn.Module):
    """Map original 64-dimensional embeddings and two availabilities to 64."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(66), nn.Linear(66, 16), nn.SiLU(), nn.Linear(16, 64))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        return self.net(context)


def _check(base, repair, x, available, enabled):
    _check_source(base)
    if (
        base.cfg.heads != 4
        or not isinstance(base.embedding, nn.Linear)
        or (base.embedding.in_features, base.embedding.out_features) != (368, 64)
        or not isinstance(base.risk_head, nn.Linear)
        or (base.risk_head.in_features, base.risk_head.out_features) != (64, 1)
    ):
        raise ValueError("the native source embedding, attention and risk head are required")
    if not isinstance(repair, TokenRepair) or type(enabled) is not bool:
        raise TypeError("TokenRepair and boolean enabled are required")
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or x.shape[1:] != (128, 23)
        or not len(x)
        or x.dtype != base.center.dtype
        or x.device != base.center.device
        or not isinstance(available, torch.Tensor)
        or available.shape != x.shape
        or available.dtype != torch.bool
        or available.device != x.device
    ):
        raise ValueError("matching source Bx128x23 values and boolean availability required")
    parameters = list(repair.parameters())
    if any(p.device != x.device or p.dtype != x.dtype for p in parameters):
        raise ValueError("repair parameters must match source device and dtype")
    if not torch.isfinite(x[available]).all() or any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("observed values and repair parameters must be finite")
    source_storage = {t.untyped_storage().data_ptr() for t in (*base.parameters(), *base.buffers())}
    if any(p.untyped_storage().data_ptr() in source_storage for p in parameters):
        raise ValueError("repair and source storage must not be shared")


def forward(base, repair, x, a, enabled=True):
    """Return native outputs and the per-token intervention for one prefix.

    ``missing_fraction`` is Bx16x2 in CIII/MNM order. It is zero for an
    all-padding patch; ``repair_tokens`` marks eligible tokens when enabled.
    Network inputs use *availability* fractions (not missing fractions).
    The no-op path returns the identical original embedding without an add.
    """
    _check(base, repair, x, a, enabled)
    grouped = a.reshape(len(x), 16, 8, 23)
    valid = grouped.any(-1)
    count = valid.sum(-1)
    fractions = grouped[..., (1, 2)].sum(-2).to(x.dtype) / count.clamp_min(1)[..., None]
    missing = torch.where(count[..., None] > 0, 1 - fractions, torch.zeros_like(fractions))
    eligible = (count > 0) & (missing > 0).any(-1)
    tokens = eligible if enabled else torch.zeros_like(eligible)
    captured = []

    def intervention(_module, _inputs, embedding):
        if captured or embedding.shape != (len(x), 16, 64) or not torch.isfinite(embedding).all():
            raise RuntimeError("one finite source Bx16x64 embedding is required")
        if not tokens.any():
            captured.append((embedding, embedding, torch.zeros_like(embedding)))
            return None
        context = torch.cat((embedding, fractions), dim=-1)[tokens]
        raw = repair(context)
        if raw.shape != (int(tokens.sum()), 64) or not torch.isfinite(raw).all():
            raise FloatingPointError("repair must produce finite 64-dimensional token corrections")
        # Clamp before sqrt: unlike norm at zero, this has finite derivatives at
        # zero initialization. Float64 avoids overflow in the squared norm.
        rms2 = raw.double().square().mean(-1, keepdim=True)
        # Leave a few dtype ulps of headroom only when clipping: otherwise the
        # rounded product can have a float64-measured RMS slightly above .25.
        ceiling = 0.25 * (1 - 4 * torch.finfo(raw.dtype).eps)
        factor = torch.where(
            rms2 > 0.25**2, ceiling / rms2.clamp_min(0.25**2).sqrt(), 1.0
        )
        # Keep the factor and product in float64: casting a very small factor
        # to float32 first can round it up enough to exceed the RMS bound.
        bounded = (raw.double() * factor).to(raw.dtype)
        delta = torch.zeros_like(embedding)
        delta[tokens] = bounded
        after = torch.where(tokens[..., None], embedding + delta, embedding)
        captured.append((embedding, after, delta))
        return after

    hook = base.embedding.register_forward_hook(intervention)
    try:
        result = base(x, a)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("exactly one source embedding call is required")
    if not torch.isfinite(result["logit"]).all() or not torch.isfinite(result["forecast"]).all():
        raise FloatingPointError("nonfinite source logit or forecast")
    before, after, delta = captured[0]
    return {
        "logit": result["logit"],
        "forecast": result["forecast"],
        "embedding_before": before,
        "embedding_after": after,
        "embedding_delta": delta,
        "missing_fraction": missing,
        "repair_tokens": tokens,
    }
