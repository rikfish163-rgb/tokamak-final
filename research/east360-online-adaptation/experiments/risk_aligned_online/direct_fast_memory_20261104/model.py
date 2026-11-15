"""A no-bias map from actual fast-weight innovation to existing token adapters."""

import torch
from torch import nn


class Injection(nn.Linear):
    def __init__(self):
        super().__init__(168, 128, bias=False)
        nn.init.zeros_(self.weight)


def levels(base, values, available):
    if values.shape != available.shape or available.dtype != torch.bool:
        raise ValueError("matching diagnostic values and boolean masks required")
    if not torch.isfinite(values[available]).all():
        raise ValueError("observed diagnostics must be finite")
    clean = torch.where(available, values, base.center)
    return torch.asinh((clean - base.center) / base.scale).masked_fill(~available, 0)


def predict(base, injection, x, a, innovation):
    if base.cfg.fast_site != "tokens" or base.cfg.dim != 64:
        raise ValueError("qualified token adaptation site required")
    if any(p.requires_grad for p in base.parameters()) or base.training:
        raise ValueError("source must remain frozen in evaluation mode")
    if innovation.shape != (len(x), 168) or not torch.isfinite(innovation).all():
        raise ValueError("finite batch-matching direct innovation required")
    fast = base.initial_fast(len(x)) + injection(innovation)
    return base(x, a, fast=fast)["logit"]
