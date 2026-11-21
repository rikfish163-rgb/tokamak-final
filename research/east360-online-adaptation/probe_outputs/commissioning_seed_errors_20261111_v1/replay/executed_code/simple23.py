"""Single encoder, single risk head, optional direct signal forecast.

Temporal patching is literature-inspired (PatchTST, arXiv:2211.14730), but this
multichannel classification model is not the channel-independent paper model.
No shared/private split, experts, auxiliary risk horizon or learned controller.
"""
from dataclasses import dataclass, asdict, replace
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


@dataclass(frozen=True)
class Config:
    window: int = 128
    patch: int = 8
    dim: int = 64
    heads: int = 4
    layers: int = 2
    horizon: int = 25
    fast_site: str = 'output'
    fast_rule: str = 'raw'
    input_view: str = 'source'

    def __post_init__(self):
        if self.fast_rule not in ('raw','normalized'):
            raise ValueError('unknown temporary update rule')
        if self.input_view not in ('source', 'window', 'hybrid', 'residual_window'):
            raise ValueError('unknown input_view')
        if self.fast_site not in ('output', 'tokens'):
            raise ValueError('fast_site must be output or tokens')
        if min(self.window, self.patch, self.dim, self.heads, self.layers, self.horizon) < 1:
            raise ValueError('positive dimensions required')
        if self.window % self.patch or self.dim % self.heads:
            raise ValueError('window/patch and dim/heads must divide')

    def to_dict(self):
        return asdict(self)


class Simple23(nn.Module):
    def __init__(self, cfg, center, scale):
        super().__init__()
        self.cfg = cfg
        center, scale = torch.as_tensor(center).float(), torch.as_tensor(scale).float()
        if center.shape != (23,) or scale.shape != (23,):
            raise ValueError('exactly 23 diagnostic channels required')
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('finite source normalization required')
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)
        self.embedding = nn.Linear(cfg.patch * (69 if cfg.input_view == 'hybrid' else 46), cfg.dim)
        tokens = cfg.window // cfg.patch
        self.position = nn.Parameter(torch.randn(1, tokens + 1, cfg.dim) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        block = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 2 * cfg.dim,
                                          dropout=0., activation='gelu',
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)
        self.fast_origin = nn.Parameter(torch.zeros(2 * cfg.dim))
        self.risk_head = nn.Linear(cfg.dim, 1)
        self.forecast_head = nn.Linear((tokens + 1) * cfg.dim, cfg.horizon * 23)
        if cfg.input_view == 'residual_window':
            self.relative_embedding = nn.Linear(cfg.patch*23, cfg.dim, bias=False)
            nn.init.zeros_(self.relative_embedding.weight)

    def initial_fast(self, batch):
        return self.fast_origin.unsqueeze(0).expand(batch, -1).clone()

    def forward(self, values, available, fast=None):
        if values.ndim != 3 or values.shape[1:] != (self.cfg.window, 23):
            raise ValueError('values must be [batch, past window, 23]')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('matching boolean availability required')
        available = available & torch.isfinite(values)
        # No per-shot full-record statistics. Window mode sees only the causal
        # input supplied for this endpoint, and ignores padding/missing values.
        clean = torch.where(available, values, self.center)
        if self.cfg.input_view in ('window', 'hybrid', 'residual_window'):
            observed = torch.where(available, values, 0.).double().detach()
            count = available.sum(1, keepdim=True).clamp_min(1)
            local_center = observed.sum(1, keepdim=True) / count
            centered = (observed-local_center).masked_fill(~available, 0.)
            variance = centered.square().sum(1, keepdim=True) / count
            # Exact constants have no scale information and map to zero. A
            # unitful epsilon would erase tiny yet valid qa_proxy variation.
            # Float64 statistics prevent float32 cancellation on constant data.
            local_scale = torch.where(variance > 0, variance.sqrt(), 1.)
            local_center = local_center.to(values.dtype)
            local_scale = local_scale.to(values.dtype)
            level = torch.asinh((clean-local_center) / local_scale).masked_fill(~available, 0.)
        else:
            level = torch.asinh((clean - self.center) / self.scale)
        if self.cfg.input_view == 'hybrid':
            absolute = torch.asinh((clean-self.center) / self.scale)
            x = torch.cat((absolute, level, available.to(level.dtype)), -1)
        elif self.cfg.input_view == 'residual_window':
            absolute = torch.asinh((clean-self.center) / self.scale)
            x = torch.cat((absolute, available.to(level.dtype)), -1)
        else:
            x = torch.cat((level, available.to(level.dtype)), -1)
        batch = len(values)
        x = self.embedding(x.reshape(batch, self.cfg.window // self.cfg.patch, -1))
        if self.cfg.input_view == 'residual_window':
            x = x + self.relative_embedding(level.reshape(batch, self.cfg.window // self.cfg.patch, -1))
        fast = self.initial_fast(batch) if fast is None else fast
        if fast.shape != (batch, 2 * self.cfg.dim):
            raise ValueError('one temporary state per shot required')
        if self.cfg.fast_site == 'tokens':
            # Center at the learned source origin: a fresh shot is identical
            # to its parent. Only observed patch tokens change, not CLS.
            dg, db = (fast-self.initial_fast(batch)).chunk(2, -1)
            x = x * (1 + .1 * dg[:, None].tanh()) + .1 * db[:, None].tanh()
        valid = available.reshape(batch, self.cfg.window // self.cfg.patch, -1).any(-1)
        x = torch.cat((x, self.cls.expand(batch, -1, -1)), 1) + self.position
        valid = torch.cat((valid, torch.ones(batch, 1, dtype=torch.bool, device=x.device)), 1)
        # Meta-training through attention needs double backward. Explicit math
        # SDPA supports it; fused attention does not in the local torch runtime.
        # This is the tested token-adaptation backend, not an error bypass.
        context = sdpa_kernel(SDPBackend.MATH) if self.cfg.fast_site == 'tokens' else nullcontext()
        with context:
            x = self.encoder(x, src_key_padding_mask=~valid)
        output_fast = self.initial_fast(batch) if self.cfg.fast_site == 'tokens' else fast
        gain, bias = output_fast.chunk(2, -1)
        # Small feature-wise corrections affect BOTH tasks; not only an alarm
        # threshold or a signal-prediction output correction.
        x = self.norm(x) * (1 + .1 * gain[:, None].tanh()) + .1 * bias[:, None].tanh()
        x = x.masked_fill(~valid[:, :, None], 0.)
        forecast = self.forecast_head(x.flatten(1)).reshape(batch, self.cfg.horizon, 23)
        if self.cfg.input_view in ('window', 'hybrid'):
            # Predict relative future values, reconstruct physical-file units
            # using PAST statistics, then use the established forecast-loss
            # coordinate. Never normalize targets using their future moments.
            future = forecast * local_scale + local_center
            forecast = torch.asinh((future-self.center) / self.scale)
        return dict(logit=self.risk_head(x[:, -1]).squeeze(-1), forecast=forecast)

    def forecast_loss(self, prediction, truth, available):
        if truth.shape != prediction.shape or available.shape != truth.shape or available.dtype != torch.bool:
            raise ValueError('forecast targets and masks must match')
        available = available & torch.isfinite(truth)
        truth = torch.where(available, truth.detach(), self.center)
        target = torch.asinh((truth - self.center) / self.scale)
        loss = F.smooth_l1_loss(prediction, target, reduction='none')
        return loss.masked_fill(~available, 0.).sum() / available.sum().clamp_min(1)


def add_relative_input(parent):
    """Zero-init extension of a source-view parent; never reinterpret its head."""
    if parent.cfg.input_view != 'source':
        raise ValueError('relative extension requires source-view parent')
    model = Simple23(replace(parent.cfg, input_view='residual_window'),
                     parent.center.detach().clone(), parent.scale.detach().clone())
    state = model.load_state_dict(parent.state_dict(), strict=False)
    if state.missing_keys != ['relative_embedding.weight'] or state.unexpected_keys:
        raise ValueError('unexpected parent state contract')
    return model.to(parent.center.device)


def fit_forecaster(model, optimizer, x, mask, truth, truth_mask):
    """Fit only the auxiliary head after source classifier selection.

    This stage cannot change source risk predictions. Its existence does not
    establish that subsequent signal-based temporary updates help warnings.
    """
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    # Only forecast-head parameters receive gradients; detach the rest of the
    # graph using a temporary, restored requires_grad scope.
    states = [(p, p.requires_grad) for p in model.parameters()]
    head = {id(p) for p in model.forecast_head.parameters()}
    try:
        for p, _ in states:
            p.requires_grad_(id(p) in head)
        out = model(x, mask)
        loss = model.forecast_loss(out['forecast'], truth, truth_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite forecast loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    finally:
        for p, enabled in states:
            p.requires_grad_(enabled)
    return float(loss.detach())
