"""Small past-window Transformer with an INTERNAL, resettable fast adapter.

No target-population fit or first-shot calibration phase.  Each prediction uses
source-fitted level information AND statistics of its own already observed
window.  Those local statistics are also the fixed reference for that origin's
direct multi-step forecast; future truth never defines its scale.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 23
    window: int = 128
    patch: int = 8
    dim: int = 64
    heads: int = 4
    layers: int = 2
    rank: int = 8
    horizon: int = 25
    fast_lr: float = 0.2
    fast_radius: float = 1.0
    grad_cap: float = 1.0
    forecast_space: str = 'linear'
    backbone: str = 'patch'

    def __post_init__(self):
        if self.backbone not in ('patch', 'variate'):
            raise ValueError('backbone must be patch or variate')
        if self.forecast_space not in ('linear', 'asinh'):
            raise ValueError('forecast_space must be linear or asinh')
        if min(self.channels, self.window, self.patch, self.dim, self.heads,
               self.rank, self.horizon) < 1 or self.layers < 2:
            raise ValueError('positive dimensions and at least two temporal blocks required')
        if self.window % self.patch or self.dim % self.heads:
            raise ValueError('window/patch and dim/heads must divide exactly')
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.fast_lr, self.fast_radius, self.grad_cap)):
            raise ValueError('positive finite adaptation bounds required')

    def to_dict(self):
        return asdict(self)


class TemporalBlock(nn.Module):
    """Explicit math attention supports the second derivatives used in training.

    Attention is bidirectional WITHIN a past-only input window. There are no
    future tokens or batch statistics. Using explicit matmul also avoids relying
    on a CUDA fused-attention kernel's double-backward support.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, valid):
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-1, -2)) / math.sqrt(dim // self.heads)
        weights = weights.masked_fill(~valid[:, None, None, :], -1e4).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        x = x + self.out(attended)
        return x + self.ff(self.norm2(x))


class RiskAlignedModel(nn.Module):
    """Head 0: J-TEXT max(25,AdvancedTime); head 1: fixed 125 ms.

    Both heads are trained using J-TEXT only before EAST deployment. The second
    head prevents silently relabelling a short-horizon source probability as an
    EAST 125-ms probability. Outputs remain learned risk scores, not guaranteed
    calibrated event probabilities.
    """
    def __init__(self, cfg, center, scale):
        super().__init__()
        self.cfg = cfg
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if center.shape != (cfg.channels,) or scale.shape != center.shape:
            raise ValueError('source scaler shape does not match channel contract')
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('source scaler must be finite with positive scales')
        self.register_buffer('source_center', center)
        self.register_buffer('source_scale', scale)
        # Variate tokens follow iTransformer's time-to-token inversion and
        # shared per-variate direct horizon projection (arXiv:2310.06625).
        # Masks, four causal views, channel identity, risk CLS and fast adapter
        # are project modifications, NOT an exact reproduction of that paper.
        embedding_size = cfg.patch * cfg.channels * 4 if cfg.backbone == 'patch' else cfg.window * 4
        self.patch_embed = nn.Linear(embedding_size, cfg.dim)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        token_count = cfg.window // cfg.patch if cfg.backbone == 'patch' else cfg.channels
        self.position = nn.Parameter(torch.randn(1, token_count + 1, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(TemporalBlock(cfg.dim, cfg.heads) for _ in range(cfg.layers))
        self.adapter_norm = nn.LayerNorm(cfg.dim)
        self.adapter_down = nn.Linear(cfg.dim, cfg.rank)
        self.adapter_up = nn.Linear(cfg.rank, cfg.dim)
        self.fast_init = nn.Parameter(torch.zeros(2 * cfg.rank))
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.risk_head = nn.Linear(cfg.dim, 2)
        forecast_size = cfg.horizon * cfg.channels if cfg.backbone == 'patch' else cfg.horizon
        self.forecast_head = nn.Linear(cfg.dim, forecast_size)
        # Controller inputs are observable loss, target availability and past
        # variation. Its usefulness is trained by L_risk AFTER the inner update.
        self.update_controller = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))

    def initial_fast(self, batch):
        return self.fast_init.unsqueeze(0).expand(batch, -1)

    def observed_features(self, values, available, level_drop=None):
        if values.ndim != 3 or values.shape[1:] != (self.cfg.window, self.cfg.channels):
            raise ValueError('input must be [batch, window, channels]')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('availability must be a matching boolean tensor')
        valid = available & torch.isfinite(values)
        clean = torch.where(valid, values, torch.zeros_like(values))
        count = valid.sum(1)
        mean = clean.sum(1) / count.clamp_min(1)
        variance = (torch.where(valid, clean - mean[:, None], 0.).square().sum(1)
                    / count.clamp_min(1))
        # Causal local reference, not a fitted EAST scaler. The floor prevents
        # constant startup channels from amplifying tiny instrument noise.
        scale = torch.maximum(variance.clamp_min(0.).sqrt(),
                              0.05 * mean.abs() + 0.01 * self.source_scale).clamp_min(1e-8)
        index = torch.arange(self.cfg.window, device=values.device)[None, :, None]
        last_index = torch.where(valid, index, -1).amax(1)
        last = clean.gather(1, last_index.clamp_min(0)[:, None]).squeeze(1)
        level = torch.asinh((clean - self.source_center) / self.source_scale)
        if level_drop is not None:
            if level_drop.shape != mean.shape:
                raise ValueError('level dropout must be [batch, channels]')
            level = level.masked_fill(level_drop[:, None], 0.)
        relative = torch.asinh((clean - mean[:, None]) / scale[:, None])
        delta = torch.zeros_like(clean)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        delta[:, 1:] = torch.where(pair_valid,
                                  torch.asinh((clean[:, 1:] - clean[:, :-1]) / scale[:, None]), 0.)
        parts = [torch.where(valid, part, 0.) for part in (level, relative, delta)]
        features = torch.cat([*parts, valid.to(values.dtype)], -1)
        variation = delta.abs().tanh().sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        return features, valid, last.detach(), scale.detach(), count > 0, variation.detach()

    def forward(self, values, available, *, fast=None, level_drop=None):
        features, valid, last, scale, channel_valid, variation = self.observed_features(
            values, available, level_drop)
        batch = values.shape[0]
        if self.cfg.backbone == 'patch':
            tokens = features.reshape(batch, self.cfg.window // self.cfg.patch, -1)
            token_valid = valid.reshape(batch, self.cfg.window // self.cfg.patch, -1).any(-1)
        else:
            # Each token contains one channel's observed history, never the
            # future forecast target. Feature concatenation order is view,C.
            tokens = features.reshape(batch, self.cfg.window, 4, self.cfg.channels)
            tokens = tokens.permute(0, 3, 1, 2).reshape(batch, self.cfg.channels, -1)
            token_valid = channel_valid
        tokens = self.patch_embed(tokens)
        tokens = torch.cat([tokens, self.cls.expand(batch, -1, -1)], 1) + self.position
        token_valid = torch.cat([token_valid, torch.ones(batch, 1, dtype=torch.bool, device=values.device)], 1)
        tokens = self.blocks[0](tokens, token_valid)
        fast = self.initial_fast(batch) if fast is None else fast
        if fast.shape != (batch, 2 * self.cfg.rank):
            raise ValueError('one fast vector per independent episode is required')
        gain, bias = fast.chunk(2, -1)
        hidden = F.gelu(self.adapter_down(self.adapter_norm(tokens)))
        tokens = tokens + self.adapter_up(hidden * (1. + gain[:, None].tanh()) + bias[:, None].tanh())
        for block in self.blocks[1:]:
            tokens = block(tokens, token_valid)
        state = self.final_norm(tokens[:, -1])
        if self.cfg.backbone == 'patch':
            forecast = self.forecast_head(state).reshape(batch, self.cfg.horizon, self.cfg.channels)
        else:
            forecast = self.forecast_head(self.final_norm(tokens[:, :-1])).transpose(1, 2)
        return dict(logits=self.risk_head(state), state=state,
                    forecast=forecast,
                    forecast_space=self.cfg.forecast_space,
                    reference_last=last, reference_scale=scale,
                    channel_valid=channel_valid, variation=variation)

    def persistent_parameters(self):
        """Small supervised update scope; the source temporal weights stay fixed."""
        yield self.fast_init
        for module in (self.risk_head, self.forecast_head, self.update_controller):
            yield from module.parameters()
