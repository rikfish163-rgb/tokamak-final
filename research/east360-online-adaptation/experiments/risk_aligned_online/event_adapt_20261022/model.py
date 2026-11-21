"""One causal encoder, one risk head that reads predicted diagnostic evolution.

This is an experimental architecture, not a validated physics model. Forecasts
are direct multi-step predictions; none are fed back as observed measurements.
The only fast parameters are an internal low-rank temporal-block matrix.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    dim: int = 48
    rank: int = 8
    horizon: int = 32
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64)
    risk_reads_forecast: bool = True

    def __post_init__(self):
        if min(self.dim, self.rank, self.horizon, *self.dilations) < 1:
            raise ValueError('positive model dimensions required')

    @property
    def window(self):
        # Each causal width-3 convolution adds 2*dilation. First differences
        # require one additional observed point beyond the receptive field.
        return max(2*sum(self.dilations)+2, self.horizon+1)

    def to_dict(self):
        return asdict(self)


class TemporalBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, dilation=dilation)
        self.mix = nn.Linear(dim, dim)

    def forward(self, h, valid):
        x = self.norm(h).masked_fill(~valid[..., None], 0.)
        x = self.conv(F.pad(x.transpose(1, 2), (2*self.dilation, 0))).transpose(1, 2)
        return (h+self.mix(F.silu(x))).masked_fill(~valid[..., None], 0.)


class EventForecastModel(nn.Module):
    def __init__(self, cfg, center, scale, change_scale, forecast_active):
        super().__init__()
        self.cfg = cfg
        for name, value in [('center', center), ('scale', scale), ('change_scale', change_scale)]:
            value = torch.as_tensor(value, dtype=torch.float32)
            if value.shape != (23,) or not torch.isfinite(value).all():
                raise ValueError('23 finite source coordinates required')
            if name != 'center' and not (value > 0).all():
                raise ValueError('positive coordinate scales required')
            self.register_buffer(name, value)
        active = torch.as_tensor(forecast_active, dtype=torch.bool)
        if active.shape != (23,):
            raise ValueError('23 forecast activity flags required')
        self.register_buffer('forecast_active', active)
        self.embedding = nn.Linear(69, cfg.dim)
        self.blocks = nn.ModuleList(TemporalBlock(cfg.dim, d) for d in cfg.dilations)
        self.fast_down = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.fast_origin = nn.Parameter(torch.zeros(cfg.rank, cfg.dim))
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.forecast_head = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(),
                                           nn.Linear(cfg.dim, cfg.horizon*23))
        self.trajectory_embedding = nn.Sequential(nn.Linear(cfg.horizon*23, cfg.dim), nn.SiLU())
        self.risk_head = nn.Sequential(nn.Linear(2*cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, 1))
        self.event_head = nn.Linear(cfg.dim, 6)

    def initial_fast(self, batch):
        return self.fast_origin.unsqueeze(0).expand(batch, -1, -1).clone()

    def coordinates(self, values, available):
        valid = available & torch.isfinite(values)
        clean = torch.where(valid, values, self.center)
        return torch.asinh((clean-self.center)/self.scale).masked_fill(~valid, 0.), valid

    def sequence(self, values, available, fast=None):
        if values.ndim != 3 or values.shape[-1] != 23 or not values.shape[1]:
            raise ValueError('nonempty [batch,time,23] causal inputs required')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('matching boolean availability required')
        z, mask = self.coordinates(values, available)
        valid = mask.any(-1)
        delta = torch.zeros_like(z)
        delta[:, 1:] = (z[:, 1:]-z[:, :-1]).masked_fill(~(mask[:, 1:] & mask[:, :-1]), 0.)
        h = self.embedding(torch.cat((z, delta, mask.float()), -1)).masked_fill(~valid[..., None], 0.)
        fast = self.initial_fast(len(values)) if fast is None else fast
        if fast.shape != (len(values), self.cfg.rank, self.cfg.dim):
            raise ValueError('one internal temporary matrix per shot required')
        for i, block in enumerate(self.blocks):
            # Fast update is INSIDE the final temporal block, before its
            # convolution/nonlinearity; it changes both forecast and risk paths.
            if i == len(self.blocks)-1:
                h = h+torch.bmm(F.silu(self.fast_down(h)), fast)/math.sqrt(self.cfg.rank)
                h = h.masked_fill(~valid[..., None], 0.)
            h = block(h, valid)
        h = self.output_norm(h).masked_fill(~valid[..., None], 0.)
        future = self.forecast_head(h).reshape(len(h), h.shape[1], self.cfg.horizon, 23)
        if self.cfg.risk_reads_forecast:
            trajectory = future
        else:
            # Same trainable readout capacity, but actual past evolution in
            # place of forecast evolution. Retrain this control from source.
            history = F.pad(z.transpose(1, 2), (self.cfg.horizon-1, 0)).unfold(2, self.cfg.horizon, 1)
            history_mask = F.pad(mask.transpose(1, 2), (self.cfg.horizon-1, 0)).unfold(2, self.cfg.horizon, 1)
            trajectory = (history.permute(0, 2, 3, 1)-z[:, :, None])/self.change_scale
            trajectory = trajectory.masked_fill(~history_mask.permute(0, 2, 3, 1), 0.)
        # A diagnostic forecast must have an observed origin and a supervised
        # continuous target. Otherwise unconstrained output slots (e.g. a mode
        # ID or a missing channel) can encode risk without predicting signals.
        usable_trajectory = mask & self.forecast_active[None, None]
        trajectory = trajectory.masked_fill(~usable_trajectory[:, :, None], 0.)
        evolution = self.trajectory_embedding(trajectory.flatten(-2))
        logit = self.risk_head(torch.cat((h, evolution), -1)).squeeze(-1)
        return dict(logit=logit, forecast=future, event=self.event_head(h))

    def forward(self, values, available, fast=None):
        return {k: v[:, -1] for k, v in self.sequence(values, available, fast).items()}


def event_loss(logits, target):
    known = target >= 0
    return F.cross_entropy(logits[known], target[known]) if known.any() else logits.sum()*0.


def arrived_loss(model, prediction, origin, origin_mask, truth, truth_mask):
    """One loss per shot; caller supplies ONLY already-arrived target entries."""
    if prediction.shape != truth.shape or truth_mask.shape != truth.shape:
        raise ValueError('direct forecast and arrived targets must match')
    start, a = model.coordinates(origin[:, None], origin_mask[:, None])
    end, b = model.coordinates(truth, truth_mask)
    mask = b & a & model.forecast_active[None, None]
    target = (end-start)/model.change_scale
    point_loss = F.smooth_l1_loss(prediction, target.detach(), reduction='none')
    numerator = point_loss.masked_fill(~mask, 0.).sum((1, 2))
    count = mask.sum((1, 2))
    return numerator/count.clamp_min(1), count


def update_fast(model, fast, x, mask, truth, truth_mask, *, lr, radius, create_graph=False):
    """Exact inner operator shared by source meta preparation and EAST use.

    Fresh differentiable forward on stored past inputs, never backprop through
    detached predictions or overwrite an earlier issued forecast. No optimizer
    state or fast matrix is persisted across shots by this function.
    """
    if lr <= 0 or radius <= 0 or not fast.requires_grad:
        raise ValueError('positive learning rate/radius and differentiable fast state required')
    output = model(x, mask, fast)
    losses, counts = arrived_loss(model, output['forecast'], x[:, -1], mask[:, -1], truth, truth_mask)
    if not (counts > 0).any():
        return fast, dict(updated=False, loss=0., observations=0, gradient_norm=0.)
    if not torch.isfinite(losses).all():
        raise FloatingPointError('nonfinite mature diagnostic loss')
    grad = torch.autograd.grad(losses.sum(), fast, create_graph=create_graph)[0]
    norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
    if not torch.isfinite(norm).all():
        raise FloatingPointError('nonfinite fast gradient')
    # Per-shot clipping: one episode must not alter another episode's update.
    candidate = fast-lr*grad/norm.clamp_min(1.)[:, None, None]
    origin = model.initial_fast(len(fast))
    difference = candidate-origin
    distance = difference.flatten(1).norm(dim=1).clamp_min(1e-12)
    candidate = origin+difference*(radius/distance).clamp_max(1.)[:, None, None]
    if not create_graph:
        candidate = candidate.detach().requires_grad_(True)
    return candidate, dict(updated=True, loss=float(losses.mean().detach()),
                           observations=int(counts.sum()), gradient_norm=float(norm.mean().detach()))
