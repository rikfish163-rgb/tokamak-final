"""Independent small causal TCN source baseline for the flat23 contract.

The encoder follows the causal, dilated residual-convolution idea from Bai et
al. (arXiv:1803.01271), but is a project-specific 23-channel baseline rather
than an exact reproduction.  It deliberately has no dependency on Simple23.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import io23


CHECKPOINT_FORMAT = 'tcn23-v1'
CHANNELS = 23
INPUT_FEATURES = 2 * CHANNELS
DILATIONS = (1, 2, 4, 8, 16)
KERNEL_SIZE = 3


@dataclass(frozen=True)
class TCNConfig:
    window: int = 128
    dim: int = 64
    horizon: int = 25
    fast_site: str = 'tokens'
    fast_rule: str = 'normalized'

    def __post_init__(self):
        if min(self.window, self.dim, self.horizon) < 1:
            raise ValueError('window, dim, and horizon must be positive')
        if self.fast_site != 'tokens':
            raise ValueError("TCN23 requires fast_site='tokens'")
        if self.fast_rule != 'normalized':
            raise ValueError("TCN23 requires fast_rule='normalized'")

    def to_dict(self):
        return asdict(self)


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-only padding, so output time t sees no future input."""

    def __init__(self, channels, *, dilation):
        self.left_padding = (KERNEL_SIZE - 1) * dilation
        super().__init__(channels, channels, KERNEL_SIZE,
                         dilation=dilation, padding=0)

    def forward(self, values):
        values = F.pad(values, (self.left_padding, 0))
        return super().forward(values)


class CausalResidualBlock(nn.Module):
    """Two causal k=3 convolutions at one dilation, with channel-only norms."""

    def __init__(self, dim, dilation):
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = CausalConv1d(dim, dilation=self.dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = CausalConv1d(dim, dilation=self.dilation)
        self.norm2 = nn.LayerNorm(dim)

    @staticmethod
    def _time_norm(values, norm):
        # LayerNorm's normalized dimension is the channel dimension only;
        # time is not pooled or normalized jointly with another time point.
        return norm(values.transpose(1, 2)).transpose(1, 2)

    def forward(self, values):
        residual = values
        values = self.conv1(values)
        values = F.relu(self._time_norm(values, self.norm1))
        values = self.conv2(values)
        values = self._time_norm(values, self.norm2)
        return F.relu(values + residual)


class TCN23(nn.Module):
    """23 values plus 23 availability bits -> risk and direct 25-step forecast."""

    receptive_field = 1 + 2 * 2 * sum(DILATIONS)

    def __init__(self, cfg, center, scale):
        super().__init__()
        if not isinstance(cfg, TCNConfig):
            raise TypeError('cfg must be a TCNConfig')
        self.cfg = cfg
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if center.shape != (CHANNELS,) or scale.shape != (CHANNELS,):
            raise ValueError('exactly 23 diagnostic channels required')
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError('finite source normalization required')
        if not (scale > 0).all():
            raise ValueError('source scales must be positive')
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)

        self.embedding = nn.Linear(INPUT_FEATURES, cfg.dim)
        self.fast_origin = nn.Parameter(torch.zeros(2 * cfg.dim))
        self.blocks = nn.ModuleList(
            [CausalResidualBlock(cfg.dim, dilation) for dilation in DILATIONS]
        )
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.risk_head = nn.Linear(cfg.dim, 1)
        self.forecast_head = nn.Linear(cfg.dim, cfg.horizon * CHANNELS)

    def initial_fast(self, batch):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        return self.fast_origin.unsqueeze(0).expand(batch, -1).clone()

    def _prepare(self, values, available):
        if values.ndim != 3 or values.shape[1:] != (self.cfg.window, CHANNELS):
            raise ValueError('values must be [batch, past window, 23]')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('matching boolean availability required')
        values = values.to(dtype=self.center.dtype)
        available = available & torch.isfinite(values)
        clean = torch.where(available, values, self.center)
        level = torch.asinh((clean - self.center) / self.scale)
        return torch.cat((level, available.to(level.dtype)), dim=-1)

    def encode(self, values, available, fast=None):
        """Return causal hidden states with shape [batch, window, dim]."""
        features = self._prepare(values, available)
        hidden = self.embedding(features)
        batch = hidden.shape[0]
        fast = self.initial_fast(batch) if fast is None else fast
        if fast.shape != (batch, 2 * self.cfg.dim):
            raise ValueError('one temporary state per shot required')
        fast = fast.to(dtype=hidden.dtype)
        gain, bias = fast.chunk(2, dim=-1)
        hidden = hidden * (1. + .1 * torch.tanh(gain)[:, None, :])
        hidden = hidden + .1 * torch.tanh(bias)[:, None, :]

        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2)
        return self.output_norm(hidden)

    def forward(self, values, available, fast=None):
        hidden = self.encode(values, available, fast=fast)
        last = hidden[:, -1]
        batch = last.shape[0]
        forecast = self.forecast_head(last).reshape(batch, self.cfg.horizon, CHANNELS)
        return dict(logit=self.risk_head(last).squeeze(-1), forecast=forecast)

    def forecast_loss(self, prediction, truth, available):
        if (truth.shape != prediction.shape or available.shape != truth.shape
                or available.dtype != torch.bool):
            raise ValueError('forecast targets and masks must match')
        available = available & torch.isfinite(truth)
        truth = torch.where(available, truth.detach(), self.center)
        target = torch.asinh((truth - self.center) / self.scale)
        loss = F.smooth_l1_loss(prediction, target, reduction='none')
        return loss.masked_fill(~available, 0.).sum() / available.sum().clamp_min(1)


def fit_forecaster(model, optimizer, values, available, truth, truth_available):
    """Fit only the direct forecast head after risk-model selection."""
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    states = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    head = {id(parameter) for parameter in model.forecast_head.parameters()}
    try:
        for parameter, _ in states:
            parameter.requires_grad_(id(parameter) in head)
        output = model(values, available)
        loss = model.forecast_loss(output['forecast'], truth, truth_available)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite forecast loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1.,
                                       error_if_nonfinite=True)
        optimizer.step()
    finally:
        for parameter, enabled in states:
            parameter.requires_grad_(enabled)
    return float(loss.detach())


def _checkpoint_format(checkpoint):
    nested = checkpoint.get('config', {})
    nested = nested.get('format') if isinstance(nested, dict) else None
    direct = checkpoint.get('format')
    if direct is not None and nested is not None and direct != nested:
        raise ValueError(f'conflicting checkpoint formats; expected {CHECKPOINT_FORMAT}')
    return direct if direct is not None else nested


def load_checkpoint(path, device):
    """Load a tcn23-v1 checkpoint, validate its contract, and return eval model."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or _checkpoint_format(checkpoint) != CHECKPOINT_FORMAT:
        raise ValueError(f'checkpoint format must be {CHECKPOINT_FORMAT}')
    config = checkpoint.get('config')
    if not isinstance(config, dict) or config.get('feature_names') != list(io23.RAW23):
        raise ValueError('tcn23-v1 feature_names order does not match io23.RAW23')
    required = ('model', 'model_config', 'source_center', 'source_scale',
                'threshold', 'validation', 'forecast_trained')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'incomplete {CHECKPOINT_FORMAT} checkpoint: {missing}')
    try:
        threshold = float(checkpoint['threshold'])
    except (TypeError, ValueError) as exc:
        raise ValueError('checkpoint threshold must be finite with 0 < threshold <= 1') from exc
    if not math.isfinite(threshold) or not 0. < threshold <= 1.:
        raise ValueError('checkpoint threshold must be finite with 0 < threshold <= 1')
    source_center = torch.as_tensor(checkpoint['source_center'], dtype=torch.float32)
    source_scale = torch.as_tensor(checkpoint['source_scale'], dtype=torch.float32)
    if (source_center.shape != (CHANNELS,) or source_scale.shape != (CHANNELS,)
            or not torch.isfinite(source_center).all()
            or not torch.isfinite(source_scale).all()
            or not (source_scale > 0).all()):
        raise ValueError('checkpoint source center/scale must be finite 23-vectors with positive scale')
    try:
        cfg = TCNConfig(**checkpoint['model_config'])
        model = TCN23(cfg, source_center, source_scale).to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
    except (TypeError, KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(f'invalid {CHECKPOINT_FORMAT} model payload') from exc
    if (not torch.isfinite(model.center).all() or not torch.isfinite(model.scale).all()
            or not (model.scale > 0).all()
            or not torch.equal(model.center, source_center.to(model.center))
            or not torch.equal(model.scale, source_scale.to(model.scale))):
        raise ValueError('checkpoint model center/scale state differs from source center/scale')
    model.eval()
    model.threshold = threshold
    model.validation = checkpoint['validation']
    model.forecast_trained = bool(checkpoint['forecast_trained'])
    model.checkpoint_format = CHECKPOINT_FORMAT
    return model


__all__ = [
    'CHECKPOINT_FORMAT',
    'CausalConv1d',
    'CausalResidualBlock',
    'TCNConfig',
    'TCN23',
    'fit_forecaster',
    'load_checkpoint',
]
