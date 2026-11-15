"""Observed-channel completion, not an additional risk-score correction.

The existing two-block Transformer is split into a shared observation encoder
(embedding + first block) and a risk readout (second block + risk head). First
fit the encoder to masked TRAIN channels; then freeze it and fit the readout.
At deployment only an ephemeral copy of the encoder is adapted, from one
arrived causal window. Missing truth is never reconstructed as a target.
This is a testable engineering architecture, not identified plasma physics.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.func import functional_call


class MaskedState(nn.Module):
    def __init__(self, base, level_scale):
        super().__init__()
        if (base.cfg.layers != 2 or base.cfg.fast_site != 'tokens'
                or base.cfg.input_view != 'source'):
            raise ValueError('two-block source-coordinate token model required')
        q = torch.as_tensor(level_scale).float()
        if q.shape != (23,) or not torch.isfinite(q).all() or not (q > 0).all():
            raise ValueError('23 finite positive TRAIN-only level scales required')
        self.base, self.cfg = base, base.cfg
        self.register_buffer('level_scale', q)
        self.reconstructor = nn.Linear(base.cfg.dim, base.cfg.patch*23)
        self.event_head = nn.Linear(base.cfg.dim, 6)

    def encoder_names(self):
        return [k for k, _ in self.named_parameters() if k.startswith(
            ('base.embedding.', 'base.encoder.layers.0.')) or k in ('base.position', 'base.cls')]

    def risk_names(self):
        return [k for k, _ in self.named_parameters() if k.startswith(
            ('base.encoder.layers.1.', 'base.norm.', 'base.risk_head.'))]

    def set_stage(self, stage):
        if stage not in ('reconstruction', 'risk', 'frozen'):
            raise ValueError('unknown training stage')
        names = set(self.encoder_names() if stage == 'reconstruction' else self.risk_names() if stage == 'risk' else [])
        for k, p in self.named_parameters():
            p.requires_grad_(k in names or (stage == 'reconstruction' and k.startswith(('reconstructor.', 'event_head.'))))
            p.grad = None
        self.eval()

    def encode(self, values, available):
        if values.shape != available.shape or available.dtype != torch.bool or values.shape[1:] != (self.cfg.window, 23):
            raise ValueError('matching observed windows and masks required')
        valid = available & torch.isfinite(values)
        clean = torch.where(valid, values, self.base.center)
        z = torch.asinh((clean-self.base.center)/self.base.scale)
        x = torch.cat((z, valid.to(z.dtype)), -1)
        x = self.base.embedding(x.reshape(len(x), self.cfg.window//self.cfg.patch, -1))
        # Source fast_origin is not a new online variable. At its learned
        # origin the parent's pre-encoder token delta is exactly zero.
        x = torch.cat((x, self.base.cls.expand(len(x), -1, -1)), 1)+self.base.position
        token_valid = valid.reshape(len(x), self.cfg.window//self.cfg.patch, -1).any(-1)
        v = torch.cat((token_valid, torch.ones(len(x), 1, device=x.device, dtype=torch.bool)), 1)
        # Same explicit math attention backend as the source model; supports
        # gradients through all eligible encoder parameters in adaptation.
        with sdpa_kernel(SDPBackend.MATH):
            x = self.base.encoder.layers[0](x, src_key_padding_mask=~v)
        return x[:, :-1], token_valid

    def forward(self, values, available, *, mode='risk'):
        if mode == 'risk':
            return self.base(values, available)
        if mode != 'reconstruct': raise ValueError('unknown forward task')
        h, valid = self.encode(values, available)
        prediction = self.reconstructor(h).reshape(len(values), self.cfg.window, 23)
        pooled = (h*valid[:, :, None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
        return dict(reconstruction=prediction, event=self.event_head(pooled))

    def predict_fast(self, values, available, fast, *, mode='risk'):
        if set(fast) != set(self.encoder_names()):
            raise ValueError('temporary state must contain only the observation encoder')
        return functional_call(self, fast, (values, available), dict(mode=mode), strict=False)


def masked_views(values, available):
    """Four deterministic whole-channel masks; no temporal target shortcut.

For every example each valid target appears in exactly one of four views.
The entire channel history is hidden, not only its last point. The modular
partition is an engineering mask schedule, NOT a new physical grouping.
"""
    if values.shape != available.shape or available.dtype != torch.bool or values.ndim != 3 or values.shape[-1] != 23:
        raise ValueError('matching B,T,23 windows required')
    x = values.repeat_interleave(4, dim=0)
    m = (available & torch.isfinite(values)).repeat_interleave(4, dim=0)
    group = torch.arange(4, device=x.device).repeat(len(values))
    held = (torch.arange(23, device=x.device)[None] % 4 == group[:, None])[:, None].expand_as(m) & m
    return x, m & ~held, held


def reconstruction_loss(model, prediction, truth, target_mask):
    if prediction.shape != truth.shape or target_mask.shape != truth.shape or target_mask.dtype != torch.bool:
        raise ValueError('matching reconstruction and boolean truth mask required')
    valid = target_mask & torch.isfinite(truth)
    clean = torch.where(valid, truth.detach(), model.base.center)
    target = torch.asinh((clean-model.base.center)/model.base.scale)
    error = (prediction-target)/model.level_scale
    losses = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
    return losses.masked_fill(~valid, 0.).sum()/valid.sum().clamp_min(1)


def adapt_observed(model, x, available, *, learning_rate, steps):
    """Re-fit a disposable encoder copy on one causal window, then discard.

No future or outcome field is accepted. Repeating this operator does not
accumulate state beyond the supplied observation window. SGD has no hidden
optimizer state; the source model and reconstruction/risk heads stay fixed.
"""
    if len(x) != 1 or available.shape != x.shape:
        raise ValueError('one observed window, never a mixed-shot batch, required')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or type(steps) is not int or steps < 1:
        raise ValueError('positive finite learning rate and integer steps required')
    names = set(model.encoder_names())
    fast = {k: p.detach().clone().requires_grad_(True) for k,p in model.named_parameters() if k in names}
    origin = {k: v.detach().clone() for k,v in fast.items()}
    xx, mm, held = masked_views(x, available)
    count = int(held.sum())
    if not count:
        return fast, dict(gradient_steps=0, parameter_delta_norm=0., observations=0, losses=[])
    losses = []
    with torch.enable_grad():
        for _ in range(steps):
            out = model.predict_fast(xx, mm, fast, mode='reconstruct')
            loss = reconstruction_loss(model, out['reconstruction'], xx, held)
            grad = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=False)
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grad):
                raise FloatingPointError('nonfinite masked adaptation')
            fast = {k:(v-learning_rate*g).detach().requires_grad_(True) for (k,v),g in zip(fast.items(), grad)}
            losses.append(float(loss.detach()))
    distance = sum((v.detach()-origin[k]).double().square().sum() for k,v in fast.items()).sqrt()
    return fast, dict(gradient_steps=steps, parameter_delta_norm=float(distance), observations=count, losses=losses)
