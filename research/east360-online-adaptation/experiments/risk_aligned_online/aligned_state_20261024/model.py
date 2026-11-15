"""Align a signal-trained first block to the qualified source's fixed readout.

The bridge acts on 64-dimensional tokens, not on the final risk score. The
second Transformer block, post-normalization, learned fast_origin and risk
head remain the original source's. Temporary updates affect the observation
encoder only; the caller owns one fast dictionary per shot and discards it.
This is a cross-task-alignment-inspired candidate, not a reproduction of CTA.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.func import functional_call

from ..masked_state_20261022.model import MaskedState, masked_views, reconstruction_loss


def first_tokens(base, values, available):
    if (base.cfg.layers != 2 or base.cfg.fast_site != 'tokens'
            or base.cfg.input_view != 'source'):
        raise ValueError('source-coordinate two-block token architecture required')
    if (values.shape != available.shape or available.dtype != torch.bool
            or values.shape[1:] != (base.cfg.window, 23)):
        raise ValueError('matching causal windows and masks required')
    valid = available & torch.isfinite(values)
    z = torch.asinh((torch.where(valid, values, base.center)-base.center)/base.scale)
    x = torch.cat((z, valid.to(z.dtype)), -1)
    x = base.embedding(x.reshape(len(x), base.cfg.window//base.cfg.patch, -1))
    x = torch.cat((x, base.cls.expand(len(x), -1, -1)), 1)+base.position
    v = valid.reshape(len(x), base.cfg.window//base.cfg.patch, -1).any(-1)
    v = torch.cat((v, torch.ones(len(x), 1, device=x.device, dtype=torch.bool)), 1)
    with sdpa_kernel(SDPBackend.MATH):
        x = base.encoder.layers[0](x, src_key_padding_mask=~v)
    return x, v


def fixed_readout(base, tokens, valid):
    with sdpa_kernel(SDPBackend.MATH):
        x = base.encoder.layers[1](tokens, src_key_padding_mask=~valid)
    gain, bias = base.fast_origin.chunk(2, -1)
    x = base.norm(x)*(1.+.1*gain[None,None].tanh())+.1*bias[None,None].tanh()
    x = x.masked_fill(~valid[:,:,None], 0.)
    return base.risk_head(x[:,-1]).squeeze(-1)


class AlignedState(MaskedState):
    # Canonical flat23 index. A stored/averaged mode index is not necessarily
    # categorical; its meaning and Euclidean regression distance are not
    # certified across devices. Retain risk input, exclude auxiliary target.
    mode_number_index = 10

    def __init__(self, base, level_scale):
        super().__init__(base, level_scale)
        self.bridge = nn.Linear(base.cfg.dim, base.cfg.dim)
        with torch.no_grad():
            self.bridge.weight.copy_(torch.eye(base.cfg.dim)); self.bridge.bias.zero_()
        keep = torch.ones(23, dtype=torch.bool); keep[self.mode_number_index] = False
        self.register_buffer('reconstruction_channels', keep)
        self.requires_grad_(False)

    def forward(self, values, available, *, mode='risk'):
        h, v = first_tokens(self.base, values, available)
        if mode == 'reconstruct':
            return dict(reconstruction=self.reconstructor(h[:,:-1]).reshape(len(values), self.cfg.window, 23))
        if mode not in ('risk', 'aligned'):
            raise ValueError('unknown aligned-state task')
        aligned = self.bridge(h)
        out = dict(logit=fixed_readout(self.base, aligned, v))
        if mode == 'aligned': out.update(tokens=aligned, valid=v)
        return out

    def new_fast(self):
        names = set(self.encoder_names())
        return {k:p.detach().clone().requires_grad_(True)
                for k,p in self.named_parameters() if k in names}

    def set_alignment(self, *, encoder):
        names = set(self.encoder_names()) if encoder else set()
        for k,p in self.named_parameters():
            p.requires_grad_(k in names or k.startswith('bridge.'))
            p.grad = None
        self.eval()


def observed_loss(model, values, available, fast=None):
    xx, mm, held = masked_views(values, available)
    held = held & model.reconstruction_channels[None,None]
    out = (model(xx, mm, mode='reconstruct') if fast is None
           else model.predict_fast(xx, mm, fast, mode='reconstruct'))
    return reconstruction_loss(model, out['reconstruction'], xx, held)


def update_fast(model, fast, values, available, *, learning_rate):
    """One causal encoder step; no labels, future tensors or teacher risk input.

    Keeping the dictionary across calls implements within-shot accumulation.
    Calling new_fast at the next shot resets all temporary learning. There is
    no optimizer momentum, bridge update, readout update or score correction.
    """
    if len(values) != 1 or values.shape != available.shape:
        raise ValueError('one shot window required')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('positive finite learning rate required')
    if set(fast) != set(model.encoder_names()):
        raise ValueError('only observation encoder parameters may adapt')
    n = int((available & torch.isfinite(values) & model.reconstruction_channels[None,None]).sum())
    if not n:
        return fast, dict(gradient_steps=0, observations=0, gradient_norm=0., loss=None)
    with torch.enable_grad():
        loss = observed_loss(model, values, available, fast)
        grads = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=False)
        norm = sum(g.double().square().sum() for g in grads).sqrt()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite observed reconstruction update')
        # Fixed numerical gradient clipping, NOT risk-dependent gating.
        factor = (1./norm.clamp_min(1.)).to(values.dtype)
        new = {k:(v-learning_rate*factor*g).detach().requires_grad_(True)
               for (k,v),g in zip(fast.items(), grads)}
    return new, dict(gradient_steps=1, observations=n, gradient_norm=float(norm), loss=float(loss.detach()))
