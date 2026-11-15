"""An offline-trained update objective; no added risk-score correction.

The original decoder-preparation experiment keeps the qualified parent
immutable and trains only its masked-channel decoder. The separate
joint_feedback experiment explicitly prepares a NEW source model and uses
track_initial for its differentiable outer gradient; it never overwrites
the original checkpoint. At evaluation an ephemeral copy of the input projection
and first Transformer block is updated using arrived observations. Paired
clean representations and labels exist only in the separate TRAIN outer
objective, never in update_observed.
"""
import math

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from ..masked_state_20261022.model import MaskedState, masked_views, reconstruction_loss


class PairedFeedback(MaskedState):
    def prepare_decoder(self):
        self.requires_grad_(False)
        self.reconstructor.requires_grad_(True)
        self.eval()

    def forward(self, values, available, *, mode='risk'):
        if mode == 'state':
            h, valid = self.encode(values, available)
            return dict(tokens=h, valid=valid)
        if mode == 'risk_state':
            # Expose the EXISTING classifier input, including CLS, without
            # adding an inference branch. Equality to base.forward is tested
            # both before and after functional encoder updates.
            valid = available & torch.isfinite(values)
            clean = torch.where(valid,values,self.base.center)
            level = torch.asinh((clean-self.base.center)/self.base.scale)
            x = torch.cat((level,valid.to(level.dtype)),-1)
            x = self.base.embedding(x.reshape(len(x),self.cfg.window//self.cfg.patch,-1))
            x = torch.cat((x,self.base.cls.expand(len(x),-1,-1)),1)+self.base.position
            tv = valid.reshape(len(x),self.cfg.window//self.cfg.patch,-1).any(-1)
            tv = torch.cat((tv,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)
            with sdpa_kernel(SDPBackend.MATH):
                x = self.base.encoder(x,src_key_padding_mask=~tv)
            gain,bias = self.base.initial_fast(len(x)).chunk(2,-1)
            x = self.base.norm(x)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
            return dict(risk_state=x[:,-1])
        return super().forward(values, available, mode=mode)


def initial_encoder(model, *, track_initial=False):
    names = set(model.encoder_names())
    if track_initial:
        result = {k:p for k,p in model.named_parameters() if k in names}
        if not all(p.requires_grad for p in result.values()):
            raise ValueError('outer training requires trainable initial encoder parameters')
        return result
    return {k:p.detach().clone().requires_grad_(True)
            for k,p in model.named_parameters() if k in names}


def representation_loss(h, reference, valid):
    if h.shape != reference.shape or valid.shape != h.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('matching token representations and validity required')
    reference = reference.detach()
    # Per-token source energy balances amplitude without erasing it: a
    # multiplicative change still has a nonzero error. This is not a new
    # physical state metric or an EAST-estimated normalization.
    error = (h-reference).square().mean(-1)/(1.+reference.square().mean(-1))
    return error.masked_fill(~valid, 0.).sum()/valid.sum().clamp_min(1)


def risk_state_loss(model,h,reference):
    if h.shape != reference.shape or h.ndim != 2 or h.shape[-1] != model.cfg.dim:
        raise ValueError('one final classifier state per example required')
    # For the fixed linear risk head z=w.h+b, Cauchy-Schwarz gives
    # (z-z_ref)^2 <= ||w||^2 ||h-h_ref||^2. This whole-state bound is an
    # OFFLINE paired teaching term, not an online risk-output anchor.
    weight = model.base.risk_head.weight.detach().square().sum()
    return weight*(h-reference.detach()).square().sum(-1).mean()


def update_observed(model, x, available, *, learning_rate, steps, create_graph=False,
                    track_initial=False):
    if len(x) != 1 or x.shape != available.shape:
        raise ValueError('one observed window per temporary state required')
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('positive finite update budget required')
    if track_initial and not create_graph:
        raise ValueError('tracking initial parameters is only for differentiable outer training')
    fast = initial_encoder(model, track_initial=track_initial)
    origin = {k:v.detach().clone() for k,v in fast.items()}
    xx, mm, target = masked_views(x, available)
    count = int(target.sum())
    if not count:
        return fast, dict(steps=0, observations=0, delta=0., first_loss=None)
    first = None
    with torch.enable_grad():
        for _ in range(steps):
            out = model.predict_fast(xx, mm, fast, mode='reconstruct')
            loss = reconstruction_loss(model, out['reconstruction'], xx, target)
            if first is None: first = loss
            gradients = torch.autograd.grad(loss, tuple(fast.values()), create_graph=create_graph,
                                            retain_graph=create_graph, allow_unused=False)
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('nonfinite paired-feedback inner update')
            fast = {k:v-learning_rate*g for (k,v),g in zip(fast.items(), gradients)}
            if not create_graph:
                fast = {k:v.detach().requires_grad_(True) for k,v in fast.items()}
    delta = sum((v.detach()-origin[k]).double().square().sum() for k,v in fast.items()).sqrt()
    return fast, dict(steps=steps, observations=count, delta=float(delta), first_loss=first)
