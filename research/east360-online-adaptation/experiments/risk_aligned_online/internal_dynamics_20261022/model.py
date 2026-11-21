"""Function-preserving internal interface plus a direct diagnostic forecast.

The original risk head and both Transformer blocks are frozen. Temporary U
changes observed PATCH representations before the last attention block; it
does not add a scalar to the risk and never edits CLS directly. A direct
multi-step diagnostic decoder supplies the inner loss. Source meta-training
can teach the basis/decoder which diagnostic-error gradients help later risk.
This structural property is not a claim of learned physical mechanisms.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class InternalDynamics(nn.Module):
    def __init__(self, parent, change_scale, active, *, rank=8):
        super().__init__()
        if parent.cfg.fast_site != 'tokens' or parent.cfg.layers < 2 or rank < 1:
            raise ValueError('token parent with >=2 blocks and positive rank required')
        self.parent = parent.eval().requires_grad_(False)
        self.cfg, self.rank = parent.cfg, rank
        q = torch.as_tensor(change_scale).float()
        a = torch.as_tensor(active).bool()
        if q.shape != (23,) or a.shape != (23,) or not torch.isfinite(q).all() or not (q > 0).all():
            raise ValueError('23 finite positive source change scales required')
        self.register_buffer('change_scale', q)
        self.register_buffer('active', a)
        self.basis = nn.Linear(self.cfg.dim, rank, bias=False)
        self.decoder = nn.Linear((self.cfg.window//self.cfg.patch+1)*self.cfg.dim,
                                 self.cfg.horizon*23)
        # Exactly persistence until fitted; no random forecast advertised as
        # useful feedback. The first warmup stage must establish predictivity.
        nn.init.zeros_(self.decoder.weight); nn.init.zeros_(self.decoder.bias)
        self.event_head = nn.Linear(rank, 6)

    def adaptation_parameters(self):
        return [*self.basis.parameters(), *self.decoder.parameters(), *self.event_head.parameters()]

    def initial_fast(self, batch):
        return torch.zeros(batch, self.rank, self.cfg.dim, device=self.change_scale.device,
                           requires_grad=True)

    def forward(self, values, available, fast=None):
        self.parent.eval()
        batch = len(values)
        fast = self.initial_fast(batch) if fast is None else fast
        if fast.shape != (batch, self.rank, self.cfg.dim):
            raise ValueError('one internal temporary matrix per independent shot required')
        valid = (available & torch.isfinite(values)).reshape(batch, -1, self.cfg.patch*23).any(-1)
        valid_cls = torch.cat((valid, torch.zeros(batch, 1, device=valid.device, dtype=torch.bool)), 1)
        latent, representation = [], []

        def insert(_module, args):
            h = args[0]
            z = F.silu(self.basis(F.layer_norm(h, (self.cfg.dim,))))
            delta = torch.bmm(z, fast) / math.sqrt(self.rank)
            delta = delta.masked_fill(~valid_cls[:, :, None], 0.)
            latent.append((z[:, :-1]*valid[:, :, None]).sum(1)/valid.sum(1, keepdim=True).clamp_min(1))
            return (h+delta, *args[1:])

        def capture(_module, args):
            representation.append(args[0])

        # Hooks preserve the parent's implemented normalization/masking/token
        # order. They live for one call only, including exception cleanup.
        hook = self.parent.encoder.layers[-1].register_forward_pre_hook(insert)
        readout = self.parent.forecast_head.register_forward_pre_hook(capture)
        try:
            # Force the SAME differentiable attention dispatch for zero and
            # adapted states. PyTorch inference fastpaths otherwise introduce
            # tiny differences that can be mistaken for a weak update.
            with torch.enable_grad():
                dummy = self.parent.initial_fast(batch).detach().requires_grad_(True)
                out = self.parent(values, available, dummy)
                forecast = self.decoder(representation[0]).reshape(batch, self.cfg.horizon, 23)
                event = self.event_head(latent[0])
                return dict(logit=out['logit'], forecast=forecast, event=event)
        finally:
            hook.remove(); readout.remove()


def feedback_target(model, origin, origin_available, truth, available):
    center, scale = model.parent.center, model.parent.scale
    valid = (available & torch.isfinite(truth) & origin_available[:, None]
             & torch.isfinite(origin)[:, None] & model.active[None, None])
    clean = torch.where(valid, truth, center)
    baseline = torch.where(origin_available & torch.isfinite(origin), origin, center)
    target = (torch.asinh((clean-center)/scale)-torch.asinh((baseline-center)/scale)[:, None])
    return target/model.change_scale, valid


def feedback_loss(model, prediction, origin, origin_available, truth, available):
    target, valid = feedback_target(model, origin, origin_available, truth, available)
    loss = F.smooth_l1_loss(prediction, target, reduction='none')
    counts = valid.sum((1, 2))
    return loss.masked_fill(~valid, 0.).sum((1, 2))/counts.clamp_min(1), counts


def temporary_step(model, fast, x, mask, truth, tm, *, step=.1, radius=1., create_graph=False):
    if not (math.isfinite(step) and step > 0 and math.isfinite(radius) and radius > 0):
        raise ValueError('finite positive temporary step and radius required')
    if not fast.requires_grad: fast = fast.detach().requires_grad_(True)
    prediction = model(x, mask, fast)['forecast']
    loss, counts = feedback_loss(model, prediction, x[:, -1], mask[:, -1], truth, tm)
    gradient, = torch.autograd.grad(loss.sum(), fast, create_graph=create_graph)
    if not torch.isfinite(gradient).all(): raise FloatingPointError('nonfinite internal fast gradient')
    norm = (gradient.square().sum((1, 2), keepdim=True)+1e-16).sqrt()
    candidate = fast-step*gradient/norm
    distance = (candidate.square().sum((1, 2), keepdim=True)+1e-16).sqrt()
    candidate = candidate*(radius/distance).clamp(max=1.)
    candidate = torch.where((counts > 0)[:, None, None], candidate, fast)
    if not create_graph: candidate = candidate.detach().requires_grad_(True)
    return candidate, dict(loss=float(loss.detach().mean()), observations=int(counts.sum()),
        gradient_norm=float(gradient.detach().norm(dim=(1, 2)).mean()),
        fast_norm=float(candidate.detach().norm(dim=(1, 2)).mean()))
