"""Risk-directed, signal-only feedback into a frozen temporal representation.

ADAPT-Z motivates contextual feature feedback, not an algorithm reproduction.
Only a temporary common offset of valid signal-patch representations changes.
Pairwise patch differences at the insertion site are preserved; this does NOT
guarantee that the following attention block preserves every dangerous trend.
The six measured diagnostic forecast targets are engineering observations,
not plasma-state equations or inferred disruption labels.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import simple_data
from .diagnostic_targets import coordinates, trajectory_loss, trajectory_target


class ContextFeedback(nn.Module):
    def __init__(self, dim=64, *, horizon=8):
        super().__init__()
        if dim < 2 or horizon < 1:
            raise ValueError('positive forecast horizon and internal dimension required')
        self.dim, self.horizon = dim, horizon
        self.forecaster = nn.Sequential(nn.LayerNorm(3*dim), nn.Linear(3*dim, dim),
                                        nn.GELU(), nn.Linear(dim, horizon*6))
        self.controller = nn.Sequential(nn.LayerNorm(4*dim), nn.Linear(4*dim, dim),
                                        nn.Tanh(), nn.Linear(dim, dim))
        # Start with ordinary normalized feature-gradient descent. The learned
        # term is fitted on source training episodes, never EAST outcome labels.
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def initial_fast(self, batch):
        return self.forecaster[-1].bias.new_zeros(batch, self.dim)

    def forward(self, parent, values, available, fast=None, *, perturb=None):
        if parent.cfg.layers < 2 or parent.cfg.dim != self.dim or parent.cfg.fast_site != 'tokens':
            raise ValueError('matching frozen multi-block temporal parent required')
        if values.ndim != 3 or values.shape[1:] != (parent.cfg.window, 23):
            raise ValueError('exact parent past-window shape required')
        if values.shape != available.shape or available.dtype != torch.bool:
            raise ValueError('matching input and boolean availability required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if fast.shape != (len(values), self.dim):
            raise ValueError('one internal temporary context per shot required')
        valid = (available & torch.isfinite(values)).reshape(len(values), -1, parent.cfg.patch*23).any(-1)
        fresh = perturb is None and not fast.requires_grad and not bool(fast.detach().count_nonzero())
        # Hook presence disables a torch fused encoder path. Preserve the
        # actual unhooked source logit bitwise at zero; use the hooked path for
        # a differentiable state. Tests check both equality and double backward.
        original = parent(values, available)['logit'] if fresh else None
        captured = {}

        def insert(_module, _args, hidden):
            patches = hidden[:, :-1]
            nuisance = torch.zeros_like(fast) if perturb is None else perturb
            shifted = patches+(fast+nuisance)[:, None]*valid[:, :, None]
            count = valid.sum(-1, keepdim=True).clamp_min(1)
            mean = shifted.masked_fill(~valid[:, :, None], 0.).sum(1)/count
            last = shifted[:, -1]-mean
            pairs = valid[:, 1:] & valid[:, :-1]
            variation = (shifted[:, 1:]-shifted[:, :-1]).abs().masked_fill(~pairs[:, :, None], 0.)
            variation = variation.sum(1)/pairs.sum(-1, keepdim=True).clamp_min(1)
            context = torch.cat((mean, last, variation), -1)
            # No CLS correction. Original risk head receives the next frozen
            # block's representation of the changed SIGNAL patches.
            result = torch.cat((shifted, hidden[:, -1:]), 1)
            captured.update(context=context, patches_after=shifted,
                            trajectory=self.forecaster(context).reshape(-1, self.horizon, 6))
            return result

        handle = parent.encoder.layers[0].register_forward_hook(insert)
        try:
            out = parent(values, available)
        finally:
            handle.remove()
        return dict(logit=original if fresh else out['logit'], **captured)

    def update(self, current, gradient, context, *, learned=True):
        if current.shape != gradient.shape or context.shape != (len(current), 3*self.dim):
            raise ValueError('matched state, gradient and current context required')
        norm = gradient.norm(dim=-1, keepdim=True)
        unit = gradient/norm.clamp_min(1e-8)
        modulation = .5*self.controller(torch.cat((context, unit), -1)).tanh() if learned else 0.
        direction = -unit+modulation
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1.)
        # No signal gradient means no adaptation, regardless of the context.
        proposal = current+.5*(norm/.01).clamp(max=1.)*direction
        return proposal*(8./proposal.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)


def signal_target(x, a, truth, mask):
    origin = coordinates(x, a, min_count=8)
    target, valid = trajectory_target(origin, truth, mask)
    # A signed, monotone coordinate avoids tiny past variance letting one
    # diagnostic dominate the whole loss. It does not clip large changes.
    return torch.asinh(target), valid


def train_step(parent, adapter, optimizer, episode, *, mode, perturb=None, unroll=1,
               target_fn=signal_target):
    if mode not in ('aligned', 'forecast') or any(p.requires_grad for p in parent.parameters()):
        raise ValueError('known source objective and frozen parent required')
    matured = unroll*(adapter.horizon+1)-1
    if unroll < 1 or episode['truth'].shape[1] != matured:
        raise ValueError('complete chronological inner-update episode required')
    optimizer.zero_grad(set_to_none=True)
    state = adapter.initial_fast(len(episode['x'])).requires_grad_(True)
    losses, fitted_losses, persistence, valid_fractions = [], [], [], []
    for inner in range(unroll):
        offset = inner*(adapter.horizon+1)
        # A later origin sees the original prefix plus ONLY signals that have
        # arrived by that origin. No future-target statistics or label input.
        x = torch.cat((episode['x'], episode['truth'][:, :offset]), 1)[:, -parent.cfg.window:]
        a = torch.cat((episode['a'], episode['mask'][:, :offset]), 1)[:, -parent.cfg.window:]
        truth = episode['truth'][:, offset:offset+adapter.horizon]
        mask = episode['mask'][:, offset:offset+adapter.horizon]
        out = adapter(parent, x, a, state, perturb=perturb)
        target, valid = target_fn(x, a, truth, mask)
        loss = trajectory_loss(out['trajectory'], target, valid)
        gradient, = torch.autograd.grad(loss.sum(), state, create_graph=True)
        updated = adapter.update(state, gradient, out['context'])
        fitted = adapter(parent, x, a, updated, perturb=perturb)
        fitted_losses.append(trajectory_loss(fitted['trajectory'], target, valid).mean())
        losses.append(loss.mean())
        persistence.append(trajectory_loss(torch.zeros_like(target), target, valid).mean())
        valid_fractions.append(valid.float().mean())
        state = updated
    query = adapter(parent, episode['qx'], episode['qa'], updated, perturb=perturb)
    risk = F.binary_cross_entropy_with_logits(query['logit'], episode['y'])
    signal = torch.stack(losses).mean()
    after_signal = torch.stack(fitted_losses).mean()
    # Compare source risk-directed training against matched signal-only fitting.
    # Forecast control learns to reduce the SAME arrived forecast loss; it
    # never sees risk labels. Both arms use identical source episode streams.
    objective = risk+.2*signal if mode == 'aligned' else after_signal+.2*signal
    if not torch.isfinite(objective):
        raise FloatingPointError('nonfinite source contextual adaptation objective')
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        base = parent(episode['qx'], episode['qa'])['logit']
        risk_before = F.binary_cross_entropy_with_logits(base, episode['y'])
    return dict(signal_loss=float(signal.detach()), after_signal=float(after_signal.detach()),
                persistence_loss=float(torch.stack(persistence).mean()),
                risk_before=float(risk_before), risk_after=float(risk.detach()),
                gradient_norm=float(norm), fast_change=float(updated.detach().norm(dim=-1).mean()),
                valid_fraction=float(torch.stack(valid_fractions).mean()),
                inner_updates=unroll, matured_signal_points=matured)


def predict_shot(parent, adapter, shot, *, arm, device='cuda', data_api=simple_data,
                 target_fn=signal_target):
    if arm not in ('static', 'learned', 'gradient', 'reverse'):
        raise ValueError('fixed contextual replay arm required')
    parent.eval()
    adapter.eval()
    state = adapter.initial_fast(1).detach()
    scores = np.empty(len(shot['values']), np.float32)
    events = []
    origin = 0
    while origin < len(scores):
        stop = min(len(scores), origin+(adapter.horizon+1 if arm != 'static' else 256))
        complete = arm != 'static' and stop == origin+adapter.horizon+1
        if complete:
            x, a = data_api.window(shot, origin, parent.cfg.window)
            x, a = torch.as_tensor(x[None], device=device), torch.as_tensor(a[None], device=device)
            candidate = state.detach().requires_grad_(True)
            issued = adapter(parent, x, a, candidate)
        windows = [data_api.window(shot, t, parent.cfg.window) for t in range(origin, stop)]
        bx = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        ba = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        with torch.no_grad():
            out = (parent(bx, ba) if arm == 'static' else
                   adapter(parent, bx, ba, state.expand(len(bx), -1)))
            probabilities = out['logit'].sigmoid()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError('nonfinite sealed risk')
            scores[origin:stop] = probabilities.cpu().numpy()
        if complete:
            truth = torch.as_tensor(shot['values'][origin+1:stop][None], device=device)
            mask = torch.as_tensor(shot['available'][origin+1:stop][None], device=device)
            target, valid = target_fn(x, a, truth, mask)
            if valid.any():
                loss = trajectory_loss(issued['trajectory'], target, valid).sum()
                gradient, = torch.autograd.grad(loss, candidate)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('nonfinite matured signal gradient')
                if arm == 'reverse':
                    gradient = -gradient  # Explicit wrong-feedback control.
                state = adapter.update(candidate.detach(), gradient.detach(), issued['context'].detach(),
                                       learned=arm != 'gradient').detach()
                events.append(dict(origin=origin, observed_through=stop-1, affects_from=stop,
                                   signal_loss=float(loss.detach()), gradient_norm=float(gradient.norm()),
                                   state_norm=float(state.norm())))
        origin = stop
    return scores, events
