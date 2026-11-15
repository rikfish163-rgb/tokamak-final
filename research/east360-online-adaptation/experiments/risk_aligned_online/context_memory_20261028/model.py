"""Fit eight shot-local variables from arrived direct-forecast errors.

The learned observation-dependent basis is fixed during deployment. Context
updates modify internal tokens, never a classifier or an alarm threshold.
No state survives trace(): persistent learning is handled separately.
"""
import math

import torch
from torch import nn

from ..delayed_recurrent_memory_20261025.model import HORIZON
from ..lifecycle_stream_20261025.model import LifecycleAdapter

CONTEXT_DIM = 8


class ContextMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(46, 8)
        self.kernel = nn.Parameter(torch.empty(9, HORIZON, 23, CONTEXT_DIM))
        nn.init.normal_(self.kernel, std=1 / math.sqrt(9 * CONTEXT_DIM))
        self.rate = nn.Parameter(torch.tensor(0.))
        self.output = nn.Linear(92, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    @property
    def eta(self):
        return .5 * self.rate.sigmoid()

    def write(self, context, basis, target, mask):
        if (context.shape != (CONTEXT_DIM,) or basis.shape != (*target.shape, CONTEXT_DIM)
                or mask.shape != target.shape or mask.dtype != torch.bool):
            raise ValueError('eight-variable context and observed forecast targets required')
        error = torch.where(mask, basis @ context - target, 0.)
        observed_basis = torch.where(mask[..., None], basis, 0.)
        gradient = (observed_basis * error[..., None]).flatten(0, -2).sum(0)
        # >= Hessian spectral norm, hence a conservative least-squares step.
        # This constrains diagnostic fitting, not the downstream alarm risk.
        candidate = context - self.eta * gradient / (1 + observed_basis.square().sum())
        return candidate / candidate.norm().clamp_min(1.)

    def key_features(self, clean, complete):
        return torch.cat((clean, complete.to(clean)), -1)

    def expected_forecast(self, clean, complete):
        # Existing context models predict the full change themselves. Optional
        # independently trained expectations are supplied only by a subclass.
        return None

    def trace(self, means, complete, *, mode, reference_norm=None, return_before=False):
        if (mode not in ('off', 'memory', 'local', 'hold', 'reverse')
                or means.ndim != 2 or means.shape[1] != 23 or complete.shape != means.shape
                or complete.dtype != torch.bool):
            raise ValueError('declared mode and observed diagnostic patches required')
        controlled = mode in ('local', 'hold', 'reverse')
        if controlled and (reference_norm is None or reference_norm.shape != means.shape[:1]):
            raise ValueError('controls require main-arm norm at every patch')
        if not torch.isfinite(means[complete]).all():
            raise ValueError('nonfinite values marked as observed')
        clean = torch.where(complete, means, 0.)
        expected = self.expected_forecast(clean, complete)
        if expected is not None and (expected.shape != (len(means), HORIZON, 23)
                                     or not torch.isfinite(expected).all()):
            raise ValueError('finite prefix-only issued expectations required')
        features = self.key_features(clean, complete)
        keys = torch.cat((self.key(features).tanh(), means.new_ones(len(means), 1)), -1)
        basis = torch.einsum('nk,khfc->nhfc', keys, self.kernel)
        pairs = [complete[h:] & complete[:-h] for h in range(1, HORIZON + 1)]
        horizon_targets = [int(p.any(-1).sum()) for p in pairs]
        write_mask = torch.zeros(len(means), dtype=torch.bool, device=means.device)
        for h, pair in enumerate(pairs, 1):
            write_mask[h:] |= pair.any(-1)
        writes = write_mask.detach().cpu().tolist()
        context = means.new_zeros(CONTEXT_DIM)
        forecasts, contexts, norms, zeros, before = [], [], [], [], []
        first_target = first_mask = None
        for j in range(len(means)):
            if return_before:
                before.append((basis[j] @ context).flatten())
            zero = torch.zeros((), dtype=torch.bool, device=means.device)
            if j:
                count = min(j, HORIZON)
                origins = torch.arange(j - 1, j - count - 1, -1, device=means.device)
                target = clean[j] - clean[origins]
                mask = complete[j] & complete[origins]
                arriving_basis = basis[origins, torch.arange(count, device=means.device)]
                if first_target is None:
                    first_target, first_mask = target, mask
                if mode != 'off' and writes[j]:
                    if mode == 'hold':
                        target = torch.zeros_like(target)
                    elif mode == 'reverse':
                        # Leave each issued basis in its original horizon.
                        # Flipping BOTH basis and target would merely reorder a
                        # sum and produce an identically unchanged gradient.
                        target = target.flip(0)
                        mask = mask & mask.flip(0)
                    if expected is not None:
                        # Use the expectation issued at each earlier origin,
                        # not a recomputed expectation from the arriving state.
                        target = target - expected[origins, torch.arange(count, device=means.device)]
                    current = torch.zeros_like(context) if mode == 'local' else context
                    context = self.write(current, arriving_basis, target, mask)
                    if controlled:
                        norm = context.norm()
                        desired = reference_norm[j].detach()
                        zero = (norm == 0) & (desired > 0)
                        context = context * desired / norm.clamp_min(torch.finfo(norm.dtype).tiny)
            forecasts.append(basis[j] @ context)
            contexts.append(context)
            norms.append(context.norm())
            zeros.append(zero)
        forecast = torch.stack(forecasts) if forecasts else means.new_zeros(0, HORIZON, 23)
        residual = forecast
        if expected is not None:
            forecast = expected + residual
        if len(means) > HORIZON:
            target = torch.stack([clean[h:len(means)-HORIZON+h] - clean[:-HORIZON]
                                  for h in range(1, HORIZON+1)], 1)
            mask = torch.stack([complete[h:len(means)-HORIZON+h] & complete[:-HORIZON]
                                for h in range(1, HORIZON+1)], 1)
            loss = ((forecast[:-HORIZON] - target).square() * mask).sum() / mask.sum().clamp_min(1)
        else:
            loss = clean.sum() * 0
        result = dict(delta=residual.flatten(1), forecast=forecast, forecast_loss=loss,
                      context=torch.stack(contexts) if contexts else means.new_zeros(0, CONTEXT_DIM),
                      norm=torch.stack(norms) if norms else means.new_zeros(0),
                      zero_direction=torch.stack(zeros) if zeros else complete.new_zeros(0),
                      updates=sum(writes) if mode != 'off' else 0,
                      first_target=first_target, first_mask=first_mask,
                      horizon_targets=horizon_targets, target_scalars=sum(int(p.sum()) for p in pairs))
        if return_before:
            result['before_delta'] = torch.stack(before) if before else means.new_zeros(0, 92)
        if expected is not None:
            result['expected_forecast'] = expected
        return result


class ContextAdapter(LifecycleAdapter):
    def __init__(self, base, *, readout='full', feedback='progressive', forecast_weights=None):
        if readout != 'full' or feedback != 'progressive':
            raise ValueError('context experiment requires full readout/progressive feedback')
        super().__init__(base, readout=readout, feedback=feedback)
        if forecast_weights is None:
            self.memory = ContextMemory(base.cfg.dim)
        else:
            from .scaled import ScaledContextMemory
            self.memory = ScaledContextMemory(base.cfg.dim, forecast_weights)
        self.memory.requires_grad_(False)
        self.memory.output.requires_grad_(True)
