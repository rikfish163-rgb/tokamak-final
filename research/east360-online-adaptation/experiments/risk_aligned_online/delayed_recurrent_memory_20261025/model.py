"""Delayed, direct diagnostic forecasts update a shot-local recurrent state.

Only arrived signals enter writes. Issued forecasts are immutable. The state
changes an internal token, never a final risk logit or alarm threshold.
"""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..internal_fast_window_20261023.model import WindowAdapter

ARMS = ('memory', 'local', 'hold', 'reverse')
PATCH = 8
HORIZON = 4


def patch_signal(signal, center, scale):
    if set(signal) != {'values', 'available'}:
        raise ValueError('only values and availability may enter online memory')
    x = torch.as_tensor(signal['values'], device=center.device, dtype=center.dtype)
    a = torch.as_tensor(signal['available'], device=center.device)
    if x.ndim != 2 or x.shape[1] != 23 or a.shape != x.shape or a.dtype != torch.bool:
        raise ValueError('flat23 time series and boolean availability required')
    a = a & torch.isfinite(x)
    x = torch.asinh((torch.where(a, x, center)-center)/scale)
    n = len(x)//PATCH
    x, a = x[:n*PATCH].reshape(n, PATCH, 23), a[:n*PATCH].reshape(n, PATCH, 23)
    complete = a.all(1)
    # Partial patches never create an unobserved full-patch target or key value.
    means = torch.where(complete, x.mean(1), 0.)
    return means, complete


def endpoint_corrections(delta, ends):
    indices = torch.div(ends, PATCH, rounding_mode='floor')-1
    if not len(delta):
        return delta.new_zeros((len(ends), 92))
    return torch.where((indices >= 0)[:, None], delta[indices.clamp_min(0)], 0.)


class DiagnosticMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(46, 8)
        self.prior = nn.Parameter(torch.zeros(9, 92))
        self.rate = nn.Parameter(torch.tensor(0.))
        self.output = nn.Linear(92, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    @property
    def eta(self):
        return .5*self.rate.sigmoid()

    def write(self, state, key, target, mask):
        error = (key@state-target)*mask
        candidate = state-self.eta*key[:, None]*error[None, :]/(1+key.square().sum())
        delta = candidate-self.prior
        return self.prior+delta/torch.linalg.vector_norm(delta).clamp_min(1.)

    def trace(self, means, complete, *, mode, reference_norm=None):
        if (mode not in ('off',)+ARMS or means.ndim != 2 or means.shape[1] != 23
                or complete.shape != means.shape or complete.dtype != torch.bool):
            raise ValueError('declared mode and observed diagnostic patches required')
        if mode in ('local','hold','reverse') and (
                reference_norm is None or reference_norm.shape != means.shape[:1]):
            raise ValueError('controls require main-arm norm at every patch')
        features = torch.cat((means*complete, complete.to(means)), -1)
        keys = torch.cat((self.key(features).tanh(), means.new_ones(len(means),1)), -1)
        w = self.prior
        deltas, forecasts, norms, zeros, targets, masks = [], [], [], [], [], []
        valid_writes = (complete[:-4] & complete[1:-3] | complete[:-4] & complete[2:-2]
                        | complete[:-4] & complete[3:-1] | complete[:-4] & complete[4:]).any(-1) if len(means)>4 else torch.zeros(0,dtype=torch.bool)
        writes = valid_writes.detach().cpu().tolist()
        for j in range(len(means)):
            zero = torch.zeros((), dtype=torch.bool, device=means.device)
            if j >= HORIZON:
                origin = j-HORIZON
                target = means[origin+1:j+1]-means[origin]
                mask = complete[origin+1:j+1] & complete[origin]
                targets.append(target); masks.append(mask)
                if mode != 'off' and writes[origin]:
                    current_target, current_mask = target, mask
                    if mode == 'hold': current_target = torch.zeros_like(target)
                    if mode == 'reverse': current_target, current_mask = target.flip(0), mask.flip(0)
                    w = self.write(self.prior if mode == 'local' else w, keys[origin],
                                   current_target.flatten(), current_mask.flatten())
                    if mode in ('local','hold','reverse'):
                        change = w-self.prior
                        norm = torch.linalg.vector_norm(change)
                        desired = reference_norm[j].detach()
                        zero = (norm == 0) & (desired > 0)
                        # A zero direction is explicitly reported, not replaced
                        # with a fabricated direction to satisfy norm matching.
                        w = self.prior+change*desired/norm.clamp_min(torch.finfo(norm.dtype).tiny)
            forecasts.append((keys[j]@w).reshape(4,23))
            deltas.append(keys[j]@(w-self.prior))
            norms.append(torch.linalg.vector_norm(w-self.prior))
            zeros.append(zero)
        forecast = torch.stack(forecasts) if forecasts else means.new_zeros(0,4,23)
        if targets:
            ys, ms = torch.stack(targets), torch.stack(masks)
            forecast_loss = ((forecast[:-4]-ys).square()*ms).sum()/ms.sum().clamp_min(1)
        else:
            forecast_loss = means.sum()*0
        return dict(delta=torch.stack(deltas) if deltas else means.new_zeros(0,92),
                    forecast=forecast, forecast_loss=forecast_loss,
                    norm=torch.stack(norms) if norms else means.new_zeros(0),
                    zero_direction=torch.stack(zeros) if zeros else torch.zeros(0,dtype=torch.bool,device=means.device),
                    updates=sum(writes) if mode != 'off' else 0,
                    first_target=targets[0] if targets else None, first_mask=masks[0] if masks else None)


class RecurrentAdapter(WindowAdapter):
    def __init__(self, base, *, memory_class=DiagnosticMemory):
        super().__init__(base)
        del self.layer
        self.memory = memory_class(base.cfg.dim)

    def internal_shift(self, last, delta):
        return self.memory.output(delta)

    def risk(self, h, valid, delta):
        if delta.shape != (len(h),92):
            raise ValueError('one causal memory read per prediction required')
        h = h.clone()
        h[:, -2] = h[:, -2]+self.internal_shift(h[:, -2], delta)*valid[:, -2, None]
        with sdpa_kernel(SDPBackend.MATH):
            h = self.base.encoder.layers[1](h, src_key_padding_mask=~valid)
        gain,bias = self.base.initial_fast(len(h)).chunk(2,-1)
        h = self.base.norm(h)*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
        return self.base.risk_head(h[:,-1]).squeeze(-1)
