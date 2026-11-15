"""Frozen-parent Q/K gradients from a fully matured, signal-only H25."""
import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch


def observed_gradients(net, signal, origin, slow):
    if (set(signal) != {'values', 'available'} or origin < 127
            or origin+25 >= len(signal['values'])):
        raise ValueError('complete past and matured H25 signal only required')
    prefix = {k: v[:origin+1] for k, v in signal.items()}
    x, available = batch(prefix, [origin], net.base.center.device)
    truth = torch.as_tensor(signal['values'][origin+1:origin+26], device=x.device, dtype=x.dtype)[None]
    mask = torch.as_tensor(signal['available'][origin+1:origin+26], device=x.device)[None].clone()
    mask &= torch.isfinite(truth) & available[:, -1:]
    mask[:, :, 10] = False
    with torch.enable_grad():
        state = x.new_zeros(1, 8192, requires_grad=True)
        result = net(x, available, state, slow, 'qk', forecast=True)
        direct = net.base.forecast_loss(result['forecast'], truth, mask)
        reverse = net.base.forecast_loss(result['forecast'], truth.flip(1), mask.flip(1))
        g, = torch.autograd.grad(direct, state, retain_graph=True)
        rg, = torch.autograd.grad(reverse, state, retain_graph=True)
        risk, = torch.autograd.grad(result['logit'].sum(), state)
    if not all(torch.isfinite(v).all() for v in (g, rg, risk)):
        raise FloatingPointError('nonfinite observed gradient')
    info = dict(observed=int(mask.sum()), signal_loss=float(direct.detach()),
                reverse_loss=float(reverse.detach()), signal_norm=float(g.norm()),
                risk_norm=float(risk.norm()), origin_logit=float(result['logit'].detach().item()))
    return g.detach(), rg.detach(), risk.detach(), info


def unit(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / torch.where(norm > 0, norm, torch.ones_like(norm))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value
