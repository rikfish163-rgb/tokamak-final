"""Causal, ordered diagnostic changes for a representation-target candidate."""

import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from .learning import ordered_context


def observation(net, signal, origin, *, reverse=False):
    if (set(signal) != {'values', 'available'} or not isinstance(origin, int)
            or origin < 127 or origin+25 >= len(signal['values'])
            or signal['values'].shape != signal['available'].shape):
        raise ValueError('signal-only origin and complete matured 25-point target required')
    prefix = {k: v[:origin+1] for k, v in signal.items()}
    x, available = batch(prefix, [origin], net.base.center.device)
    with torch.no_grad():
        h, _, _ = net.first(x, available, x.new_zeros(1, 8192), 'qk')
        truth = torch.as_tensor(signal['values'][origin+1:origin+26], device=x.device, dtype=x.dtype)[None]
        mask = torch.as_tensor(signal['available'][origin+1:origin+26], device=x.device, dtype=torch.bool)[None].clone()
        mask &= torch.isfinite(truth) & available[:, -1:] & torch.isfinite(x[:, -1:])
        mask[:, :, 10] = False
        center, scale = net.base.center, net.base.scale
        # Values outside the available mask are not treated as real zero observations.
        arrived = torch.asinh((torch.where(mask, truth, center)-center)/scale)
        anchor = torch.asinh((torch.where(mask, x[:, -1:], center)-center)/scale)
        change = torch.where(mask, arrived-anchor, 0.)
        if reverse:
            change, mask = change.flip(1), mask.flip(1)
        context, support = ordered_context(h[:, -1], h[:, -2], change, mask)
    return dict(context=context, support=support, x=x, available=available,
                reference=h[:, -1].detach())
