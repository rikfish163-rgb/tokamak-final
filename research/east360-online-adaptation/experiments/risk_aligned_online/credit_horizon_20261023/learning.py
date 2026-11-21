"""Same cumulative forward updates, explicitly different derivative horizons."""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..increment_forecast_20261023.learning import feedback_loss, target_mask
from ..paired_feedback_20261022.model import initial_encoder


def observed_step(model, signal, count, fast, *, device, create_graph):
    if set(signal) != {'values', 'available'}:
        raise ValueError('only arrived signal fields allowed')
    prefix = {k:v[:count] for k,v in signal.items()}
    x, mask, truth, tm, _ = forecast_examples(prefix, count, model.cfg, device)
    x, truth = x.to(model.base.center.dtype), truth.to(model.base.center.dtype)
    if not target_mask(x, mask, truth, tm).any():
        return fast, model.base.forecast_head.weight.sum()*0.  # Explicitly absent measured target.
    loss = feedback_loss(model, model.predict_fast(x, mask, fast)['forecast'], x, mask, truth, tm)
    gradient = torch.autograd.grad(loss, tuple(fast.values()), create_graph=create_graph)
    norm = sum(g.square().sum() for g in gradient).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite observed update')
    # Below the clipping boundary the operator is exactly SGD. Explicitly
    # select that branch to avoid sqrt's undefined derivative at zero.
    factor = 1./norm if bool(norm.detach() > 1.) else norm.new_tensor(1.)
    result = {k:v-.001*factor*g for (k,v),g in zip(fast.items(), gradient)}
    if not create_graph:
        result = {k:v.detach().requires_grad_(True) for k,v in result.items()}
    return result, loss


def episode(model, signal, count, ends, disruptive, *, boundary, tail, device):
    if (type(count) != int or count < 256 or count % 128 or not 0 < tail <= 8
            or not ends or ends != sorted(set(ends)) or min(ends) < count
            or max(ends) >= min(count+128, len(signal['values']))):
        raise ValueError('fixed-clock episode and positive derivative tail up to eight required')
    counts = list(range(256, count+1, 128)); retained = counts[-tail:]
    fast = initial_encoder(model)
    for seen in counts:
        fast, loss = observed_step(model, signal, seen, fast, device=device, create_graph=seen in retained)
    windows = [simple_data.window(signal, end, 128) for end in ends]
    x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
    x = x.to(model.base.center.dtype)
    h = model.predict_fast(x, mask, fast, mode='risk_state')['risk_state']
    margin = model.base.risk_head(h).flatten().max()-boundary
    risk = F.softplus(-margin if disruptive else margin)
    with torch.no_grad():
        h0 = model.predict_fast(x, mask, initial_encoder(model), mode='risk_state')['risk_state']
        margin0 = model.base.risk_head(h0).flatten().max()-boundary
        before = F.softplus(-margin0 if disruptive else margin0)
    return dict(risk=risk, before=before, forecast=loss, fast=fast,
                all_counts=counts, differentiated=retained)
