"""Same-state decomposition, not a learned or deployable update policy."""
import numpy as np
import torch

from ..increment_forecast_20261023.learning import target_mask


def lead_ms(times, down):
    # Match simple_train.metrics exactly, including strict >30 ms at decimal
    # boundaries. Binary float noise must not make a 30 ms endpoint timely.
    return np.asarray([float(round((down-t)*1000.,9)) for t in times])


def losses(model, prediction, x, available, truth, truth_mask):
    mask = target_mask(x, available, truth, truth_mask)
    flat = x[:, -1:].expand_as(truth)
    real = model.base.forecast_loss(prediction, truth, mask)
    constant = model.base.forecast_loss(prediction, flat, mask)
    return dict(real=real, flat=constant, change=real-constant)


def vector_geometry(values, reference):
    if len(values) != len(reference) or not values:
        raise ValueError('matching nonempty gradient tuples required')
    if any(a.shape != b.shape for a, b in zip(values, reference)):
        raise ValueError('matching gradient shapes required')
    norm = sum(v.double().square().sum() for v in values).sqrt()
    refnorm = sum(v.double().square().sum() for v in reference).sqrt()
    dot = sum((a.double()*b.double()).sum() for a, b in zip(values, reference))
    if not all(torch.isfinite(v) for v in (norm, refnorm, dot)):
        raise FloatingPointError('nonfinite gradient geometry')
    cosine = float(dot/(norm*refnorm)) if norm > 0 and refnorm > 0 else None
    return dict(norm=float(norm), reference_norm=float(refnorm), dot=float(dot), cosine=cosine)


def trial_step(fast, gradients):
    norm = sum(g.square().sum() for g in gradients).sqrt()
    factor = (1. / norm.clamp_min(1.)).detach()
    return {k: (v-.001*factor*g).detach().requires_grad_(True)
            for (k,v),g in zip(fast.items(), gradients)}
