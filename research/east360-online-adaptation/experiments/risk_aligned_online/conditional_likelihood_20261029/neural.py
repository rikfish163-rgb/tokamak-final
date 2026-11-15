"""Independent causal Q/K steps from delayed generative feedback.

The following 25 points are evaluated only after the update is constructed.
They do not form the gradient. These reset episodes are not a streaming policy.
"""

import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from .density import mixture_feedback


def proposal(logit, gradient, logs, counts):
    if (logit.ndim != 1 or gradient.shape != (len(logit), 8192)
            or logs.shape != (len(logit), 2) or counts.shape != logit.shape
            or (counts < 0).any() or not torch.isfinite(counts).all()
            or not torch.isfinite(gradient).all()):
        raise ValueError('finite per-example logits, Q/K gradients, densities and counts required')
    _, _, feedback = mixture_feedback(logit, logs)
    feedback = torch.where(counts > 0, feedback, torch.zeros_like(feedback))
    norm = gradient.norm(dim=1)
    # An exactly zero gradient has no possible local parameter update. This
    # is an explicit no-op, not a fabricated gradient or successful adaptation.
    divisor = torch.where(norm > 0, norm, torch.ones_like(norm))
    state = -.05*feedback.to(gradient.dtype)[:, None]*gradient/divisor[:, None]
    if not torch.isfinite(state).all():
        raise FloatingPointError('nonfinite temporary Q/K proposal')
    return state.detach(), feedback.detach()


def _predict(net, signal, ends, state, slow):
    predictions = []
    with torch.no_grad():
        for lo in range(0, len(ends), 32):
            x, m = batch(signal, ends[lo:lo+32], net.base.center.device)
            z = net(x, m, state[lo:lo+32], slow, 'qk', forecast=False)['logit']
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite subsequent risk')
            predictions.append(z.detach().cpu().numpy())
    return np.concatenate(predictions)


def finite_step(net, signal, ends, slow, log_densities, counts):
    ends = np.asarray(ends)
    if (set(signal) != {'values', 'available'} or ends.ndim != 1
            or ends.dtype.kind not in 'iu' or not len(ends) or ends.min() < 127
            or ends.max()+50 >= len(signal['values'])
            or signal['values'].shape != signal['available'].shape
            or signal['available'].dtype != bool or not log_densities
            or set(log_densities) != set(counts)):
        raise ValueError('signal-only complete past, arrived H25 and subsequent H25 required')
    for arm in log_densities:
        if np.shape(log_densities[arm]) != (len(ends), 2) or np.shape(counts[arm]) != (len(ends),):
            raise ValueError('one pair of densities and support count per endpoint required')
    device = net.base.center.device
    rows = []
    for lo in range(0, len(ends), 8):
        origins = ends[lo:lo+8]
        # Only past windows enter the computation graph; no future input has
        # been formed when the temporary state below is decided.
        x, m = batch(signal, origins, device)
        state = x.new_zeros(len(x), 8192, requires_grad=True)
        origin_z = net(x, m, state, slow, 'qk', forecast=False)['logit']
        gradient, = torch.autograd.grad(origin_z.sum(), state)
        states, evidence = {}, {}
        for arm, logs in log_densities.items():
            states[arm], evidence[arm] = proposal(origin_z.detach(), gradient.detach(),
                torch.as_tensor(logs[lo:lo+len(x)], device=device, dtype=torch.float64),
                torch.as_tensor(counts[arm][lo:lo+len(x)], device=device))
        later = (origins[:, None]+np.arange(26, 51)).reshape(-1)
        baseline = _predict(net, signal, later, x.new_zeros(len(later), 8192), slow)
        result = dict(baseline_logits=baseline.reshape(len(x), 25),
                      origin_logit=origin_z.detach().cpu().numpy())
        for arm, delta in states.items():
            later_state = delta.repeat_interleave(25, dim=0)
            result[arm+'_logits'] = _predict(net, signal, later, later_state, slow).reshape(len(x), 25)
            result[arm+'_state_norm'] = delta.norm(dim=1).cpu().numpy()
            result[arm+'_evidence_gradient'] = evidence[arm].cpu().numpy()
            result[arm+'_changed'] = (delta.norm(dim=1) > 0).cpu().numpy()
        rows.append(result)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
