"""Risk-only learning of the existing shot-local context update operator.

Examples contain frozen first-block tokens and diagnostic patch means. Their
preparation, allowed EAST history, optimizer, and fit budget are caller-owned.
"""

import torch

from ..delayed_recurrent_memory_20261025.model import PATCH, endpoint_corrections
from ..lifecycle_stream_20261025.learning import window_loss


def configure_operator_training(model):
    """Freeze the entire adapter except the four context-operator parameters."""
    parameters = [model.memory.key.weight, model.memory.key.bias,
                  model.memory.kernel, model.memory.rate]
    model.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _risk_logits_and_trace(model, example, mode):
    if mode not in ('memory', 'off'):
        raise ValueError('mode must be memory or the zero-delta off control')
    try:
        means, complete = example['means'], example['complete']
        h, valid, ends = example['h'], example['valid'], example['ends']
    except (KeyError, TypeError) as exc:
        raise ValueError('means, complete, h, valid and ends are required') from exc

    if (not isinstance(means, torch.Tensor) or not means.is_floating_point()
            or means.ndim != 2 or means.shape[1] != 23
            or not isinstance(complete, torch.Tensor)
            or complete.shape != means.shape or complete.dtype != torch.bool):
        raise ValueError('flat23 diagnostic means and matching boolean completeness required')
    if (not isinstance(h, torch.Tensor) or not h.is_floating_point() or h.ndim != 3
            or not len(h) or h.shape[1:] != (model.cfg.window // model.cfg.patch + 1, model.cfg.dim)
            or not isinstance(valid, torch.Tensor) or valid.shape != h.shape[:2]
            or valid.dtype != torch.bool):
        raise ValueError('nonempty first-block tokens and matching boolean validity required')
    if (means.device != complete.device or h.device != valid.device
            or means.device != h.device):
        raise ValueError('example tensors must share a device')
    if not torch.isfinite(h).all():
        raise ValueError('nonfinite first-block tokens')
    try:
        ends = torch.as_tensor(ends, device=means.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError('one integer endpoint per token window required') from exc
    if (ends.ndim != 1 or len(ends) != len(h)
            or ends.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('one integer endpoint per token window required')
    ends = ends.to(dtype=torch.long)
    # Incomplete final patches may contain up to PATCH-1 additional samples.
    if (ends < 0).any() or (ends >= PATCH * (len(means) + 1)).any():
        raise ValueError('endpoints must lie within the available patch history')

    # Do not detach the trace: risk gradients must traverse the actual writes.
    trace = model.memory.trace(means, complete, mode=mode)
    if (not torch.isfinite(trace['delta']).all() or not torch.isfinite(trace['norm']).all()
            or not torch.isfinite(trace['forecast_loss'])):
        raise ValueError('nonfinite context memory trace')
    delta = endpoint_corrections(trace['delta'], ends)
    z = torch.cat([model.risk(h[i:i + 32], valid[i:i + 32], delta[i:i + 32])
                   for i in range(0, len(h), 32)])
    if z.shape != (len(h),) or not torch.isfinite(z).all():
        raise ValueError('one finite risk logit per endpoint required')
    return z, trace


def risk_logits(model, example, mode='memory'):
    """Return differentiable risk logits; ``off`` is a zero-delta control."""
    return _risk_logits_and_trace(model, example, mode)[0]


def objective(model, example):
    """Apply the existing revealed-shot window loss to memory-mode logits."""
    try:
        y, flag, h = example['y'], example['flag'], example['h']
    except (KeyError, TypeError) as exc:
        raise ValueError('revealed window labels and shot flag are required') from exc
    if not isinstance(flag, bool):
        raise ValueError('shot flag must be a boolean')
    if (not isinstance(h, torch.Tensor) or h.ndim == 0
            or not isinstance(y, torch.Tensor) or not y.is_floating_point()
            or y.shape != (len(h),) or y.device != h.device
            or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all()):
        raise ValueError('one finite binary label per endpoint is required')
    if not flag and (y != 0).any():
        raise ValueError('a safe shot cannot contain positive labels')
    z, trace = _risk_logits_and_trace(model, example, 'memory')
    loss = window_loss(z, y, flag, 'disruptive_positive_only')
    if not torch.isfinite(loss):
        raise ValueError('nonfinite revealed-shot risk loss')
    detail = dict(writes=trace['updates'],
                  max_state_norm=trace['norm'].max().item() if len(trace['norm']) else 0.,
                  forecast_loss=trace['forecast_loss'].item())
    return loss, detail
