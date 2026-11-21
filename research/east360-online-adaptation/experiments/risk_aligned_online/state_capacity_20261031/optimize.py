"""Label-known oracle capacity probe, never a deployable or online score.

The final token is CLS. The preceding 16 are data tokens; slow and the
context-memory output normally affect only the last of these. The free
sequence path changes valid data tokens at exactly that same block boundary.
The free-path radius is a context-ball operator norm, not a safety limit.
"""

import math
import numbers

import torch
from torch.nn import functional as F

from ..context_memory_20261028.model import ContextAdapter
from ..delayed_recurrent_memory_20261025.model import RecurrentAdapter
from ..lifecycle_stream_20261025.model import LifecycleAdapter


def _check_model(model, h, valid):
    if not isinstance(model, ContextAdapter):
        raise ValueError('a ContextAdapter with the original risk path is required')
    if (getattr(model.risk, '__func__', None) is not LifecycleAdapter.risk
            or getattr(model.internal_shift, '__func__', None) is not RecurrentAdapter.internal_shift
            or model.readout != 'full' or model.memory.output.bias is not None):
        raise ValueError('original full context readout and bias-free memory output required')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('all model modules must be eval and all parameters frozen')
    if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[1:] != (17, 64) or not len(h):
        raise ValueError('h must be nonempty [N,17,64]')
    if h.dtype not in (torch.float32, torch.float64) or not torch.isfinite(h).all():
        raise ValueError('h must be finite float32 or float64')
    if (not isinstance(valid, torch.Tensor) or valid.shape != h.shape[:2]
            or valid.dtype != torch.bool or valid.device != h.device or not valid[:, -1].all()):
        raise ValueError('valid must be a matching boolean mask with valid final CLS')
    for value in list(model.parameters()) + list(model.buffers()):
        if value.device != h.device or (value.is_floating_point() and value.dtype != h.dtype):
            raise ValueError('model, h and floating model state must share device and dtype')
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError('model state must be finite')
    if model.memory.output.weight.shape != (64, 92) or model.slow.weight.shape != (64, 64):
        raise ValueError('expected 92-to-64 memory output and 64-to-64 slow layer')


def _check_float(name, value, shape, h):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != h.dtype
            or value.device != h.device or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must have shape {shape}, and match h dtype/device with finite values')


def _risk_with_shift(model, h, valid, shift):
    # Reconstruct LifecycleAdapter.risk's slow operation first. Calling the
    # RecurrentAdapter implementation directly avoids applying slow twice.
    last = h[:, -2] + model.slow(h[:, -2]) * valid[:, -2, None]
    data = torch.cat((h[:, :-2], last[:, None]), dim=1)
    data = data + shift * valid[:, :-1, None]
    adapted = torch.cat((data, h[:, -1:]), dim=1)
    return RecurrentAdapter.risk(model, adapted, valid, h.new_zeros((len(h), 92)))


def risk_with_shift(model, h, valid, shift):
    """Replay [N,16,64] shifts after slow, masking padding and protecting CLS.

    The function does not change model parameters, gradients or train/eval
    flags. It remains differentiable with respect to h and shift.
    """
    _check_model(model, h, valid)
    _check_float('shift', shift, (len(h), 16, 64), h)
    return _risk_with_shift(model, h, valid, shift)


def _project(parameter, radius, mask):
    parameter = parameter * mask
    norm = parameter.flatten(1).norm(dim=1)
    factor = (radius / norm.clamp_min(torch.finfo(parameter.dtype).tiny)).clamp(max=1)
    return parameter * factor.reshape((-1,) + (1,) * (parameter.ndim - 1))


def _summary(logits, loss, y, threshold):
    pred, label = logits.sigmoid() >= threshold, y.bool()
    return dict(count=len(y), mean_loss=float(loss.mean()), sum_loss=float(loss.sum()),
                selected_endpoint_positive_targets=int(label.sum()),
                selected_endpoint_negative_targets=int((~label).sum()),
                selected_endpoint_positive_met=int((pred & label).sum()),
                selected_endpoint_negative_met=int((~pred & ~label).sum()))


def _run_path(name, starts, init_labels, radius, mask, predict, to_shift,
              y, threshold, steps, lr, progress):
    n = len(y)
    best_loss = y.new_full((n,), float('inf'))
    best_parameter = torch.zeros_like(starts[0])
    best_step = torch.zeros(n, dtype=torch.long, device=y.device)
    best_init = torch.zeros_like(best_step)
    for init, start in enumerate(starts):
        parameter = torch.nn.Parameter(_project(start.detach(), radius, mask))
        optimizer = torch.optim.Adam([parameter], lr=lr)
        for step in range(steps + 1):
            logits = predict(parameter)
            losses = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
                raise FloatingPointError(f'nonfinite oracle loss in {name}')
            with torch.no_grad():
                better = losses < best_loss
                best_loss = torch.where(better, losses.detach(), best_loss)
                select = better.reshape((-1,) + (1,) * (parameter.ndim - 1))
                best_parameter = torch.where(select, parameter.detach(), best_parameter)
                best_step = torch.where(better, step, best_step)
                best_init = torch.where(better, init, best_init)
            if progress is not None and step and step % 20 == 0:
                progress(dict(path=name, init=init, step=step, steps=steps,
                              best_mean_loss=float(best_loss.mean())))
            if step == steps:
                break
            grad, = torch.autograd.grad(losses.sum(), parameter)
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f'nonfinite oracle gradient in {name}')
            parameter.grad = grad
            optimizer.step()
            with torch.no_grad():
                parameter.copy_(_project(parameter, radius, mask))
    # Recompute through the same shift replay interface given to the caller.
    # predict may use model.risk for context8, hence to_shift owns replay too.
    with torch.no_grad():
        best_shift, best_logits = to_shift(best_parameter)
        replay_loss = F.binary_cross_entropy_with_logits(best_logits, y, reduction='none')
    return dict(best_parameter=best_parameter.detach(), best_shift=best_shift.detach(),
                best_logits=best_logits.detach(), best_loss=replay_loss.detach(),
                selection_loss=best_loss.detach(),
                best_step=best_step.detach(), best_init=best_init.detach(),
                init_labels=init_labels, parameter_radius=radius.detach(),
                summary=_summary(best_logits, replay_loss, y, threshold))


def probe(model, h, valid, basis, actual_context, y, *, enabled, threshold,
          steps=100, lr=.1, progress=None):
    """Optimize three per-example oracle paths with known binary labels.

    ``enabled`` is supplied by the caller's unchanged >=16 clock. Two fixed
    starts per path, step 0 plus ``steps`` Adam updates, are compared per case.
    ``progress``, if supplied, receives a scalar-only event every 20 updates;
    logger exceptions propagate and abort the probe. Returned tensors remain
    on the input device and can be saved by torch.save. Nothing here selects a
    model, checkpoint or deployable online state.
    """
    _check_model(model, h, valid)
    n = len(h)
    _check_float('basis', basis, (n, 92, 8), h)
    _check_float('actual_context', actual_context, (n, 8), h)
    if (not isinstance(enabled, torch.Tensor) or enabled.shape != (n,)
            or enabled.dtype != torch.bool or enabled.device != h.device):
        raise ValueError('enabled must be a boolean [N] tensor on h device')
    if (not isinstance(y, torch.Tensor) or y.shape != (n,) or y.device != h.device
            or y.is_complex() or not torch.isfinite(y).all()
            or not ((y == 0) | (y == 1)).all()):
        raise ValueError('y must contain one finite binary label per case on h device')
    if not isinstance(threshold, numbers.Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite probability in [0,1]')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not isinstance(lr, numbers.Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')
    if torch.is_inference_mode_enabled():
        raise ValueError('oracle optimization requires gradients; disable inference_mode')

    # Input graphs are irrelevant to this post-hoc optimization and must not
    # receive gradients. No model requires_grad flags or modes are modified.
    h, valid = h.detach(), valid.detach()
    basis, actual_context = basis.detach(), actual_context.detach()
    enabled, y = enabled.detach(), y.detach().to(dtype=h.dtype)
    if (actual_context.norm(dim=-1)[enabled] > 1 + 1e-6).any():
        raise ValueError('enabled actual_context must lie in the unit ball')
    with torch.no_grad():
        # This radius defines a comparison constraint, not an approximate
        # learned feature. The real61629 case exceeded the declared independent
        # double-precision check when CUDA FP32 matmul/SVD was used (v1 audit).
        # Compute both matrix product and singular value in float64, then round
        # only the final radius to the model dtype used for projected search.
        radius_matrix = model.memory.output.weight.double() @ basis.double()
        radius = torch.linalg.svdvals(radius_matrix)[:, 0].to(h.dtype)
        if not torch.isfinite(radius).all():
            raise FloatingPointError('nonfinite context-ball output radius')
        radius = radius * enabled
        actual = actual_context * enabled[:, None]
        actual_delta = torch.bmm(basis, actual.unsqueeze(-1)).squeeze(-1)
        zero_logits = model.risk(h, valid, h.new_zeros(n, 92))
        actual_logits = model.risk(h, valid, actual_delta)
        if not torch.isfinite(zero_logits).all() or not torch.isfinite(actual_logits).all():
            raise FloatingPointError('nonfinite reference logits')

    data_mask = (valid[:, :-1] & enabled[:, None])[:, :, None]
    context_mask = enabled[:, None]
    last_mask = (enabled & valid[:, -2])[:, None]
    context_radius = enabled.to(h.dtype)

    def context_delta(c):
        return torch.bmm(basis, c.unsqueeze(-1)).squeeze(-1)

    def context_shift(c):
        shift = h.new_zeros((n, 16, 64))
        shift[:, -1] = model.memory.output(context_delta(c)) * last_mask
        return shift

    def last_shift(p):
        shift = h.new_zeros((n, 16, 64))
        shift[:, -1] = p * last_mask
        return shift

    def replay(shift):
        return shift, _risk_with_shift(model, h, valid, shift)

    with torch.enable_grad():
        context = _run_path('context8', (torch.zeros_like(actual), actual), ('zero', 'actual_context'),
                            context_radius, context_mask,
                            lambda c: model.risk(h, valid, context_delta(c)),
                            lambda c: replay(context_shift(c)), y, threshold, steps, lr,
                            progress)
        last = _run_path('free_last', (h.new_zeros((n, 64)), context['best_shift'][:, -1]),
                         ('zero', 'best_context8'), radius, last_mask,
                         lambda p: _risk_with_shift(model, h, valid, last_shift(p)),
                         lambda p: replay(last_shift(p)), y, threshold, steps, lr,
                         progress)
        sequence = _run_path('free_sequence', (h.new_zeros((n, 16, 64)), last['best_shift']),
                             ('zero', 'best_free_last'), radius, data_mask,
                             lambda p: _risk_with_shift(model, h, valid, p * data_mask),
                             lambda p: replay(p * data_mask), y, threshold, steps, lr,
                             progress)
    for path in (context, last, sequence):
        path['radius'] = radius.detach()
    return dict(diagnostic='label-known post-hoc oracle; not online performance or a deployable checkpoint',
                radius_definition='largest singular value of memory.output.weight @ basis; not a safety limit',
                threshold=float(threshold), steps=steps, lr=float(lr), labels=y,
                enabled=enabled, radius=radius.detach(), actual_logits=actual_logits.detach(),
                zero_logits=zero_logits.detach(), actual_context=actual_context,
                actual_context_norm=actual_context.norm(dim=-1),
                effective_actual_context=actual,
                paths=dict(context8=context, free_last=last, free_sequence=sequence))
