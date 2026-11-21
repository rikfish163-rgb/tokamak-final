"""Source-aligned encoder-copy probe; no source model or data is modified.

The caller supplies only feedback that has already matured.  This kernel does
not establish that an update improves disruption warnings, and it does not
enforce an episode's clock or retain temporary state between episodes.
"""

from copy import deepcopy
import math

import torch
from torch.func import functional_call

from experiments.risk_aligned_online.simple23 import Simple23


def _representation_names(model):
    """The shared representation, excluding both heads and fast_origin."""
    return tuple(name for name, _ in model.named_parameters()
                 if name in ('position', 'cls')
                 or name.startswith(('embedding.', 'encoder.', 'norm.')))


def copy_for_training(base):
    """Copy a frozen source-view Simple23 and open only representation/forecast.

    The risk head, fast origin, and source normalization remain fixed.  The
    original is never changed, and the copy begins with identical state.
    """
    if not isinstance(base, Simple23) or base.cfg.input_view != 'source':
        raise ValueError('a source-view Simple23 is required')
    if base.training or any(parameter.requires_grad for parameter in base.parameters()):
        raise ValueError('the source must be frozen and in eval mode')
    model = deepcopy(base)
    names = set(_representation_names(model))
    if not names:
        raise ValueError('source representation parameters not found')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names or name.startswith('forecast_head.'))
    return model


def forward_features(model, x, available, *, parameters=None, return_tokens=False):
    """Run Simple23 and capture differentiable inputs to its two heads.

    ``risk_features`` is the risk-head input.  Optional ``tokens`` are the
    complete post-normalization, post-fast-state, masked token representation
    that feeds the forecast head; its final token also feeds the risk head.
    Hooks are temporary, so calls on the same model must not run concurrently.
    ``parameters`` is a functional_call parameter mapping when provided.
    """
    risk, flattened = [], []
    handles = []
    try:
        handles.append(model.risk_head.register_forward_pre_hook(
            lambda _module, args: risk.append(args[0])))
        if return_tokens:
            handles.append(model.forecast_head.register_forward_pre_hook(
                lambda _module, args: flattened.append(args[0])))
        output = (model(x, available) if parameters is None else
                  functional_call(model, parameters, (x, available)))
    finally:
        for handle in handles:
            handle.remove()
    if len(risk) != 1 or (return_tokens and len(flattened) != 1):
        raise RuntimeError('expected one invocation of each Simple23 head')
    result = dict(output, risk_features=risk[0])
    if return_tokens:
        result['tokens'] = flattened[0].reshape(
            x.shape[0], model.cfg.window // model.cfg.patch + 1, model.cfg.dim)
    return result


def _norm(tensors):
    # Accumulate in float64 for a stable global parameter-space norm.
    return torch.linalg.vector_norm(torch.cat([x.detach().reshape(-1).double()
                                                for x in tensors]))


def encoder_parameter_norm(model):
    """Euclidean norm of the representation parameters, excluding both heads."""
    names = set(_representation_names(model))
    if not names:
        raise ValueError('representation parameters not found')
    return _norm([parameter for name, parameter in model.named_parameters() if name in names])


def inner_gradient(model, x0, a0, truth, mask):
    """One independent episode's forecast gradient in representation space.

    ``truth`` is the raw future signal, not a change or a shape potential;
    Simple23.forecast_loss applies the original asinh normalization and mask.
    Every representation leaf is detached from the model.  Neither autograd
    nor the functional forward mutates model parameters or accumulates grads.
    """
    if x0.ndim != 3 or x0.shape[0] != 1:
        raise ValueError('a single independent episode is required')
    if model.training:
        raise ValueError('the copy must be in eval mode for the inner update')
    names = _representation_names(model)
    if not names:
        raise ValueError('representation parameters not found')
    name_set = set(names)
    parameters = {name: p.detach().clone().requires_grad_(name in name_set)
                  for name, p in model.named_parameters()}
    with torch.enable_grad():
        prediction = functional_call(model, parameters, (x0, a0))['forecast']
        loss = model.forecast_loss(prediction, truth, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite forecast loss')
        values = torch.autograd.grad(loss, tuple(parameters[name] for name in names))
    gradients = dict(zip(names, values))
    if not all(torch.isfinite(gradient).all() for gradient in values):
        raise FloatingPointError('nonfinite representation gradient')
    return dict(parameters=parameters, gradients=gradients, loss=loss.detach(),
                gradient_norm=_norm(values))


def _budget_values(reference_norm, budget):
    reference_norm, budget = float(reference_norm), float(budget)
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError('reference_norm must be positive and finite')
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('budget must be nonnegative and finite')
    return reference_norm, budget


def _step(gradient_result, reference_norm, budget, *, common_norm=None):
    norm = float(gradient_result['gradient_norm'])
    if not math.isfinite(norm):
        raise FloatingPointError('nonfinite global representation gradient norm')
    scale = budget / max(norm, reference_norm)
    if common_norm is not None:
        scale = 0. if norm == 0. else common_norm / norm
    if norm == 0. or scale == 0.:
        # Avoid even adding signed zero to preserve the exact base tensor bits.
        delta = {name: torch.zeros_like(g) for name, g in gradient_result['gradients'].items()}
        parameters = dict(gradient_result['parameters'])
    else:
        if common_norm is None:
            delta = {name: -scale * g for name, g in gradient_result['gradients'].items()}
        else:
            # Form the matched direction in float64 so a tiny nonzero wrong
            # gradient cannot overflow a low-precision scaling factor.
            delta = {name: (-(g.double() / norm) * common_norm).to(g.dtype)
                     for name, g in gradient_result['gradients'].items()}
        parameters = {name: (value + delta[name] if name in delta else value)
                      for name, value in gradient_result['parameters'].items()}
    return dict(parameters=parameters, delta=delta, loss=gradient_result['loss'],
                gradient_norm=gradient_result['gradient_norm'], update_norm=_norm(delta.values()))


def inner_step(model, x0, a0, truth, mask, *, reference_norm, budget):
    """Take a bounded, non-mutating representation-only step for one episode.

    The update is ``-budget * gradient / max(||gradient||, reference_norm)``.
    ``reference_norm`` must be fixed from source TRAIN rather than tuned on the
    evaluation episode.  Use returned ``parameters`` with functional_call.
    """
    reference_norm, budget = _budget_values(reference_norm, budget)
    return _step(inner_gradient(model, x0, a0, truth, mask), reference_norm, budget)


def paired_steps(model, x0, a0, truth, wrong_truth, mask, *, reference_norm, budget):
    """Keep the real update and norm-match the wrong-feedback direction.

    A zero wrong gradient cannot be matched to a nonzero real update: in that
    case the real arm is unchanged, the wrong arm is a no-op, and
    ``unmatched_control`` is true.  Both targets must have exactly the same
    observed finite-value mask.  Norms agree up to tensor dtype roundoff.
    """
    reference_norm, budget = _budget_values(reference_norm, budget)
    if truth.shape != wrong_truth.shape or mask.shape != truth.shape or mask.dtype != torch.bool:
        raise ValueError('matching feedback tensors and boolean mask required')
    if not torch.equal(mask & torch.isfinite(truth), mask & torch.isfinite(wrong_truth)):
        raise ValueError('real and wrong feedback must have the same effective mask')
    real = inner_gradient(model, x0, a0, truth, mask)
    wrong = inner_gradient(model, x0, a0, wrong_truth, mask)
    real_step = _step(real, reference_norm, budget)
    wrong_step = _step(wrong, reference_norm, budget, common_norm=float(real_step['update_norm']))
    unmatched = bool(real_step['update_norm'] > 0 and wrong_step['update_norm'] == 0)
    return dict(real=real_step, wrong=wrong_step, unmatched_control=unmatched)
