"""Differentiable, accumulated AdamW for a single parameter matrix."""

import math

import torch


def adam_step(weight, gradient, history=(), *, lr=1e-4):
    """Apply an AdamW step using a tuple of previously clipped gradients.

    The optimizer uses betas (0.9, 0.999), epsilon 1e-8, zero weight
    decay, and global gradient-norm clipping at 1. The returned history keeps
    its computation graph and can be passed directly to the next step.
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    total_norm = torch.linalg.vector_norm(gradient)
    if not torch.isfinite(total_norm):
        raise RuntimeError("The total gradient norm is non-finite, so it cannot be clipped")
    clip = (1.0 / (total_norm + 1e-6)).clamp(max=1.0)
    history = history + (gradient * clip,)

    step = len(history)
    exp_avg = torch.zeros_like(gradient)
    exp_avg_sq = torch.zeros_like(gradient)
    weighted_history = []
    for index, clipped in enumerate(history):
        exp_avg = torch.lerp(exp_avg, clipped, 1 - beta1)
        exp_avg_sq = torch.addcmul(beta2 * exp_avg_sq, clipped, clipped, value=1 - beta2)
        scale = math.sqrt((1 - beta2) * beta2 ** (step - index - 1))
        weighted_history.append(clipped * scale)

    # The norm has a finite first derivative at an all-zero history, unlike
    # differentiating sqrt(exp_avg_sq) when its argument is zero.
    root_second_moment = torch.linalg.vector_norm(torch.stack(weighted_history), dim=0)
    denominator = root_second_moment / math.sqrt(1 - beta2 ** step) + eps
    updated = torch.addcdiv(weight, exp_avg, denominator, value=-lr / (1 - beta1 ** step))
    return {
        "weight": updated,
        "history": history,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }


def direction_candidate(weight, momentum, adam_weight):
    """The same momentum-direction map for functional and actual optimizers.

    Both callers supply a nonnegative-learning-rate Adam candidate. The zero
    branch preserves that candidate's first-derivative convention; it is not
    a claim that a cancelled momentum is differentiable.
    """
    delta = adam_weight - weight
    momentum_scale = momentum.abs().max()
    delta_scale = delta.abs().max()
    if momentum_scale == 0:
        momentum_norm = torch.linalg.vector_norm(momentum)
        adam_step_norm = torch.linalg.vector_norm(delta)
        updated = adam_weight
    else:
        scaled_momentum = momentum / momentum_scale
        scaled_momentum_norm = torch.linalg.vector_norm(scaled_momentum)
        momentum_norm = momentum_scale * scaled_momentum_norm
        if delta_scale == 0:
            adam_step_norm = torch.linalg.vector_norm(delta)
            updated = adam_weight
        else:
            adam_step_norm = delta_scale * torch.linalg.vector_norm(delta / delta_scale)
            updated = weight - adam_step_norm * (scaled_momentum / scaled_momentum_norm)
    return dict(weight=updated, adam_weight=adam_weight,
                adam_step_norm=adam_step_norm, momentum_norm=momentum_norm)


def direction_step(weight, gradient, history=(), *, lr=1e-4):
    """Replace Adam's direction with its momentum, keeping Adam's step length.

    The clipped history and moments are exactly those returned by adam_step.
    No smoothing is applied to nonzero steps. At exactly zero momentum the
    unchanged Adam candidate supplies the autodiff convention: its first
    derivative is correct for an all-zero history, but the exact mapping is
    generally not twice differentiable there. If nonzero history cancels to
    zero momentum, even the first derivative may not exist; Adam's derivative
    there is a convention, not an exact gradient of the directional mapping.
    """
    result = adam_step(weight, gradient, history, lr=lr)
    return {**result, **direction_candidate(weight, result['exp_avg'], result['weight'])}
