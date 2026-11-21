"""Differentiable first AdamW step for the single-matrix adaptation experiment."""

import torch
from torch.func import functional_call
from torch.nn import functional as F

from joint_motion_20261107 import OrderedTrajectory
from meta_energy_20261107 import joint_energy_no_self


WEIGHT_KEY = 'encoder.layers.0.linear2.weight'


class MetaOrderedTrajectory(OrderedTrajectory):
    """The same level/change score with zero self-pairs removed for Hessians."""

    def loss(self, encoded, probabilities, target, available):
        level = joint_energy_no_self(encoded, probabilities, target, available)
        observed = torch.where(available[:, None], encoded, 0.0)
        truth = torch.where(available, target, 0.0)
        change = joint_energy_no_self(
            observed[:, :, 8:] - observed[:, :, :-8], probabilities,
            truth[:, 8:] - truth[:, :-8], available[:, 8:] & available[:, :-8])
        return level + self.motion_weight * change


def first_adam_step(weight, gradient, *, lr=1e-4):
    """Match zero-state AdamW with norm clip 1, betas (.9, .999), and no decay.

    This is only the first step, not an approximation to accumulated AdamW.
    Writing the bias-corrected denominator as abs(g) avoids differentiating
    sqrt(g**2) at zero; eps=1e-8 is the actual optimizer's epsilon.
    """
    clip = (1.0 / (torch.linalg.vector_norm(gradient) + 1e-6)).clamp(max=1.0)
    clipped = gradient * clip
    return {
        "weight": weight - lr * clipped / (clipped.abs() + 1e-8),
        "exp_avg": .1 * clipped,
        "exp_avg_sq": .001 * clipped.square(),
    }


def teacher_step(model, episode, query, query_available, labels, *, stop_update=False):
    """Source-only outer classification after one actual signal-gradient step.

    The caller must use the math attention backend for double backward. Labels
    belong only to the outer loss. The signal update cannot access them, and no
    model state or optimizer is mutated. The per-shot prior has zero gradient
    at this first step because the initial and temporary weights coincide.
    """
    signal_loss = model.signal_loss(episode)
    gradient, = torch.autograd.grad(signal_loss, model.weight, create_graph=True)
    step = first_adam_step(model.weight, gradient)
    fast = step['weight']
    if stop_update:
        fast = model.weight + (fast - model.weight).detach()
    logit = functional_call(model.risk, {WEIGHT_KEY: fast},
                            (query, query_available))['logit']
    return dict(signal_loss=signal_loss, risk_loss=F.binary_cross_entropy_with_logits(logit, labels),
                logit=logit, gradient=gradient, **step)
