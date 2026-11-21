"""Ended-shot supervision of one source-risk encoder matrix."""

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import simple_data
from risk_meta_step_20261107 import WEIGHT_KEY


def train_inter(base, shots, records, schedule, initial_weight, *, device='cpu', lr=1e-4):
    """Update the persistent base from the caller's ended-shot draw schedule.

    ``base`` must already be on ``device``; ``initial_weight`` is an independent
    snapshot of the original matrix. Each history entry describes the objective
    and unclipped gradient norm before that step's optimizer update.
    """
    base.eval().requires_grad_(False)
    for parameter in base.parameters():
        parameter.grad = None
    try:
        weight = dict(base.named_parameters())[WEIGHT_KEY]
        if initial_weight.shape != weight.shape:
            raise ValueError('initial_weight shape must match the updated weight')
        if (initial_weight.device == weight.device and
                initial_weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()):
            raise ValueError('initial_weight must have independent storage')
        anchor = initial_weight.detach().to(device=weight.device, dtype=weight.dtype).clone()
        weight.requires_grad_(True)
        optimizer = torch.optim.AdamW([weight], lr=lr, weight_decay=0.)
        history = []
        for draw in schedule:
            selected = [records[i] for i in draw['indices']]
            windows = [simple_data.window(shots[row['shot']], row['query'], 128)
                       for row in selected]
            x = torch.as_tensor(np.stack([window[0] for window in windows]), device=device)
            available = torch.as_tensor(np.stack([window[1] for window in windows]), device=device)
            labels = torch.tensor([row['y'] for row in selected], dtype=x.dtype, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = base(x, available)['logit']
            risk_loss = F.binary_cross_entropy_with_logits(logits, labels)
            prior = (weight - anchor).square().sum()
            loss = risk_loss + .01 * prior
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([weight], 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=draw['step'], loss=float(loss.detach()),
                                risk_loss=float(risk_loss.detach()), prior=float(prior.detach()),
                                gradient_norm=float(norm)))
    finally:
        base.requires_grad_(False)
        for parameter in base.parameters():
            parameter.grad = None
    return history
