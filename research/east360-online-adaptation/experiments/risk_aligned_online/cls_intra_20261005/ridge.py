"""Masked mean-MSE ridge initialization on source-fit shots only."""
import numpy as np
import torch

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows


def solve_ridge(gram, rhs, count, *, penalty):
    if penalty <= 0 or not np.isfinite(penalty): raise ValueError('positive ridge penalty required')
    empty = count == 0
    regularizer = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    regularizer[-1, -1] = 0.  # intercept is unpenalized
    system = gram + count[:, None, None] * penalty * regularizer
    # An unobserved coordinate is explicitly untrained, not an arbitrary solve.
    system = system.clone(); system[empty] = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    coef = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    coef[empty] = 0.
    if not torch.isfinite(coef).all(): raise FloatingPointError('nonfinite ridge coefficients')
    return coef, torch.nonzero(empty).flatten().cpu().tolist()


def fit_decoder(model, shots):
    if not shots or any(s['device'] != 'J-TEXT' for s in shots):
        raise ValueError('source fit shots only')
    device = model.center.device
    gram = torch.zeros(575, 65, 65, device=device, dtype=torch.float64)
    rhs = torch.zeros(575, 65, device=device, dtype=torch.float64)
    count = torch.zeros(575, device=device, dtype=torch.long)
    total = 0
    for i, shot in enumerate(shots):
        origins = list(range(0, len(shot['times']) - 25, 26))
        signals = {k: shot[k] for k in ('values', 'available')}
        for start in range(0, len(origins), 256):
            batch = origins[start:start + 256]
            x, a = _batch_windows(simple_data, signals, batch, model, device)
            cls = model(x, a, model.initial_delta())['cls'].detach().double()
            design = torch.cat((cls, torch.ones(len(cls), 1, device=device, dtype=cls.dtype)), dim=1)
            raw = torch.as_tensor(np.stack([shot['values'][o+1:o+26] for o in batch]), device=device, dtype=model.center.dtype)
            valid = torch.as_tensor(np.stack([shot['available'][o+1:o+26] for o in batch]), device=device) & torch.isfinite(raw)
            target = torch.asinh((torch.where(valid, raw, model.center) - model.center) / model.scale).flatten(1).double()
            mask = valid.flatten(1)
            gram += torch.einsum('bo,bi,bj->oij', mask.double(), design, design)
            rhs += torch.einsum('bo,bi->oi', target.masked_fill(~mask, 0.), design)
            count += mask.sum(0); total += len(batch)
        if (i + 1) % 100 == 0: print(f'ridge_source_shots={i+1} windows={total}', flush=True)
    coef, untrained = solve_ridge(gram, rhs, count, penalty=.001)
    with torch.no_grad():
        model.decoder.weight.copy_(coef[:, :-1].to(model.decoder.weight.dtype))
        model.decoder.bias.copy_(coef[:, -1].to(model.decoder.bias.dtype))
    return dict(shots=len(shots), windows=total, untrained_output_coordinates=untrained,
        observed_target_counts=count.cpu().tolist(), ridge_penalty=.001,
        objective='mean squared source-asinh error + .001 squared weight; unpenalized intercept')
