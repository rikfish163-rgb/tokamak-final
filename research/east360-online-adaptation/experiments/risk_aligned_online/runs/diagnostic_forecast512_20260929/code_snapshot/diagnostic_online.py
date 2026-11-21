"""Signal-only within-shot adaptation in interpretable diagnostic coordinates.

The forecast at an origin is issued before its future signal arrives. The
complete 25-point measured trajectory is the target, never a generated input.
Only later predictions use the updated state. State is discarded per shot.
"""
import numpy as np
import torch

from . import simple_data, simple_train
from .diagnostic_targets import coordinates, trajectory_loss, trajectory_target


def predict_shot(parent, adapter, shot, *, intra, device='cuda', data_api=simple_data):
    from .diagnostic_adapter import bounded_step

    parent.eval()
    adapter.eval()
    horizon = parent.cfg.horizon
    if horizon != 25:
        raise ValueError('direct H25 forecast contract required')
    fast = adapter.initial_fast(1).detach()
    scores = np.empty(len(shot['values']), np.float32)
    events = []
    origin = 0
    while origin < len(scores):
        stop = min(len(scores), origin+(26 if intra else 256))
        windows = [data_api.window(shot, end, parent.cfg.window) for end in range(origin, stop)]
        x = torch.as_tensor(np.stack([row[0] for row in windows]), device=device)
        mask = torch.as_tensor(np.stack([row[1] for row in windows]), device=device)
        # No update during this block; each risk window has its own past only.
        with torch.no_grad():
            out = adapter(parent, x, mask, fast.expand(len(x), -1))
            probabilities = out['logit'].sigmoid()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError('nonfinite within-shot prediction')
            scores[origin:stop] = probabilities.cpu().numpy()
        if not intra or stop != origin+horizon+1:
            origin = stop
            continue
        candidate = fast.detach().requires_grad_(True)
        # Re-evaluation uses precisely the origin's state and observed window;
        # no model/parameter changes occurred in the block. The emitted values
        # are saved below so this deterministic replay can be independently read.
        prediction = adapter(parent, x[:1], mask[:1], candidate)['trajectory']
        coordinate = coordinates(x[:1], mask[:1])
        truth = torch.as_tensor(shot['values'][origin+1:stop][None], device=device)
        available = torch.as_tensor(shot['available'][origin+1:stop][None], device=device)
        target, valid = trajectory_target(coordinate, truth, available)
        if valid.any():
            loss = trajectory_loss(prediction, target, valid).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite matured diagnostic loss')
            gradient, = torch.autograd.grad(loss, candidate)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('nonfinite diagnostic state gradient')
            updated = bounded_step(candidate.detach(), gradient.detach())
            events.append(dict(origin=origin, observed_through=stop-1,
                               affects_from=stop, loss=float(loss.detach()),
                               persistence_loss=float(trajectory_loss(torch.zeros_like(target), target, valid)),
                               gradient_norm=float(gradient.norm()),
                               before=fast[0].cpu().tolist(), after=updated[0].cpu().tolist(),
                               eligible=coordinate['eligible'][0].cpu().tolist(),
                               predicted_coefficients=prediction[0, -1].detach().cpu().tolist()))
            fast = updated.detach()
        origin = stop
    return scores, events


def metric_rows(parent, adapter, shots, *, intra, device='cuda'):
    rows, event_rows = [], []
    for i, shot in enumerate(shots):
        scores, events = predict_shot(parent, adapter, shot, intra=intra, device=device)
        valid = shot['valid']
        rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                         times=shot['times'][valid], scores=scores[valid]))
        event_rows.append(dict(shot=shot['shot'], events=events))
        if (i+1) % 40 == 0:
            print(f'within-shot evaluated {i+1}/{len(shots)}', flush=True)
    return rows, event_rows


def evaluate(parent, adapter, shots, threshold, *, lead=10., device='cuda'):
    static, _ = metric_rows(parent, adapter, shots, intra=False, device=device)
    intra, events = metric_rows(parent, adapter, shots, intra=True, device=device)
    return dict(static=simple_train.metrics(static, threshold, lead=lead),
                intra=simple_train.metrics(intra, threshold, lead=lead), events=events)
