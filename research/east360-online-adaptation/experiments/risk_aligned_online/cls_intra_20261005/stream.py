"""Risk-first source-CLS experiment. Temporary FFN updates reset every shot."""
import hashlib
import time

import numpy as np
import torch

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows
from ..partial_intra_20261005.stream import snapshot
from .feedback import segment_step


def predict_shot(model, shot, *, use_task, intra=True, offline_source=False, data_api=simple_data):
    if shot.get('device') != ('J-TEXT' if offline_source else 'EAST'):
        raise ValueError('source access must be explicit and offline; otherwise EAST only')
    signals = {k: np.asarray(shot[k]) for k in ('values', 'available')}
    times = np.asarray(shot['times']); n = len(times)
    if (not n or times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or signals['values'].shape != (n, 23) or signals['available'].shape != (n, 23)
            or signals['available'].dtype != bool):
        raise ValueError('increasing timestamps and flat23 observations required')
    device = model.center.device
    delta = model.initial_delta(); scores = np.empty(n, np.float32)
    events = []; pending = None; next_origin = 0; start = 0
    started = time.perf_counter()

    def update(segment, end, truth, observed):
        nonlocal delta
        candidate, row = segment_step(model, delta, pending['x'], pending['a'], truth, observed,
                                      segment, use_task=use_task)
        delta = candidate.detach().requires_grad_()
        events.append(dict(**row, origin_index=pending['origin'], update_index=end,
            effective_index=end + 1 if end + 1 < n else None,
            delta_norm=float(delta.detach().double().norm())))

    while start < n:
        stop = min(n, start + 256)
        if intra:
            stop = min(stop, next_origin + 1)
        if pending is not None:
            stop = min(stop, pending['origin'] + (6 if 'early' not in pending else 26))
        x, a = _batch_windows(data_api, signals, range(start, stop), model, device)
        with torch.enable_grad():
            out = model(x, a, delta)
            if not torch.isfinite(out['logit']).all():
                raise FloatingPointError('nonfinite emitted risk')
            scores[start:stop] = out['logit'].detach().sigmoid().cpu().numpy()
            end = stop - 1
            if intra and end == next_origin:
                pending = dict(origin=end, x=x[-1:].detach().clone(), a=a[-1:].detach().clone())
                next_origin += 26
            if pending is not None and end == pending['origin'] + 5:
                pending['early'] = snapshot(signals, pending['origin'], 'early', device, model.center.dtype)
                update('early', end, *pending['early'])
            if pending is not None and end == pending['origin'] + 25:
                late = snapshot(signals, pending['origin'], 'late', device, model.center.dtype)
                # Preserve the earlier arrival snapshot; h5 is already in it.
                truth = torch.cat((pending['early'][0], late[0][:, 1:]), dim=1)
                observed = torch.cat((pending['early'][1], late[1][:, 1:]), dim=1)
                update('late', end, truth, observed)
                pending = None
        start = stop
    return scores, dict(use_task=use_task, intra=intra, events=events,
        accepted_updates=sum(r['state_changed'] for r in events), temporary_reset=True,
        outcome_access=False, alarm_adaptation=False,
        final_delta_sha256=hashlib.sha256(delta.detach().cpu().numpy().tobytes()).hexdigest(),
        final_delta_norm=float(delta.detach().double().norm()), parameter_count=model.delta_count,
        parameter_names=model.parameter_names, stride=26, radius=1., segment_steps=[.02, .08],
        elapsed_seconds=time.perf_counter() - started,
        latency_scope='offline replay, not a real-time performance claim')
