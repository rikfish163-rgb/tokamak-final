"""Causal P5+20 and same-head delayed control. No outcome-dependent operations.

Risk is emitted first. Origin inputs and arrived targets are immutable snapshots.
Both clocks execute the same two operators per completed cycle, in the same order;
partial executes the first earlier. Incomplete terminal cycles need not match.
"""
import hashlib
import time

import numpy as np
import torch

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows
from .feedback import segment_step


def snapshot(signals, origin, segment, device, dtype):
    """Caller has reached o+5 (early) or o+25 (late); copy only that block."""
    if segment not in ('early', 'late'):
        raise ValueError('early or late segment required')
    start, stop = (origin + 1, origin + 6) if segment == 'early' else (origin + 5, origin + 26)
    truth = torch.as_tensor(signals['values'][start:stop], dtype=dtype, device=device)[None].clone()
    observed = torch.as_tensor(signals['available'][start:stop], device=device)[None].clone()
    return truth, observed & torch.isfinite(truth)


def predict_shot(model, shot, *, mode, offline_source=False, data_api=simple_data):
    if mode not in ('static', 'partial', 'delayed'):
        raise ValueError('declare static, partial or delayed clock')
    if shot.get('device') != ('J-TEXT' if offline_source else 'EAST'):
        raise ValueError('EAST stream only; source requires explicit offline auxiliary access')
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

    def update(segment, end):
        nonlocal delta
        truth, observed = pending[segment]
        candidate, row = segment_step(model, delta, pending['x'], pending['a'], truth, observed, segment)
        delta = candidate.detach().requires_grad_()
        events.append(dict(**row, origin_index=pending['origin'],
            snapshot_index=pending['origin'] + (5 if segment == 'early' else 25),
            update_index=end, effective_index=end + 1 if end + 1 < n else None,
            delta_norm=float(delta.detach().double().norm())))

    while start < n:
        stop = min(n, start + 256)
        if mode != 'static':
            stop = min(stop, next_origin + 1)
        if pending is not None:
            # Even delayed execution captures the early target AT its arrival.
            stop = min(stop, pending['origin'] + (6 if 'early' not in pending else 26))
        x, a = _batch_windows(data_api, signals, range(start, stop), model, device)
        with torch.enable_grad():
            out = model(x, a, delta)
            if not torch.isfinite(out['logit']).all():
                raise FloatingPointError('nonfinite emitted risk')
            scores[start:stop] = out['logit'].detach().sigmoid().cpu().numpy()
            end = stop - 1
            if mode != 'static' and end == next_origin:
                pending = dict(origin=end, x=x[-1:].detach().clone(), a=a[-1:].detach().clone())
                next_origin += 26
            if pending is not None and end == pending['origin'] + 5:
                pending['early'] = snapshot(signals, pending['origin'], 'early', device, model.center.dtype)
                if mode == 'partial':
                    update('early', end)
            if pending is not None and end == pending['origin'] + 25:
                pending['late'] = snapshot(signals, pending['origin'], 'late', device, model.center.dtype)
                if mode == 'delayed':
                    update('early', end)
                update('late', end)
                pending = None
        start = stop
    return scores, dict(mode=mode, events=events, accepted_updates=sum(r['state_changed'] for r in events),
        temporary_reset=True, outcome_access=False, alarm_adaptation=False,
        final_delta_sha256=hashlib.sha256(delta.detach().cpu().numpy().tobytes()).hexdigest(),
        final_delta_norm=float(delta.detach().double().norm()), parameter_count=model.delta_count,
        parameter_names=model.parameter_names, stride=26, radius=1., segment_steps=[.02, .08],
        elapsed_seconds=time.perf_counter() - started,
        latency_scope='batched offline replay wall time; no real-time feasibility claim')
