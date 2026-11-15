"""Single-label, flat23 data for the rebuilt lane; no old model imports."""
import numpy as np

from . import io23


def source_roles():
    return io23.source_roles()


def advanced_times():
    return io23.advanced_times()


def read_shot(device, sid, *, rate_contract='strict'):
    # The strict default preserves source training and archived callers. The
    # redesigned EAST runner opts in explicitly; the same raw grid is retained.
    raw = (io23.read_signals(device, sid) if rate_contract == 'strict' else
           io23.read_signals(device, sid, rate_contract=rate_contract))
    values, available = raw['v'].copy(), raw['a'].copy()
    if device == 'J-TEXT':
        values[:, io23.RAW23.index('ip')] *= 1000.
    else:
        available[:, io23.RAW23.index('MNM')] = False
    available &= np.isfinite(values)
    values[~available] = 0.
    return dict(shot=int(sid), device=device, values=values, available=available,
                times=raw['times'], planning_ip=raw['planning_ip'] * (1000 if device == 'J-TEXT' else 1),
                planning_Bt=raw['planning_Bt'])


def attach_outcome(s, device, advanced):
    if device != s['device']:
        raise ValueError('device mismatch')
    disruptive, down = io23.read_outcome(device, s['shot'])
    horizon = io23.effective_horizon_ms(s['shot'], advanced)[0] if device == 'J-TEXT' else 125.
    y, valid = io23.labels_at(s['times'], disruptive, down, horizon)
    return dict(s, y=y, valid=valid, is_disrupt=disruptive, down=down, horizon=horizon)


def window(s, end, length):
    if length < 1 or not 0 <= end < len(s['values']):
        raise ValueError('invalid past window')
    x, mask = np.zeros((length, 23), np.float32), np.zeros((length, 23), bool)
    start = max(0, end-length+1)
    x[-(end-start+1):] = s['values'][start:end+1]
    mask[-(end-start+1):] = s['available'][start:end+1]
    return x, mask


def fit_source_scaler(shots):
    if not shots or any(s['device'] != 'J-TEXT' for s in shots):
        raise ValueError('source training shots only')
    centers, scales = [], []
    for c in range(23):
        samples = []
        for s in shots:
            v = s['values'][s['valid'] & s['available'][:, c], c]
            if len(v):
                samples.append(v[::max(1, len(v)//128)])
        if not samples:
            raise ValueError(f'no source observations for channel {c}')
        q25, median, q75 = np.quantile(np.concatenate(samples), [.25, .5, .75])
        centers.append(median)
        scales.append(max(q75-q25, abs(median)*.01, 1e-6))
    return dict(center=np.asarray(centers, np.float32), scale=np.asarray(scales, np.float32))
