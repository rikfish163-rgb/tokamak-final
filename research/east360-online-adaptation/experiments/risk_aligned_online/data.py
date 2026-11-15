"""Source-labelled J-TEXT data and label-free signal readers for the online lane."""
from __future__ import annotations

import numpy as np

from experiments.fast_online_idea import pipeline as P

CHANNELS = tuple(P.RAW23)


def _device(device):
    if device not in ('J-TEXT', 'EAST'):
        raise ValueError('device must be J-TEXT or EAST')


def read_shot(device, sid):
    """Read signals only. The returned stream has no outcome or future-fit statistics."""
    _device(device)
    raw = P.read_signals(device, int(sid), 'flat23')
    values = np.array(raw['v'], dtype=np.float32, copy=True)
    available = np.array(raw['a'], dtype=bool, copy=True) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != 23 or available.shape != values.shape:
        raise ValueError('flat23 signal shape mismatch')
    if device == 'J-TEXT':
        values[:, CHANNELS.index('ip')] *= 1000.  # J-TEXT kA -> A, once.
    else:
        # EAST MNM status has uncertain cross-device semantics; retain other
        # genuine flat23 channels, including ne_nG and qa_proxy.
        available[:, CHANNELS.index('MNM')] = False
    values[~available] = 0.
    times = np.asarray(raw['times'], dtype=np.float64)
    if len(times) != len(values) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('invalid signal clock')
    return dict(shot=int(sid), device=device, values=values, available=available,
                times=times, rate=float(raw['rate']),
                planning_ip=np.asarray(raw['planning_ip'], dtype=np.float32) *
                (1000. if device == 'J-TEXT' else 1.),
                planning_Bt=np.asarray(raw['planning_Bt'], dtype=np.float32))


def attach_outcome(s, device, advanced):
    """Explicit source/offline access; online callers may call only AFTER sealing."""
    _device(device)
    if s.get('device', device) != device:
        raise ValueError('shot/device mismatch')
    disruptive, down = P.read_outcome(device, s['shot'])
    horizon = (P.JL.effective_horizon_ms(s['shot'], advanced)[0]
               if device == 'J-TEXT' else 125.)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('invalid AdvancedTime')
    y, valid = P.labels_at(s['times'], disruptive, down, horizon)
    y125, valid125 = P.labels_at(s['times'], disruptive, down, 125.)
    if not np.array_equal(valid, valid125):
        raise AssertionError('horizon changed validity')
    s.update(y=y, y125=y125, valid=valid, is_disrupt=bool(disruptive),
             down=down, horizon=horizon)
    return s


def window(s, end, length=128):
    if length <= 0 or not 0 <= end < len(s['values']):
        raise IndexError(end)
    values = np.zeros((length, 23), np.float32)
    available = np.zeros((length, 23), bool)
    lo = max(0, end - length + 1)
    history = s['values'][lo:end+1]
    mask = s['available'][lo:end+1]
    values[-len(history):] = history
    available[-len(history):] = mask
    return values, available


def fit_source_scaler(shots):
    """Observed, valid TRAIN J-TEXT samples only; no VAL/TEST or target fit."""
    if not shots:
        raise ValueError('empty source training set')
    chunks = [[] for _ in CHANNELS]
    for s in shots:
        if s.get('device') != 'J-TEXT':
            raise ValueError('scaler requires J-TEXT training shots')
        values, available = s['values'], s['available']
        valid = np.asarray(s['valid'], bool)
        if values.shape != available.shape or values.shape[1] != 23 or len(valid) != len(values):
            raise ValueError('invalid source scaler shot')
        for c in range(23):
            used = values[valid & available[:, c] & np.isfinite(values[:, c]), c]
            if len(used):
                chunks[c].append(used[::max(1, len(used)//128)])
    center = np.empty(23, np.float32)
    scale = np.empty(23, np.float32)
    for c, parts in enumerate(chunks):
        if not parts:
            raise ValueError(f'no observed source samples for channel {c}')
        q25, median, q75 = np.quantile(np.concatenate(parts), [.25, .5, .75])
        center[c] = median
        scale[c] = max(q75-q25, abs(median)*.01, 1e-6)
    return dict(center=center, scale=scale)


def source_roles():
    return P.SRC.load_canonical_jtext_roles()


def advanced_times():
    return P.JL.load_advanced_time_ms(P.SRC.JTEXT_ADVANCED_TIME_PATH)


def build_order(device, ids, order='low_to_high'):
    _device(device)
    if device == 'J-TEXT':
        return [dict(shot=int(sid)) for sid in sorted(ids)]
    if order not in ('low_to_high', 'high_to_low'):
        raise ValueError('unsupported signal-only order')
    rows = []
    for sid in ids:
        s = read_shot(device, sid)
        if len(s['values']) < 100:
            raise ValueError(f'EAST shot {sid} has fewer than 100 samples')
        ip_prefix = np.asarray(s['planning_ip'][:100], np.float64)
        bt_prefix = np.asarray(s['planning_Bt'][:100], np.float64)
        if len(ip_prefix) < 100 or len(bt_prefix) < 100 or not np.isfinite(ip_prefix).all() or not np.isfinite(bt_prefix).all():
            raise ValueError(f'EAST shot {sid} has nonfinite or short 100-ms planning prefix')
        ip = float(np.median(np.abs(ip_prefix)))
        bt = float(np.median(np.abs(bt_prefix)))
        # HDF5 Bt has no unit attribute. This is a raw-channel ordering proxy,
        # not a verified Tesla value or quantitative disruption-damage estimate.
        rows.append(dict(shot=int(sid), initial_ip_A=ip, initial_abs_Bt_raw=abs(bt),
                         proxy=(ip/100_000.)**1.2*bt))
    return sorted(rows, key=lambda r: (r['proxy'], r['shot']),
                  reverse=(order == 'high_to_low'))
