"""Audit available EAST measurements at a declared causal release clock."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from . import io23


FIRST, STRIDE, WINDOW = 127, 32, 25
FAMILIES = {
    'motion': ('dx_a', 'dy_a'),
    'magnetic': ('Mir_avg_amp', 'n=1 amplitude'),
    'radiation': ('sxr_var', 'sxr_skew', 'xuv_var', 'xuv_skew'),
    'electrical': ('v_loop',),
}
ROUTES = {
    'motion_magnetic': ('motion', 'magnetic'),
    'radiation_electrical': ('radiation', 'electrical'),
}
TREND = {-1: 'decreasing', 0: 'stable', 1: 'increasing', 2: 'unknown'}


def signal_windows(values, available):
    q = np.arange(FIRST, len(values)-WINDOW-1, STRIDE)
    past = available[q[:, None] + np.arange(-31, 1)].sum(axis=1)
    arrived = available[q[:, None] + np.arange(1, WINDOW+1)].sum(axis=1)
    supported = {}
    for group, names in FAMILIES.items():
        j = [io23.RAW23.index(k) for k in names]
        supported[group] = ((past[:, j] >= 16) & (arrived[:, j] >= 13)).any(axis=1)
    for route, (g, h) in ROUTES.items():
        supported[route] = supported[g] & supported[h]
    ip, bt = [io23.RAW23.index(k) for k in ('ip', 'Bt')]
    a = q[:, None]+np.arange(-31, -23)
    b = q[:, None]+np.arange(-7, 1)
    earlier = np.median(np.abs(values[a, ip]), axis=1)
    later = np.median(np.abs(values[b, ip]), axis=1)
    valid = available[a, ip].all(axis=1) & available[b, ip].all(axis=1)
    valid &= available[q, bt] & (earlier > 0.)
    # Descriptive observed current trend, not PCS ramp intent or a risk label.
    trend = np.where(later < .99*earlier, -1, np.where(later > 1.01*earlier, 1, 0))
    return dict(release=q+WINDOW+1, supported=supported,
                current_trend=np.where(valid, trend, 2))


def summarize(windows, selection):
    trend = windows['current_trend'][selection]
    return dict(windows=int(selection.sum()),
                support={k: int((v & selection).sum())
                         for k, v in windows['supported'].items()},
                current_trend={v: int((trend == k).sum()) for k, v in TREND.items()})


def totals(rows, key):
    return dict(shots=len(rows),
                shots_with_windows=sum(r[key]['windows'] > 0 for r in rows),
                windows=sum(r[key]['windows'] for r in rows),
                by_group_or_route={k: {
                    'shots': sum(r[key]['support'][k] > 0 for r in rows),
                    'windows': sum(r[key]['support'][k] for r in rows),
                } for k in (*FAMILIES, *ROUTES)},
                current_trend={v: {
                    'shots': sum(r[key]['current_trend'][v] > 0 for r in rows),
                    'windows': sum(r[key]['current_trend'][v] for r in rows),
                } for v in TREND.values()})


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def run(reference, output):
    parent = reference.read_bytes()
    protocol = json.loads(parent)
    ids, startup = protocol['order'], protocol['startup']
    if len(set(ids)) != 360 or len(ids) != 360 or startup != 20:
        raise ValueError('expected locked EAST360 with cold20')
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(
        scope='EAST360_MEASUREMENT_COVERAGE_NOT_A_PREDICTION_EXPERIMENT',
        parent=str(reference), parent_sha256=hashlib.sha256(parent).hexdigest(),
        order=ids, startup=startup, families=FAMILIES, routes=ROUTES,
        first=FIRST, stride=STRIDE, window=WINDOW, support_definition=(
            'same channel: at least 16/32 prior and 13/25 arrived finite samples; '
            'one channel per family and both families per route'),
        current_trend_definition='relative change of two prior eight-point medians of |Ip|, +/-1%',
        physical_acquisition_latency_verified=False, event_labels_read=False))
    counts = {k: Counter() for k in io23.RAW23}
    attributes = {k: set() for k in io23.RAW23}
    fields, computed = set(), []
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome not yet sealed')):
        for sid in ids:
            s = io23.read_signals('EAST', sid, rate_contract='nominal')
            values, available = s['v'], s['a']
            for c, k in enumerate(io23.RAW23):
                v = values[available[:, c], c]
                counts[k]['finite_shots'] += bool(v.size)
                counts[k]['nonconstant_shots'] += bool(v.size and v.min() < v.max())
                counts[k]['all_finite_zero_shots'] += bool(v.size and not v.any())
            with h5py.File(io23.path_for('EAST', sid), 'r') as f:
                fields.update(f['data'].keys())
                for k in io23.RAW23:
                    attributes[k].update(f['data'][k].attrs.keys())
            windows = signal_windows(values, available)
            t = s['times'][windows['release']]
            computed.append((sid, t, windows))
    # Keep the measurement-only stream independently inspectable before joining outcomes.
    np.savez_compressed(
        output/'measurement_seal.npz', shot=np.array(ids),
        offsets=np.r_[0, np.cumsum([len(t) for _, t, _ in computed])],
        times=np.concatenate([t for _, t, _ in computed]),
        current_trend=np.concatenate([w['current_trend'] for _, _, w in computed]),
        **{k: np.concatenate([w['supported'][k] for _, _, w in computed])
           for k in (*FAMILIES, *ROUTES)})
    rows = []
    for sid, t, w in computed:
        d, down = io23.read_outcome('EAST', sid)
        lead = (down-t)*1000. if d else np.zeros_like(t)
        rows.append(dict(shot=sid, disrupt=d, all=summarize(w, np.ones(len(t), bool)),
                         early=summarize(w, (lead > 30.+1e-6) & (lead <= 125.+1e-6)),
                         later=summarize(w, (lead > 0.) & (lead <= 30.+1e-6))))
    suffix = rows[startup:]
    disruptive = [r for r in suffix if r['disrupt']]
    safe = [r for r in suffix if not r['disrupt']]
    if sum(r['disrupt'] for r in rows) != 180:
        raise ValueError('HDF5 outcomes must match the locked balanced cohort')
    write(output/'per_shot.json', rows)
    write(output/'result.json', dict(
        data_fields=sorted(fields), channel_attributes={k: sorted(v) for k, v in attributes.items()},
        raw_channel_coverage=counts, entire_stream=totals(rows, 'all'),
        cold_disruptive=sum(r['disrupt'] for r in rows[:startup]),
        suffix_safe=totals(safe, 'all'), suffix_disruptive_timely=totals(disruptive, 'early'),
        suffix_disruptive_late=totals(disruptive, 'later')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.reference_protocol.resolve(), args.output.resolve())
