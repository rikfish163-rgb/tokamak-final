"""Unlabelled eight-sample radiation estimate audit; CPU, signal-only.

Run from any directory, for example:
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 TMPDIR=/absolute/worktree/outputs \
    /home/hetaisheng/miniconda3/envs/rl_env/bin/python -B /absolute/path/forecast.py

The only output is result.json beside this script. The index offset deliberately
begins at TRAIN, after the TEST section, and is checked before parsing.
"""

import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = ROOT / 'artifacts/canonical/jtext_source_bundle_p131_v1/index.json'
PROTOCOL = ROOT / 'outputs/east360_observation_clock_20261114_v2/protocol.json'
MAPPING = ROOT / 'probe_outputs/stratified_cold20_context360_v1/input_map.json'
CHECKPOINT = ROOT / 'experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt'
FEATURES = ('sxr_kurt', 'sxr_mean', 'sxr_skew', 'sxr_var',
            'xuv_kurt', 'xuv_ratio', 'xuv_skew', 'xuv_var')
COLS = (14, 15, 16, 17, 19, 20, 21, 22)
METHODS = ('source_center', 'hold_anchor', 'damped_trend')
INDEX_TRAIN_OFFSET = 645107


def source_ids():
    # No scan of the preceding index section: it contains TEST identities.
    result = {}
    with INDEX.open('rb') as f:
        f.seek(INDEX_TRAIN_OFFSET)
        for role, expected in (('train', 1353), ('val', 160)):
            header = f.readline().decode().strip()
            if header != f'"{role}": [':
                raise ValueError(f'canonical index role/offset changed: {role}: {header!r}')
            ids = []
            while True:
                line = f.readline().decode().strip()
                if line in ('],', ']'):
                    break
                ids.append(int(line.rstrip(',')))
            if len(ids) != expected or len(set(ids)) != expected:
                raise ValueError(f'canonical index count changed: {role}')
            result[role] = ids
    if set(result['train']) & set(result['val']):
        raise ValueError('TRAIN and VAL overlap')
    return result


def cold_ids():
    # Stop as soon as 20 IDs have been read; do not inspect the suffix.
    with PROTOCOL.open() as f:
        for line in f:
            if line.strip() == '"order": [':
                ids = [int(next(f).strip().rstrip(',')) for _ in range(20)]
                break
        else:
            raise ValueError('protocol order not found')
    if len(set(ids)) != 20:
        raise ValueError('protocol cold prefix is not 20 distinct shots')
    return ids


def fixed_parameters(ids):
    # Check only the fixed calibration map, never outcomes or target signals.
    with MAPPING.open() as f:
        mapping = json.load(f)
    if mapping['calibration_ids'] != ids or mapping['first_affected_index'] != 20:
        raise ValueError('fixed map does not correspond to this cold prefix')
    gain = np.asarray(mapping['gain'], dtype=np.float64)[list(COLS)]
    offset = np.asarray(mapping['offset'], dtype=np.float64)[list(COLS)]
    if not (np.array_equal(gain, np.ones(8)) and np.array_equal(offset, np.zeros(8))):
        raise ValueError('radiation map no longer identity; use the actual fixed transform')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    c = np.asarray(checkpoint['source_center'], dtype=np.float64)[list(COLS)]
    s = np.asarray(checkpoint['source_scale'], dtype=np.float64)[list(COLS)]
    model_c = np.asarray(checkpoint['model']['center'].cpu())[list(COLS)]
    model_s = np.asarray(checkpoint['model']['scale'].cpu())[list(COLS)]
    if not (np.array_equal(c, model_c) and np.array_equal(s, model_s)
            and np.isfinite(c).all() and np.isfinite(s).all() and (s > 0).all()):
        raise ValueError('checkpoint source coordinates do not match model buffers')
    return c, s


def load_radiation(device, sid):
    path = ROOT / 'data' / device / 'unified_hdf5' / str(sid // 100 * 100) / f'{sid}.hdf5'
    with h5py.File(path, 'r') as f:
        data = f['data']
        ip = data['ip']  # Only the length and acquisition-clock attributes.
        n = len(ip)
        rate = float(np.asarray(ip.attrs['SampleRate']).reshape(-1)[0])
        start = float(np.asarray(ip.attrs['StartTime']).reshape(-1)[0])
        tol = 1e-9 if device == 'J-TEXT' else 0.5
        if not math.isfinite(rate) or not math.isfinite(start) or abs(rate - 1000) > tol:
            raise ValueError(f'invalid declared sample clock: {rate}, {start}')
        cols = []
        for feature in FEATURES:
            ds = data[feature]
            if len(ds) != n:
                raise ValueError(f'unaligned length for {feature}')
            for attr, expected in (('SampleRate', rate), ('StartTime', start)):
                if attr not in ds.attrs or not np.isclose(
                        float(np.asarray(ds.attrs[attr]).reshape(-1)[0]), expected):
                    raise ValueError(f'unaligned {attr} for {feature}')
            cols.append(np.asarray(ds[:], dtype=np.float32))
    # This is exactly the radiation portion of simple_data.read_shot and the
    # locked post-cold20 transform. No /meta group is opened.
    return np.column_stack(cols)


def calc(raw, center, scale):
    mask = np.isfinite(raw)
    z = np.arcsinh((np.where(mask, raw, center) - center) / scale)
    q = np.arange(127, len(z), 32, dtype=np.int64)
    if not len(q):
        return None
    anchor = z[q - 8]
    previous = z[q - 16]
    truth = z[q[:, None] + np.arange(-7, 1)[None, :]]
    anchored = mask[q - 8]
    observed = mask[q[:, None] + np.arange(-7, 1)[None, :]] & anchored[:, None, :]
    full = observed.all(axis=(1, 2))
    change = .25 * (anchor - previous)[:, None, :] * (np.arange(1, 9)[None, :, None] / 8.)
    prior_available = mask[q - 16] & anchored
    change = np.where(prior_available[:, None, :], np.clip(change, -.25, .25), 0.)
    estimates = (np.zeros_like(truth), np.broadcast_to(anchor[:, None, :], truth.shape),
                 anchor[:, None, :] + change)
    errors = np.stack([np.abs(truth - p) for p in estimates])
    return dict(q=q, observed=observed, anchored=anchored, full=full, errors=errors,
                prior_available=prior_available,
                clipped=(np.abs(.25 * (anchor - previous)[:, None, :]
                                * (np.arange(1, 9)[None, :, None] / 8.)) > .25)
                        & prior_available[:, None, :] & observed)


def make_domain(device, ids, center, scale):
    rows, issues = [], []
    all_feature, all_horizon = [[] for _ in FEATURES], [[] for _ in range(8)]
    totals = dict(q_total=0, q_any_comparable=0, q_all_channels_have_anchor=0,
                  q_fully_observed=0, comparable_cells=0, full_case_cells=0,
                  trend_anchors_with_prior=0, trend_eligible_anchors=0,
                  trend_clipped_cells=0)
    for index, sid in enumerate(ids, 1):
        try:
            item = calc(load_radiation(device, sid), center, scale)
            if item is None:
                issues.append(dict(shot=sid, problem='no q=127+32k in signal'))
                continue
            observed, full, errors = item['observed'], item['full'], item['errors']
            cells = int(observed.sum())
            q_any = observed.any(axis=(1, 2))
            totals['q_total'] += len(item['q'])
            totals['q_any_comparable'] += int(q_any.sum())
            totals['q_fully_observed'] += int(full.sum())
            totals['comparable_cells'] += cells
            totals['full_case_cells'] += int(full.sum()) * 64
            totals['trend_anchors_with_prior'] += int(item['prior_available'].sum())
            totals['trend_eligible_anchors'] += int(item['anchored'].sum())
            totals['q_all_channels_have_anchor'] += int(item['anchored'].all(axis=1).sum())
            totals['trend_clipped_cells'] += int(item['clipped'].sum())
            if not cells:
                issues.append(dict(shot=sid, problem='no comparable observed anchor/target cells'))
                continue
            means = [float(x[observed].mean()) for x in errors]
            row = dict(shot=sid, q_total=len(item['q']), q_comparable=int(q_any.sum()),
                       q_complete=int(full.sum()), cells=cells,
                       mae=dict(zip(METHODS, means)))
            if full.any():
                row['mae_complete'] = dict(zip(METHODS, [float(x[full].mean()) for x in errors]))
            for j in range(8):
                mask_j = observed[:, :, j]
                if mask_j.any():
                    all_feature[j].append([float(x[:, :, j][mask_j].mean()) for x in errors])
                mask_h = observed[:, j, :]
                if mask_h.any():
                    all_horizon[j].append([float(x[:, j, :][mask_h].mean()) for x in errors])
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(dict(shot=sid, problem=f'{type(exc).__name__}: {exc}'))
        if index % 200 == 0 or index == len(ids):
            print(f'{device}: {index}/{len(ids)}, comparable shots={len(rows)}, issues={len(issues)}', flush=True)

    def summary(key):
        used = [r for r in rows if key in r]
        if not used:
            return dict(shots=0)
        v = np.array([[r[key][m] for m in METHODS] for r in used])
        return dict(shots=len(used), equal_shot_mean=dict(zip(METHODS, v.mean(axis=0).tolist())),
                    equal_shot_median=dict(zip(METHODS, np.median(v, axis=0).tolist())),
                    hold_vs_center=dict(better=int((v[:, 1] < v[:, 0]).sum()),
                                        equal=int((v[:, 1] == v[:, 0]).sum()),
                                        worse=int((v[:, 1] > v[:, 0]).sum())),
                    trend_vs_hold=dict(better=int((v[:, 2] < v[:, 1]).sum()),
                                       equal=int((v[:, 2] == v[:, 1]).sum()),
                                       worse=int((v[:, 2] > v[:, 1]).sum())))

    def breakdown(values, names):
        result = {}
        for name, shots in zip(names, values):
            result[str(name)] = (dict(shots=len(shots), equal_shot_mean=dict(
                zip(METHODS, np.mean(shots, axis=0).tolist()))) if shots else dict(shots=0))
        return result

    return dict(requested_shots=len(ids), counts=totals, all_comparable=summary('mae'),
                complete_case=summary('mae_complete'),
                feature_equal_shot=breakdown(all_feature, FEATURES),
                horizon_equal_shot=breakdown(all_horizon, range(1, 9)),
                anomalies=issues, per_shot=rows)


def main():
    start = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the empty string')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1' or Path(os.environ.get('TMPDIR', '')).resolve() != ROOT / 'outputs':
        raise RuntimeError('Set PYTHONDONTWRITEBYTECODE=1 and TMPDIR to the worktree outputs directory')
    ids = source_ids()
    cold = cold_ids()
    center, scale = fixed_parameters(cold)
    result = dict(
        scope='UNLABELLED_JTEXT_TRAIN_VAL_AND_EAST_COLD20_RADIATION_ONLY',
        input_contract=dict(features=FEATURES, q='127+32k within stored signal',
            delay_samples=8, anchor='q-8, true finite observation required per feature',
            estimated_points='q-7 through q inclusive, relative to anchor horizons 1..8',
            coordinates='asinh((feature - fixed source model center) / fixed source model scale)',
            comparisons='identical finite anchor/target cells for every method; no label cropping',
            complete_case='all 8 channels at anchor and all 8 pending timestamps finite',
            methods=dict(source_center='zero in source asinh coordinates',
                hold_anchor='same-channel true value at q-8, held across the 8 pending points',
                damped_trend='anchor + clip(0.25*(z[q-8]-z[q-16])*h/8, -0.25, 0.25); '
                             'if q-16 unavailable use anchor; fixed before data read'),
            shot_aggregation='mean absolute error over comparable channel-time cells within each shot, '
                             'then equal mean/median over shots',
            target_mapping='existing post-cold20 density_gain mapping; radiation gains=1, offsets=0; '
                           'applied retrospectively to cold20 only for shared coordinates',
            qualification='finite raw features; no /meta, outcome, event, post-termination or risk-based filter',
            source_split='canonical index train/val arrays only; TEST section not read',
            east_ids=cold, fixed_center=center.tolist(), fixed_scale=scale.tolist(),
            parent=str(CHECKPOINT), source_index=str(INDEX), source_index_train_byte_offset=INDEX_TRAIN_OFFSET,
            east_protocol=str(PROTOCOL), fixed_map=str(MAPPING)),
        interpretation='Unlabelled reconstruction MAE is not first-alarm performance, '
                       'timely-detection benefit, false-alarm rate, or safety evidence. '
                       'Cold20 is a retrospective diagnostic under the already fixed later map; '
                       'no change to historical cold-start behavior is implied.',
        results={})
    for role, device, shots in (('jtext_train', 'J-TEXT', ids['train']),
                                ('jtext_val', 'J-TEXT', ids['val']),
                                ('east_cold20', 'EAST', cold)):
        result['results'][role] = make_domain(device, shots, center, scale)
        result['elapsed_seconds'] = time.monotonic() - start
        with (HERE / 'result.json').open('w') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
    print(f"completed in {result['elapsed_seconds']:.2f}s", flush=True)


if __name__ == '__main__':
    main()
