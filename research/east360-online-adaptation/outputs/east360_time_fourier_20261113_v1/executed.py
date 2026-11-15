"""Compare equal-size causal time and frequency readouts on the fixed EAST 360."""

import argparse
import csv
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from . import commissioning_write_delay_20261112 as delay
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition


group, lag = delay.group, delay.lag
ARCHIVE = io23.ROOT / 'probe_outputs/commissioning_write_delay_east360_20261112_v2'
ARMS = ('original', 'time', 'fourier')
CHANNELS = ('n=1 amplitude', 'dx_a', 'dy_a', 'v_loop', 'sxr_skew', 'sxr_var', 'xuv_skew', 'xuv_var')
COLS = [io23.RAW23.index(name) for name in CHANNELS]
W, RECENT, STRIDE, BATCH = 128, 32, 4, 256
BANDS = ((0., 31.25), (31.25, 125.), (125., 500.))
DESCRIPTORS = (('mean_last32_minus_prev96', 'diff_rms_last32', 'diff_rms_last128'),
               ('power_0_31.25', 'power_31.25_125', 'power_125_500'))
STEPS, LR, RIDGE, Z_LIMIT = 160, .02, .10, 3.
HANN = np.hanning(W)
FREQ = np.fft.rfftfreq(W, .001)
BAND_MASK = [(FREQ > a) & (FREQ <= b) for a, b in BANDS]
MAIN = '0.2'


def features(signal, scale, ends):
    """At index t use exactly [t-127, t]. A gap disables only its own channel."""
    values = sliding_window_view(signal['values'][:, COLS], W, axis=0) if len(signal['values']) >= W else None
    result = np.zeros((2, len(ends), len(COLS), 3), np.float32)
    known = np.zeros((len(ends), len(COLS)), bool)
    rows = np.flatnonzero(ends >= W-1)
    if not len(rows):
        return result, known
    masks = sliding_window_view(signal['available'][:, COLS], W, axis=0)
    for i in range(0, len(rows), BATCH):
        dest = rows[i:i+BATCH]
        starts = ends[dest] - W + 1
        ok = masks[starts].all(axis=-1)
        z = np.where(ok[..., None], values[starts], 0.).astype(np.float64)
        z -= z[..., :1]
        z /= np.asarray(scale)[COLS, None]
        change = z[..., -RECENT:].mean(-1) - z[..., :-RECENT].mean(-1)
        d2 = np.diff(z, axis=-1)**2
        result[0, dest] = np.stack((np.arcsinh(change),
            np.log1p(np.sqrt(d2[..., -(RECENT-1):].mean(-1))),
            np.log1p(np.sqrt(d2.mean(-1)))), -1)
        spectrum = np.fft.rfft((z-z.mean(-1, keepdims=True))*HANN, axis=-1)
        power = (spectrum.real**2 + spectrum.imag**2) / (W*np.square(HANN).sum())
        power[..., 1:-1] *= 2.
        result[1, dest] = np.stack([np.log1p(np.sqrt(power[..., b].sum(-1))) for b in BAND_MASK], -1)
        known[dest] = ok
    return result, known


def shot_weights(labels, disruptive):
    disruptive = np.asarray(disruptive, dtype=bool)
    masses = .5 / np.bincount(disruptive.astype(int))[disruptive.astype(int)]
    rows = []
    for label, mass in zip(labels, masses, strict=True):
        counts = np.bincount(label.astype(int), minlength=2)
        rows.append(mass / ((counts > 0).sum() * counts[label.astype(int)]))
    return rows


def standardized(x, known, fit):
    z = (x - np.asarray(fit['center'])[:, None]) / np.asarray(fit['scale'])[:, None]
    return np.where(known[None, :, :, None], np.clip(z, -Z_LIMIT, Z_LIMIT), 0.)


def apply_extra(original, x, known, fit):
    parts = standardized(x, known, fit) * np.asarray(fit['coef'])[:, None]
    delta = parts.sum((2, 3))
    updated = expit(logit(original.astype(np.float64))[None] + delta).astype(original.dtype)
    return np.where(delta != 0., updated, original[None]), parts


@torch.no_grad()
def source_risk(base, signal, ends):
    return np.concatenate([base(*lag._windows(base, signal, ends[i:i+BATCH]))['logit'].cpu().numpy()
                           for i in range(0, len(ends), BATCH)])


def training_data(base, ids, scale):
    advanced = io23.advanced_times()
    data = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        y = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        t = np.flatnonzero(y['valid'])[::STRIDE]
        if not len(t):
            raise ValueError(f'no J-TEXT training points for {sid}')
        positive = y['y'][t].astype(bool)
        timely = positive & (np.round((y['down']-raw['times'][t])*1000., 9) > 10.) if y['is_disrupt'] else positive
        x, known = features(raw, scale, t)
        data.append((y['is_disrupt'], positive, timely, x, known, source_risk(base, raw, t)))
        if i % 300 == 0 or i == len(ids):
            print(f'J-TEXT TRAIN features: {i}/{len(ids)}', flush=True)
    d, labels, timely, x, known, raw = zip(*data, strict=True)
    owner = np.repeat(np.arange(len(ids)), [len(y) for y in labels])
    return (np.asarray(d), np.concatenate(labels), np.concatenate(timely), np.concatenate(x, axis=1),
            np.concatenate(known), np.concatenate(raw), owner, np.concatenate(shot_weights(labels, d)))


def fit_readouts(data, original_threshold):
    disruptive, labels, timely, x, observed, base_logits, owner, weight = data
    known_weights = (observed * weight[:, None])[None, :, :, None]
    norm = known_weights.sum(axis=1)
    if (norm <= 0).any():
        raise ValueError('feature standardization requires observed TRAIN windows')
    center = (x*known_weights).sum(axis=1) / norm
    var = ((x-center[:, None])**2 * known_weights).sum(axis=1) / norm
    scale = np.sqrt(np.maximum(var, 1e-12))
    fit = dict(center=center.tolist(), scale=scale.tolist())
    safe, positives = np.flatnonzero(~disruptive), np.unique(owner[timely])

    def tensor(v, dtype=None):
        return torch.as_tensor(v, device='cuda', dtype=dtype)

    design = tensor(standardized(x, observed, fit).reshape(2, len(labels), -1), torch.float32)
    base = tensor(base_logits-logit(original_threshold), torch.float32)[None]
    is_positive = tensor(labels, torch.float32)[None].expand_as(base.expand(2, -1))
    w, t, p, s = tensor(weight, torch.float32), tensor(timely), tensor(positives), tensor(safe)
    owner_gpu = tensor(owner)[None].expand(2, -1)
    coeff = torch.nn.Parameter(torch.zeros((2, design.shape[-1]), device='cuda'))

    def objective():
        z = base + torch.bmm(design, coeff[..., None]).squeeze(-1)
        point = F.binary_cross_entropy_with_logits(z, is_positive, reduction='none') @ w
        board = z.new_full((2, len(disruptive)), -torch.inf)
        hit = board.scatter_reduce(1, owner_gpu, z.masked_fill(~t, -torch.inf), 'amax')[:, p]
        false = board.scatter_reduce(1, owner_gpu, z, 'amax')[:, s]
        peaks = .5*(F.softplus(-hit).mean(-1) + F.softplus(false).mean(-1))
        return .5*(point + peaks) + RIDGE*coeff.square().sum(-1)

    optim = torch.optim.Adam([coeff], lr=LR)
    before = objective().detach().tolist()
    for _ in range(STEPS):
        optim.zero_grad(set_to_none=True)
        objective().sum().backward()
        optim.step()
    after = objective().detach().tolist()
    if not np.isfinite(after).all() or np.any(np.asarray(after) > np.asarray(before)):
        raise ValueError('shared TRAIN objective did not improve for both readouts')
    fit.update(coef=coeff.detach().cpu().numpy().reshape(2, len(COLS), 3).tolist(),
               loss_before=before, loss_after=after,
               train=dict(shots=len(disruptive), disruptive=int(disruptive.sum()), safe=len(safe),
                          disrupted_weight=float(weight[disruptive[owner]].sum()),
                          safe_weight=float(weight[~disruptive[owner]].sum()),
                          positive_points=int(labels.sum()), all_points=len(labels),
                          timely_supported_shots=len(positives), observed_points_by_channel=observed.sum(0).tolist()))
    return fit


def original_curve(signal, rank, args, decoder, path, weights):
    base = args[0]
    if rank <= 20:
        z = source_risk(base, signal, np.arange(len(signal['values'])))
        p = torch.as_tensor(z, device=base.center.device).sigmoid().cpu().numpy()
        return p, np.empty(0, int), np.empty(0, bool), np.zeros_like(p)
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    eligible, fields, wave = group.feedback_waveform(signal, q, args[2], z.device)
    states, common = group._estimate_innovation(base, signal, q, eligible, phase, correction,
                                                prior, fields, wave, decoder, weights)
    cur, _, dose, *_ = delay.render(base, signal, q, h, z, beta, phase, correction, states, common, waits=(5,))
    return cur[0], q+lag.H+1, common, dose[0]


def replay(raw, rank, training, scale, mapping, *assets):
    signal = lag.prepare_actual(raw, rank, mapping)
    original, release, eligible, dose = original_curve(signal, rank, *assets)
    n = len(original)
    if rank <= 20:
        scores = np.tile(original, (3, 1))
        available = np.zeros((n, len(COLS)), bool)
        delta = np.zeros((2, n))
    else:
        x, available = features(signal, scale, np.arange(n))
        score, parts = apply_extra(original, x, available, training)
        scores = np.vstack((original, score))
        delta = parts.sum((2, 3))
    return scores, release, eligible, dose, available, delta


def prefix_check(raw, rank, full, threshold, training, scale, mapping, *assets):
    scores, release, eligible, dose, available, delta = full
    above = scores >= threshold
    differing = np.flatnonzero(np.any(above[1:] != above[:1], axis=0))
    cross = np.flatnonzero(above[0])
    pivot = int(differing[0] if len(differing) else cross[0] if len(cross) else min(550, len(dose)-1))
    records = []
    for stop in sorted({max(1, pivot), pivot+1}):
        cut = dict(raw, **{k: raw[k][:stop] for k in ('values', 'available', 'times', 'planning_Bt', 'planning_ip')})
        short, rel, ok, norm, known, dz = replay(cut, rank, training, scale, mapping, *assets)
        np.testing.assert_allclose(short, scores[:, :stop], atol=3e-6, rtol=0.)
        np.testing.assert_array_equal(short >= threshold, above[:, :stop])
        np.testing.assert_array_equal(rel, release[release < stop])
        np.testing.assert_array_equal(ok, eligible[release < stop])
        np.testing.assert_allclose(norm, dose[:stop], atol=3e-6, rtol=1e-6)
        np.testing.assert_array_equal(known, available[:stop])
        np.testing.assert_allclose(dz, delta[:, :stop], atol=2e-14, rtol=0.)
        records.append(dict(points=stop, score_max_abs=float(np.max(abs(short-scores[:, :stop]))),
                            feature_correction_max_abs=float(np.max(abs(dz-delta[:, :stop])))))
    return records


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    original, clock, args, decoder, path = group.load_reference(lock)
    p, archive = (read(ARCHIVE / f'{k}.json') for k in ('protocol', 'seal'))
    teach_path = delay.HISTORY / 'teaching.json'
    if (any(p[k] != original[k] for k in ('order', 'startup', 'original', 'calibration', 'source_sha256'))
            or p['startup'] != 20 or len(p['order']) != 360 or len(archive['curve_sha256']) != 360
            or p['reference_seal'] != sha(group.REFERENCE / 'seal.json')
            or p['history_seal'] != sha(delay.HISTORY / 'seal.json')
            or read(delay.HISTORY / 'protocol.json')['teaching_sha256'] != sha(teach_path)):
        raise ValueError('existing EAST 360 Step 5 bindings changed')
    scale = args[0].scale.detach().cpu().numpy()
    weights = torch.tensor(read(teach_path)['weights'], device='cuda', dtype=torch.float32)
    frozen = [state_digest(m) for m in args[:2]]
    train = io23.source_roles()['train']
    allowed = set(train)
    raw_outcome = io23.read_outcome

    def train_outcome(device, sid):
        if device != 'J-TEXT' or sid not in allowed:
            raise ValueError('new readout can only see J-TEXT TRAIN outcomes')
        return raw_outcome(device, sid)

    (out / 'shots').mkdir(parents=True)
    copyfile(Path(__file__), out / 'executed.py')
    write(out / 'protocol.json', dict(
        **{k: p[k] for k in ('order', 'startup', 'calibration', 'original', 'source_sha256')},
        arms=ARMS, main=MAIN, window=W, recent=RECENT, channels=CHANNELS, channel_source_scale=scale.tolist(),
        fourier_bands_hz=BANDS, descriptors=DESCRIPTORS, train_shots=train, train_stride=STRIDE,
        train_steps=STEPS, train_lr=LR, ridge=RIDGE, standardized_clip=Z_LIMIT, train_intercept=False,
        objective='half point logistic with equal class and shot mass; half balanced source shot peak logistic',
        cold='original replay and threshold, no branch yet', source_labels='J-TEXT TRAIN HDF5 only',
        step5='original feedback only, wait 5 extra points; output residual never feeds back',
        reference_sha256=sha(ARCHIVE / 'seal.json'), executed_sha256=sha(out / 'executed.py')))
    with patch.object(io23, 'read_outcome', side_effect=train_outcome):
        fit = fit_readouts(training_data(args[0], train, scale), p['original'])
    if (fit['train']['disruptive'], fit['train']['safe']) != (188, 1165):
        raise ValueError('class counts changed in source TRAIN')
    write(out / 'fit.json', fit)
    print(dict(stage='readouts_sealed', loss=fit['loss_after'], train=fit['train'],
               seconds=round(time.monotonic()-start, 1)), flush=True)
    digests, witnesses = {}, {}
    baseline_error, max_dose_error = 0., 0.
    active, point_support, nonzero_correction = 0, np.zeros(len(COLS), np.int64), np.zeros(2, np.int64)
    live_points = 0
    assets = (args, decoder, path, weights)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('EAST outcomes forbidden before score seal')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            a, b = ARCHIVE / 'shots' / name, group.REFERENCE / 'shots' / name
            if (sha(a) != archive['curve_sha256'][name] or sha(b) != clock['curve_sha256'][name]
                    or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]):
                raise ValueError(f'raw or sealed reference changed: {sid}')
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            full = replay(raw, rank, fit, scale, lock['mapping'], *assets)
            score, rel, ok, dose, available, delta = full
            tau = p['original'] if rank <= p['startup'] else p['calibration'][MAIN]
            with np.load(a, allow_pickle=False) as ref, np.load(b, allow_pickle=False) as t:
                np.testing.assert_array_equal(raw['times'], t['times'])
                np.testing.assert_array_equal(rel, ref['release'])
                np.testing.assert_array_equal(ok, ref['eligible'])
                np.testing.assert_allclose(score[0], ref['current_5'], atol=3e-6, rtol=0.)
                for level in (tau,) if rank <= p['startup'] else set(p['calibration'].values()):
                    np.testing.assert_array_equal(score[0] >= level, ref['current_5'] >= level)
                baseline_error = max(baseline_error, float(np.max(abs(score[0]-ref['current_5']))))
                np.testing.assert_array_equal(dose > 0., ref['dose'][-1] > 0.)
                np.testing.assert_allclose(dose, ref['dose'][-1], atol=3e-6, rtol=1e-6)
                max_dose_error = max(max_dose_error, float(np.max(abs(dose-ref['dose'][-1]))))
            if sid in (67900, 70320) or rank in (21, 360):
                witnesses[str(sid)] = prefix_check(raw, rank, full, tau, fit, scale, lock['mapping'], *assets)
            if rank > 20:
                live_points += score.shape[1]
                point_support += available.sum(0)
                nonzero_correction += np.count_nonzero(delta, axis=1)
            active += np.count_nonzero(dose)
            dest = out / 'shots' / name
            with dest.open('xb') as f:
                np.savez_compressed(f, times=raw['times'], dose=dose, release=rel, eligible=ok,
                                    delta_logit=delta.astype(np.float32), **dict(zip(ARMS, score, strict=True)))
            digests[name] = sha(dest)
            if rank % 60 == 0 or rank == 21:
                print(dict(stage='replayed', rank=rank, seconds=round(time.monotonic()-start, 1)), flush=True)
    if frozen != [state_digest(m) for m in args[:2]]:
        raise ValueError('a frozen EAST or J-TEXT model changed')
    write(out / 'seal.json', dict(curve_sha256=digests, fit_sha256=sha(out / 'fit.json'),
                                protocol_sha256=sha(out / 'protocol.json'), prefix_witnesses=witnesses,
                                baseline_max_error=baseline_error, max_dose_error=max_dose_error,
                                step5_active_points=int(active), endpoint_support=point_support.tolist(),
                                live_points=live_points, nonzero_correction_points=nonzero_correction.tolist(),
                                elapsed_s=time.monotonic()-start))


def score(out):
    p, seal = (read(out / f'{k}.json') for k in ('protocol', 'seal'))
    if (len(seal['curve_sha256']) != 360 or seal['protocol_sha256'] != sha(out / 'protocol.json')
            or seal['fit_sha256'] != sha(out / 'fit.json') or p['reference_sha256'] != sha(ARCHIVE / 'seal.json')):
        raise ValueError('no EAST scoring before all predictions and fitted weights have been sealed')
    prior = read(ARCHIVE / 'per_shot.json')
    per = {band: {arm: [] for arm in ARMS} for band in p['calibration']}
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        f = out / 'shots' / name
        if sha(f) != seal['curve_sha256'][name]:
            raise ValueError('sealed prediction changed')
        disruptive, down, t = group.hdf._raw_truth(sid)
        with np.load(f, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['times'], t)
            for band, thresh in p['calibration'].items():
                tau = p['original'] if rank <= p['startup'] else thresh
                for arm in ARMS:
                    row = dict(shot=sid, disruptive=disruptive, down_s=down,
                               **score_curve(t, z[arm], disruptive, down, tau))
                    if arm == 'original' and row != prior[band]['current_5'][rank-1]:
                        raise ValueError(f'original first alarm differs: {sid}')
                    valid = (down-t)*1000 >= -1e-6 if disruptive else np.ones(len(t), bool)
                    hits = np.flatnonzero(valid & (z[arm] >= tau))
                    row.update(rank=rank, first_idx=int(hits[0]) if len(hits) else None,
                               first_s=float(t[hits[0]]) if len(hits) else None,
                               peak_score=float(z[arm][valid].max()), threshold=tau)
                    per[band][arm].append(row)
    metrics = {b: {k: timing_decomposition(v) for k, v in rows.items()} for b, rows in per.items()}
    if (metrics[MAIN]['original']['tp'], metrics[MAIN]['original']['fp']) != (146, 34):
        raise ValueError('original Step 5 is not reproduced')
    comparisons = {b: {k: dict(flips=flips(v['original'], v[k]), lead=lead.paired(v['original'], v[k]))
                       for k in ARMS[1:]} for b, v in per.items()}
    write(out / 'per_shot.json', per)
    write(out / 'result.json', dict(metrics=metrics, comparisons=comparisons,
                                   verified_curves=360*len(ARMS)*len(per), physical_realtime_verified=False))
    with (out / 'per_shot.csv').open('x', encoding='utf-8-sig', newline='') as f:
        fields = ('alarm', 'timely', 'first_s', 'lead_ms', 'peak_score')
        writer = csv.writer(f)
        writer.writerow(('rank', 'shot', 'disruptive', 'down_s', 'threshold',
                         *(f'{k}_{v}' for k in ARMS for v in fields)))
        for a, t, f in zip(*(per[MAIN][k] for k in ARMS), strict=True):
            writer.writerow([a[k] for k in ('rank', 'shot', 'disruptive', 'down_s', 'threshold')]
                            + [row[k] for row in (a, t, f) for k in fields])
    print({b: {k: (v['tp'], v['fp']) for k, v in rows.items()} for b, rows in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.out.resolve())
    score(args.out.resolve())
