"""Check Fourier-only rises with other diagnostics, using source-frozen rules."""

import argparse
import csv
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from scipy.special import expit

from . import commissioning_time_fourier_20261113 as prior
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


REFERENCE = io23.ROOT / 'outputs/east360_time_fourier_20261113_v1'
CHANNELS = ('ip', 'ne_nG', 'Mir_VV')
COLS = [io23.RAW23.index(k) for k in CHANNELS]
W, BLOCK, K, RIDGE = 128, 32, 4, .10
NAMES = ('last32_vs_prev96', 'last32_vs_prior32', 'ordered_change')
ARMS = ('original', 'fourier', 'positive_only', 'confirmed_current', 'confirmed')


def features(signal, ends):
    """Three signed changes for each held-out sensor, computed on the same past."""
    x = np.abs(np.asarray(signal['values'][:, COLS], dtype=np.float64))
    present = signal['available'][:, COLS]
    ans = np.zeros((len(ends), len(COLS), len(NAMES)))
    known = np.zeros((len(ends), len(COLS)), bool)
    rows = np.flatnonzero(ends >= W-1)
    if not len(rows):
        return ans, known
    windows = sliding_window_view(x, W, axis=0)
    masks = sliding_window_view(present, W, axis=0)
    for start in range(0, len(rows), 1024):
        idx = rows[start:start+1024]
        chosen = ends[idx]+1-W
        frame = windows[chosen]
        ok = masks[chosen].all(axis=-1)
        blocks = frame.reshape(len(idx), len(COLS), 4, BLOCK).mean(-1)
        norm = frame.mean(-1)
        ok &= norm > 0.
        recent, previous = blocks[..., -1], blocks[..., -2]
        totals = np.stack((recent-blocks[..., :3].mean(-1), recent-previous,
                           blocks @ np.array([-.3, -.1, .1, .3])), axis=-1)
        ans[idx] = np.divide(totals, norm[..., None], out=np.zeros_like(totals), where=ok[..., None])
        known[idx] = ok
    return ans, known


def gate(parent, fourier, evidence, available, cut):
    return np.where((fourier > parent) & available & (evidence > cut), fourier, parent)


def safety_cut(scores):
    """Admit at most a quarter of source validation safe proposal shots; ties fail closed."""
    if not len(scores):
        raise ValueError('no source safe proposal shots to set the confirmation cutoff')
    return float(np.sort(scores)[len(scores)-1-len(scores)//4])


def evidence(x, model):
    z = np.clip((x-np.asarray(model['center']))/np.asarray(model['scale']), -3., 3.)
    return (z.reshape(len(x), len(model['beta']))*np.asarray(model['beta'])).sum(axis=1)


def source_scores(ids, base, scale, fixed_fit, threshold, *, training, outcome_reader):
    advanced = io23.advanced_times()
    out = []
    allowed = set(ids)

    def outcome(device, shot):
        if device != 'J-TEXT' or shot not in allowed:
            raise ValueError('confirmation may read only its stated source role')
        return outcome_reader(device, shot)

    with patch.object(io23, 'read_outcome', side_effect=outcome):
        for sid in ids:
            s = simple_data.read_shot('J-TEXT', sid)
            truth = simple_data.attach_outcome(s, 'J-TEXT', advanced)
            t = np.flatnonzero(truth['valid'])[::prior.STRIDE if training else 1]
            x, available = features(s, t)
            f, support = prior.features(s, scale, t)
            offset = (prior.standardized(f, support, fixed_fit)[1]*np.asarray(fixed_fit['coef'][1])).sum((1, 2))
            logits = prior.source_risk(base, s, t).astype(np.float64)
            original = expit(logits)
            fourier = expit(logits+offset)
            d = truth['is_disrupt']
            if d:
                h = np.round((truth['down']-s['times'][t])*1000., 9)
                pos = (h>10.) & truth['y'][t].astype(bool)
            else:
                pos = np.ones(len(t), dtype=bool)
            valid = available.all(1) & pos
            if training:
                # Source model already sees most disruptions. Learn from the
                # most alert-like legal window in both classes, even there.
                ix = np.flatnonzero(valid)
                ix = ix[np.argsort(-fourier[ix], kind='stable')[:K]]
                out.append((sid, d, bool(np.any(valid)), x[ix]))
            else:
                candidate = (original < threshold) & (fourier >= threshold)
                out.append((sid, d, candidate & pos, available.all(1), x))
    return out


def fit_models(train, val):
    labels = np.array([row[1] for row in train if row[2]])
    rows = [row[3] for row in train if row[2]]
    weights = prior.shot_weights([np.full(len(r), tag, bool) for r, tag in zip(rows, labels, strict=True)], labels)
    y = np.repeat(labels.astype(float), [len(r) for r in rows])
    w = np.concatenate(weights)
    x = np.concatenate(rows)
    center = np.sum(x*w[:, None, None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-center)**2*w[:, None, None], axis=0)), 1e-8)
    z = np.clip((x-center)/scale, -3., 3.).reshape(len(x), -1)

    def fit(use):
        def loss(b):
            a = z[:, use]@b
            return (float(w@(np.logaddexp(0., a)-y*a) + RIDGE*(b@b)/2.),
                    z[:, use].T@(w*(expit(a)-y))+RIDGE*b)

        got = minimize(loss, np.zeros(int(sum(use))), jac=True, method='L-BFGS-B', options={'gtol': 1e-9})
        if not got.success:
            raise ValueError(got.message)
        beta = np.zeros(len(use))
        beta[use] = got.x
        model = dict(center=center.tolist(), scale=scale.tolist(), beta=beta.tolist(), train_loss=float(got.fun))
        positives, safe = [], []
        rejected = [0, 0]
        for _, d, proposed, present, vx in val:
            if not proposed.any():
                continue
            if not (proposed & present).any():
                rejected[int(d)] += 1
                continue
            (positives if d else safe).append(evidence(vx[proposed & present], model).max())
        cut = safety_cut(safe)
        return dict(model, cutoff=cut, validation=dict(safe_proposal_shots=len(safe),
            safe_admitted_shots=int(np.count_nonzero(np.asarray(safe)>cut)),
            disruptive_proposal_shots=len(positives), disruptive_admitted_shots=int(np.count_nonzero(np.asarray(positives)>cut)),
            source_proposal_shots_missing_confirmation=rejected))

    full = np.ones(len(COLS)*len(NAMES), bool)
    current = np.arange(len(full)) < len(NAMES)
    return dict(confirmed_current=fit(current), confirmed=fit(full),
                train=dict(disruptive_used=int(labels.sum()), safe_used=int((~labels).sum()), windows=len(w),
                           disruptive_weight=float(w[y==1.].sum()), safe_weight=float(w[y==0.].sum()),
                           fit_ids=[row[0] for row in train if row[2]],
                           no_usable_window=[row[0] for row in train if not row[2]]))


def predict_curves(parent, fourier, samples, models):
    x, known = features(samples, np.arange(len(parent)))
    present = known.all(1)
    scores = dict(original=parent, fourier=fourier, positive_only=np.maximum(parent, fourier))
    values = {k: evidence(x, models[k]) for k in ARMS[-2:]}
    for k in ARMS[-2:]:
        scores[k] = gate(parent, fourier, values[k], present, models[k]['cutoff'])
    return scores, values, present


def predict(out):
    p, old_seal, fixed_fit = (read(REFERENCE/f'{k}.json') for k in ('protocol', 'seal', 'fit'))
    lock = prior.lag.load_inputs()
    if p['order'] != lock['ids'] or len(old_seal['curve_sha256']) != 360 or p['startup'] != 20:
        raise ValueError('expected 360 sealed Step 5 and Fourier curves')
    if old_seal['fit_sha256'] != sha(REFERENCE/'fit.json') or old_seal['protocol_sha256'] != sha(REFERENCE/'protocol.json'):
        raise ValueError('original calibration or Fourier fit changed')
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed.py')
    roles = io23.source_roles()
    if p['train_shots'] != roles['train']:
        raise ValueError('source role mismatch')
    write(out/'protocol.json', dict(order=p['order'], startup=p['startup'], original=p['original'],
        calibration=p['calibration'], arms=ARMS, primary='confirmed', window=W, block=BLOCK, features=NAMES,
        sensors=CHANNELS, train_shots=roles['train'], source_calibration_shots=roles['val'], ridge=RIDGE,
        train_windows_per_shot=K, source_safe_proposal_budget=.25,
        rules='source-frozen confirmation is only applied to positive Fourier corrections; preserve original alarms',
        reference_seal=sha(REFERENCE/'seal.json'), cold='original preparation unchanged'))
    prior.lag.configure()
    base, ckpt = load_parent(DEFAULT_PARENT)
    base.cuda()
    if ckpt['threshold'] != p['original']:
        raise ValueError('expected fixed source threshold')
    scale = np.asarray(p['channel_source_scale'])
    mapping = lock['mapping']
    reader = io23.read_outcome
    train, val = (source_scores(roles[role], base, scale, fixed_fit, p['original'],
                  training=role=='train', outcome_reader=reader) for role in ('train','val'))
    models = fit_models(train, val)
    write(out/'fit.json', models)
    print({k: models[k]['validation'] for k in ARMS[-2:]}, flush=True)
    files, checks = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('EAST truth unavailable while predicting')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            src = REFERENCE/'shots'/name
            if sha(src) != old_seal['curve_sha256'][name]:
                raise ValueError('original replay no longer matches its seal')
            with np.load(src, allow_pickle=False) as f:
                t, original, fourier = (f[k] for k in ('times', 'original', 'fourier'))
            if rank <= p['startup']:
                np.testing.assert_array_equal(original, fourier)
                scores = dict.fromkeys(ARMS, original)
                yes = np.zeros(len(t), dtype=bool)
                values = dict.fromkeys(ARMS[-2:], np.zeros(len(t)))
            else:
                if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('confirmation signals differ from the original replay input')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(t, raw['times'])
                sig = prior.lag.prepare_actual(raw, rank, mapping)
                scores, values, yes = predict_curves(original, fourier, sig, models)
                if rank in (21,120,360):
                    for stop in (127, min(len(t), 551)):
                        short = {k: sig[k][:stop] for k in ('values', 'available')}
                        a, b, c = predict_curves(original[:stop], fourier[:stop], short, models)
                        for arm in ARMS:
                            np.testing.assert_array_equal(a[arm], scores[arm][:stop])
                        for arm in values:
                            np.testing.assert_array_equal(b[arm], values[arm][:stop])
                        np.testing.assert_array_equal(c, yes[:stop])
                    checks[str(sid)] = [127,min(len(t),551)]
            for k in ARMS[-2:]:
                if not ((scores[k] >= original) & (scores[k] <= scores['positive_only'])).all():
                    raise ValueError('confirmation cannot lower the original or create a boost')
            f = out/'shots'/name
            with f.open('xb') as file:
                np.savez_compressed(file, times=t, available=yes, evidence_current=values[ARMS[-2]],
                                    evidence=values[ARMS[-1]], **scores)
            files[name] = sha(f)
            if rank%120 == 0:
                print(f'EAST predictions sealed: {rank}/360', flush=True)
    write(out/'seal.json',dict(curve_sha256=files, fit_sha256=sha(out/'fit.json'),
                               protocol_sha256=sha(out/'protocol.json'), prefix=checks))


def score(out):
    p, seal = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if (len(seal['curve_sha256']) != 360 or seal['fit_sha256'] != sha(out/'fit.json')
            or seal['protocol_sha256'] != sha(out/'protocol.json')):
        raise ValueError('complete frozen replay required before scoring EAST outcomes')
    old = read(REFERENCE/'per_shot.json')
    per = {b:{k:[] for k in ARMS} for b in p['calibration']}
    for rank,sid in enumerate(p['order'],1):
        f = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f) != seal['curve_sha256'][f.name]:
            raise ValueError('sealed prediction changed')
        d, down, clock = prior.group.hdf._raw_truth(sid)
        with np.load(f, allow_pickle=False) as z:
            t = z['times']
            np.testing.assert_array_equal(clock, t)
            valid = (down-t)*1000 >= -1e-6 if d else np.ones(len(t),bool)
            for b, tau in p['calibration'].items():
                cut = p['original'] if rank <= p['startup'] else tau
                for arm in ARMS:
                    row = dict(shot=sid, disruptive=d, down_s=down, **score_curve(t, z[arm], d, down, cut))
                    hit = np.flatnonzero(valid & (z[arm] >= cut))
                    row.update(rank=rank, first_idx=int(hit[0]) if len(hit) else None,
                           first_s=float(t[hit[0]]) if len(hit) else None, threshold=cut)
                    if arm in ('original','fourier') and any(row[k] != old[b][arm][rank-1][k] for k in row):
                        raise ValueError('sealed reference shot changed during scoring')
                    per[b][arm].append(row)
    metrics = {b:{k:timing_decomposition(rows) for k,rows in arms.items()} for b,arms in per.items()}
    compare = {b:{k:dict(flips=flips(arms['original'], arms[k]), lead=lead.paired(arms['original'], arms[k]))
                  for k in ARMS[1:]} for b,arms in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(metrics=metrics, comparisons=compare,
                                verified_curves=360*len(ARMS)*len(metrics)))
    with (out/'per_shot.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        fields = ('alarm','timely','first_s','lead_ms')
        w.writerow(('rank','shot','disruptive','down_s','threshold',*(f'{k}_{v}' for k in ARMS for v in fields)))
        for rows in zip(*(per[prior.MAIN][k] for k in ARMS),strict=True):
            w.writerow([rows[0][k] for k in ('rank','shot','disruptive','down_s','threshold')]+
                       [row[k] for row in rows for k in fields])
    print({b:{k:(v['tp'],v['fp']) for k,v in m.items()} for b,m in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    predict(args.out.resolve())
    score(args.out.resolve())
