"""Separate diagnostic levels, within-window shape, and projection cross-talk."""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data
from .commissioning_roughness_20261110 import coordinates
from .history_gradient_metric_20261102.run import read, sha, write


NAMES = ('raw_level', 'raw_shape', 'aligned_shape', 'aligned_full',
         'only8_patch', 'full_patch_no_intra', 'full_patch_with_intra')


def component(row, kind):
    values = row.reshape(-1, len(probe.COLUMNS), probe.WIDTH//probe.BIN)
    view = values[..., 0] if kind == 'level' else values[..., 1:]
    return view.reshape(len(row), int(np.prod(view.shape[1:])))


def value_subspace(weight):
    """Map a patch-constant value in each selected lane to its real token."""
    lane = weight.unflatten(1, (-1, 46))[..., list(probe.COLUMNS)].sum(1)
    return lane, torch.linalg.pinv(lane)


def bins(z):
    value = z.reshape(len(z), probe.WIDTH//probe.BIN, -1, z.shape[-1]).mean(2)
    return torch.cat((value.mean(1, keepdim=True), value.diff(dim=1)), dim=1).transpose(1, 2).flatten(1)


def patch_views(patches, weight, bias, inverse):
    """Decompose the existing projection; this does not change the model."""
    selected = patches.unflatten(-1, (-1, 46))[..., list(probe.COLUMNS)]
    w = weight.unflatten(-1, (-1, 46))[..., list(probe.COLUMNS)]
    values = selected.reshape(len(patches), probe.WIDTH, len(probe.COLUMNS))
    actual = F.linear(patches, weight, bias) - bias
    only = F.linear(selected.flatten(2), w.flatten(1))
    return dict(raw=bins(values).cpu().numpy(),
                selected=bins(F.linear(only, inverse)).cpu().numpy(),
                lane=bins(F.linear(actual, inverse)).cpu().numpy())


def later_subset(valid, on_time):
    return np.flatnonzero(valid & on_time)


@torch.no_grad()
def features(signal, base, inverse, ends, track=None):
    w, bias = base.embedding.weight, base.embedding.bias
    center, scale = (v.cpu().numpy()[list(probe.COLUMNS)] for v in (base.center, base.scale))
    waveform, observed = coordinates(signal['values'][:, probe.COLUMNS], signal['available'][:, probe.COLUMNS],
                                     center, scale)
    raw, valid = probe.trajectory(waveform, observed, ends)
    operating, enough = probe.operating(signal, ends)
    data = dict(raw_level=component(raw, 'level'), raw_shape=component(raw, 'shape'))
    aligned, projected, lane0, lane1, untouched = [], [], [], [], []
    for j in range(0, len(ends), probe.BATCH):
        t = ends[j:j+probe.BATCH]
        values, available = probe.lag._windows(base, signal, t)
        # load_parent locks the source-asinh, 46-lane input view.
        absolute = ((torch.where(available, values, base.center)-base.center)/base.scale).asinh()
        patches = torch.cat((absolute, available.to(absolute.dtype)), dim=-1).reshape(
            len(t), probe.WIDTH//base.cfg.patch, -1)
        original = patch_views(patches, w, bias, inverse)
        if track is None:
            after = original
        else:
            q, phase, offset = track
            slot = probe.lag.delayed_slots(q, t)[0]
            corrected = probe.layer.patcher(phase[slot], offset[slot])(patches)
            after = patch_views(corrected, w, bias, inverse)
        untouched.append(original['raw'])
        aligned.append(after['raw'])
        projected.append(after['selected'])
        lane0.append(original['lane'])
        lane1.append(after['lane'])

    width = raw.shape[1]
    def stack(parts):
        return np.concatenate(parts) if parts else np.empty((0, width), np.float32)

    np.testing.assert_allclose(stack(untouched), raw, atol=3e-6, rtol=1e-6)
    x = stack(aligned)
    return operating.astype(np.float32), raw, valid & enough, dict(**data,
        aligned_shape=component(x, 'shape'), aligned_full=x, only8_patch=stack(projected),
        full_patch_no_intra=stack(lane0), full_patch_with_intra=stack(lane1))


def fit_source(base, inverse):
    rows = []
    advanced = io23.advanced_times()
    for sid in simple_data.source_roles()['train']:
        raw, ends, disrupted = probe.source_training_points(sid, advanced)
        if not len(ends):
            continue
        op, _, valid, x = features(raw, base, inverse, ends)
        if valid.any():
            rows.append(dict(op=op[valid], shot=np.full(valid.sum(), sid),
                target=np.full(valid.sum(), disrupted, bool), **{k: v[valid] for k,v in x.items()}))
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return {k: probe.fit_probe(data['op'], data[k], data['shot'], data['target']) for k in NAMES}, (
        probe.train_counts(data['shot'], data['target']))


def scores(op, features, models):
    return np.column_stack([probe.apply_probe(op, features[k], models[k]) for k in NAMES]).astype(np.float32)


def observe(original, out):
    p, sealed, prior_fit = (read(original/f'{k}.json') for k in ('protocol', 'seal', 'fit'))
    lock, _, _, args, decoder, path = probe.history.inputs()
    if p['order'] != lock['ids'] or len(sealed['curves']) != 360:
        raise ValueError('need the completed same-stream probe')
    base = args[0]
    subspace, inverse = value_subspace(base.embedding.weight.detach())
    sv = torch.linalg.svdvals(subspace).cpu().numpy()
    models, counts = fit_source(base, inverse)
    if counts != prior_fit['usable']:
        raise ValueError('source fit changed its window support')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__, out/'executed.py')
    write(out/'fit.json', dict(models=models, counts=counts, singular_values=sv.tolist(),
        base_matrix_rank=int(torch.linalg.matrix_rank(subspace)), lanes=[io23.RAW23[c] for c in probe.COLUMNS]))
    write(out/'protocol.json', dict(scope='OBSERVATION_ONLY_EAST360_STAGE_LOCALIZATION',
        reference=str(original), order=p['order'], compared=NAMES,
        endpoint=p['endpoint'], startup=p['startup'],
        source='TRAIN only; same weighted examples, regularization and sampling as the original probe',
        alignment='exact inherited phase and historical bias at the endpoint; before the added token update',
        readout='same eight constant-within-patch directions of the actual 368-to-64 linear map; '
                'only8 uses selected value contributions, full includes all other value and availability lanes',
        limits='offline probes only, unchanged sealed predictions'))
    files = {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('observations are preseal')):
        val = []
        for sid in simple_data.source_roles()['val']:
            raw = simple_data.read_shot('J-TEXT', sid)
            t = probe.end_grid(len(raw['times']))
            op, _, valid, x = features(raw, base, inverse, t)
            val.append(dict(shot=np.full(len(t), sid), ends=t, valid=valid, score=scores(op, x, models)))
        for rank, sid in enumerate(p['order'], 1):
            key = f'{rank:04d}_{sid}.npz'
            if sha(original/'shots'/key) != sealed['curves'][key]:
                raise ValueError('original probe support is no longer sealed')
            with np.load(original/'shots'/key, allow_pickle=False) as old:
                t = old['ends']
                if rank <= p['startup']:
                    value = np.empty((0, len(NAMES)), np.float32)
                else:
                    current = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                    signal = probe.lag.prepare_actual(current, rank, lock['mapping'])
                    q, _, _, _, _, phase, correction = probe.group.parent_track(signal, args, decoder, path)
                    op, waveform, valid, x = features(signal, base, inverse, t, (q, phase, correction))
                    np.testing.assert_array_equal(valid, old['valid'])
                    ref = probe.apply_probe(op, waveform, prior_fit['probes']['raw_trajectory'])
                    np.testing.assert_allclose(ref, old['score'][:, 1], atol=1e-6, rtol=1e-6)
                    value = scores(op, x, models)
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f, ends=t, score=value)
            files[key] = sha(out/'shots'/key)
            if rank in (120, 240, 360):
                print(dict(stage='sealed', rank=rank), flush=True)
        with (out/'val.npz').open('xb') as f:
            np.savez_compressed(f, **{k: np.concatenate([r[k] for r in val]) for k in val[0]})
    write(out/'seal.json', dict(shots=files, val=sha(out/'val.npz')))


def source_auc(out):
    with np.load(out/'val.npz', allow_pickle=False) as f:
        data = dict(f)
    advanced = io23.advanced_times()
    targets, peaks = [], []
    for sid in simple_data.source_roles()['val']:
        disrupted, down = io23.read_outcome('J-TEXT', sid)
        k = np.flatnonzero((data['shot'] == sid) & data['valid'])
        if disrupted:
            raw = simple_data.read_shot('J-TEXT', sid)
            ld = np.round((down - raw['times'][data['ends'][k]]) * 1000., 9)
            k = k[(ld > 10.) & (ld <= io23.effective_horizon_ms(sid, advanced)[0])]
        if len(k):
            targets.append(disrupted)
            peaks.append(data['score'][k].max(0))
    target = np.asarray(targets)
    n = int(target.sum())
    auc = (rankdata(peaks, axis=0)[target].sum(0) - n*(n+1)/2) / (n*(len(target)-n))
    return dict(positive=n, safe=len(target)-n, descriptive_peak_auc=dict(zip(NAMES, auc.tolist(), strict=True)))


def pairs(entries, joined, raw_prior, first_alarm, *, hard):
    names = (*probe.VIEWS, *NAMES)
    before, after, intime, later = [], [], [], []
    by_shot = {}
    for row in entries:
        p, q = row['precursor_shot'], row['safe_shot']
        values, subsequent, on_time = [], [], []
        for i, (shot, end) in enumerate(((p, row['precursor_index']), (q, row['safe_index']))):
            current = joined[shot]
            j = int(np.searchsorted(current['ends'], end))
            if j == len(current['ends']) or current['ends'][j] != end:
                raise ValueError('comparison must use the exact old pair')
            values.append(current['score'][j])
            disrupted, down, clock = raw_prior[shot]
            if disrupted != (i == 0):
                raise ValueError('matched outcome differs from original HDF5')
            if not disrupted and hard and (first_alarm[shot] == len(clock) or end > first_alarm[shot]):
                raise ValueError('hard match must precede a real false alarm')
            k, _ = probe.later_rows(current['ends'], np.array([j]))
            valid = len(k) and current['valid'][k[0]]
            if valid:
                point = current['ends'][k[0]]
                if disrupted:
                    lead = round((down-clock[point])*1000., 9)
                    valid = lead >= -1e-6
                    available = lead > 30.
                else:
                    available = point <= first_alarm[shot]
            if valid:
                subsequent.append(current['score'][k[0]])
                on_time.append(available)
        d = values[0]-values[1]
        before.append(d)
        ok = len(subsequent) == 2
        later.append(ok)
        intime.append(ok and all(on_time))
        after.append(subsequent[0]-subsequent[1] if ok else np.zeros(len(names)))
        by_shot[f'{p}/{q}'] = dict(before={k: float(x) for k,x in zip(names,d,strict=True)},
            later_jointly_in_time=intime[-1])
    baseline, future = np.asarray(before), np.asarray(after)
    selected = later_subset(np.asarray(later), np.asarray(intime))
    scores_at_t = probe.describe_difference(baseline, names)
    adjacent = {k: dict(
        fixes=int(np.sum((baseline[:,j]>0) & (baseline[:,1]<=0))),
        regresses=int(np.sum((baseline[:,j]<=0) & (baseline[:,1]>0))))
        for j,k in enumerate(names)}
    return dict(pairs=len(entries), scores=scores_at_t, against_initial_raw=adjacent,
        same_pairs_plus32_in_time=dict(pairs=len(selected), before=probe.describe_difference(baseline[selected], names),
                                       later=probe.describe_difference(future[selected], names)),
        details=by_shot)


def score(original, out):
    seal, protocol = (read(out/f'{k}.json') for k in ('seal','protocol'))
    original_seal, objective = (read(original/f'{k}.json') for k in ('seal','result'))
    if (len(seal['shots']) != 360 or len(original_seal['curves']) != 360
            or seal['val'] != sha(out/'val.npz') or protocol['reference'] != str(original)):
        raise ValueError('complete sealed comparison inputs and source VAL required')
    archive = read(probe.history.ARCHIVE/'seal.json')
    tau = read(original/'protocol.json')['calibration']['0.2']
    joined, truth, first_alarm = {}, {}, {}
    relevant = {r[field] for arm in objective['comparisons'].values() for r in arm['pair_rows']
                for field in ('precursor_shot','safe_shot')}
    for rank, sid in enumerate(protocol['order'], 1):
        key = f'{rank:04d}_{sid}.npz'
        if (sha(out/'shots'/key) != seal['shots'][key] or sha(original/'shots'/key) != original_seal['curves'][key]):
            raise ValueError('matched support changed')
        if sid in relevant:
            with np.load(out/'shots'/key, allow_pickle=False) as added, np.load(original/'shots'/key, allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ends'], added['ends'])
                joined[sid] = dict(ends=a['ends'], valid=a['valid'] & (a['dose'] > 0),
                                   score=np.c_[a['score'], added['score']])
            truth[sid] = probe.group.hdf._raw_truth(sid)
            if not truth[sid][0]:
                curve = probe.history.archived_shot(rank, sid, archive)['current_5']
                hits = np.flatnonzero(curve >= tau)
                first_alarm[sid] = int(hits[0]) if len(hits) else len(curve)
    reports = {name: pairs(data['pair_rows'], joined, truth, first_alarm, hard=(name == 'difficult'))
               for name, data in objective['comparisons'].items()}
    result = dict(source=source_auc(out), comparisons=reports)
    write(out/'result.json', result)
    for name, row in reports.items():
        print(name, row['pairs'], {k: v['correct'] for k, v in row['scores'].items()},
              'still timely +32', row['same_pairs_plus32_in_time']['pairs'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('original', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--score', action='store_true')
    args = cli.parse_args()
    (score if args.score else observe)(args.original.resolve(), args.output.resolve())
