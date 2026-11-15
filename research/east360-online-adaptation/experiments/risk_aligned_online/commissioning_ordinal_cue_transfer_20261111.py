"""Post-seal diagnostic of whether ordered source feedback transfers to EAST."""

import argparse
from pathlib import Path

import numpy as np
import torch

from . import commissioning_ordinal_20261111 as ordinal
from . import commissioning_ordinal_pairs_20261111 as pairs
from . import commissioning_surplus_20261111 as surplus
from . import commissioning_weak_shape_source_20261111 as weak
from . import io23, simple_data
from .commissioning_lag_state_20261110 import LagState
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_repair_20261104.east_evaluation import load_inputs, prepare_actual
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


def eligible_endpoints(indices, track, groups):
    obs = track['observed']
    ok = np.column_stack([obs[:, groups == g].any(1) for g in np.unique(groups)]).sum(1) >= 2
    first = track['origins'][ok][:1] + surplus.HORIZON + 1
    if not len(first):
        return np.zeros(len(indices), bool)
    return (indices[:, 0] >= first[0]) & track['matchable'].reshape(-1, 2).all(1)


def compare(rows, matched, selection, prediction):
    pos, neg = (v[selection] for v in matched[:2])
    shots, inv, count = np.unique(rows['shot'][pos], return_inverse=True, return_counts=True)
    info = dict(pairs=len(pos), disruptive_shots=len(shots),
                safe_donors=len(np.unique(rows['shot'][neg])), views={})
    if not len(pos):
        return info
    wins = {}
    for k, v in prediction.items():
        delta = ordinal.ordered_difference(v[pos], v[neg])
        wins[k] = delta > 0.
        per_shot = np.bincount(inv, weights=wins[k], minlength=len(shots)) / count
        info['views'][k] = dict(progression_wins=int(wins[k].sum()),
                                ties=int((delta == 0.).sum()),
                                shot_mean_progression_win_rate=float(per_shot.mean()),
                                nearer_vs_safe_wins=int((v[pos, 1] > v[neg, 1]).sum()),
                                median_progression_margin=float(np.median(delta)))
    a, b = wins['latest'], wins['previous']
    change = np.bincount(inv, weights=a.astype(int) - b, minlength=len(shots))
    info['latest_vs_previous'] = dict(pair_win_flips=[int((a & ~b).sum()), int((b & ~a).sum())],
                                     shot_win_flips=[int((change > 0).sum()), int((change < 0).sum())])
    return info


def run(root):
    p, seal, held, fit, audit = (read(root / k) for k in (
        'protocol.json', 'seal.json', 'source_validation.json', 'fit.json', 'independent_audit.json'))
    locked = load_inputs()
    ref = Path(p['reference'])
    if (p['scope'] != 'EAST360_FIXED_HEAD_EARLY_ORDINAL_TEACHING_DEVELOPMENT'
            or p['order'] != locked['ids'] or len(seal['curve_sha256']) != 360
            or audit['status'] != 'INDEPENDENT_HDF5_RECOUNT'
            or p['data_bindings'] != locked['bound']
            or any(sha(root / k) != v for k, v in seal['fit_assets'].items())
            or any(sha(ref / k) != v for k, v in p['reference_assets'].items())
            or p['source_label_sha256'] != sha(io23.JTEXT_ADVANCED_TIME_PATH)):
        raise ValueError('independently recounted and sealed fixed EAST360 required')
    for f in (ordinal, pairs, surplus, surplus.feedback):
        path = Path(f.__file__)
        if p['code_sha256'][path.name] != sha(path):
            raise ValueError(f'sealed source changed: {path.name}')
    rp = read(ref / 'protocol.json')
    parent = Path(rp['reference'])
    if sha(parent / 'protocol.json') != rp['reference_assets']['protocol.json']:
        raise ValueError('sealed readout protocol changed')
    source = Path(read(parent / 'protocol.json')['source_prior'])
    if any(sha(source / k) != v for k, v in p['source_assets'].items()):
        raise ValueError('sealed source checkpoint changed')
    source_result = read(source / 'result.json')
    if weak.code_bindings() != read(source / 'protocol.json')['code_sha256']:
        raise ValueError('source shape implementation changed')
    net = LagState(source_result['inputs'], source_result['edges']).eval().requires_grad_(False)
    with np.load(source / 'model.npz', allow_pickle=False) as f:
        net.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy() for v in (base.center, base.scale))
    with np.load(root / 'fit.npz', allow_pickle=False) as f:
        noise = f['noise_scale']
    match_scale = np.asarray(source_result['match_scale'])
    roles, advanced = io23.source_roles(), io23.advanced_times()
    tr = ordinal.collect(net, roles['train'], advanced, center, scale, noise, 'train')
    m = pairs.match(tr['context'], tr['disruptive'], tr['tail'], match_scale)
    train_d = np.unique(tr['shot'][m[0]]).tolist()
    if len(m[0]) != fit['pairs'] or train_d != fit['disruptive_shots']:
        raise ValueError('source training matched sample mismatch')
    cue = ordinal.cue_probe_fit(tr, m)
    del tr
    val = ordinal.collect(net, roles['val'], advanced, center, scale, noise, 'val')
    m = pairs.match(val['context'], val['disruptive'], val['tail'], match_scale)
    reproduces = {
        k: {j: int((ordinal.cue_direction(val, m, j) @ c > 0.).sum()) for j in ('intra', 'lag')}
        for k, c in cue.items()}
    if len(m[0]) != held['summary']['all']['n'] or reproduces != held['cue_only_progression_probe']:
        raise ValueError('source validation cue probe differs from its sealed counts')
    del val

    suffix, groups, rows = p['order'][p['startup']:], np.asarray(surplus.feedback.GROUPS), []
    baseline = read(root / 'per_shot.json')['0.2']['fixed']
    missed = []
    for i, sid in enumerate(suffix, p['startup'] + 1):
        path = root / 'shots' / f'{i:04d}_{sid}.npz'
        if (sha(path) != seal['curve_sha256'][path.name]
                or sha(io23.path_for('EAST', sid)) != locked['hashes'][str(sid)]):
            raise ValueError(f'prediction or raw data changed for {sid}')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        t = raw['times']
        d, down = io23.read_outcome('EAST', sid)
        b = baseline[i - 1]
        if (b['shot'], b['disruptive'], b['down_s']) != (sid, d, down):
            raise ValueError(f'recounted outcome mismatch for {sid}')
        if d and not b['timely']:
            missed.append(sid)
        _, valid = io23.labels_at(t, d, down, 125.)
        # A zero exclusion permits analysis inside the target's 125 ms window.
        # The ground truth only selects post-seal diagnostic endpoints.
        ix = pairs.endpoints(t, valid, d, down, 0., first_effect=surplus.FIRST_EFFECT)
        if len(ix):
            signal = prepare_actual(raw, i, locked['mapping'])
            track = surplus.trace(net, signal, center, scale, ix.ravel())
            mask = eligible_endpoints(ix, track, groups)
            if mask.any():
                ds, _ = surplus.state(track, noise)
                context = track['context'].reshape(-1, 2, 7)[mask]
                ix = ix[mask]
                near = ix[:, 1]
                with np.load(path, allow_pickle=False) as f:
                    np.testing.assert_array_equal(t, f['times'])
                    curve = {k: f[k][ix].astype(np.float64) for k in ('fixed', 'ordered', 'ordered_lag')}
                prob = {k: np.clip(v, 1e-7, 1. - 1e-7) for k, v in curve.items()}
                rows.append(dict(context=context, tail=(not d) & (t[near] >= t[-1] - weak.TAIL_S)
                                 & (context[:, 1, 2] < 0.),
                                 lead=np.round((down - t[near]) * 1000., 9) if d else np.zeros(len(near)),
                                 shot=np.full(len(ix), sid), disruptive=np.full(len(ix), d), indices=ix,
                                 **{k: np.log(v) - np.log1p(-v) for k, v in prob.items()},
                                 **{k: ds[k].reshape(-1, 2, surplus.PHI_DIM)[mask] for k in ('intra', 'lag')}))
        if i % 100 == 0 or i == len(p['order']):
            print(dict(stage='sealed_target_diagnostic', shots=i, total=len(p['order'])), flush=True)
    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    m = pairs.match(rows['context'], rows['disruptive'], rows['tail'], match_scale)
    pos, neg, tail, distance = m
    missed_mask = np.isin(rows['shot'][pos], missed)
    if len(missed) != audit['reach']['cases']:
        raise ValueError('static miss population disagrees with independent HDF5 recount')
    pred = dict(latest=rows['intra'] @ cue['ordered'], previous=rows['lag'] @ cue['ordered'],
                scrambled_latest=rows['intra'] @ cue['scrambled'],
                scrambled_previous=rows['lag'] @ cue['scrambled'],
                frozen_risk=rows['fixed'], latest_risk=rows['ordered'], previous_risk=rows['ordered_lag'])
    groups = dict(all=np.ones(len(pos), bool), static_missed=missed_mask,
                  static_timely=~missed_mask, falling_tail=tail,
                  missed_vs_falling_tail=missed_mask & tail, lead_over_60_ms=rows['lead'][pos] > 60.)
    summary = {k: compare(rows, m, v, pred) for k, v in groups.items()}
    result = dict(scope='POST_SEAL_CUE_TRANSFER_DIAGNOSTIC_NOT_AN_ONLINE_ARM',
                  source_train_pairs=fit['pairs'], source_train_disruptions=len(train_d),
                  source_validation_pairs=held['summary']['all']['n'], source_validation=reproduces,
                  target_suffix_size=len(suffix), static_missed_population=len(missed),
                  static_missed_not_matched=sorted(set(missed) - set(rows['shot'][pos].tolist())),
                  source_match_rms_median=held['median_rms_distance'],
                  target_match_rms_median=float(np.median(distance)),
                  risks='sealed scores in log-odds, not refitted',
                  interpretation='target outcomes are used only to choose and score sealed diagnostic pairs',
                  target_training_or_threshold_selection=False, safe_donors_reused=True, summary=summary,
                  bindings={k: sha(root / k) for k in ('protocol.json', 'seal.json', 'independent_audit.json',
                                                        'per_shot.json')},
                  analysis_code_sha256=sha(Path(__file__)),
                  matched_pairs=[[int(rows['shot'][a]), int(rows['shot'][b]), rows['indices'][a].tolist(),
                                  rows['indices'][b].tolist(), float(rows['lead'][a]), bool(c), float(d)]
                                 for a, b, c, d in zip(pos, neg, tail, distance, strict=True)])
    write(root / 'cue_transfer_diagnostic.json', result)
    for k in ('all', 'static_missed', 'static_timely'):
        g = summary[k]
        print(k, g['pairs'], {j: v['progression_wins'] for j, v in g['views'].items()},
              g['latest_vs_previous'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sealed', type=Path, required=True)
    run(cli.parse_args().sealed.resolve())
