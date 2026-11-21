"""Test transferable inter-diagnostic ordering without changing a risk alarm."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_coupling_20261111 as cue
from . import commissioning_ordinal_pairs_20261111 as matching
from . import io23, simple_data
from .commissioning_intra_20261110 import STARTUP
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


FIRST, STRIDE = 153, 8
STEPS = 350
MODES = ('marginal', 'undirected', 'ordered', 'mirror')


def observe(signal, grid, center, scale):
    z, m, c, cm = shot_coordinates(signal, center, scale)
    return cue.observed_features(z, m, c, cm, grid - 1)


def select(track, times, grid, disruptive, down, *, official=None):
    if disruptive:
        near = cue.outcome_endpoints(times, grid, track['eligible'], down)
        row = np.searchsorted(grid, near)
    else:
        row = np.flatnonzero(track['eligible'])
    if not len(row):
        return None
    q = grid[row]
    fields = dict(shot=np.full(len(row), track['shot']), endpoint=q,
                  disruptive=np.full(len(row), disruptive),
                  tail=(not disruptive) & (times[q] >= times[-1] - .125)
                  & (track['context'][row, -1, 2] < 0))
    if official is not None:
        fields['official_positive'] = official[q].astype(bool)
    return {**fields, **{k: v[row] for k, v in track.items() if k not in ('shot', 'eligible')}}


def stack(rows):
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def collect_source(ids, advanced, center, scale, role):
    rows = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        actual = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        grid = np.arange(FIRST, len(raw['times']), STRIDE)
        if len(grid):
            track = dict(observe(raw, grid, center, scale), shot=sid)
            row = select(track, raw['times'], grid, actual['is_disrupt'], actual['down'],
                         official=actual['y'])
            if row is not None:
                rows.append(row)
        if i % 450 == 0 or i == len(ids):
            print(dict(stage=f'collect_{role}', shots=i, total=len(ids)), flush=True)
    return stack(rows)


def views(rows, context_scale):
    signed = rows['odd']
    context = np.tanh(rows['context'][:, -1, :3] / context_scale[-1, :3])
    # The same direction may matter differently at a different Ip, Bt or Ip trend.
    interaction = (signed[:, :, None] * context[:, None, :]).reshape(len(signed), 3*signed.shape[1])
    directed = np.column_stack((signed, interaction))
    marginal = rows['marginal']
    even = np.column_stack((marginal, rows['even']))
    return dict(marginal=marginal, undirected=even,
                ordered=np.column_stack((even, directed)),
                mirror=np.column_stack((even, -directed)))


def matches(rows, scale):
    return matching.match(rows['context'], rows['disruptive'], rows['tail'], scale)


def fit(rows, paired, context_scale):
    pos, neg = paired[:2]
    _, inv, counts = np.unique(rows['shot'][pos], return_inverse=True, return_counts=True)
    shot_weights = torch.as_tensor(1. / counts[inv], dtype=torch.float32, device='cuda')
    shot_weights /= shot_weights.sum()
    fitted = {}
    for k, x in views(rows, context_scale).items():
        if k == 'mirror':
            continue
        scale = np.maximum(x.std(0), .025)
        diff = torch.as_tensor((x[pos] - x[neg]) / scale, dtype=torch.float32, device='cuda')
        beta = torch.nn.Parameter(torch.zeros(diff.shape[1], device='cuda'))
        opt = torch.optim.Adam([beta], lr=.025)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            loss = (F.softplus(-diff.mv(beta)) * shot_weights).sum() + .05 * beta.square().sum()
            loss.backward()
            opt.step()
        fitted[k] = beta.detach().cpu().numpy() / scale
    return fitted


def score(rows, fitted, context_scale):
    return np.column_stack([x @ fitted['ordered' if k == 'mirror' else k]
                            for k, x in views(rows, context_scale).items()])


def summary(rows, paired, predictions, *, strata=None):
    pos, neg, tail, dist = paired
    diff = predictions[pos] - predictions[neg]
    cols = (*MODES, 'archived_static') if diff.shape[1] == len(MODES) + 1 else MODES
    groups = dict(all=np.ones(len(pos), bool), falling_safe=tail, other_safe=~tail)
    groups.update({k: mask[pos] for k, mask in (strata or {}).items()})
    if 'official_positive' in rows:
        groups['inside_source_positive'] = rows['official_positive'][pos]
        groups['outside_source_positive'] = ~groups['inside_source_positive']
    result = {}
    for name, keep in groups.items():
        if not keep.any():
            result[name] = dict(pairs=0, disruptive_shots=0)
            continue
        val = diff[keep]
        wins = (val > 0) + .5 * (val == 0)
        ids, inv, n = np.unique(rows['shot'][pos[keep]], return_inverse=True, return_counts=True)
        weight = 1. / n[inv] / len(ids)
        distinct = (val[:, 2] > 0) != (val[:, 3] > 0)
        result[name] = dict(
            pairs=len(val), disruptive_shots=len(ids), safe_shots=len(np.unique(rows['shot'][neg[keep]])),
            median_match_distance=float(np.median(dist[keep])),
            wins=dict(zip(cols, (val > 0).sum(0).tolist(), strict=True)),
            ties=dict(zip(cols, (val == 0).sum(0).tolist(), strict=True)),
            shot_balanced_concordance=dict(zip(cols, (weight @ wins).tolist(), strict=True)),
            order_disagreements=dict(
                true_only=int(((val[:, 2] > 0) & distinct).sum()),
                mirror_only=int(((val[:, 3] > 0) & distinct).sum())))
    return result


def run(reference, out):
    configure()
    start = time.monotonic()
    locked = load_inputs()
    p, seal = (read(reference / k) for k in ('protocol.json', 'seal.json'))
    ids = locked['ids']
    if (p['scope'] != 'EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or p['order'] != ids or p['startup'] != STARTUP or p['source_sha256'] != PARENT_SHA256
            or len(seal['curve_sha256']) != len(ids)
            or any(sha(reference / k) != v for k, v in seal['artifacts'].items())):
        raise ValueError('sealed fixed comparator required')
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    roles, advanced = io23.source_roles(), io23.advanced_times()
    out.mkdir(parents=True)
    code = (Path(__file__), Path(cue.__file__), Path(matching.__file__),
            Path(shot_coordinates.__code__.co_filename),
            Path(cue._window_mean.__code__.co_filename))
    write(out / 'protocol.json', dict(
        scope='CAUSAL_RELATION_FEATURES_OFFLINE_MATCHED_DIAGNOSTIC_NO_ALARM',
        source=PARENT_SHA256, fit_source_ids=roles['train'], held_source_ids=roles['val'],
        advanced_time_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        code_sha256={f.name: sha(f) for f in code}, reference=str(reference),
        reference_assets={k: sha(reference/k) for k in ('protocol.json', 'seal.json', 'per_shot.json')},
        data_bindings=locked['bound'], east_order=ids, startup=STARTUP,
        no_east_fit=True, no_alarm_changes=True, model_selection_on_suffix=False,
        grid=[FIRST, STRIDE], trailing_bins=[cue.BINS+1, cue.BIN], families=cue.FAMILIES,
        feature_boundary='all inputs through q-1; readout sealed before accessing EAST outcomes',
        pairs='one nearest falling-tail and one nearest other safe endpoint per selected D endpoint',
        positive_endpoint_scope='diagnostic matched bins in (30,125] ms, not J-TEXT hard relabeling',
        controls=MODES, steps=STEPS))
    train = collect_source(roles['train'], advanced, center, scale, 'train')
    match_scale = np.maximum(train['context'].std(0), .025)
    train_pairs = matches(train, match_scale)
    fitted = fit(train, train_pairs, match_scale)
    with (out / 'weights.npz').open('xb') as f:
        np.savez(f, **fitted, match_scale=match_scale)
    val = collect_source(roles['val'], advanced, center, scale, 'val')
    source_summary = dict(train=summary(train, train_pairs, score(train, fitted, match_scale)),
                          val=summary(val, matches(val, match_scale), score(val, fitted, match_scale)))
    write(out / 'source.json', source_summary)
    print(dict(stage='held_source', **source_summary['val']['all']), flush=True)

    # No current or future EAST outcome is consulted while constructing predictors.
    signals, receipt = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(ids, 1):
            path = io23.path_for('EAST', sid)
            if sha(path) != locked['hashes'][str(sid)]:
                raise ValueError('EAST signal changed')
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            actual = prepare_actual(raw, i, locked['mapping'])
            name = f'{i:04d}_{sid}.npz'
            path = reference / 'shots' / name
            if sha(path) != seal['curve_sha256'][name]:
                raise ValueError('reference curve changed')
            with np.load(path, allow_pickle=False) as old:
                np.testing.assert_array_equal(raw['times'], old['times'])
                t, fixed = old['times'], old['fixed']
            grid = np.arange(FIRST, len(t), STRIDE)
            r = observe(actual, grid, center, scale)
            r['prediction'] = np.column_stack((score(r, fitted, match_scale), fixed[grid]))
            signals[sid] = (t, grid, dict(r, shot=sid))
            receipt[str(sid)] = dict(rows=len(grid), eligible=int(r['eligible'].sum()))
            if i in (21, 120, 360):
                sample = grid[:8]
                small = {k: actual[k][:sample[-1]] for k in ('values', 'available')}
                truncated = observe(small, sample, center, scale)
                for k, v in truncated.items():
                    np.testing.assert_array_equal(v, r[k][:len(sample)])
            if i % 120 == 0:
                print(dict(stage='seal_feature_rows', shots=i), flush=True)
    # Store the light, outcome-free feature scores once, without duplicating HDF5.
    with (out / 'sealed_scores.npz').open('xb') as f:
        np.savez_compressed(f, **stack([dict(
            shot=np.full(len(g), sid), endpoint=g, times=t[g], context=r['context'],
            eligible=r['eligible'], scores=r['prediction']) for sid, (t, g, r) in signals.items()]))
    write(out / 'seal.json', dict(rows=receipt, artifacts={
        k: sha(out/k) for k in ('weights.npz', 'source.json', 'sealed_scores.npz')},
        prefix_witness_ranks=[21, 120, 360]))
    archived = {r['shot']: r for r in read(reference / 'per_shot.json')['0.2']['fixed']}
    cold, suffix = [], []
    for i, sid in enumerate(ids, 1):
        t, g, r = signals[sid]
        disruptive, down = io23.read_outcome('EAST', sid)
        a = archived[sid]
        if (a['disruptive'], a['down_s']) != (disruptive, down):
            raise ValueError('archived HDF5 metadata disagreement')
        picked = select(r, t, g, disruptive, down)
        if picked is not None:
            (cold if i <= STARTUP else suffix).append(picked)
    results = {}
    for name, data in (('cold20', cold), ('suffix340', suffix)):
        rows = stack(data)
        paired = matches(rows, match_scale)
        missed = np.array([not archived[int(sid)]['timely'] for sid in rows['shot']])
        results[name] = summary(rows, paired, rows['prediction'], strata={'static_not_timely': missed})
        if name == 'suffix340':
            pos, neg = paired[:2]
            with (out / 'matched_suffix.npz').open('xb') as f:
                np.savez_compressed(f, positive_shot=rows['shot'][pos], safe_shot=rows['shot'][neg],
                                    positive_index=rows['endpoint'][pos], safe_index=rows['endpoint'][neg],
                                    margins=rows['prediction'][pos] - rows['prediction'][neg],
                                    safe_falling=paired[2], match_distance=paired[3],
                                    static_not_timely=missed[pos])
    write(out / 'result.json', dict(status='SEALED_EAST360_DEVELOPMENT_RELATION_PROBE',
                                   source=source_summary, east=results,
                                   alarm_predictions_changed=0, seconds=time.monotonic()-start))
    print(results, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    a = cli.parse_args()
    run(a.reference.resolve(), a.output.resolve())
