"""Measure matured block order information in shot-disjoint matched source pairs."""

import argparse
from pathlib import Path

import numpy as np

from .commissioning_event_pairs_20261110 import fit_axis, match, pair_agreement
from .commissioning_source_dynamics_20261110 import candidates
from .history_gradient_metric_20261102.run import read, sha, write
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


SHUFFLES = tuple(range(131, 147))


def trajectory_shape(y, mask, order=None):
    v = np.where(mask, y, 0.)
    strength = np.abs(v).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    if order is not None:
        if mask.all():
            v = v[order]
        else:
            priority = np.argsort(order)
            for c in range(v.shape[1]):
                available = np.flatnonzero(mask[:, c])
                v[available, c] = v[available[np.argsort(priority[available])], c]
    n0, n1 = mask[:8].sum(axis=0), mask[-9:].sum(axis=0)
    early = v[:8].sum(axis=0) / np.maximum(n0, 1)
    late = v[-9:].sum(axis=0) / np.maximum(n1, 1)
    use = (n0 >= 4) & (n1 >= 4)
    denom = strength + .05
    drift = np.where(use, np.clip((late-early)/denom, -3., 3.), 0.)
    extension = np.where(use, np.clip((abs(late)-abs(early))/denom, -3., 3.), 0.)
    return np.concatenate((np.arcsinh(strength), drift, extension))


def rank_stats(pairs, fit_pairs):
    axis = fit_axis(fit_pairs)
    v = np.stack([d['feature'] - s['feature'] for d, s in pairs]) @ axis
    return dict(count=len(v), wins=int((v > 0).sum()), ties=int((v == 0).sum()),
                family_equal_agreement=pair_agreement(pairs, axis))


def run(out):
    base, _ = load_parent(DEFAULT_PARENT)
    data = candidates(*(v.detach().numpy() for v in (base.center, base.scale)))
    original = read(out / 'source_activity.json')
    full = match(data[True], data[False])
    if [[[r['shot'], r['family'], r['endpoint']] for r in p] for p in full] != original['pairs']:
        raise ValueError('source pairs no longer match the sealed experiment')
    partition = {sid: f['fold'] for f in original['held_source'] for sid in f['held_shots']}
    # The producer sorted every eligible shot by class before assigning folds;
    # a donor not selected in a held match still belongs to that same fold.
    assignment = {s: j % 4 for group in data.values() for j, s in enumerate(sorted({r['shot'] for r in group}))}
    if any(assignment[s] != fold for s, fold in partition.items()):
        raise ValueError('changed source fold split')
    folds = []
    for f in original['held_source']:
        fold = f['fold']
        train = match(*([r for r in data[d] if assignment[r['shot']] != fold] for d in (True, False)))
        held = match(*([r for r in data[d] if assignment[r['shot']] == fold] for d in (True, False)))
        if (len(held), len(train)) != (f['held_pairs'], f['train_pairs']):
            raise ValueError('changed held pair population')
        folds.append((train, held))

    def evaluate(kind):
        derived = {}

        def describe(r):
            key = r['shot'], r['endpoint']
            if key not in derived:
                order = (np.random.default_rng([kind, *key]).permutation(len(r['y']))
                         if isinstance(kind, int) else None)
                x = trajectory_shape(r['y'], r['m'], order)
                if kind == 'amplitude':
                    x[len(x)//3:] = 0.
                derived[key] = dict(r, feature=x)
            return derived[key]

        def use(pairs):
            return [(describe(d), describe(s)) for d, s in pairs]

        cells = [rank_stats(use(held), use(train)) for train, held in folds]
        n = sum(r['count'] for r in cells)
        return dict(folds=cells, total_pairs=n, pooled_agreement=sum(r['wins'] + .5*r['ties'] for r in cells)/n)

    results = dict(
        scope='EXPLORATORY_SOURCE_TRAIN_SHOT_DISJOINT_ORDER_PROBE_NO_EAST',
        source_sha256=PARENT_SHA256, source_activity_sha256=sha(out / 'source_activity.json'),
        producer_sha256=sha(Path(__file__)),
        amplitude=evaluate('amplitude'), chronological=evaluate('chronological'),
        nulls={s: evaluate(s) for s in SHUFFLES})
    write(out / 'source_order_probe.json', results)
    print({k: results[k]['pooled_agreement'] for k in ('amplitude', 'chronological')})
    print('permutation range',
          min(r['pooled_agreement'] for r in results['nulls'].values()),
          max(r['pooled_agreement'] for r in results['nulls'].values()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output.resolve())
