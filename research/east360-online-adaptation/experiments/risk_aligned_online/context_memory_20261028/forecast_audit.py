"""Read-only source VAL forecast check against last-observation persistence.

This recomputes the baseline from cached signals and compares it with the
recorded epoch forecasts on exactly the same masks and four future horizons.
It does not rerun training, select an epoch, or evaluate EAST alarms.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import objective_weight, read, SCALED_PROFILE


def persistence_error(means, complete, *, weights=None):
    if (means.ndim != 2 or means.shape[1] != 23 or len(means) <= 4
            or complete.shape != means.shape or complete.dtype != torch.bool):
        raise ValueError('nonempty full four-horizon forecast origins required')
    if not torch.isfinite(means[complete]).all():
        raise ValueError('finite observed values required')
    clean = torch.where(complete, means, 0.)
    n = len(means)
    target = torch.stack([clean[h:n-4+h] - clean[:-4] for h in range(1, 5)], 1)
    mask = torch.stack([complete[h:n-4+h] & complete[:-4] for h in range(1, 5)], 1)
    if not mask.any():
        raise ValueError('at least one observed target required')
    if weights is not None:
        from .scaled import validate_weights
        validate_weights(weights)
        w = mask*weights.to(means)
        return float((target.square()*w).sum()/w.sum())
    return float((target.square() * mask).sum() / mask.sum())


def summarize(rows):
    if not rows:
        raise ValueError('nonempty source subset required')
    fields = ('persistence', 'epoch0', 'epoch1', 'epoch2')
    ratios = [r['epoch2']/r['persistence'] for r in rows if r['persistence'] > 0]
    return dict(shots=len(rows), statistics={key: dict(
        mean=float(np.mean([r[key] for r in rows])),
        median=float(np.median([r[key] for r in rows]))) for key in fields},
        epoch2_better_than_persistence=sum(r['epoch2'] < r['persistence'] for r in rows),
        epoch2_better_than_epoch0=sum(r['epoch2'] < r['epoch0'] for r in rows),
        ratio_defined_shots=len(ratios),
        epoch2_over_persistence_median=float(np.median(ratios)) if ratios else None)


def audit(root):
    output = root / 'forecast_persistence_audit.json'
    if output.exists():
        raise FileExistsError(output)
    p, done, checked = (read(root, n) for n in ('protocol.json', 'complete.json', 'audit.json'))
    objective_weight(p)
    if (not checked['complete'] or not done['complete']
            or done['target_or_test_signals_read'] != 0 or not done['source_unchanged']
            or sha(root/'complete.json') != checked['complete_sha256']
            or sha(root/'context8_epoch2.pt') != done['checkpoint_sha256']
            or p['val_ids'] != simple_data.source_roles()['val']):
        raise ValueError('audited complete source context8 with canonical VAL160 required')
    cache = Path(p['cache_root'])
    if sha(cache/'inventory.json') != p['cache_inventory_sha256']:
        raise ValueError('source cache inventory changed')
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    weights = None
    if p['profile'] == SCALED_PROFILE:
        if sha(root/'forecast_weights.json') != p['forecast_weights_sha256']:
            raise ValueError('sealed TRAIN forecast scale changed')
        weights = torch.tensor(read(root,'forecast_weights.json')['weights'])
    evaluations = {j: read(root, f'val_epoch{j}.json')['per_shot'] for j in (0, 1, 2)}
    for v in evaluations.values():
        if [r['shot'] for r in v] != p['val_ids']:
            raise ValueError('identical ordered VAL160 evaluations required')
    rows = []
    for i, sid in enumerate(p['val_ids']):
        path = cache/'cache'/f'{sid}.pt'
        if sha(path) != inventory[sid]['cache_sha256'] or inventory[sid]['role'] != 'val':
            raise ValueError('source validation cache changed')
        e = torch.load(path, map_location='cpu', weights_only=True)
        losses = {f'epoch{j}': evaluations[j][i]['forecast_loss'] for j in (0, 1, 2)}
        if (any(evaluations[j][i]['flag'] != e['flag'] for j in (0, 1, 2))
                or any(not np.isfinite(v) or v < 0 for v in losses.values())):
            raise ValueError('consistent source labels and finite forecast losses required')
        rows.append(dict(shot=sid, disruptive=e['flag'],
                         persistence=persistence_error(e['means'], e['complete'], weights=weights), **losses))
        if weights is not None:
            rows[-1].update(unweighted_persistence=persistence_error(e['means'],e['complete']),
                            unweighted_epoch2=evaluations[2][i]['unweighted_forecast_loss'])
    result = dict(scope='SOURCE_VAL160_FORECAST_DIAGNOSTIC_NOT_EAST_PERFORMANCE',
        model_updates=0, east_records_read=0, test220_records_read=0,
        baseline='hold last observed normalized diagnostic patch, zero future increment',
        metric='mean squared normalized increment error; identical observed masks; 4 direct horizons',
        aggregation='each shot equally weighted; ratio excludes exact zero baseline explicitly',
        trained_errors='recorded source epoch evaluations, not newly rerun',
        input_hashes={n: sha(root/n) for n in ('protocol.json', 'complete.json', 'audit.json',
            'val_epoch0.json', 'val_epoch1.json', 'val_epoch2.json')},
        code_sha256=sha(Path(__file__)),
        summary={name: summarize(sub) for name, sub in (
            ('all', rows), ('D', [r for r in rows if r['disruptive']]),
            ('S', [r for r in rows if not r['disruptive']]))}, shots=rows)
    if weights is not None:
        result.update(metric='TRAIN-scaled squared increment error; identical weighted observed denominator',
                      forecast_weights_sha256=p['forecast_weights_sha256'])
    _durable_json(output, result)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
