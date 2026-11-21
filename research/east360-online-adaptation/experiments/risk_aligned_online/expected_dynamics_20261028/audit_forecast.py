"""Independent CPU reconstruction of Stage A and channel contributions."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..io23 import RAW23
from ..context_memory_20261028.checkpoint import read
from ..simple_online import _durable_json
from ..simple_train import sha


def run(root):
    output = root / 'independent_forecast_audit.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    p, done = read(root, 'protocol.json'), read(root, 'complete.json')
    checkpoint = root / 'forecast_epoch10.pt'
    if not done['complete'] or sha(checkpoint) != done['checkpoint_sha256']:
        raise ValueError('completed unchanged prediction checkpoint required')
    record = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if record['protocol_sha256'] != sha(root / 'protocol.json'):
        raise ValueError('checkpoint protocol differs')
    state = record['model']; cache = Path(p['cache'])
    weights_path = Path(p['reference']) / 'forecast_weights.json'
    if sha(weights_path) != p['forecast_weights_sha256']:
        raise ValueError('training scales changed')
    weights = torch.tensor(read(Path(p['reference']), 'forecast_weights.json')['weights'])
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    saved = {r['shot']: r for r in read(root, 'val_epoch10.json')['rows']}
    rows = []; max_difference = 0.
    with torch.no_grad():
        for sid in p['val_ids']:
            path = cache / 'cache' / f'{sid}.pt'
            if inventory[sid]['role'] != 'val' or sha(path) != inventory[sid]['cache_sha256']:
                raise ValueError('source validation cache differs')
            e = torch.load(path, map_location='cpu', weights_only=True)
            a = e['complete']; x = torch.where(a, e['means'], 0.)
            pair = torch.zeros_like(a); dx = torch.zeros_like(x)
            pair[1:] = a[1:] & a[:-1]
            dx[1:] = torch.where(pair[1:], x[1:] - x[:-1], 0.)
            features = torch.cat((x, a.to(x), dx, pair.to(x)), -1)[:-4]
            hidden = F.silu(F.linear(features, state['net.0.weight'], state['net.0.bias']))
            predicted = F.linear(hidden, state['net.2.weight'], state['net.2.bias']).reshape(-1, 4, 23)
            target = torch.stack([x[h:len(x)-4+h] - x[:-4] for h in range(1, 5)], 1)
            valid = torch.stack([a[h:len(x)-4+h] & a[:-4] for h in range(1, 5)], 1)
            w = valid * weights
            # Contributions use each shot's original denominator, so their sum
            # reproduces its metric. They are not unweighted per-channel MSEs.
            predicted_error = ((predicted-target).square()*w).sum((0, 1))/w.sum()
            held_error = (target.square()*w).sum((0, 1))/w.sum()
            prediction, persistence = float(predicted_error.sum()), float(held_error.sum())
            max_difference = max(max_difference, abs(prediction-saved[sid]['prediction']))
            if (not np.isclose(prediction, saved[sid]['prediction'], atol=1e-7, rtol=1e-6)
                    or not np.isclose(persistence, saved[sid]['persistence'], atol=1e-7, rtol=1e-6)):
                raise ValueError('independent CPU reconstruction differs')
            rows.append(dict(shot=sid, disruptive=bool(e['flag']), prediction=prediction, persistence=persistence,
                             channel_prediction=predicted_error.tolist(), channel_persistence=held_error.tolist()))
    groups = {}
    for name in ('all', 'D', 'S'):
        subset = [r for r in rows if name == 'all' or r['disruptive'] == (name == 'D')]
        predicted = np.array([r['channel_prediction'] for r in subset])
        held = np.array([r['channel_persistence'] for r in subset])
        benefit = held.sum(1)-predicted.sum(1)
        rng = np.random.default_rng(131)
        boot = benefit[rng.integers(len(subset), size=(10000, len(subset)))].mean(1)
        groups[name] = dict(shots=len(subset), prediction=float(predicted.sum(1).mean()),
            persistence=float(held.sum(1).mean()), improved=int((benefit > 0).sum()),
            relative_reduction=float(benefit.mean()/held.sum(1).mean()),
            paired_mean_benefit=float(benefit.mean()),
            descriptive_iid_shot_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
            channels=[dict(channel=channel, prediction=float(predicted[:, j].mean()),
                           persistence=float(held[:, j].mean()), benefit=float((held[:, j]-predicted[:, j]).mean()),
                           improved=int((held[:, j] > predicted[:, j]).sum())) for j, channel in enumerate(RAW23)])
    _durable_json(output, dict(scope='SOURCE_VAL_COMPONENT_ONLY_SINGLE_SEED_DEVELOPMENT',
        checkpoint_sha256=sha(checkpoint), code_sha256=sha(Path(__file__)), max_reconstruction_difference=max_difference,
        interval_caveat='Descriptive shot resampling conditional on one model; excludes training-seed variability, selection and temporal dependence.',
        method='CPU functional linear/SiLU reconstruction; no DirectForecast/direct_examples/evaluation helper reused',
        groups=groups, rows=rows))
    for name, g in groups.items():
        print(name, {k: v for k, v in g.items() if k != 'channels'})
        print('channel benefits', [(r['channel'], round(r['benefit'], 7)) for r in g['channels']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
