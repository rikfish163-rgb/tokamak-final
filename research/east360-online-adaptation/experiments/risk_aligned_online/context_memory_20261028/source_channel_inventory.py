"""Read-only source TRAIN channel-loss inventory; no model selection or fitting.

Short records receive zero forecast loss in the actual trainer. They remain in
the cohort denominator but are explicitly ineligible for prediction-error
statistics. Zero loss is not evidence of a correct forecast on those records.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..io23 import RAW23
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import read
from .forecast_channels import channel_errors


def record(means, complete):
    if (means.ndim != 2 or means.shape[1] != 23 or complete.shape != means.shape
            or complete.dtype != torch.bool or not torch.isfinite(means[complete]).all()):
        raise ValueError('finite observed 23-channel patches and matching mask required')
    if len(means) <= 4:
        status = 'too_short_for_complete_four_horizon_loss'
    elif not any(bool((complete[h:len(means)-4+h] & complete[:-4]).any()) for h in range(1, 5)):
        status = 'no_observed_four_horizon_pairs'
    else:
        e = channel_errors(means, complete, means.new_zeros(len(means), 4, 23))
        return dict(status='eligible', mse=e['mse'], training_loss_value=e['mse'],
                    persistence_sse=e['persistence_sse'].tolist(), count=e['count'].tolist())
    return dict(status=status, mse=None, training_loss_value=0.,
                persistence_sse=[0.] * 23, count=[0] * 23)


def run(root):
    output = root/'train_channel_inventory.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    p = read(root, 'protocol.json')
    cache = Path(p['cache_root'])
    if sha(cache/'inventory.json') != p['cache_inventory_sha256']:
        raise ValueError('source cache inventory changed')
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    if len(p['train_ids']) != 1353 or len(set(p['train_ids'])) != 1353:
        raise ValueError('full canonical TRAIN1353 required')
    rows = []
    for sid in p['train_ids']:
        path = cache/'cache'/f'{sid}.pt'
        if inventory[sid]['role'] != 'train' or sha(path) != inventory[sid]['cache_sha256']:
            raise ValueError('source training cache changed')
        e = torch.load(path, map_location='cpu', weights_only=True)
        rows.append(dict(shot=sid, patches=len(e['means']), disruptive=e['flag'],
                         **record(e['means'], e['complete'])))
    counts = np.asarray([r['count'] for r in rows])
    sse = np.asarray([r['persistence_sse'] for r in rows])
    # Per-record zero denominators are the explicitly recorded ineligible cases.
    contribution = sse / counts.sum(1).clip(min=1)[:, None]
    mean = float(np.mean([r['training_loss_value'] for r in rows]))
    channels = [dict(channel=name, observed_shots=int((counts[:, k] > 0).sum()),
                     mean_loss_contribution=float(contribution[:, k].mean()),
                     fraction_of_total=float(contribution[:, k].mean()/mean) if mean > 0 else None)
                for k, name in enumerate(RAW23)]
    if not np.isclose(sum(c['mean_loss_contribution'] for c in channels), mean, atol=1e-12, rtol=1e-12):
        raise ValueError('per-channel contributions do not recover equal-shot loss')
    summary = dict(shots=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                   mean_loss_including_explicit_zero_loss_records=mean, channels=channels)
    _durable_json(output, dict(scope='POSTHOC_TRAIN_PERSISTENCE_CHANNEL_INVENTORY',
        normalization='unchanged cached source asinh-normalized patch means',
        model_updates=0, east_records_read=0, test220_records_read=0,
        protocol_sha256=sha(root/'protocol.json'), code_sha256=sha(Path(__file__)),
        summary=summary, rows=rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
