"""Read-only, all-channel decomposition of the fixed source forecast trial.

Every channel is retained. Missing observations contribute zero to sums but
have zero count, never an invented zero MSE. This is a post-hoc diagnosis;
it cannot change the previously sealed admission rule or select an epoch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..io23 import RAW23
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import load_context_memory, read
from .model import ContextAdapter


def channel_errors(means, complete, forecast):
    if (means.ndim != 2 or means.shape[1] != 23 or len(means) <= 4
            or complete.shape != means.shape or complete.dtype != torch.bool
            or forecast.shape != (len(means), 4, 23)):
        raise ValueError('complete four-horizon origins and 23 channels required')
    if not torch.isfinite(means[complete]).all():
        raise ValueError('finite observed values required')
    clean = torch.where(complete, means, 0.).double()
    n = len(means)
    target = torch.stack([clean[h:n-4+h] - clean[:-4] for h in range(1, 5)], 1)
    valid = torch.stack([complete[h:n-4+h] & complete[:-4] for h in range(1, 5)], 1)
    if not valid.any() or not torch.isfinite(forecast[:-4][valid]).all():
        raise ValueError('finite predictions and at least one observed target required')
    sse = torch.where(valid, (forecast[:-4].double()-target).square(), 0.).sum((0, 1))
    baseline = torch.where(valid, target.square(), 0.).sum((0, 1))
    count = valid.sum((0, 1))
    return dict(sse=sse, persistence_sse=baseline, count=count,
                mse=float(sse.sum()/count.sum()), persistence_mse=float(baseline.sum()/count.sum()))


def summarize(rows):
    count = np.array([r['count'] for r in rows])
    b = np.array([r['persistence_sse'] for r in rows])
    m = np.array([r['sse'] for r in rows])
    delta = np.array([r['persistence_mse']-r['mse'] for r in rows])
    channels = []
    for k, name in enumerate(RAW23):
        observed = count[:, k] > 0
        # Contributions have the original all-observation denominator per shot;
        # their sum exactly recovers the equal-shot aggregate improvement.
        contribution = (b[:, k]-m[:, k])/count.sum(1)
        channels.append(dict(channel=name, observed_shots=int(observed.sum()),
            target_scalars=int(count[:, k].sum()),
            persistence_mse=float(np.mean(b[observed, k]/count[observed, k])) if observed.any() else None,
            memory_mse=float(np.mean(m[observed, k]/count[observed, k])) if observed.any() else None,
            improved_shots=int((m[observed, k] < b[observed, k]).sum()),
            mean_contribution_to_overall_improvement=float(contribution.mean())))
    total = float(delta.sum())
    return dict(shots=len(rows), better_shots=int((delta > 0).sum()),
                mean_improvement=float(delta.mean()), median_improvement=float(np.median(delta)),
                top5_share_of_net_improvement=float(np.sort(delta)[-5:].sum()/total) if total > 0 else None,
                channels=channels)


@torch.no_grad()
def run(root):
    output = root/'forecast_channel_diagnosis.json'
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('same local CUDA environment required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    p = read(root, 'protocol.json')
    admission = read(root, 'source_admission.json')
    if sha(root/'forecast_persistence_audit.json') != admission['forecast_audit_sha256']:
        raise ValueError('source forecast audit changed')
    cache = Path(p['cache_root'])
    if sha(cache/'inventory.json') != p['cache_inventory_sha256']:
        raise ValueError('source cache inventory changed')
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    if [r['shot'] for r in admission['rows']] != p['val_ids']:
        raise ValueError('complete ordered source VAL160 required')
    base, _ = load_parent(DEFAULT_PARENT)
    model = ContextAdapter(base.cuda()).cuda().eval()
    receipt = load_context_memory(model, root/'context8_epoch2.pt')
    model.requires_grad_(False)
    rows = []
    for recorded in admission['rows']:
        sid = recorded['shot']; path = cache/'cache'/f'{sid}.pt'
        if inventory[sid]['role'] != 'val' or sha(path) != inventory[sid]['cache_sha256']:
            raise ValueError('validation cache changed')
        e = torch.load(path, map_location='cpu', weights_only=True)
        x, a = e['means'].cuda(), e['complete'].cuda()
        trace = model.memory.trace(x, a, mode='memory')
        errors = channel_errors(x, a, trace['forecast'])
        for new, old in [('mse', 'memory'), ('persistence_mse', 'persistence')]:
            if not np.isclose(errors[new], recorded[old], atol=1e-7, rtol=1e-6):
                raise ValueError('channel decomposition disagrees with archived aggregate')
        rows.append(dict(shot=sid, disruptive=e['flag'], **{
            k: v.cpu().tolist() if torch.is_tensor(v) else v for k, v in errors.items()}))
    summary = {name: summarize(sub) for name, sub in (
        ('all', rows), ('D', [r for r in rows if r['disruptive']]),
        ('S', [r for r in rows if not r['disruptive']]))}
    result = dict(scope='POSTHOC_SOURCE_CHANNEL_DIAGNOSIS_NOT_EAST_OR_PHYSICS_PROOF',
        normalization='existing cached source asinh-normalized patch means; no new scaling',
        model_updates=0, east_records_read=0, test220_records_read=0, source_memory=receipt,
        source_admission_sha256=sha(root/'source_admission.json'),
        code_sha256=sha(Path(__file__)), summary=summary, rows=rows)
    _durable_json(output, result)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
