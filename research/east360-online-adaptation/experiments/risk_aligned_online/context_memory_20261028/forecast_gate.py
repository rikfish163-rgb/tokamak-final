"""Fixed source-only admission check, not an EAST success criterion."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import FORECAST_PROFILE, SCALED_PROFILE, load_context_memory, read
from .model import ContextAdapter


def decide(before, after, forecast):
    values = list(before.values()) + list(after.values()) + list(forecast.values())
    if (not np.isfinite(values).all() or forecast['shots'] != 160
            or forecast['persistence_mean'] <= 0 or min(values) < 0
            or not 0 <= forecast['better_shots'] <= 160):
        raise ValueError('complete finite source validation diagnostics required')
    checks = dict(
        forecast_improvement_at_least_one_percent=forecast['memory_mean'] <= .99*forecast['persistence_mean'],
        majority_shots_improved=forecast['better_shots'] > 80,
        better_than_horizon_reverse=forecast['memory_mean'] < forecast['reverse_mean'],
        source_bag_not_worse=after['bag'] <= before['bag'],
        source_local_not_worse=after['local'] <= before['local'])
    return {'pass': all(checks.values()), 'checks': checks,
            'scope': 'SOURCE_ADMISSION_ONLY_NOT_PHYSICAL_LAW_OR_EAST_BENEFIT'}


@torch.no_grad()
def run(root):
    output = root/'source_admission.json'
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('same local CUDA environment required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    p = read(root, 'protocol.json')
    if p['profile'] not in (FORECAST_PROFILE,SCALED_PROFILE):
        raise ValueError('forecast-trained context8 profile required')
    diagnostics = read(root, 'forecast_persistence_audit.json')
    for name, digest in diagnostics['input_hashes'].items():
        if sha(root/name) != digest:
            raise ValueError('source diagnostic inputs changed')
    if [r['shot'] for r in diagnostics['shots']] != p['val_ids']:
        raise ValueError('canonical complete ordered VAL160 required')
    base, _ = load_parent(DEFAULT_PARENT)
    model = ContextAdapter(base.cuda()).cuda().eval()
    receipt = load_context_memory(model, root/'context8_epoch2.pt')
    model.requires_grad_(False)
    cache = Path(p['cache_root'])
    if sha(cache/'inventory.json') != p['cache_inventory_sha256']:
        raise ValueError('source cache inventory changed')
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    rows = []
    for recorded in diagnostics['shots']:
        sid = recorded['shot']; path = cache/'cache'/f'{sid}.pt'
        if sha(path) != inventory[sid]['cache_sha256'] or inventory[sid]['role'] != 'val':
            raise ValueError('validation cache changed')
        e = torch.load(path, map_location='cpu', weights_only=True)
        x, a = e['means'].cuda(), e['complete'].cuda()
        actual = model.memory.trace(x, a, mode='memory')
        reverse = model.memory.trace(x, a, mode='reverse', reference_norm=actual['norm'])
        real_loss = float(actual['forecast_loss'])
        if not np.isclose(real_loss, recorded['epoch2'], atol=1e-7, rtol=1e-6):
            raise ValueError('replayed forecast differs from source training evaluation')
        rows.append(dict(shot=sid, persistence=recorded['persistence'],
                         memory=real_loss, reverse=float(reverse['forecast_loss']),disruptive=e['flag']))
    summary = dict(shots=len(rows), **{f'{k}_mean': float(np.mean([r[k] for r in rows]))
                   for k in ('persistence', 'memory', 'reverse')},
                   better_shots=sum(r['memory'] < r['persistence'] for r in rows))
    before = read(root, 'val_epoch0.json')['summary']
    after = read(root, 'val_epoch2.json')['summary']
    decision = decide(before, after, summary)
    if p['profile'] == SCALED_PROFILE:
        safe = [r for r in rows if not r['disruptive']]
        if len(safe) != 80:
            raise ValueError('canonical 80 safe VAL shots required')
        summary['safe_shots'] = len(safe)
        summary['safe_better_shots'] = sum(r['memory']<r['persistence'] for r in safe)
        decision['checks']['majority_safe_shots_improved'] = summary['safe_better_shots'] > 40
        decision['pass'] = all(decision['checks'].values())
    result = dict(**decision, forecast=summary, risk_before=before, risk_after=after,
                  source_memory=receipt, rows=rows, code_sha256=sha(Path(__file__)),
                  model_updates=0, east_records_read=0, test220_records_read=0,
                  forecast_audit_sha256=sha(root/'forecast_persistence_audit.json'),
                  independent_validation=False)
    _durable_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
