"""Post-hoc source-only ablation of learned internal readout directions.

Weight singular energy and input-driven hidden energy are separate quantities.
No checkpoint/parameter/threshold is changed; every arm uses production risk().
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..operational_bag_loss_20261025.learning import bag_loss
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device
from .checkpoint import read, load_context_memory
from .model import ContextAdapter


def split_first_direction(weight, delta):
    if (weight.ndim != 2 or delta.ndim != 2 or weight.shape[1] != delta.shape[1]
            or not torch.isfinite(weight).all() or not torch.isfinite(delta).all()):
        raise ValueError('compatible finite matrices required')
    _, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    if float(s.square().sum()) == 0:
        raise ValueError('zero readout has no identified leading direction')
    v = vh[0].to(delta)
    first = (delta @ v)[:, None] * v
    rest = delta - first
    hidden = delta.double() @ weight.double().T
    h1 = first.double() @ weight.double().T
    hrest = rest.double() @ weight.double().T
    total = float(hidden.square().sum())
    info = dict(weight_first_energy_fraction=float(s[0].square()/s.square().sum()),
                hidden_total_energy=total, hidden_first_energy=float(h1.square().sum()),
                hidden_rest_energy=float(hrest.square().sum()),
                hidden_first_energy_fraction=float(h1.square().sum())/total if total else None,
                orthogonality_error=float((h1*hrest).sum().abs()))
    return first, rest, info


@torch.no_grad()
def evaluate_geometry(model, e, threshold):
    t = model.memory.trace(e['means'], e['complete'], mode='memory')
    delta = endpoint_corrections(t['delta'], e['ends'])
    first, rest, geometry = split_first_direction(model.memory.output.weight, delta)
    # Respect the production last-token availability gate also in energy counts.
    _, _, active = split_first_direction(model.memory.output.weight, delta*e['valid'][:, -2, None])
    geometry.update(active)
    variants = dict(full=delta, off=torch.zeros_like(delta), first=first,
                    rest=rest, flip_first=rest-first)
    h, v = e['h'], e['valid']
    logits = {mode: torch.cat([model.risk(h[i:i+32], v[i:i+32], d[i:i+32])
                              for i in range(0, len(h), 32)]) for mode, d in variants.items()}
    modes = {}
    shift = math.log(threshold/(1-threshold))
    for mode, z in logits.items():
        local = F.binary_cross_entropy_with_logits(z-shift, e['labels'], reduction='none')
        difference = z-logits['off']
        modes[mode] = dict(bag=float(bag_loss(z.max(), e['flag'], threshold)),
            local=float(local[e['local_mask']].mean()) if e['local_mask'].any() else 0.,
            mean_logit=float(z.mean()), max_logit=float(z.max()),
            mean_logit_change=float(difference.mean()),
            positive_shift_fraction=float((difference > 1e-7).float().mean()))
    full_rms = float((logits['full']-logits['off']).double().square().mean().sqrt())
    for mode in ('first', 'rest', 'flip_first'):
        err = float((logits[mode]-logits['full']).double().square().mean().sqrt())
        geometry[f'{mode}_vs_full_logit_rms_ratio'] = err/full_rms if full_rms else None
    geometry['full_logit_change_rms'] = full_rms
    return dict(geometry=geometry, modes=modes)


def summarize(rows):
    out = {}
    for name, selected in (('all', rows), ('D', [r for r in rows if r['disruptive']]),
                           ('S', [r for r in rows if not r['disruptive']])):
        total = sum(r['geometry']['hidden_total_energy'] for r in selected)
        item = dict(shots=len(selected),
                    pooled_hidden_first_fraction=sum(r['geometry']['hidden_first_energy'] for r in selected)/total if total else None)
        for metric in ('hidden_first_energy_fraction', 'first_vs_full_logit_rms_ratio'):
            values = [r['geometry'][metric] for r in selected if r['geometry'][metric] is not None]
            item[metric] = dict(defined=len(values), median=float(np.median(values)) if values else None,
                                mean=float(np.mean(values)) if values else None)
        item['modes'] = {mode: {key: float(np.mean([r['modes'][mode][key] for r in selected]))
                              for key in selected[0]['modes'][mode]} for mode in selected[0]['modes']}
        out[name] = item
    return out


def run(root):
    output = root/'readout_geometry.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    p = read(root, 'protocol.json')
    base, _ = load_parent(DEFAULT_PARENT)
    model = ContextAdapter(base.cuda()).cuda().eval()
    receipt = load_context_memory(model, root/'context8_epoch2.pt')
    model.requires_grad_(False)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    cache, reference = Path(p['cache_root']), Path(p['reference_root'])
    if (sha(cache/'inventory.json') != p['cache_inventory_sha256']
            or sha(reference/'local_targets.pt') != p['reference_hashes']['local_targets.pt']):
        raise ValueError('source input provenance changed')
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    targets = torch.load(reference/'local_targets.pt', map_location='cpu', weights_only=True)
    controls = read(root, 'source_risk_controls.json')
    if controls['source_memory'] != receipt or [r['shot'] for r in controls['rows']] != p['val_ids']:
        raise ValueError('same audited model and complete VAL160 required')
    rows = []
    for saved in controls['rows']:
        sid = saved['shot']; path = cache/'cache'/f'{sid}.pt'
        if inventory[sid]['role'] != 'val' or sha(path) != inventory[sid]['cache_sha256']:
            raise ValueError('source validation example changed')
        e = torch.load(path, map_location='cpu', weights_only=True); e.update(targets[sid])
        result = evaluate_geometry(model, to_device(e, 'cuda'), p['threshold'])
        for mode, old in (('full', 'memory'), ('off', 'off')):
            for metric in ('bag', 'local', 'mean_logit', 'max_logit'):
                if not np.isclose(result['modes'][mode][metric], saved['modes'][old][metric], atol=1e-7, rtol=1e-6):
                    raise ValueError('readout ablation does not reproduce sealed source risk')
        rows.append(dict(shot=sid, disruptive=e['flag'], **result))
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, initial[k], atol=0, rtol=0)
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError('source classifier changed')
    result = dict(scope='POSTHOC_SOURCE_VAL160_SAMPLED_ENDPOINTS_NOT_EAST',
                  parameter_updates=0, east_records_read=0, test220_records_read=0,
                  source_memory=receipt, code_sha256=sha(Path(__file__)),
                  source_controls_sha256=sha(root/'source_risk_controls.json'),
                  definition='right singular projection of internal readout; neither input permutation nor deployed policy',
                  summary=summarize(rows), rows=rows)
    _durable_json(output, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
