"""Post-hoc source diagnosis: mapping capacity versus causal context fitting.

The future-informed oracle is a lower bound on *diagnostic forecast error* for
each origin's fixed basis and unit-ball state. It is never a risk predictor or
an online arm. Causal local least squares uses only already arrived targets;
this alternative was not used for training and is not qualified for EAST.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..operational_bag_loss_20261025.learning import bag_loss
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device
from .model import TrajectoryAdapter


def ball_least_squares(basis, target, weights):
    if (basis.ndim != 3 or basis.shape[-1] != 8 or target.shape != basis.shape[:2]
            or weights.shape != target.shape or not torch.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError('batched observed eight-variable least squares required')
    observed = weights > 0
    a = torch.where(observed[..., None], basis, 0.).double()
    y = torch.where(observed, target, 0.).double()
    w = weights.double()
    if not torch.isfinite(a).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite observed least-squares system')
    normal = a.transpose(1, 2) @ (a * w[..., None])
    rhs = (a * (w * y)[..., None]).sum(1)
    eigen, rotation = torch.linalg.eigh(normal)
    eigen = eigen.clamp_min(0.)
    coordinate = (rotation.transpose(1, 2) @ rhs[..., None]).squeeze(-1)
    tolerance = (eigen.amax(-1, keepdim=True) * 1e-10).clamp_min(1e-14)
    unconstrained = torch.where(eigen > tolerance, coordinate / eigen.clamp_min(1e-300), 0.)
    constrained = unconstrained.norm(dim=-1) > 1
    lo = torch.zeros_like(constrained, dtype=torch.float64)
    hi = rhs.norm(dim=-1).clamp_min(1e-14)
    for _ in range(60):
        midpoint = (lo + hi) * .5
        trial = coordinate / (eigen + midpoint[:, None]).clamp_min(1e-300)
        too_large = trial.norm(dim=-1) > 1
        lo = torch.where(too_large, midpoint, lo)
        hi = torch.where(too_large, hi, midpoint)
    boundary = coordinate / (eigen + hi[:, None]).clamp_min(1e-300)
    chosen = torch.where(constrained[:, None], boundary, unconstrained)
    answer = (rotation @ chosen[..., None]).squeeze(-1)
    return answer.to(basis)


def causal_local_context(x, valid, basis, weights):
    if (x.ndim != 2 or x.shape[-1] != 23 or valid.shape != x.shape
            or valid.dtype != torch.bool or basis.shape != (len(x), 4, 23, 8)
            or weights.shape != (4, 23)):
        raise ValueError('ordered observed diagnostic sequence and fixed issued bases required')
    n = len(x)
    a = basis.new_zeros(n, 92, 8)
    target = x.new_zeros(n, 92); w = x.new_zeros(n, 92)
    clean = torch.where(valid, x, 0.)
    for h in range(1, 5):
        if n <= h:
            continue
        block = slice((h-1)*23, h*23)
        a[h:, block] = basis[:-h, h-1]
        target[h:, block] = clean[h:] - clean[:-h]
        w[h:, block] = (valid[h:] & valid[:-h]) * weights[h-1]
    return ball_least_squares(a, target, w)


@torch.no_grad()
def measure(model, e, threshold):
    memory = model.memory
    x, valid = e['means'], e['complete']
    clean = torch.where(valid, x, 0.)
    keys = torch.cat((memory.key(memory.key_features(clean, valid)).tanh(), x.new_ones(len(x), 1)), -1)
    basis = torch.einsum('nk,khfc->nhfc', keys, memory.kernel)
    trace = memory.trace(x, valid, mode='memory')
    causal = causal_local_context(x, valid, basis, memory.forecast_weights)
    norm = causal.norm(dim=-1)
    matched = causal * (trace['norm'] / norm.clamp_min(torch.finfo(norm.dtype).tiny))[:, None]
    state = dict(production=trace['context'], causal_local_exact=causal, causal_local_norm=matched)
    n = len(x)
    if n <= 4:
        raise ValueError('four matured direct horizons required for this diagnosis')
    target = torch.stack([clean[h:n-4+h] - clean[:-4] for h in range(1, 5)], 1)
    w = torch.stack([valid[h:n-4+h] & valid[:-4] for h in range(1, 5)], 1) * memory.forecast_weights
    oracle = ball_least_squares(basis[:-4].flatten(1, 2), target.flatten(1), w.flatten(1))

    def forecast_error(prediction):
        return float(((prediction - target).square() * w).sum() / w.sum())

    errors = dict(persistence=forecast_error(torch.zeros_like(target)),
                  future_oracle_DO_NOT_DEPLOY=forecast_error(torch.einsum('nhfc,nc->nhf', basis[:-4], oracle)))
    risks = {}; h, v = e['h'], e['valid']
    for name, c in state.items():
        forecasts = torch.einsum('nhfc,nc->nhf', basis, c)
        errors[name] = forecast_error(forecasts[:-4])
        d = endpoint_corrections(forecasts.flatten(1), e['ends'])
        z = torch.cat([model.risk(h[i:i+32], v[i:i+32], d[i:i+32]) for i in range(0, len(h), 32)])
        local = F.binary_cross_entropy_with_logits(z-math.log(threshold/(1-threshold)), e['labels'], reduction='none')
        risks[name] = dict(bag=float(bag_loss(z.max(), e['flag'], threshold)),
                           local=float(local[e['local_mask']].mean()) if e['local_mask'].any() else 0.)
    if errors['future_oracle_DO_NOT_DEPLOY'] > min(errors.values()) + 1e-7:
        raise ValueError('unit-ball oracle failed its forecast-error bound')
    return dict(forecast=errors, risk=risks,
                production_norm=float(trace['norm'].mean()), exact_norm=float(norm.mean()),
                norm_match_zero_direction=int(((norm == 0) & (trace['norm'] > 0)).sum()))


def run(root):
    output = root / 'solver_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    protocol, complete = read(root, 'protocol.json'), read(root, 'complete.json')
    if not complete['complete'] or complete['gate']['eligible']:
        raise ValueError('post-hoc diagnosis of the completed rejected pair only')
    cache = Path(protocol['cache']); targets = Path(protocol['targets'])
    if sha(cache/'inventory.json') != protocol['cache_inventory_sha256'] or sha(targets) != protocol['targets_sha256']:
        raise ValueError('source examples changed')
    labels = torch.load(targets, map_location='cpu', weights_only=True)
    inventory = {r['shot']: r for r in read(cache, 'inventory.json')}
    weights = torch.tensor(read(Path(protocol['reference']), 'forecast_weights.json')['weights'])
    base, _ = load_parent(DEFAULT_PARENT); base.cuda()
    all_results = {}
    for kind in ('level', 'trajectory'):
        model = TrajectoryAdapter(base, weights, trajectory=kind == 'trajectory').cuda().eval()
        checkpoint = root / kind / 'memory_epoch2.pt'
        if sha(checkpoint) != complete['checkpoints'][kind]:
            raise ValueError('sealed checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.memory.load_state_dict(state['memory']); model.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        recorded = {r['shot']: r for r in read(root / kind, 'val_epoch2.json')['rows']}
        rows = []
        for sid in protocol['val_ids']:
            path = cache / 'cache' / f'{sid}.pt'
            if inventory[sid]['role'] != 'val' or sha(path) != inventory[sid]['cache_sha256']:
                raise ValueError('source VAL example differs')
            e = torch.load(path, map_location='cpu', weights_only=True); e.update(labels[sid])
            row = measure(model, to_device(e, 'cuda'), protocol['threshold'])
            for metric in ('bag', 'local'):
                if not np.isclose(row['risk']['production'][metric], recorded[sid]['modes']['memory'][metric],
                                  atol=1e-7, rtol=1e-6):
                    raise ValueError('production reconstruction failed')
            rows.append(dict(shot=sid, disruptive=e['flag'], **row))
        summary = {}
        for part in ('all', 'D', 'S'):
            selected = [r for r in rows if part == 'all' or r['disruptive'] == (part == 'D')]
            summary[part] = dict(
                forecast={key: float(np.mean([r['forecast'][key] for r in selected])) for key in rows[0]['forecast']},
                risk={key: {metric: float(np.mean([r['risk'][key][metric] for r in selected]))
                            for metric in ('bag', 'local')} for key in rows[0]['risk']},
                production_mean_norm=float(np.mean([r['production_norm'] for r in selected])),
                exact_mean_norm=float(np.mean([r['exact_norm'] for r in selected])))
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], atol=0, rtol=0)
        all_results[kind] = dict(summary=summary, rows=rows)
        print(json.dumps(dict(arm=kind, summary=summary), indent=2), flush=True)
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError('frozen source model changed')
    _durable_json(output, dict(scope='POSTHOC_SOURCE_VAL160_NOT_A_NEW_TRAINED_MODEL_OR_EAST_RESULT',
        parameter_updates=0, source_unchanged=True, east_records_read=0, test220_records_read=0,
        oracle='Future diagnostics used only for forecast lower bound; never passed to risk().',
        causal_rule='Reestimate context from targets arriving now at horizons 1..4; no earlier history beyond these pairs.',
        norm_control='Causal solution direction rescaled to production norm; zero directions reported, not fabricated.',
        result=all_results, code_sha256=sha(Path(__file__)),
        input_hashes={f: sha(root/f) for f in ('protocol.json', 'complete.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    run(p.parse_args().root)
