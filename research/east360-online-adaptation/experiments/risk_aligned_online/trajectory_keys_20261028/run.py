"""Fixed two-epoch source mechanism pair. Not an EAST deployment checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..context_memory_20261028.checkpoint import read
from ..context_memory_20261028.conditioned_run import summarize
from ..context_memory_20261028.forecast_audit import persistence_error
from ..context_memory_20261028.source_controls import evaluate_modes
from ..direction_teaching_20261026.learning import loss_from_example
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device, validate_roles
from .model import TrajectoryAdapter


def screen(candidate, forecast, control):
    checks = {}
    for metric in ('bag', 'local'):
        modes = candidate['all']
        for baseline in ('off', 'reverse'):
            checks[f'{metric}_better_than_{baseline}'] = modes['memory'][metric] < modes[baseline][metric] - 1e-6
        checks[f'{metric}_better_than_level'] = modes['memory'][metric] < control['all']['memory'][metric] - 1e-6
        for group in ('D', 'S'):
            checks[f'{group}_{metric}_not_worse'] = candidate[group]['memory'][metric] <= candidate[group]['off'][metric] + 1e-6
    checks.update(forecast_better_than_hold=forecast['mean'] < forecast['persistence_mean'],
                  forecast_majority=forecast['improved'] >= 81,
                  forecast_safe_majority=forecast['safe_improved'] >= 41)
    return dict(eligible=all(checks.values()), checks=checks,
                scope='SOURCE_MECHANISM_SCREEN_NOT_EAST_ACCEPTANCE')


def run(output, reference):
    output = output.absolute()
    if output.parent != io23.ROOT / 'probe_outputs' or output.exists():
        raise ValueError('new output under this worktree/probe_outputs only')
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    roles = simple_data.source_roles(); validate_roles(roles)
    p, completed = read(reference, 'protocol.json'), read(reference, 'complete.json')
    if (not completed['complete'] or not completed['source_unchanged']
            or p['source_sha256'] != PARENT_SHA256 or p['train_ids'] != roles['train']
            or p['val_ids'] != roles['val'] or completed['target_or_test_signals_read'] != 0):
        raise ValueError('completed source-only reference required')
    cache, targets_path = Path(p['cache_root']), Path(p['reference_root']) / 'local_targets.pt'
    if (sha(cache / 'inventory.json') != p['cache_inventory_sha256']
            or sha(targets_path) != p['reference_hashes']['local_targets.pt']
            or sha(reference / 'forecast_weights.json') != p['forecast_weights_sha256']):
        raise ValueError('source cache, target or TRAIN scale provenance changed')
    inventory = read(cache, 'inventory.json')
    targets = torch.load(targets_path, map_location='cpu', weights_only=True)
    weights_record = read(reference, 'forecast_weights.json')
    if weights_record['train_ids'] != roles['train']:
        raise ValueError('only canonical TRAIN may estimate diagnostic scales')
    weights = torch.tensor(weights_record['weights'])
    examples = {}
    for r in inventory:
        sid = r['shot']; path = cache / 'cache' / f'{sid}.pt'
        if r['role'] not in ('train', 'val') or sid not in roles[r['role']] or sha(path) != r['cache_sha256']:
            raise ValueError('audited TRAIN/VAL cache only; no TEST or EAST')
        e = torch.load(path, map_location='cpu', weights_only=True)
        e.update(targets[sid]); examples[sid] = e
    if set(examples) != set(roles['train'] + roles['val']):
        raise ValueError('complete TRAIN1353 and VAL160 required')
    base, parent = load_parent(DEFAULT_PARENT); base.cuda()
    frozen = classifier_state(base)
    models = {kind: TrajectoryAdapter(base, weights, trajectory=kind == 'trajectory').cuda().eval()
              for kind in ('level', 'trajectory')}
    models['trajectory'].load_state_dict(models['level'].state_dict())
    for k, value in models['level'].state_dict().items():
        torch.testing.assert_close(value, models['trajectory'].state_dict()[k], rtol=0, atol=0)
    plan = io23.ROOT / 'TRAJECTORY_PLAN.md'
    output.mkdir(parents=True)
    imported = {Path(m.__file__) for name, m in list(sys.modules.items())
                if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    protocol = dict(profile='trajectory_key_pair_v1_NOT_DEPLOYABLE', source_sha256=PARENT_SHA256,
        threshold=float(parent['threshold']), train_ids=roles['train'], val_ids=roles['val'],
        cache_inventory_sha256=sha(cache / 'inventory.json'), targets_sha256=sha(targets_path),
        forecast_weights_sha256=sha(reference / 'forecast_weights.json'), reference=str(reference),
        cache=str(cache), targets=str(targets_path), plan_sha256=sha(plan),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in imported},
        seed=131, epochs=2, learning_rate=.001, accumulation_shots=8, gradient_clip=1.,
        forecast_weight=1., objective='detached_direction_plus_scaled_forecast',
        parameter_count={kind: sum(q.numel() for q in m.memory.parameters()) for kind, m in models.items()},
        effective_capacity_equal=False, initialization_equal=True, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), source_classifier_frozen=True,
        east_records_read=0, test220_records_read=0, source_event_labels_read=0)
    _durable_json(output / 'protocol.json', protocol)
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(imported):
            archive.write(f, str(f.relative_to(io23.ROOT)))
        archive.write(plan, plan.name)
    threshold = protocol['threshold']; start = time.monotonic()

    def example(sid):
        return to_device(examples[sid], 'cuda')

    @torch.no_grad()
    def evaluate(model):
        rows, forecasts = [], []
        for sid in roles['val']:
            e = example(sid)
            rows.append(dict(shot=sid, disruptive=e['flag'], modes=evaluate_modes(model, e, threshold)))
            trace = model.memory.trace(e['means'], e['complete'], mode='memory')
            persistence = persistence_error(e['means'], e['complete'], weights=weights.cuda())
            forecasts.append(dict(shot=sid, disruptive=e['flag'], error=float(trace['forecast_loss']),
                                  persistence=persistence))
        forecast = dict(mean=float(np.mean([r['error'] for r in forecasts])),
            persistence_mean=float(np.mean([r['persistence'] for r in forecasts])),
            improved=sum(r['error'] < r['persistence'] for r in forecasts),
            safe_improved=sum(not r['disruptive'] and r['error'] < r['persistence'] for r in forecasts))
        return dict(summary=summarize(rows), forecast=forecast, rows=rows, forecast_rows=forecasts,
                    scope='SAMPLED_SOURCE_VAL160_NOT_FULL_SHOT_TPR_FPR')

    counts = {flag: sum(examples[sid]['flag'] == flag for sid in roles['train']) for flag in (False, True)}
    results = {}
    for kind, model in models.items():
        arm = output / kind; arm.mkdir()
        _durable_json(arm / 'val_epoch0.json', evaluate(model))
        optimizer = torch.optim.Adam(model.memory.parameters(), lr=.001)
        history = []; rng = np.random.default_rng(131)
        for epoch in (1, 2):
            ids = rng.permutation(roles['train']).tolist()
            _durable_json(arm / f'order_epoch{epoch}.json', ids)
            for lo in range(0, len(ids), 8):
                group = ids[lo:lo+8]; optimizer.zero_grad(set_to_none=True)
                losses = []
                for sid in group:
                    e = example(sid)
                    loss, _ = loss_from_example(model, e, threshold, 'detached', forecast_weight=1.)
                    (loss * (.5 * len(ids) / counts[e['flag']] / len(group))).backward()
                    losses.append(float(loss.detach()))
                gradient = float(torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1., error_if_nonfinite=True))
                extra_gradient = float(model.memory.key.weight.grad[:, 46:].norm())
                optimizer.step()
                history.append(dict(epoch=epoch, ids=group, losses=losses,
                                    gradient_norm=gradient, extra_key_gradient_norm=extra_gradient))
                if time.monotonic() - start > 3000:
                    raise TimeoutError('50-minute fixed two-arm probe bound')
                if len(history) % 20 == 0 or lo + len(group) == len(ids):
                    progress = dict(arm=kind, epoch=epoch, shots=lo+len(group), seconds=time.monotonic()-start)
                    _durable_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            # Fix epoch2 in advance; do not select on interim VAL outcomes.
            assert_classifier_unchanged(base, frozen)
        result = evaluate(model)
        _durable_json(arm / 'val_epoch2.json', result)
        _durable_json(arm / 'training.json', history)
        for epoch in (1, 2):
            seen = [sid for r in history if r['epoch'] == epoch for sid in r['ids']]
            if len(seen) != 1353 or set(seen) != set(roles['train']):
                raise ValueError('incomplete source teaching epoch')
        if not any(r['gradient_norm'] > 0 for r in history):
            raise ValueError('source learning did not occur')
        if kind == 'trajectory' and not any(r['extra_key_gradient_norm'] > 0 for r in history):
            raise ValueError('new trajectory key never trained')
        if kind == 'level' and any(r['extra_key_gradient_norm'] != 0 for r in history):
            raise ValueError('level control consumed trajectory inputs')
        torch.save(dict(profile=protocol['profile'], kind=kind,
                        protocol_sha256=sha(output / 'protocol.json'),
                        memory={k: v.cpu() for k, v in model.memory.state_dict().items()}), arm / 'memory_epoch2.pt')
        results[kind] = dict(summary=result['summary'], forecast=result['forecast'])
    for epoch in (1, 2):
        if read(output / 'level', f'order_epoch{epoch}.json') != read(output / 'trajectory', f'order_epoch{epoch}.json'):
            raise ValueError('paired train orders differ')
    for k in ('level', 'trajectory'):
        for f, value in protocol['code_sha256'].items():
            if sha(io23.ROOT / f) != value:
                raise ValueError('experiment code changed while running')
        assert_classifier_unchanged(base, frozen)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(plan) != protocol['plan_sha256']:
        raise ValueError('source parent or preregistered plan changed')
    gate = screen(results['trajectory']['summary'], results['trajectory']['forecast'], results['level']['summary'])
    done = dict(complete=True, results=results, gate=gate, seconds=time.monotonic()-start,
        source_classifier_unchanged=True, east_records_read=0, test220_records_read=0,
        checkpoints={k: sha(output / k / 'memory_epoch2.pt') for k in models})
    _durable_json(output / 'complete.json', done)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    run(args.output, args.reference)
