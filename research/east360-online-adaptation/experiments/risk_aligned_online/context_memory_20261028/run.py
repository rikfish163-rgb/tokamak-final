"""Teach a fixed eight-variable context on TRAIN1353, never EAST/TEST signals."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..direction_teaching_20261026.learning import loss_from_example
from ..direction_teaching_20261026.run import CACHE
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device, validate_roles
from .checkpoint import PROFILE, FORECAST_PROFILE, SCALED_PROFILE, read, verify_gradients
from .model import ContextAdapter

REFERENCE = io23.ROOT / 'experiments/risk_aligned_online/runs/zero_prior_source_service_20261028'
PLAN = io23.ROOT / 'docs/research/source_event_rebuild_20261023/八维工况状态_事前实验约定_20261028.md'


def example(sid, targets):
    e = torch.load(CACHE / 'cache' / f'{sid}.pt', map_location='cpu', weights_only=True)
    e.update(targets[sid])
    return to_device(e, 'cuda')


@torch.no_grad()
def evaluate(model, ids, targets, threshold, *, forecast_weight=0.):
    rows = []
    for sid in ids:
        e = example(sid, targets)
        total, info = loss_from_example(model, e, threshold, 'detached', forecast_weight=forecast_weight)
        rows.append(dict(shot=sid, flag=e['flag'], total=float(total), **info))
    summary = {metric: float(np.mean([np.mean([r[metric] for r in rows if r['flag'] == flag])
                                      for flag in (False, True)]))
               for metric in ('bag', 'local', 'direction')}
    return dict(summary=summary, per_shot=rows, scope='SAMPLED_SOURCE_VAL_NOT_FULL_SHOT_METRICS')


def main(root, *, forecast_objective=False, scaled_forecast=False):
    forecast_objective = forecast_objective or scaled_forecast
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    roles = simple_data.source_roles(); validate_roles(roles)
    previous, completed, checked = (read(REFERENCE, f) for f in ('protocol.json', 'complete.json', 'audit.json'))
    if (not checked['complete'] or not checked['source_unchanged']
            or not completed['complete'] or completed['target_or_test_signals_read'] != 0
            or previous['source_sha256'] != PARENT_SHA256 or previous['train_ids'] != roles['train']
            or previous['val_ids'] != roles['val'] or checked['complete_sha256'] != sha(REFERENCE / 'complete.json')
            or completed['local_targets_sha256'] != sha(REFERENCE / 'local_targets.pt')):
        raise ValueError('canonical audited source target/cache reference required')
    inventory = read(CACHE, 'inventory.json')
    if sha(CACHE / 'inventory.json') != previous['cache_inventory_sha256']:
        raise ValueError('source cache inventory changed')
    for row in inventory:
        if (sha(CACHE / 'cache' / f"{row['shot']}.pt") != row['cache_sha256']
                or sha(io23.path_for('J-TEXT', row['shot'])) != row['data_sha256']):
            raise ValueError('source raw/cache example changed')
    targets = torch.load(REFERENCE / 'local_targets.pt', map_location='cpu', weights_only=True)
    if set(targets) != set(roles['train'] + roles['val']):
        raise ValueError('all TRAIN and VAL local targets required')
    base, parent = load_parent(DEFAULT_PARENT); base.cuda()
    original = classifier_state(base); threshold = float(parent['threshold'])
    weights, scale_record = None, None
    if scaled_forecast:
        from .scaled import estimate
        weights, scale_record = estimate(CACHE, roles['train'])
    model = ContextAdapter(base, forecast_weights=weights).cuda().eval()
    model.slow.requires_grad_(False); model.memory.requires_grad_(True)
    profile = FORECAST_PROFILE if forecast_objective else PROFILE
    objective = 'detached_forecast' if forecast_objective else 'detached'
    weight = float(forecast_objective)
    plan = (PLAN.parent / '八维诊断预测教学_事前实验约定_20261028.md') if forecast_objective else PLAN
    if scaled_forecast:
        profile, objective = SCALED_PROFILE, 'detached_scaled_forecast'
        plan = PLAN.parent/'变化尺度反馈_事前实验约定_20261028.md'
    root.mkdir(parents=True)
    scale_digest = None
    if scaled_forecast:
        _durable_json(root/'forecast_weights.json', scale_record)
        scale_digest = sha(root/'forecast_weights.json')
    start = time.monotonic()
    files = {Path(m.__file__) for name, m in list(sys.modules.items())
             if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    files.update(Path(__file__).parent.glob('*.py'))
    p = dict(profile=profile, source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'],
             test_ids_not_read=roles['test'], context_dim=8, seed=131, epochs=2, objective=objective,
             forecast_weight=weight,
             forecast_weights_sha256=scale_digest,
             learning_rate=.001, accumulation_shots=8, gradient_clip=1., feedback='progressive', readout='full',
             threshold=threshold, cache_root=str(CACHE), cache_inventory_sha256=sha(CACHE / 'inventory.json'),
             reference_root=str(REFERENCE), reference_hashes={f: sha(REFERENCE / f) for f in
                 ('protocol.json', 'complete.json', 'audit.json', 'local_targets.pt')},
             plan_sha256=sha(plan), code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
             fast_state_parameters=8, source_memory_parameters=sum(p.numel() for p in model.memory.parameters()),
             torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(root / 'protocol.json', p)
    with zipfile.ZipFile(root / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))

    def emit(**kwargs):
        if time.monotonic() - start > 5400:
            raise TimeoutError('declared 90-minute source teaching limit')
        row = dict(seconds=round(time.monotonic()-start, 2), **kwargs)
        _durable_json(root / 'progress.json', row)
        print(json.dumps(row), flush=True)

    _durable_json(root / 'val_epoch0.json', evaluate(model, roles['val'], targets, threshold, forecast_weight=weight))
    counts = {flag: sum(r['role'] == 'train' and r['disruptive'] == flag for r in inventory) for flag in (False, True)}
    if not all(counts.values()):
        raise ValueError('both source TRAIN classes required')
    optimizer = torch.optim.Adam(model.memory.parameters(), lr=.001)
    rng = np.random.default_rng(131); history = []
    for epoch in (1, 2):
        ids = rng.permutation(roles['train']).tolist()
        _durable_json(root / f'order_epoch{epoch}.json', ids)
        for lo in range(0, len(ids), 8):
            group = ids[lo:lo+8]; optimizer.zero_grad(set_to_none=True)
            examples = []
            for sid in group:
                e = example(sid, targets)
                loss, detail = loss_from_example(model, e, threshold, 'detached', forecast_weight=weight)
                (loss * (.5 * len(ids) / counts[e['flag']] / len(group))).backward()
                examples.append(dict(shot=sid, loss=float(loss.detach()), **detail))
            gradients = {name: None if v.grad is None else float(v.grad.norm()) for name, v in model.memory.named_parameters()}
            norm = float(torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            history.append(dict(epoch=epoch, ids=group, examples=examples, gradients=gradients, gradient_norm=norm))
            if len(history) % 5 == 0 or lo + len(group) == len(ids):
                _durable_json(root / 'training.json', history)
                emit(epoch=epoch, shots=lo+len(group), total=len(ids))
        validation = evaluate(model, roles['val'], targets, threshold, forecast_weight=weight)
        _durable_json(root / f'val_epoch{epoch}.json', validation)
        emit(epoch=epoch, phase='validation', summary=validation['summary'])
        torch.save(dict(profile=profile, kind='context8', context_dim=8, epoch=epoch, objective=objective,
                        forecast_weight=weight,
                        forecast_weights_sha256=scale_digest,
                        feedback='progressive', readout='full', source_sha256=PARENT_SHA256,
                        train_ids=roles['train'], val_ids=roles['val'], protocol_sha256=sha(root / 'protocol.json'),
                        memory={k: v.detach().cpu() for k, v in model.memory.state_dict().items()}),
                   root / f'context8_epoch{epoch}.pt')
    verify_gradients(history, roles['train'])
    assert_classifier_unchanged(base, original)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(root / 'training.json', history)
    done = dict(complete=True, source_unchanged=True, target_or_test_signals_read=0, train=1353, val=160,
                final_epoch=2, total_seconds=time.monotonic()-start, checkpoint_sha256=sha(root / 'context8_epoch2.pt'))
    _durable_json(root / 'complete.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forecast-objective', action='store_true', help='explicit source-only diagnostic forecast loss')
    parser.add_argument('--scaled-forecast', action='store_true', help='source TRAIN increment weights in both teaching and causal writes')
    args = parser.parse_args()
    main(args.output, forecast_objective=args.forecast_objective, scaled_forecast=args.scaled_forecast)
