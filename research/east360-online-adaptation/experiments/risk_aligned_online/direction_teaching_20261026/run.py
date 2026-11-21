"""Matched progressive source teaching: bag control versus local write objective."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..simple_online import _durable_json
from ..simple_train import sha
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..source_task_memory_20261026.learning import to_device, validate_roles, source_endpoints
from .learning import make_adapter, local_targets, loss_from_example, PROFILE_KINDS, reference_policy, prior_policy

CACHE = io23.ROOT/'experiments/risk_aligned_online/runs/source_task_memory_v2_20261026'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/逐次更新方向教学_源域与360对照约定_20261026.md'
REFERENCE_PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/参照梯度单因素对照_20261027.md'
ZERO_PRIOR_PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/零初始记忆_完整配对事前约定_20261028.md'


def load_example(sid, targets):
    e = torch.load(CACHE/'cache'/f'{sid}.pt', map_location='cpu', weights_only=True)
    e.update(targets[sid])
    return to_device(e, 'cuda')


@torch.no_grad()
def evaluate(models, ids, targets, threshold):
    rows = {k: [] for k in models}
    for sid in ids:
        e = load_example(sid, targets)
        for k, m in models.items():
            total, info = loss_from_example(m, e, threshold, k)
            rows[k].append(dict(shot=sid, flag=e['flag'], total=float(total), **info))
    summary = {}
    for k, data in rows.items():
        summary[k] = {metric: float(np.mean([np.mean([r[metric] for r in data if r['flag'] == flag])
                                           for flag in (False, True)]))
                      for metric in ('bag', 'local', 'direction')}
        summary[k].update(direction_count=sum(r['direction_count'] for r in data),
                          harmful_writes=sum(r['harmful_writes'] for r in data))
    return dict(summary=summary, per_shot=rows, scope='SAMPLED_VAL_NOT_FULL_SHOT_TPR')


def main(root, reference_ablation=False, zero_prior_ablation=False):
    if reference_ablation and zero_prior_ablation:
        raise ValueError('choose one explicit ablation')
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    roles = simple_data.source_roles()
    validate_roles(roles)
    previous = json.loads((CACHE/'protocol.json').read_text())
    audit = json.loads((CACHE/'audit.json').read_text())
    if (previous['source_sha256'] != PARENT_SHA256 or previous['train_ids'] != roles['train']
            or previous['val_ids'] != roles['val'] or not audit['complete']):
        raise ValueError('canonical source-only cache audit required')
    root.mkdir(parents=True)
    start = time.monotonic()

    def emit(**fields):
        if time.monotonic()-start > 5400:
            raise TimeoutError('declared90-minute source teaching limit')
        data = dict(seconds=round(time.monotonic()-start, 2), **fields)
        _durable_json(root/'progress.json', data)
        print(json.dumps(data), flush=True)

    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    original = classifier_state(base)
    threshold = float(parent['threshold'])
    profile = ('zero_prior_ablation_v3' if zero_prior_ablation else
               'reference_gradient_ablation_v2' if reference_ablation else 'direction_teaching_v1')
    kinds = PROFILE_KINDS[profile]
    models = {k: make_adapter(base, zero_prior=k == 'zero_prior').cuda() for k in kinds}
    models[kinds[1]].load_state_dict(models[kinds[0]].state_dict())
    files = {Path(m.__file__) for name, m in list(sys.modules.items())
             if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    files.update(Path(__file__).parent.glob('*.py'))
    protocol = dict(profile=profile, source_sha256=PARENT_SHA256,
        train_ids=roles['train'], val_ids=roles['val'], test_ids_not_read=roles['test'],
        feedback='progressive', readout='full', kinds=kinds, threshold=threshold,
        epochs=2, fixed_final_epoch=2, seed=131, learning_rate=.001, accumulation_shots=8,
        local_weight=.5, direction_weight=.5, direction_temperature=.02,
        cache_root=str(CACHE), cache_inventory_sha256=sha(CACHE/'inventory.json'),
        cache_audit_sha256=sha(CACHE/'audit.json'),
        plan_sha256=sha(ZERO_PRIOR_PLAN if zero_prior_ablation else REFERENCE_PLAN if reference_ablation else PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files})
    if reference_ablation or zero_prior_ablation:
        protocol['reference_gradient'] = {k: reference_policy(k) for k in kinds}
    if zero_prior_ablation:
        protocol['prior_policy'] = {k: prior_policy(k) for k in kinds}
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    inventory = json.loads((CACHE/'inventory.json').read_text())
    advanced = simple_data.advanced_times()
    targets, counts = {}, {False: 0, True: 0}
    for i, row in enumerate(inventory, 1):
        sid = row['shot']
        if (sha(CACHE/'cache'/f'{sid}.pt') != row['cache_sha256']
                or sha(io23.path_for('J-TEXT', sid)) != row['data_sha256']):
            raise ValueError('changed cached/raw source example')
        s = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        ends = source_endpoints(s)
        if ends.tolist() != row['ends']:
            raise ValueError('source endpoint contract changed')
        targets[sid] = local_targets(s, ends)
        if row['role'] == 'train':
            counts[s['is_disrupt']] += 1
        if i % 100 == 0 or i == len(inventory):
            emit(phase='source_target_labels', shots=i, total=len(inventory))
    if set(targets) != set(roles['train']+roles['val']):
        raise ValueError('complete TRAIN+VAL cache required')
    torch.save(targets, root/'local_targets.pt')
    _durable_json(root/'inventory.json', inventory)
    optimizers = {k: torch.optim.Adam([p for p in m.memory.parameters() if p.requires_grad], lr=.001)
                  for k, m in models.items()}
    initial = evaluate(models, roles['val'], targets, threshold)
    _durable_json(root/'val_epoch0.json', initial)
    rng, history = np.random.default_rng(131), []
    for epoch in (1, 2):
        ids = rng.permutation(roles['train']).tolist()
        _durable_json(root/f'order_epoch{epoch}.json', ids)
        for lo in range(0, len(ids), 8):
            group = ids[lo:lo+8]
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            record = dict(epoch=epoch, ids=group, arms={})
            for kind, model in models.items():
                examples = []
                for sid in group:
                    e = load_example(sid, targets)
                    total, info = loss_from_example(model, e, threshold, kind)
                    (total*(.5*len(ids)/counts[e['flag']]/len(group))).backward()
                    examples.append(dict(shot=sid, total=float(total.detach()), **info))
                gradients = {n: None if p.grad is None else float(p.grad.norm())
                             for n, p in model.memory.named_parameters()}
                norm = float(torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1., error_if_nonfinite=True))
                optimizers[kind].step()
                record['arms'][kind] = dict(examples=examples, gradients=gradients, gradient_norm=norm)
            history.append(record)
            if len(history) % 5 == 0 or lo+len(group) == len(ids):
                emit(phase='train', epoch=epoch, shots=lo+len(group), total=len(ids))
                _durable_json(root/'training.json', history)
        validation = evaluate(models, roles['val'], targets, threshold)
        _durable_json(root/f'val_epoch{epoch}.json', validation)
        emit(phase='validation', epoch=epoch, summary=validation['summary'])
        for kind, model in models.items():
            checkpoint = dict(profile=profile, kind=kind, feedback='progressive',
                source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'],
                epoch=epoch, readout='full', protocol_sha256=sha(root/'protocol.json'),
                memory={k: v.detach().cpu() for k, v in model.memory.state_dict().items()})
            if reference_ablation or zero_prior_ablation:
                checkpoint['reference_gradient'] = reference_policy(kind)
            if zero_prior_ablation:
                checkpoint['prior_policy'] = prior_policy(kind)
                if kind == 'zero_prior' and model.memory.prior.count_nonzero():
                    raise AssertionError('frozen zero prior changed')
            torch.save(checkpoint, root/f'{kind}_epoch{epoch}.pt')
    assert_classifier_unchanged(base, original)
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source file changed')
    _durable_json(root/'training.json', history)
    done = dict(complete=True, source_unchanged=True, total_seconds=time.monotonic()-start,
                final_epoch=2, train=1353, val=160, target_or_test_signals_read=0,
                local_targets_sha256=sha(root/'local_targets.pt'),
                checkpoints={k: sha(root/f'{k}_epoch2.pt') for k in kinds})
    _durable_json(root/'complete.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--reference-ablation', action='store_true',
                   help='Matched direction versus detached-reference objectives; new explicit v2 profile.')
    group.add_argument('--zero-prior-ablation', action='store_true',
                       help='Matched learned versus fixed-zero prior with identical detached-reference loss.')
    args = p.parse_args()
    main(args.output, args.reference_ablation, args.zero_prior_ablation)
