"""Three matched source-only arms; event head is never deployed on EAST."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn

from .. import io23, simple_data
from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events
from ..source_task_memory_20261026.learning import to_device, validate_roles, source_endpoints
from ..direction_teaching_20261026.learning import make_adapter, local_targets
from ..direction_teaching_20261026.run import CACHE
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_train import sha
from ..simple_online import _durable_json
from .learning import KINDS, family_counts, shuffle_intervals, event_targets, loss_from_example

PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/事件教学内部适配_三臂事前约定_20261026.md'


def load_example(sid, targets, kind):
    e = torch.load(CACHE/'cache'/f'{sid}.pt', map_location='cpu', weights_only=True)
    e.update(targets['risk'][sid])
    e.update(targets[kind][sid])
    return to_device(e, 'cuda')


@torch.no_grad()
def evaluate(models, heads, ids, targets, threshold):
    rows = {k: [] for k in KINDS}
    for sid in ids:
        # All validation event measurements use true labels; no VAL shuffling/fitting.
        e = load_example(sid, targets, 'true')
        for k, m in models.items():
            risk, _, info = loss_from_example(m, heads[k], e, threshold)
            rows[k].append(dict(shot=sid, flag=e['flag'], risk=float(risk), **info))
    summary = {}
    for kind, data in rows.items():
        summary[kind] = {key: float(np.mean([np.mean([r[key] for r in data if r['flag'] == f])
                                            for f in (False, True)]))
                         for key in ('risk', 'bag', 'local', 'direction')}
        summary[kind].update({key: sum(r[key] for r in data) for key in
            ('direction_count', 'harmful_writes', 'event_points', 'event_loss',
             'event_weight', 'event_correct_weight')})
    return dict(summary=summary, per_shot=rows, scope='SAMPLED_SOURCE_VAL_DEVELOPMENT_NOT_EAST')


def main(root):
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
    audited = json.loads((CACHE/'audit.json').read_text())
    if (previous['source_sha256'] != PARENT_SHA256 or previous['train_ids'] != roles['train']
            or previous['val_ids'] != roles['val'] or not audited['complete']
            or sha(DEFAULT_LABELS) != LABEL_SHA256):
        raise ValueError('canonical audited source cache and annotation required')
    root.mkdir(parents=True)
    started = time.monotonic()

    def emit(**fields):
        if time.monotonic()-started > 5400:
            raise TimeoutError('declared90-minute source teaching limit')
        r = dict(seconds=round(time.monotonic()-started, 2), **fields)
        _durable_json(root/'progress.json', r)
        print(json.dumps(r), flush=True)

    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    original = classifier_state(base)
    threshold = float(parent['threshold'])
    models = {k: make_adapter(base).cuda() for k in KINDS}
    heads = {k: nn.Linear(base.cfg.dim, 6).cuda() for k in KINDS}
    for kind in KINDS[1:]:
        models[kind].load_state_dict(models['none'].state_dict())
        heads[kind].load_state_dict(heads['none'].state_dict())
    files = {Path(m.__file__) for name, m in list(sys.modules.items())
             if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    files.update(Path(__file__).parent.glob('*.py'))
    p = dict(profile='event_teaching_v1', source_sha256=PARENT_SHA256,
        train_ids=roles['train'], val_ids=roles['val'], test_ids_not_read=roles['test'],
        feedback='progressive', readout='full', kinds=KINDS, threshold=threshold,
        epochs=2, fixed_final_epoch=2, seed=131, learning_rate=.001, accumulation_shots=8,
        event_weight=.1, event_head='training_only_64_to6_from_frozen_risk_head_input',
        unknown_event_label=-1, shuffle='whole_intervals_within_TRAIN_outcome',
        risk_objective='direction_bag_plus0.5local_plus0.5write_direction',
        event_weighting='mean_over_families_then_intervals_then_sampled_points',
        event_path=str(DEFAULT_LABELS), event_sha256=LABEL_SHA256,
        cache_root=str(CACHE), cache_inventory_sha256=sha(CACHE/'inventory.json'),
        cache_audit_sha256=sha(CACHE/'audit.json'), plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files})
    _durable_json(root/'protocol.json', p)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
    inventory = json.loads((CACHE/'inventory.json').read_text())
    targets = {'risk': {}, 'true': {}, 'shuffled': {}}
    times_by_role = {'train': {}, 'val': {}}
    flags = {}
    advanced = simple_data.advanced_times()
    for i, row in enumerate(inventory, 1):
        sid = row['shot']
        if (sha(CACHE/'cache'/f'{sid}.pt') != row['cache_sha256']
                or sha(io23.path_for('J-TEXT', sid)) != row['data_sha256']):
            raise ValueError('changed source cache/raw data')
        s = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        ends = source_endpoints(s)
        if ends.tolist() != row['ends'] or sid not in roles[row['role']]:
            raise ValueError('source endpoint or role changed')
        targets['risk'][sid] = local_targets(s, ends)
        times_by_role[row['role']][sid] = s['times'][ends]
        if row['role'] == 'train':
            flags[sid] = s['is_disrupt']
        if i % 100 == 0 or i == len(inventory):
            emit(phase='source_labels', shots=i, total=len(inventory))
    if set(targets['risk']) != set(roles['train']+roles['val']):
        raise ValueError('all TRAIN+VAL examples required')
    events, _ = read_events(DEFAULT_LABELS)
    shuffled = shuffle_intervals(events, flags, seed=131)
    manifest = dict(family_counts={}, intervals_shuffled={str(s): [asdict(e) for e in es]
                                                        for s, es in shuffled.items() if es})
    for role, clock in times_by_role.items():
        counts = family_counts(clock, events)
        if (counts == 0).any():
            raise ValueError('six represented source families required for macro event objective')
        manifest['family_counts'][role] = counts.tolist()
        for sid, t in clock.items():
            targets['true'][sid] = event_targets(t, events.get(sid, []), counts=counts)
        if role == 'train':
            scounts = family_counts(clock, shuffled)
            if not np.array_equal(counts, scounts):
                raise ValueError('shuffle changed sampled family counts')
            for sid, t in clock.items():
                targets['shuffled'][sid] = event_targets(t, shuffled.get(sid, []), counts=scounts)
    targets['none'] = targets['true']
    torch.save(targets, root/'targets.pt')
    _durable_json(root/'events.json', manifest)
    _durable_json(root/'inventory.json', inventory)
    train_counts = {f: sum(v == f for v in flags.values()) for f in (False, True)}
    parameters = {k: list(m.memory.parameters())+list(heads[k].parameters()) for k, m in models.items()}
    optimizers = {k: torch.optim.Adam(parameters[k], lr=.001) for k in KINDS}
    _durable_json(root/'val_epoch0.json', evaluate(models, heads, roles['val'], targets, threshold))
    history, rng = [], np.random.default_rng(131)
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
                    e = load_example(sid, targets, kind)
                    risk, event, info = loss_from_example(model, heads[kind], e, threshold)
                    total = risk*(.5*len(ids)/train_counts[e['flag']]/len(group))
                    if kind != 'none':
                        total = total+.1*len(ids)*event/len(group)
                    if not torch.isfinite(total):
                        raise FloatingPointError('nonfinite teaching objective')
                    total.backward()
                    examples.append(dict(shot=sid, total=float(total.detach()), **info))
                gradients = {n: None if v.grad is None else float(v.grad.norm())
                             for n, v in model.memory.named_parameters()}
                head_gradient = None if heads[kind].weight.grad is None else float(heads[kind].weight.grad.norm())
                norm = float(torch.nn.utils.clip_grad_norm_(parameters[kind], 1., error_if_nonfinite=True))
                optimizers[kind].step()
                record['arms'][kind] = dict(examples=examples, gradients=gradients,
                                           head_gradient=head_gradient, gradient_norm=norm)
            history.append(record)
            if len(history) % 5 == 0 or lo+len(group) == len(ids):
                emit(phase='train', epoch=epoch, shots=lo+len(group), total=len(ids))
                _durable_json(root/'training.json', history)
        report = evaluate(models, heads, roles['val'], targets, threshold)
        _durable_json(root/f'val_epoch{epoch}.json', report)
        emit(phase='validation', epoch=epoch, summary=report['summary'])
        for kind, m in models.items():
            torch.save(dict(profile='event_teaching_v1', kind=kind, feedback='progressive', readout='full',
                source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'], epoch=epoch,
                protocol_sha256=sha(root/'protocol.json'), event_sha256=LABEL_SHA256,
                memory={k: v.detach().cpu() for k, v in m.memory.state_dict().items()},
                event_head_training_only={k: v.detach().cpu() for k, v in heads[kind].state_dict().items()}),
                root/f'{kind}_epoch{epoch}.pt')
    assert_classifier_unchanged(base, original)
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    done = dict(complete=True, source_unchanged=True, final_epoch=2, train=1353, val=160,
        target_or_test_signals_read=0, total_seconds=time.monotonic()-started,
        targets_sha256=sha(root/'targets.pt'), events_sha256=sha(root/'events.json'),
        checkpoints={k: sha(root/f'{k}_epoch2.pt') for k in KINDS})
    _durable_json(root/'complete.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
