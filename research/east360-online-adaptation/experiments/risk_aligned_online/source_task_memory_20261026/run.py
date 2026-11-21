"""Two fixed epochs on TRAIN1353; VAL160 descriptive, no EAST/TEST read."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import simple_data, io23
from ..causal_input import apply_input_contract
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_train import sha
from ..simple_online import _durable_json
from .learning import validate_roles, make_adapter, prepare_example, to_device, loss_from_example

PLAN = io23.ROOT / 'docs/research/source_event_rebuild_20261023/源域教学炮内学习任务_实验约定_20261026.md'


def load_example(root, sid):
    return to_device(torch.load(root/'cache'/f'{sid}.pt', map_location='cpu', weights_only=True), 'cuda')


def emit(root, start, **fields):
    if time.monotonic()-start > 5400:
        raise TimeoutError('source task teaching exceeded declared 90 minutes')
    row = dict(seconds=round(time.monotonic()-start, 2), **fields)
    _durable_json(root/'progress.json', row)
    print(json.dumps(row), flush=True)


@torch.no_grad()
def evaluate(root, models, ids, threshold):
    rows = {k: [] for k in models}
    for sid in ids:
        e = load_example(root, sid)
        for kind, model in models.items():
            loss, detail = loss_from_example(model, e, threshold)
            rows[kind].append(dict(shot=int(sid), flag=e['flag'], loss=float(loss),
                                   baseline=e['baseline'], **detail))
    summaries = {}
    for kind, data in rows.items():
        d, s = [np.mean([r['loss'] for r in data if r['flag'] == flag]) for flag in (True, False)]
        summaries[kind] = dict(disrupt_loss=float(d), safe_loss=float(s), balanced_loss=float((d+s)/2))
    return dict(summary=summaries, per_shot=rows, scope='SAMPLED_SOURCE_VAL_LOSS_NOT_FULL_SHOT_METRICS')


def main(root):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    start = time.monotonic()
    roles = simple_data.source_roles(); validate_roles(roles)
    advanced = simple_data.advanced_times()
    base, parent = load_parent(DEFAULT_PARENT); base.cuda()
    original = classifier_state(base); threshold = float(parent['threshold'])
    models = {k: make_adapter(base, k).cuda() for k in ('learned', 'fixed')}
    models['fixed'].memory.load_state_dict(models['learned'].memory.state_dict())
    root.mkdir(parents=True); (root/'cache').mkdir()
    files = {Path(m.__file__) for n, m in list(sys.modules.items())
             if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    protocol = dict(scope='SOURCE_TRAIN_ONLY_TASK_TEACHING', source_sha256=PARENT_SHA256,
                    train_ids=roles['train'], val_ids=roles['val'], test_ids_not_read=roles['test'],
                    threshold=threshold, epochs=2, seed=131, learning_rate=.001, accumulation_shots=8,
                    readout='full', selected_epoch='FIXED_FINAL_2_NOT_SELECTED',
                    code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in files},
                    plan_sha256=sha(PLAN), torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json', protocol)
    inventory = []
    for role in ('train', 'val'):
        for i, sid in enumerate(roles[role]):
            shot = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT', sid)),
                                              'J-TEXT', advanced)
            e = prepare_example(models['learned'], shot, threshold)
            path = root/'cache'/f'{sid}.pt'; torch.save(e, path)
            inventory.append(dict(shot=int(sid), role=role, disruptive=e['flag'],
                                  ends=e['ends'].tolist(), patches=len(e['means']),
                                  source_horizon_ms=e['source_horizon_ms'], cache_sha256=sha(path),
                                  data_sha256=sha(io23.path_for('J-TEXT', sid))))
            if (i+1) % 25 == 0 or i+1 == len(roles[role]):
                emit(root, start, phase='cache', role=role, shots=i+1, total=len(roles[role]))
    _durable_json(root/'inventory.json', inventory)
    optimizers = {k: torch.optim.Adam([p for p in m.memory.parameters() if p.requires_grad], lr=.001)
                  for k, m in models.items()}
    class_counts = {f: sum(r['disruptive'] == f and r['role'] == 'train' for r in inventory)
                    for f in (False, True)}
    if not all(class_counts.values()):
        raise ValueError('both source TRAIN classes required')
    rng = np.random.default_rng(131)
    history = []
    baseline = evaluate(root, models, roles['val'], threshold)
    _durable_json(root/'val_epoch0.json', baseline)
    for epoch in (1, 2):
        ids = rng.permutation(roles['train']).tolist()
        _durable_json(root/f'order_epoch{epoch}.json', ids)
        for start_index in range(0, len(ids), 8):
            group = ids[start_index:start_index+8]
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            record = dict(epoch=epoch, shots_seen=start_index+len(group), ids=group, arms={})
            for kind, model in models.items():
                losses = []
                for sid in group:
                    e = load_example(root, sid)
                    loss, info = loss_from_example(model, e, threshold)
                    weight = .5*len(ids)/class_counts[e['flag']]/len(group)
                    (loss*weight).backward()
                    losses.append(dict(shot=int(sid), loss=float(loss.detach()), **info))
                gradients = {n: None if p.grad is None else float(p.grad.norm())
                             for n, p in model.memory.named_parameters()}
                norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.memory.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
                optimizers[kind].step()
                record['arms'][kind] = dict(examples=losses, gradient_norm=norm, gradients=gradients,
                                           rate=float(model.memory.eta.detach()))
            history.append(record)
            if len(history) % 5 == 0 or record['shots_seen'] == len(ids):
                emit(root, start, phase='train', epoch=epoch, shots=record['shots_seen'], total=len(ids))
                _durable_json(root/'training.json', history)
        validation = evaluate(root, models, roles['val'], threshold)
        _durable_json(root/f'val_epoch{epoch}.json', validation)
        emit(root, start, phase='validation', epoch=epoch, summary=validation['summary'])
        for kind, model in models.items():
            torch.save(dict(kind=kind, source_sha256=PARENT_SHA256, train_ids=roles['train'],
                            val_ids=roles['val'], epoch=epoch, readout='full',
                            memory={k:v.detach().cpu() for k,v in model.memory.state_dict().items()},
                            protocol_sha256=sha(root/'protocol.json')),
                       root/f'{kind}_epoch{epoch}.pt')
    assert_classifier_unchanged(base, original)
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source checkpoint changed on disk')
    _durable_json(root/'training.json', history)
    _durable_json(root/'complete.json', dict(complete=True, total_seconds=time.monotonic()-start,
                  source_unchanged=True, train_shots=len(roles['train']), val_shots=len(roles['val']),
                  test_or_east_signals_read=0, final_epoch=2,
                  checkpoints={k:sha(root/f'{k}_epoch2.pt') for k in models}))
    print(json.dumps(validation['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
