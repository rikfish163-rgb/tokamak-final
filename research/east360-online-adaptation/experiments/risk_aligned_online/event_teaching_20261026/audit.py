"""Audit source access, complete matched training and regenerated event targets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..simple_train import sha
from ..simple_online import _durable_json
from ..event_prefix_probe import read_events, LABEL_SHA256
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..source_task_memory_20261026.learning import validate_roles, source_endpoints
from ..direction_teaching_20261026.learning import local_targets
from .learning import KINDS, family_counts, shuffle_intervals, event_targets
from .checkpoint import validate_metadata


def validate_training(rows, ids):
    for epoch in (1, 2):
        seen = [sid for r in rows if r['epoch'] == epoch for sid in r['ids']]
        if len(seen) != len(ids) or set(seen) != set(ids):
            raise ValueError('nonduplicated full TRAIN coverage required')
    active = {}
    for kind in KINDS:
        active[kind] = {}
        for row in rows:
            if [e['shot'] for e in row['arms'][kind]['examples']] != row['ids']:
                raise ValueError('identical per-arm shot order required')
        for name in ('key.weight', 'prior', 'rate', 'output.weight'):
            values = [r['arms'][kind]['gradients'][name] for r in rows]
            if any(v is None or not np.isfinite(v) for v in values) or not any(v > 0 for v in values):
                raise ValueError('finite nontrivial learning-rule gradient required')
            active[kind][name] = sum(v > 0 for v in values)
        head = [r['arms'][kind]['head_gradient'] for r in rows]
        if kind == 'none':
            if any(v is not None for v in head):
                raise ValueError('no-event control must not train event head')
        elif (not any(v is not None and v > 0 for v in head)
              or any(v is not None and not np.isfinite(v) for v in head)):
            raise ValueError('finite nontrivial event head gradient required')
    return dict(complete_epochs=2, nonzero_gradient_steps=active)


def audit(root):
    if (root/'audit.json').exists():
        raise FileExistsError(root/'audit.json')
    def read(name):
        return json.loads((root/name).read_text())
    p, done = read('protocol.json'), read('complete.json')
    if not done['complete'] or done['final_epoch'] != 2 or done['target_or_test_signals_read'] != 0:
        raise ValueError('completed source-only teaching required')
    roles = simple_data.source_roles()
    validate_roles(roles)
    assert p['train_ids'] == roles['train'] and p['val_ids'] == roles['val']
    assert p['source_sha256'] == sha(DEFAULT_PARENT) == PARENT_SHA256
    assert p['event_sha256'] == sha(Path(p['event_path'])) == LABEL_SHA256
    cache = Path(p['cache_root'])
    assert sha(cache/'inventory.json') == p['cache_inventory_sha256']
    assert sha(cache/'audit.json') == p['cache_audit_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for path, digest in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    training = read('training.json')
    verified = validate_training(training, roles['train'])
    for epoch in (1, 2):
        assert read(f'order_epoch{epoch}.json') == [sid for r in training if r['epoch'] == epoch for sid in r['ids']]
    inventory = read('inventory.json')
    assert len(inventory) == len({r['shot'] for r in inventory}) == 1513
    for role in ('train', 'val'):
        assert {r['shot'] for r in inventory if r['role'] == role} == set(roles[role])
    assert sha(root/'targets.pt') == done['targets_sha256']
    assert sha(root/'events.json') == done['events_sha256']
    labels = torch.load(root/'targets.pt', map_location='cpu', weights_only=True)
    advanced = simple_data.advanced_times()
    clocks, flags = {'train': {}, 'val': {}}, {}
    for row in inventory:
        sid = row['shot']
        assert sha(cache/'cache'/f'{sid}.pt') == row['cache_sha256']
        assert sha(io23.path_for('J-TEXT', sid)) == row['data_sha256']
        s = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        ends = source_endpoints(s)
        assert ends.tolist() == row['ends']
        expected = local_targets(s, ends)
        assert all(torch.equal(v, labels['risk'][sid][k]) for k, v in expected.items())
        clocks[row['role']][sid] = s['times'][ends]
        if row['role'] == 'train':
            flags[sid] = s['is_disrupt']
    events, _ = read_events(Path(p['event_path']))
    shuffled = shuffle_intervals(events, flags, seed=p['seed'])
    manifest = read('events.json')
    assert manifest['intervals_shuffled'] == {str(s): [asdict(e) for e in es]
                                             for s, es in shuffled.items() if es}
    for flag in (False, True):
        assert sorted(e.family for s in flags if flags[s] == flag for e in events.get(s, [])) == sorted(
            e.family for s in flags if flags[s] == flag for e in shuffled[s])
    for role, clock in clocks.items():
        counts = family_counts(clock, events)
        assert counts.tolist() == manifest['family_counts'][role]
        if role == 'train':
            assert np.array_equal(counts, family_counts(clock, shuffled))
        for sid, times in clock.items():
            expected = event_targets(times, events.get(sid, []), counts=counts)
            for kind in ('none', 'true'):
                assert all(torch.equal(v, labels[kind][sid][k]) for k, v in expected.items())
            if role == 'train':
                expected = event_targets(times, shuffled.get(sid, []), counts=counts)
                assert all(torch.equal(v, labels['shuffled'][sid][k]) for k, v in expected.items())
    assert set(labels['shuffled']) == set(roles['train'])
    for epoch in (0, 1, 2):
        report = read(f'val_epoch{epoch}.json')
        for kind in KINDS:
            rows = report['per_shot'][kind]
            assert [r['shot'] for r in rows] == roles['val']
            for metric in ('risk', 'bag', 'local', 'direction'):
                expected = float(np.mean([np.mean([r[metric] for r in rows if r['flag'] == f])
                                          for f in (False, True)]))
                assert expected == report['summary'][kind][metric]
            for metric in ('direction_count', 'harmful_writes', 'event_points', 'event_loss',
                           'event_weight', 'event_correct_weight'):
                assert sum(r[metric] for r in rows) == report['summary'][kind][metric]
    for kind in KINDS:
        path = root/f'{kind}_epoch2.pt'
        assert sha(path) == done['checkpoints'][kind]
        a = torch.load(path, map_location='cpu', weights_only=True)
        validate_metadata(a, SimpleNamespace(feedback='progressive', readout='full'), roles)
        assert a['protocol_sha256'] == sha(root/'protocol.json')
        assert all(torch.isfinite(v).all() for v in a['memory'].values())
    result = dict(complete=True, source_unchanged=True, source_train=1353, source_val=160,
                  complete_sha256=sha(root/'complete.json'), verified_training=verified,
                  event_sha256=LABEL_SHA256, source_input_cache_labels_verified=1513,
                  family_counts=manifest['family_counts'],
                  scope='SOURCE_ONLY_NOT_EAST_EFFECTIVENESS')
    _durable_json(root/'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    audit(p.parse_args().root)
