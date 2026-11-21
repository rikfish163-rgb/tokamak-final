"""Verify source coverage, raw/cache hashes, actual gradients and saved summaries."""
import argparse
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
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..source_task_memory_20261026.learning import validate_roles, source_endpoints
from .learning import local_targets, PROFILE_KINDS, reference_policy, prior_policy
from .checkpoint import validate_metadata


def validate_training(rows, ids, kinds=('bag', 'direction')):
    if tuple(kinds) not in PROFILE_KINDS.values() or any(set(r['arms']) != set(kinds) for r in rows):
        raise ValueError('explicit matched source arms required')
    for epoch in (1, 2):
        seen = [sid for r in rows if r['epoch'] == epoch for sid in r['ids']]
        if len(seen) != len(ids) or set(seen) != set(ids):
            raise ValueError('nonduplicated full TRAIN coverage required')
    active, frozen = {}, {}
    for kind in kinds:
        active[kind] = {}
        for name in ('key.weight', 'prior', 'rate', 'output.weight'):
            v = [r['arms'][kind]['gradients'][name] for r in rows]
            if kind == 'zero_prior' and name == 'prior':
                if any(x is not None for x in v):
                    raise ValueError('frozen prior must have no gradient')
                frozen[kind] = ['prior']
                continue
            if any(x is None or not np.isfinite(x) for x in v) or not any(x > 0 for x in v):
                raise ValueError('finite nontrivial learning-rule gradient required')
            active[kind][name] = sum(x > 0 for x in v)
    result = dict(complete_epochs=2, nonzero_gradient_steps=active)
    if frozen:
        result['frozen_parameters'] = frozen
    return result


def audit(root):
    def read(name):
        return json.loads((root/name).read_text())
    p, done = read('protocol.json'), read('complete.json')
    kinds = PROFILE_KINDS.get(p['profile'])
    if kinds is None or tuple(p['kinds']) != kinds:
        raise ValueError('explicit matched source profile required')
    if p['profile'] in ('reference_gradient_ablation_v2', 'zero_prior_ablation_v3'):
        assert p['reference_gradient'] == {k: reference_policy(k) for k in kinds}
    if p['profile'] == 'zero_prior_ablation_v3':
        assert p['prior_policy'] == {k: prior_policy(k) for k in kinds}
    if not done['complete'] or done['final_epoch'] != 2 or done['target_or_test_signals_read'] != 0:
        raise ValueError('completed source-only teaching required')
    roles = simple_data.source_roles()
    validate_roles(roles)
    assert p['train_ids'] == roles['train'] and p['val_ids'] == roles['val']
    assert p['source_sha256'] == sha(DEFAULT_PARENT) == PARENT_SHA256
    cache = Path(p['cache_root'])
    assert sha(cache/'inventory.json') == p['cache_inventory_sha256']
    assert sha(cache/'audit.json') == p['cache_audit_sha256']
    with zipfile.ZipFile(root/'code_snapshot.zip') as archive:
        for path, digest in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    training = read('training.json')
    verified = validate_training(training, roles['train'], kinds)
    for epoch in (1, 2):
        assert read(f'order_epoch{epoch}.json') == [sid for r in training if r['epoch'] == epoch for sid in r['ids']]
    inventory = read('inventory.json')
    assert len(inventory) == len({r['shot'] for r in inventory}) == 1513
    for role in ('train', 'val'):
        assert {r['shot'] for r in inventory if r['role'] == role} == set(roles[role])
    assert sha(root/'local_targets.pt') == done['local_targets_sha256']
    labels = torch.load(root/'local_targets.pt', map_location='cpu', weights_only=True)
    advanced = simple_data.advanced_times()
    for row in inventory:
        sid = row['shot']
        assert sha(cache/'cache'/f'{sid}.pt') == row['cache_sha256']
        assert sha(io23.path_for('J-TEXT', sid)) == row['data_sha256']
        s = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        ends = source_endpoints(s)
        assert ends.tolist() == row['ends']
        expected = local_targets(s, ends)
        assert all(torch.equal(v, labels[sid][k]) for k, v in expected.items())
    for epoch in (0, 1, 2):
        report = read(f'val_epoch{epoch}.json')
        for kind in kinds:
            rows = report['per_shot'][kind]
            assert [r['shot'] for r in rows] == roles['val']
            for metric in ('bag', 'local', 'direction'):
                expected = float(np.mean([np.mean([r[metric] for r in rows if r['flag'] == f])
                                          for f in (False, True)]))
                assert expected == report['summary'][kind][metric]
            for metric in ('direction_count', 'harmful_writes'):
                assert sum(r[metric] for r in rows) == report['summary'][kind][metric]
    for kind in kinds:
        path = root/f'{kind}_epoch2.pt'
        assert sha(path) == done['checkpoints'][kind]
        a = torch.load(path, map_location='cpu', weights_only=True)
        validate_metadata(a, SimpleNamespace(feedback='progressive', readout='full'), roles)
        assert a['profile'] == p['profile'] and a['kind'] == kind
        assert a['protocol_sha256'] == sha(root/'protocol.json') and a['feedback'] == 'progressive'
        assert a['train_ids'] == roles['train'] and a['val_ids'] == roles['val']
    result = dict(complete=True, source_unchanged=True, source_train=1353, source_val=160,
                  complete_sha256=sha(root/'complete.json'), verified_training=verified,
                  source_input_cache_local_labels_verified=1513,
                  scope='SOURCE_ONLY_NOT_EAST_EFFECTIVENESS')
    if (root/'audit.json').exists():
        raise FileExistsError(root/'audit.json')
    _durable_json(root/'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    audit(p.parse_args().root)
