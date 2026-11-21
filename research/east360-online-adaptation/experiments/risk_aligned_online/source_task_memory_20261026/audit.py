"""Independent inventory/gradient audit after source-only teaching completes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..simple_train import sha
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .learning import make_adapter, validate_roles


def verify_training(rows, train_ids):
    for epoch in (1, 2):
        seen = [sid for r in rows if r['epoch'] == epoch for sid in r['ids']]
        if len(seen) != len(train_ids) or set(seen) != set(train_ids):
            raise ValueError('complete nonduplicated TRAIN coverage required')
    for r in rows:
        fixed = r['arms']['fixed']['gradients']
        if any(v is not None for k, v in fixed.items() if k != 'output.weight'):
            raise ValueError('fixed control changed learning-rule parameters')
    active = {}
    for name in ('key.weight', 'prior', 'rate', 'output.weight'):
        vals = [r['arms']['learned']['gradients'][name] for r in rows]
        if any(v is None or not np.isfinite(v) for v in vals) or not any(v > 0 for v in vals):
            raise ValueError('actual finite rule gradient required: '+name)
        active[name] = sum(v > 0 for v in vals)
    return dict(complete_epochs=2, rule_gradient_steps=active, fixed_rule_gradients_absent=True)


def audit(root):
    read = lambda name: json.loads((root/name).read_text())
    done, protocol = read('complete.json'), read('protocol.json')
    if not done['complete'] or done['final_epoch'] != 2:
        raise ValueError('completed final two-epoch run required')
    roles = simple_data.source_roles(); validate_roles(roles)
    assert protocol['train_ids'] == roles['train'] and protocol['val_ids'] == roles['val']
    assert protocol['source_sha256'] == sha(DEFAULT_PARENT) == PARENT_SHA256
    with zipfile.ZipFile(root/'code_snapshot.zip') as z:
        for path, digest in protocol['code_sha256'].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == digest
    records = read('training.json'); gradients = verify_training(records, roles['train'])
    for epoch in (1, 2):
        assert read(f'order_epoch{epoch}.json') == [sid for r in records if r['epoch'] == epoch for sid in r['ids']]
    inventory = read('inventory.json')
    assert len(inventory) == 1513 and len({r['shot'] for r in inventory}) == 1513
    for role in ('train', 'val'):
        assert {r['shot'] for r in inventory if r['role'] == role} == set(roles[role])
    for row in inventory:
        assert sha(root/'cache'/f"{row['shot']}.pt") == row['cache_sha256']
        assert sha(io23.path_for('J-TEXT', row['shot'])) == row['data_sha256']
    validation = {}
    for epoch in (0, 1, 2):
        data = read(f'val_epoch{epoch}.json')
        for kind in ('fixed', 'learned'):
            rows = data['per_shot'][kind]
            assert [r['shot'] for r in rows] == roles['val']
            d, s = [float(np.mean([r['loss'] for r in rows if r['flag'] == f])) for f in (True, False)]
            assert data['summary'][kind] == dict(disrupt_loss=d, safe_loss=s, balanced_loss=(d+s)/2)
        validation[str(epoch)] = data['summary']
    torch.manual_seed(131)
    base, _ = load_parent(DEFAULT_PARENT)
    initial = make_adapter(base, 'learned').memory.state_dict()
    changes = {}
    for kind in ('fixed', 'learned'):
        path = root/f'{kind}_epoch2.pt'
        assert sha(path) == done['checkpoints'][kind]
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        assert artifact['train_ids'] == roles['train'] and artifact['val_ids'] == roles['val']
        changes[kind] = {k: float((artifact['memory'][k]-v).norm()) for k, v in initial.items()}
        if kind == 'fixed':
            assert all(v == 0 for k, v in changes[kind].items() if k != 'output.weight')
    output = dict(complete=True, source_unchanged=True, source_train=1353, source_val=160,
                  input_and_cache_hashes_verified=1513, gradients=gradients,
                  changes=changes, validation=validation, total_seconds=done['total_seconds'],
                  scope='SOURCE_SAMPLED_OBJECTIVE_NOT_EAST_EFFECTIVENESS')
    with (root/'audit.json').open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    audit(p.parse_args().root)
