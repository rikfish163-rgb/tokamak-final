"""Strict context8 source artifact contract; never reinterpret matrix memory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles

PROFILE = 'context8_v1'
FORECAST_PROFILE = 'context8_forecast_v2'
SCALED_PROFILE = 'context8_scaled_forecast_v3'
PARAMETERS = {'key.weight': (8, 46), 'key.bias': (8,), 'kernel': (9, 4, 23, 8),
              'rate': (), 'output.weight': (64, 92)}


def read(root, name):
    return json.loads((root / name).read_text())


def objective_weight(record):
    # The archived v1 schema predates forecast_weight. Its absent field means
    # zero only under the explicit v1/detached contract; v2 requires the field.
    if (record.get('profile') == PROFILE and record.get('objective') == 'detached'
            and record.get('forecast_weight', 0.) == 0.):
        return 0.
    if (record.get('profile') == FORECAST_PROFILE and record.get('objective') == 'detached_forecast'
            and record.get('forecast_weight') == 1.):
        return 1.
    if (record.get('profile') == SCALED_PROFILE and record.get('objective') == 'detached_scaled_forecast'
            and record.get('forecast_weight') == 1.):
        return 1.
    raise ValueError('explicit context objective profile and forecast weight required')


def validate_metadata(a, roles):
    validate_roles(roles)
    objective_weight(a)
    if (a.get('kind') != 'context8' or a.get('context_dim') != 8
            or a.get('epoch') != 2
            or a.get('feedback') != 'progressive' or a.get('readout') != 'full'
            or a.get('source_sha256') != PARENT_SHA256
            or a.get('train_ids') != roles['train'] or a.get('val_ids') != roles['val']):
        raise ValueError('canonical context8 source-only final-epoch artifact required')
    expected = set(PARAMETERS)
    if a['profile'] == SCALED_PROFILE:
        from .scaled import validate_weights
        expected.add('forecast_weights')
        validate_weights(a['memory']['forecast_weights'])
        digest = a.get('forecast_weights_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('sealed TRAIN scale provenance required')
    if set(a['memory']) != expected:
        raise ValueError('explicit context8 parameter layout required')
    if any(tuple(a['memory'][k].shape) != shape or not torch.isfinite(a['memory'][k]).all()
           for k, shape in PARAMETERS.items()):
        raise ValueError('finite context8 parameter tensors required')


def verify_gradients(rows, ids):
    for epoch in (1, 2):
        seen = [sid for r in rows if r['epoch'] == epoch for sid in r['ids']]
        if len(seen) != len(ids) or set(seen) != set(ids):
            raise ValueError('complete nonduplicated TRAIN epoch required')
    result = {}
    for name in PARAMETERS:
        values = [r['gradients'][name] for r in rows]
        if any(v is None or not np.isfinite(v) for v in values) or not any(v > 0 for v in values):
            raise ValueError('finite nontrivial context learning gradient required: ' + name)
        result[name] = sum(v > 0 for v in values)
    return result


def audit(root):
    p, done = read(root, 'protocol.json'), read(root, 'complete.json')
    roles = simple_data.source_roles()
    validate_roles(roles)
    objective_weight(p)
    if (not done['complete'] or not done['source_unchanged'] or done['target_or_test_signals_read'] != 0
            or p['train_ids'] != roles['train'] or p['val_ids'] != roles['val']):
        raise ValueError('complete source-only context teaching required')
    assert sha(DEFAULT_PARENT) == p['source_sha256'] == PARENT_SHA256
    with zipfile.ZipFile(root / 'code_snapshot.zip') as archive:
        for filename, digest in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == digest
    reference = Path(p['reference_root'])
    for name, digest in p['reference_hashes'].items():
        assert sha(reference / name) == digest
    cache = Path(p['cache_root'])
    assert sha(cache / 'inventory.json') == p['cache_inventory_sha256']
    inventory = read(cache, 'inventory.json')
    assert len(inventory) == len({r['shot'] for r in inventory}) == 1513
    for role in ('train', 'val'):
        assert {r['shot'] for r in inventory if r['role'] == role} == set(roles[role])
    for r in inventory:
        assert sha(cache / 'cache' / f"{r['shot']}.pt") == r['cache_sha256']
        assert sha(io23.path_for('J-TEXT', r['shot'])) == r['data_sha256']
    training = read(root, 'training.json')
    gradients = verify_gradients(training, roles['train'])
    for epoch in (1, 2):
        assert read(root, f'order_epoch{epoch}.json') == [s for r in training if r['epoch'] == epoch for s in r['ids']]
    path = root / 'context8_epoch2.pt'
    assert sha(path) == done['checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    validate_metadata(checkpoint, roles)
    assert checkpoint['profile'] == p['profile'] and objective_weight(checkpoint) == objective_weight(p)
    assert checkpoint['protocol_sha256'] == sha(root / 'protocol.json')
    if p['profile'] == SCALED_PROFILE:
        from .scaled import estimate
        weights, estimated = estimate(cache, roles['train'])
        assert read(root, 'forecast_weights.json') == estimated
        assert p['forecast_weights_sha256'] == checkpoint['forecast_weights_sha256'] == sha(root/'forecast_weights.json')
        assert torch.equal(checkpoint['memory']['forecast_weights'], weights)
    for epoch in (0, 1, 2):
        v = read(root, f'val_epoch{epoch}.json')
        assert [r['shot'] for r in v['per_shot']] == roles['val']
        for metric in ('bag', 'local', 'direction'):
            expected = np.mean([np.mean([r[metric] for r in v['per_shot'] if r['flag'] == flag])
                                for flag in (False, True)])
            assert float(expected) == v['summary'][metric]
    result = dict(complete=True, source_unchanged=True, source_train=1353, source_val=160,
                  source_cache_verified=1513, complete_sha256=sha(root / 'complete.json'),
                  nonzero_gradient_steps=gradients, scope='SOURCE_ONLY_NOT_EAST_EFFECTIVENESS')
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    _durable_json(root / 'audit.json', result)
    print(json.dumps(result), flush=True)


def load_context_memory(model, path):
    path = Path(path)
    a = torch.load(path, map_location='cpu', weights_only=True)
    validate_metadata(a, simple_data.source_roles())
    root = path.parent
    p, done, checked = read(root, 'protocol.json'), read(root, 'complete.json'), read(root, 'audit.json')
    if (p['profile'] != a['profile'] or objective_weight(p) != objective_weight(a)
            or not done['complete'] or not checked['complete']
            or not checked['source_unchanged'] or sha(path) != done['checkpoint_sha256']
            or sha(root / 'complete.json') != checked['complete_sha256']
            or sha(root / 'protocol.json') != a['protocol_sha256']):
        raise ValueError('audited context8 source checkpoint required')
    if a['profile'] == SCALED_PROFILE:
        from .scaled import ScaledContextMemory
        if (sha(root/'forecast_weights.json') != a['forecast_weights_sha256']
                or p['forecast_weights_sha256'] != a['forecast_weights_sha256']
                or read(root, 'forecast_weights.json')['train_ids'] != p['train_ids']):
            raise ValueError('immutable source TRAIN forecast scales required')
        old = model.memory
        # Loading a new representation must not perturb downstream RNG state.
        with torch.random.fork_rng(devices=[]):
            replacement = ScaledContextMemory(64, a['memory']['forecast_weights'])
        replacement.to(next(old.parameters()).device).train(old.training)
        for name, parameter in replacement.named_parameters():
            parameter.requires_grad_(dict(old.named_parameters())[name].requires_grad)
        model.memory = replacement
    model.memory.load_state_dict(a['memory'], strict=True)
    return dict(path=str(path.resolve()), sha256=sha(path), profile=a['profile'], kind='context8', epoch=2,
                training_feedback='progressive', context_dim=8,
                source_protocol_sha256=a['protocol_sha256'], audit_sha256=sha(root / 'audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
