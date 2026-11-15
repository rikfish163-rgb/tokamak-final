"""Explicit new checkpoint contract; no conversion of legacy source artifacts."""
import json
from pathlib import Path

import torch

from .. import simple_data
from ..redesign_20261004.prepare_prior import PARENT_SHA256
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .learning import PROFILE_KINDS, reference_policy, prior_policy


def validate_metadata(a, model, roles):
    validate_roles(roles)
    if a['kind'] not in PROFILE_KINDS.get(a['profile'], ()):
        raise ValueError('explicit direction teaching profile required')
    if (a['profile'] in ('reference_gradient_ablation_v2', 'zero_prior_ablation_v3')
            and a.get('reference_gradient') != reference_policy(a['kind'])):
        raise ValueError('explicit matching reference gradient policy required')
    if a['profile'] == 'zero_prior_ablation_v3':
        if a.get('prior_policy') != prior_policy(a['kind']):
            raise ValueError('explicit matching prior policy required')
        if a['kind'] == 'zero_prior' and (a['memory']['prior'].shape != (9, 92)
                                         or a['memory']['prior'].count_nonzero()):
            raise ValueError('fixed zero prior changed')
    if a['feedback'] != 'progressive' or model.feedback != 'progressive':
        raise ValueError('matched progressive training/deployment feedback required')
    if a['epoch'] != 2 or a['readout'] != 'full' or model.readout != 'full':
        raise ValueError('fixed epoch2/full readout required')
    if (a['source_sha256'] != PARENT_SHA256 or a['train_ids'] != roles['train']
            or a['val_ids'] != roles['val']):
        raise ValueError('source identity/roles changed')


def load_direction_memory(model, path):
    path = Path(path)
    a = torch.load(path, map_location='cpu', weights_only=True)
    validate_metadata(a, model, simple_data.source_roles())
    root = path.parent
    done = json.loads((root/'complete.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['profile'] != a['profile'] or tuple(protocol['kinds']) != PROFILE_KINDS[a['profile']]:
        raise ValueError('checkpoint and source protocol profile mismatch')
    if (a['profile'] in ('reference_gradient_ablation_v2', 'zero_prior_ablation_v3')
            and protocol.get('reference_gradient') != {k: reference_policy(k) for k in protocol['kinds']}):
        raise ValueError('source protocol reference gradient mismatch')
    if (a['profile'] == 'zero_prior_ablation_v3'
            and protocol.get('prior_policy') != {k: prior_policy(k) for k in protocol['kinds']}):
        raise ValueError('source protocol prior policy mismatch')
    if (not done['complete'] or not audit['complete'] or not audit['source_unchanged']
            or sha(path) != done['checkpoints'][a['kind']]
            or sha(root/'protocol.json') != a['protocol_sha256']
            or audit['complete_sha256'] != sha(root/'complete.json')):
        raise ValueError('completed audited source teaching receipt required')
    if not all(torch.isfinite(v).all() for v in a['memory'].values()):
        raise ValueError('nonfinite memory checkpoint')
    model.memory.load_state_dict(a['memory'], strict=True)
    receipt = dict(path=str(path.resolve()), sha256=sha(path), kind=a['kind'],
                profile=a['profile'], epoch=2, training_feedback='progressive',
                source_protocol_sha256=a['protocol_sha256'], audit_sha256=sha(root/'audit.json'))
    if a['profile'] in ('reference_gradient_ablation_v2', 'zero_prior_ablation_v3'):
        receipt['reference_gradient'] = a['reference_gradient']
    if a['profile'] == 'zero_prior_ablation_v3':
        receipt['prior_policy'] = a['prior_policy']
    return receipt
