"""Deploy only audited internal-memory weights, never the source event head."""
import json
from pathlib import Path

import torch

from .. import simple_data
from ..event_prefix_probe import LABEL_SHA256
from ..redesign_20261004.prepare_prior import PARENT_SHA256
from ..simple_train import sha
from ..source_task_memory_20261026.learning import validate_roles
from .learning import KINDS


def validate_metadata(a, model, roles):
    validate_roles(roles)
    if a['profile'] != 'event_teaching_v1' or a['kind'] not in KINDS:
        raise ValueError('explicit event teaching profile required')
    if a['feedback'] != 'progressive' or model.feedback != 'progressive':
        raise ValueError('matched progressive training/deployment feedback required')
    if a['epoch'] != 2 or a['readout'] != 'full' or model.readout != 'full':
        raise ValueError('fixed epoch2/full readout required')
    if (a['source_sha256'] != PARENT_SHA256 or a['train_ids'] != roles['train']
            or a['val_ids'] != roles['val'] or a['event_sha256'] != LABEL_SHA256):
        raise ValueError('source identity/roles/event annotation changed')


def load_event_memory(model, path):
    path = Path(path)
    a = torch.load(path, map_location='cpu', weights_only=True)
    validate_metadata(a, model, simple_data.source_roles())
    root = path.parent
    done = json.loads((root/'complete.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    if (not done['complete'] or not audit['complete'] or not audit['source_unchanged']
            or sha(path) != done['checkpoints'][a['kind']]
            or sha(root/'protocol.json') != a['protocol_sha256']
            or audit['complete_sha256'] != sha(root/'complete.json')
            or audit['event_sha256'] != a['event_sha256']):
        raise ValueError('completed audited source event teaching receipt required')
    if not all(torch.isfinite(v).all() for v in a['memory'].values()):
        raise ValueError('nonfinite memory checkpoint')
    model.memory.load_state_dict(a['memory'], strict=True)
    return dict(path=str(path.resolve()), sha256=sha(path), kind=a['kind'],
                profile=a['profile'], epoch=2, training_feedback='progressive',
                source_protocol_sha256=a['protocol_sha256'], audit_sha256=sha(root/'audit.json'))
