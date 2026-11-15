"""Explicit provenance gate for locally trained source-only memory weights."""
from pathlib import Path

import torch

from .. import simple_data
from ..redesign_20261004.prepare_prior import PARENT_SHA256
from ..simple_train import sha
from .learning import validate_roles


def load_source_memory(model, path, *, roles=None):
    path = Path(path)
    roles = simple_data.source_roles() if roles is None else roles
    validate_roles(roles)
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact['source_sha256'] != PARENT_SHA256:
        raise ValueError('source checkpoint identity mismatch')
    if artifact['train_ids'] != roles['train'] or artifact['val_ids'] != roles['val']:
        raise ValueError('canonical TRAIN/VAL roster mismatch')
    if (artifact['epoch'] != 2 or artifact['readout'] != 'full' or model.readout != 'full'
            or artifact['kind'] not in ('fixed', 'learned')):
        raise ValueError('fixed final epoch2/full readout source artifact required')
    if not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in artifact['memory'].values()):
        raise ValueError('finite memory tensors required')
    model.memory.load_state_dict(artifact['memory'], strict=True)
    return dict(path=str(path.resolve()), sha256=sha(path), kind=artifact['kind'],
                epoch=artifact['epoch'], source_protocol_sha256=artifact['protocol_sha256'])
