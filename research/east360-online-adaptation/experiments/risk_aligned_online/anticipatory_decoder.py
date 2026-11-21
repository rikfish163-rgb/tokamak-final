"""Reuse the archived TRAIN-only change decoder under its exact contract."""
import torch

from .persistent_ffn import PersistentFfnModel
from .decoder_intra_20261005.model import last_observation


class AnchoredPersistentFfn(PersistentFfnModel):
    def forward(self, values, available, fast=None):
        out = super().forward(values, available, fast)
        anchor, _ = last_observation(values, available, self.center, self.scale)
        return dict(logit=out['logit'], forecast=anchor[:, None]+out['forecast'])


def load_anchored_decoder(base, path, *, parent_sha256, train_ids):
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    protocol = artifact['protocol']
    train, fit, selection = (protocol.get(key, []) for key in ('source_train_ids', 'fit_ids', 'selection_ids'))
    if (protocol.get('parent_sha256') != parent_sha256
            or protocol.get('decoder_kind') != 'last_observation_plus_direct_increment'
            or any(protocol.get(key) is not True for key in ('source_only', 'source_classifier_frozen',
                'source_scaler_frozen', 'original_source_labels_unchanged', 'no_east_reads'))
            or len(train) != len(set(train)) or set(train) != set(train_ids)
            or not fit or not selection or set(fit) & set(selection)
            or len(fit) != len(set(fit)) or len(selection) != len(set(selection))
            or set(fit) | set(selection) != set(train)):
        raise ValueError('compatible source TRAIN-only anchored decoder required')
    head = artifact['forecast_head']
    expected = base.forecast_head.state_dict()
    if (set(head) != set(expected) or any(head[k].shape != expected[k].shape
            or head[k].dtype != expected[k].dtype or not torch.isfinite(head[k]).all() for k in expected)):
        raise ValueError('finite source decoder head with original shape/dtype required')
    base.forecast_head.load_state_dict(head, strict=True)
    return AnchoredPersistentFfn(base)

