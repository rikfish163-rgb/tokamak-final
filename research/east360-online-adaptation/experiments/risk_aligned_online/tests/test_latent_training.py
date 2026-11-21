import numpy as np
import torch

from experiments.risk_aligned_online.latent_training import pack, metric_rows, selection_rank


def row(n, sid):
    return dict(shot=sid, features=np.ones((n, 4), np.float32),
                base_logits=np.zeros(n, np.float32), y=np.zeros(n, np.float32),
                observed=np.ones(n, bool), times=np.arange(n)/1000., is_disrupt=False,
                down=None)


def test_pack_masks_padding_and_preserves_signal_availability():
    one, two = row(6, 1), row(3, 2)
    two['observed'][1] = False
    batch = pack([one, two], 'cpu')
    assert batch['features'].shape == (2, 6, 4)
    assert batch['valid'].sum(1).tolist() == [6, 3]
    assert batch['observed'].sum(1).tolist() == [6, 2]
    assert not batch['observed'][1, 3:].any()


def test_selection_enforces_source_tpr_and_fpr_before_gain():
    valid = dict(tpr=.9, fpr=.1, J=.8)
    invalid = dict(tpr=.89, fpr=.01, J=.88)
    assert selection_rank(valid) < selection_rank(invalid)


def test_metric_rows_reset_state_and_use_same_base_path():
    class Identity(torch.nn.Module):
        def forward(self, features, base_logits, observed=None, state=None, adapt=True):
            assert state is None
            return {'logits': base_logits}
    rows = [row(6, 1), row(3, 2)]
    base = metric_rows(None, rows, 'cpu')
    adapted = metric_rows(Identity(), rows, 'cpu')
    for left, right in zip(base, adapted):
        np.testing.assert_array_equal(left['scores'], right['scores'])
