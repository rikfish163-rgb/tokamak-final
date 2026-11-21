import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ttt_training import pack_shots, balanced_risk_loss, score


def shot(n, positive=False):
    return dict(shot=n, device='J-TEXT', values=np.ones((n, 23), np.float32),
                available=np.ones((n, 23), bool), valid=np.ones(n, bool),
                y=np.asarray([0.] * (n-1) + [float(positive)], np.float32))


def test_pack_keeps_complete_prefix_and_masks_only_padding():
    first, second = shot(9, True), shot(5)
    second['valid'][-1] = False
    batch = pack_shots([first, second], 'cpu')
    assert batch['values'].shape == (2, 9, 23)
    assert batch['valid'].sum(1).tolist() == [9, 4]
    assert not batch['available'][1, 4:].any()
    assert batch['labels'][0, 8] == 1
    assert batch['values'][0, 0, 0] == 1


def test_pack_rejects_nonprefix_label_validity():
    s = shot(6)
    s['valid'][1] = False
    with pytest.raises(ValueError, match='prefix'):
        pack_shots([s], 'cpu')


def test_balanced_loss_ignores_padding_and_remains_defined_for_safe_only():
    labels = torch.tensor([[0., 0., 1., 0.]])
    valid = torch.tensor([[True, True, True, False]])
    logits = torch.zeros_like(labels, requires_grad=True)
    loss = balanced_risk_loss(logits, labels, valid)
    loss.backward()
    assert logits.grad[0, 3] == 0
    assert logits.grad[0, 2] < 0 and logits.grad[0, 0] > 0
    labels = torch.zeros_like(labels)
    assert torch.isfinite(balanced_risk_loss(logits, labels, valid))
    altered = logits.detach().clone()
    altered[:, 3] = 1000
    torch.testing.assert_close(balanced_risk_loss(altered, labels, valid),
                               balanced_risk_loss(logits, labels, valid))


def test_scoring_resets_every_shot_and_never_reads_after_terminal():
    class Probe(torch.nn.Module):
        def forward(self, values, available, *, state, adapt):
            assert state is None
            assert adapt
            assert values.max() == 1
            return dict(logits=values[:, :, 0] * 0)

    first, second = shot(9, True), shot(5)
    for item in (first, second):
        item.update(times=np.arange(len(item['y'])) / 1000,
                    is_disrupt=True, down=.01)
    first['valid'][-2:] = False
    first['values'][-2:] = 1000
    result = score(Probe(), [first, second], 'cpu', adapt=True)
    assert [len(row['scores']) for row in result] == [7, 5]
    np.testing.assert_array_equal(result[0]['times'], first['times'][:7])
