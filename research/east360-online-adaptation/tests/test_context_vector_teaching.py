import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030.teaching import (
    context_controls, stratified_folds, teaching_mask,
)


def test_early_disruptive_rows_are_not_labeled_safe():
    records = [dict(disruptive=True, point_label=0.),
               dict(disruptive=True, point_label=1.),
               dict(disruptive=False, point_label=0.)]
    np.testing.assert_array_equal(teaching_mask(records), [False, True, True])


def test_three_folds_cover_all_shots_and_keep_one_safe_each():
    ids = list(range(20))
    labels = {sid: sid < 17 for sid in ids}
    folds = stratified_folds(ids, labels)
    assert len(folds) == 3
    assert sorted(s for f in folds for s in f['held']) == ids
    for fold in folds:
        assert set(fold['fit']).isdisjoint(fold['held'])
        assert len(fold['fit']) + len(fold['held']) == 20
        assert sum(not labels[s] for s in fold['held']) == 1
        assert sum(not labels[s] for s in fold['fit']) == 2
    with pytest.raises(ValueError):
        stratified_folds(ids, {sid: True for sid in ids})


def test_controls_only_reverse_arrived_context_not_origin():
    context = torch.arange(2 * 358, dtype=torch.float32).reshape(2, 358)
    mean = torch.zeros(358)
    outputs = context_controls(context, mean)
    assert set(outputs) == {'real', 'reverse', 'mean_feedback', 'constant'}
    assert torch.equal(outputs['real'], context)
    assert torch.equal(outputs['reverse'][:, :128], context[:, :128])
    assert torch.equal(outputs['reverse'][:, 128:].reshape(2, 5, 46),
                       context[:, 128:].reshape(2, 5, 46).flip(1))
    assert torch.equal(outputs['mean_feedback'][:, :128], context[:, :128])
    assert torch.count_nonzero(outputs['mean_feedback'][:, 128:]) == 0
    assert torch.count_nonzero(outputs['constant']) == 0
    outputs['reverse'][0, 0] = -10
    assert context[0, 0] == 0  # No in-place mutation of the observed history.
