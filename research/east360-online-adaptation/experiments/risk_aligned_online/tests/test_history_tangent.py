import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.history_tangent_20261023.learning import (
    basis_from_rows, permute_basis, project, matched_step, predict, ARMS,
)


def test_basis_dependent_rows_and_zero_rejected():
    rows = torch.tensor([[1., 2., 0.], [2., 4., 0.], [0., 0., 3.]])
    q = basis_from_rows(rows)
    assert q.shape == (2, 3)
    torch.testing.assert_close(q @ q.T, torch.eye(2), atol=1e-6, rtol=1e-6)
    for invalid in (torch.zeros(2, 3), torch.tensor([[float('nan'), 1.]])):
        with pytest.raises(ValueError):
            basis_from_rows(invalid)


def test_projection_orthogonal_and_nonexpansive():
    q = basis_from_rows(torch.tensor([[1., 2., 3.], [2., 1., 0.]]))
    g = torch.tensor([2., 4., -3.])
    v = project(g, q)
    torch.testing.assert_close(q @ v, torch.zeros(2), atol=1e-6, rtol=0)
    assert v.norm() <= g.norm()
    torch.testing.assert_close(project(v, q), v)
    torch.testing.assert_close(g @ v, v @ v)


def test_permutation_preserves_gram_and_tensor_partition():
    q = basis_from_rows(torch.arange(21.).reshape(3, 7) + torch.eye(3, 7))
    params = {'a': torch.ones(3), 'b': torch.ones(2, 2)}
    p = permute_basis(q, params)
    torch.testing.assert_close(p @ p.T, q @ q.T)
    torch.testing.assert_close(p, permute_basis(q, params))
    for lo, hi in ((0, 3), (3, 7)):
        torch.testing.assert_close(p[:, lo:hi].sort().values, q[:, lo:hi].sort().values)


def test_same_target_length_and_zero_no_fallback():
    g = torch.tensor([2., -4., 3.])
    for v in (g, g.flip(0)):
        step = matched_step(v, .0007)
        assert abs(float(step.norm()) - .0007) < 1e-9
    assert not matched_step(torch.zeros(3), .0007).any()
    assert not matched_step(g, 0.).any()
    with pytest.raises(ValueError):
        matched_step(g, -1.)


def tiny_model():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    return small_model()


def test_signal_causality_reset_and_frozen_parent():
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model = tiny_model()
    old = copy.deepcopy(model.state_dict())
    fast = initial_encoder(model)
    d = sum(v.numel() for v in fast.values())
    q = basis_from_rows(torch.randn(2, d))
    bases = {'history': q, 'permuted': permute_basis(q, fast)}
    rng = np.random.default_rng(131)
    signal = {'values': rng.normal(size=(420, 23)).astype('float32'),
              'available': np.ones((420, 23), dtype=bool)}
    curves, receipts, _ = predict(model, signal, bases, device='cpu')
    changed = copy.deepcopy(signal); changed['values'][300:] += 100
    later, _, _ = predict(model, changed, bases, device='cpu')
    repeat, _, _ = predict(model, signal, bases, device='cpu')
    for a in ARMS:
        np.testing.assert_array_equal(curves[a][:300], later[a][:300])
        np.testing.assert_array_equal(curves[a], repeat[a])
        np.testing.assert_array_equal(curves[a][:256], curves['history'][:256])
        assert len(receipts[a]) == 2
        for r, h in zip(receipts[a], receipts['history']):
            assert r['first_affected_prediction'] > r['last_observation']
            assert abs(r['requested_step_norm'] - h['requested_step_norm']) < 1e-12
            assert abs(r['parameter_delta_norm'] - h['parameter_delta_norm']) < 2e-7
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, old[k], atol=0, rtol=0)
    with pytest.raises(ValueError):
        predict(model, dict(signal, outcome=1), bases, device='cpu')


def test_missing_targets_are_not_zero_truth():
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model = tiny_model(); params = initial_encoder(model)
    q = basis_from_rows(torch.randn(2, sum(v.numel() for v in params.values())))
    signal = {'values': np.zeros((300, 23), dtype='float32'),
              'available': np.zeros((300, 23), dtype=bool)}
    _, receipts, _ = predict(model, signal, {'history': q, 'permuted': q}, device='cpu')
    for a in ARMS:
        assert receipts[a][0]['parameter_delta_norm'] == 0
        assert receipts[a][0]['gradient_steps'] == 0
