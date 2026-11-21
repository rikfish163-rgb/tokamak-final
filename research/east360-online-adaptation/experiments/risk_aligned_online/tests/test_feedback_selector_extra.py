"""Additional post-implementation checks; not edits to sealed run sources."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.tests.test_lag_bias_feedback import fixture
from experiments.risk_aligned_online.tests.test_feedback_selector import rows
from experiments.risk_aligned_online.feedback_selector_20261023.policy import fit_policy
from experiments.risk_aligned_online.feedback_selector_20261023.execution import predict_policy
from experiments.risk_aligned_online.feedback_selector_20261023 import run


def test_dynamic_policy_prefix_future_invariance_and_reset():
    model, signal = fixture()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    policy = fit_policy(rows(), allowed_ids=[1, 2, 3, 4])
    first, receipt = predict_policy(model, signal, policy, device='cpu')
    changed = {k: v.copy() for k, v in signal.items()}
    changed['values'][400:] = 1000.; changed['available'][400:] = False
    other, _ = predict_policy(model, changed, policy, device='cpu')
    repeated, receipt2 = predict_policy(model, signal, policy, device='cpu')
    for arm in first:
        np.testing.assert_array_equal(first[arm][:400], other[arm][:400])
        np.testing.assert_array_equal(first[arm], repeated[arm])
    assert receipt == receipt2
    for k, v in original.items(): torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)


def test_invalid_prefix_cannot_read_outcome_or_write_gate(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run.simple_data, 'attach_outcome', lambda *args, **kw: calls.append('outcome'))
    monkeypatch.setattr(run, '_durable_json', lambda *args, **kw: calls.append('write'))
    with pytest.raises(ValueError):
        run.prepare_policy(None, [], {}, tmp_path, [], list(range(360)))
    assert calls == []
