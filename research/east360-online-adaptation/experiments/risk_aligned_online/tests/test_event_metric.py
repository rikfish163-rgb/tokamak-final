import pytest
import torch

from experiments.risk_aligned_online.event_metric import EventMetricModel, metric_loss, source_step
from experiments.risk_aligned_online.tests.test_event_dynamics import make_model
from experiments.risk_aligned_online.redesign_20261004.intra import _attempt, _record


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_metric():
    original = make_model()
    return EventMetricModel(original.base, original.state_directions,
        original.feature_mean, original.feature_scale)


def test_identity_metric_equals_original_state_loss():
    torch.manual_seed(131)
    prediction, target = torch.randn(10, 5), torch.randn(10, 5)
    actual = metric_loss(prediction, target, torch.eye(5))
    torch.testing.assert_close(actual, torch.nn.functional.smooth_l1_loss(prediction, target), rtol=0, atol=0)


def test_metric_budget_and_invalid_values():
    m = make_metric()
    with torch.no_grad():
        m.metric.mul_(3)
    m.project_metric()
    torch.testing.assert_close(m.metric.square().sum(), torch.tensor(5.), rtol=1e-6, atol=1e-6)
    with torch.no_grad(): m.metric.zero_()
    with pytest.raises(ValueError): m.project_metric()
    with pytest.raises(ValueError): metric_loss(torch.zeros(2, 4), torch.zeros(2, 4), torch.eye(5))


def test_zero_fast_preserves_original_risk_and_direct_forecast():
    m = make_metric().requires_grad_(False)
    x = torch.randn(1, 128, 23); mask = torch.ones_like(x, dtype=torch.bool)
    original = make_model()
    original.load_state_dict({k: v for k, v in m.state_dict().items() if k != 'metric'})
    a, b = m(x, mask), original(x, mask)
    for key in ('logit', 'forecast'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_outer_risk_gradient_reaches_metric_not_frozen_source():
    torch.manual_seed(131)
    m = make_metric().requires_grad_(False)
    m.metric.requires_grad_(True)
    initial = {k: v.clone() for k, v in m.state_dict().items() if k != 'metric'}
    x, truth, qx = torch.randn(1, 128, 23), torch.randn(1, 25, 23), torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool); mask = torch.ones_like(truth, dtype=torch.bool)
    e = dict(x=x, a=a, truth=truth, mask=mask)
    fast, row, loss = source_step(m, e, m.initial_fast(1).requires_grad_(), create_graph=True)
    assert row['accepted']
    assert row['maturity_offset'] == 25 and row['earliest_query_offset'] == 26
    assert row['loss_after'] < row['loss_before']
    risk = torch.nn.functional.softplus(-m(qx, a, fast)['logit']).mean()
    risk.backward()
    assert m.metric.grad is not None and torch.isfinite(m.metric.grad).all() and m.metric.grad.norm() > 0
    assert all(p.grad is None for n, p in m.named_parameters() if n != 'metric')
    for k, v in m.state_dict().items():
        if k != 'metric': torch.testing.assert_close(v, initial[k], rtol=0, atol=0)


def test_targets_are_detached_and_receipt_has_no_outcome():
    m = make_metric()
    x = torch.randn(1, 128, 23); truth = torch.randn(1, 25, 23, requires_grad=True)
    a = torch.ones_like(x, dtype=torch.bool); mask = torch.ones_like(truth, dtype=torch.bool)
    pred = torch.randn(1, 25, 23, requires_grad=True)
    objective, receipt = m.prepare_matured_forecast_loss(pred, truth, mask, values=x, observed=a)
    objective(pred, truth, mask).backward()
    assert truth.grad is None and pred.grad is not None
    assert receipt['target_detached'] and not receipt['outcome_access']
    assert not receipt['predicted_signals_used_for_risk']


def test_source_training_step_matches_runtime_update_exactly():
    torch.manual_seed(131)
    m = make_metric().requires_grad_(False)
    values = torch.randn(128, 23).numpy()
    import numpy as np
    signals = dict(values=values, available=np.ones_like(values, dtype=bool))
    times = np.arange(128) / 1000.
    x = torch.as_tensor(values[:1]).expand(128, -1).clone()[None]
    a = torch.ones_like(x, dtype=torch.bool)
    truth = torch.as_tensor(values[1:26])[None]
    mask = torch.ones_like(truth, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    issued = m(x, a, fast)['forecast'].detach()
    pending = (_record(0, times, fast, m), x, a, issued)
    runtime, receipt = _attempt(m, pending, signals, times, fast, .05, 1., 'level', 'cpu')
    source, row, _ = source_step(m, dict(x=x, a=a, truth=truth, mask=mask), fast, create_graph=False)
    torch.testing.assert_close(source, runtime, rtol=0, atol=0)
    assert row['accepted'] == receipt['accepted']
