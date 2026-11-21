import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.task_state_feedback import TaskStateFfnModel, task_basis
from experiments.risk_aligned_online.tests.test_event_dynamics import make_model
from experiments.risk_aligned_online.redesign_20261004 import intra


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_task(mode='event_risk'):
    m = make_model()
    return TaskStateFfnModel(m.base, m.state_directions, m.feature_mean, m.feature_scale,
                            mode=mode).requires_grad_(False)


@pytest.mark.parametrize('mode,n', [('risk', 1), ('event_risk', 6), ('random_risk', 6)])
def test_basis_is_orthonormal_and_starts_with_original_risk(mode, n):
    m = make_task(mode)
    assert m.state_directions.shape == (n, 192)
    d = m.state_directions
    torch.testing.assert_close(d @ d.T, torch.eye(n), rtol=2e-6, atol=2e-6)
    raw = torch.cat((m.base.risk_head.weight[0] * m.feature_scale[:64], torch.zeros(128)))
    torch.testing.assert_close(d[0], raw / raw.norm())


def test_invalid_or_degenerate_risk_is_not_replaced_with_a_random_direction():
    m = make_model()
    with pytest.raises(ValueError):
        task_basis(torch.zeros(1, 64), m.feature_scale, m.state_directions, mode='event_risk')
    with pytest.raises(ValueError):
        task_basis(m.base.risk_head.weight, m.feature_scale, m.state_directions, mode='wrong')


def test_first_coordinate_difference_is_exactly_scaled_frozen_logit_difference():
    m = make_task()
    x = torch.randn(2, 128, 23); mask = torch.ones_like(x, dtype=torch.bool)
    c = m.coordinates(x, mask)[:, 0]
    logits = m.teacher(x, mask)['logit']
    scale = (m.base.risk_head.weight[0] * m.feature_scale[:64]).norm()
    torch.testing.assert_close(c[1]-c[0], (logits[1]-logits[0])/scale, rtol=3e-5, atol=3e-6)


def test_direct_feedback_is_detached_from_future_targets_not_from_fast_parameters():
    m = make_task()
    original = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randn(1, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    truth = torch.randn(1, 25, 23, requires_grad=True); mask = torch.ones_like(truth, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    out = m(x, a, fast)
    loss, receipt = m.prepare_matured_forecast_loss(out['forecast'], truth, mask, values=x, observed=a)
    loss(out['forecast'], truth, mask).backward()
    assert fast.grad.norm() > 0 and truth.grad is None
    assert receipt['state_dimensions'] == 6 and receipt['risk_coordinate_count'] == 1
    assert not receipt['predicted_signals_used_for_risk'] and not receipt['outcome_access']
    assert not receipt['risk_teacher_is_physical_truth']
    for k, v in m.state_dict().items(): torch.testing.assert_close(v, original[k], rtol=0, atol=0)
    assert all(p.grad is None for p in m.parameters())


def test_zero_fast_leaves_parent_risk_and_forecast_exact():
    original = make_model().requires_grad_(False)
    m = TaskStateFfnModel(original.base, original.state_directions,
        original.feature_mean, original.feature_scale, mode='risk').requires_grad_(False)
    x = torch.randn(1, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    actual, expected = m(x, a), original(x, a)
    for k in ('logit', 'forecast'): torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)


def test_runtime_matures_all25_points_before_task_feedback_affects_risk():
    m = make_task('risk')
    values = np.random.default_rng(7).normal(size=(50, 23)).astype(np.float32)
    signals = dict(values=values, available=np.ones_like(values, dtype=bool))
    times = np.arange(50)/1000.
    x = torch.as_tensor(values[:1]).expand(128, -1).clone()[None]
    a = torch.ones_like(x, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    issued = m(x, a, fast)['forecast'].detach()
    pending = (intra._record(0, times, fast, m), x, a, issued)
    _, receipt = intra._attempt(m, pending, signals, times, fast, .05, 1., 'level', 'cpu')
    assert receipt['maturity_index'] == receipt['predictions_sealed_through_index'] == 25
    if receipt['accepted']: assert receipt['effective_index'] == 26
    assert receipt['matured_objective']['mode'] == 'risk'
