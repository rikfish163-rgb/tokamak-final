import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.progressive_feedback_20261026.model import ProgressiveDiagnosticMemory
from experiments.risk_aligned_online.update_direction_20261026.analysis import (
    select_endpoints, selected_states, channel_read_ablation, direction_summary,
)


def test_endpoint_selection_strict_deadline_and_no_cherry_picking():
    times = np.array([0., .100, .170, .200])
    scores = np.array([.7, .6, .99, 1.])
    rows = select_endpoints(times, scores, True, .200)
    assert rows == [{'end': 0, 'roles': ['primary']}, {'end': 1, 'roles': ['late125']}]
    assert select_endpoints(times, scores, False, None) == [{'end': 3, 'roles': ['primary']}]


def test_endpoint_deduplication_and_absent_window():
    rows = select_endpoints(np.array([0., .1]), np.array([.2, .3]), True, .2)
    assert rows == [{'end': 1, 'roles': ['primary', 'late125']}]
    assert select_endpoints(np.array([.18, .2]), np.array([.2, .3]), True, .2) == []


@pytest.mark.parametrize('missing', [False, True])
def test_reconstructed_states_equal_production_and_causal_prefix(missing):
    torch.manual_seed(41)
    memory = ProgressiveDiagnosticMemory(64).double()
    means = torch.randn(13, 23, dtype=torch.float64) * 5
    complete = torch.ones_like(means, dtype=torch.bool)
    if missing:
        complete[:, 2] = False
        complete[4:7] = False
    with torch.no_grad():
        memory.prior.normal_(std=.03)
        trace = memory.trace(means, complete, mode='memory')
    before = {k: v.clone() for k, v in memory.state_dict().items()}
    states = selected_states(memory, means, complete, [0, 1, 4, 6, 12])
    for j, row in states.items():
        assert torch.equal(row['key'] @ (row['after'] - memory.prior), trace['delta'][j])
        prefix = selected_states(memory, means[:j+1], complete[:j+1], [j])[j]
        assert torch.equal(prefix['after'], row['after'])
        assert (row['after'] - memory.prior).norm() <= 1 + 1e-12
        assert torch.isfinite(row['before']).all()
    assert all(torch.equal(before[k], v) for k, v in memory.state_dict().items())
    with pytest.raises(ValueError, match='patch'):
        selected_states(memory, means, complete, [13])


def test_channel_ablation_removes_only_that_diagnostic_at_four_horizons():
    delta = torch.arange(92.).reshape(1, 92)
    result = channel_read_ablation(delta)
    assert result.shape == (23, 92)
    for channel in range(23):
        expected = delta.clone().reshape(4, 23)
        expected[:, channel] = 0
        assert torch.equal(result[channel], expected.flatten())


def test_direction_sign_and_zero_step_not_fabricated():
    gradient = torch.tensor([1., -2.])
    good = direction_summary(gradient, -gradient)
    assert good['first_order_loss_change'] == -5
    assert good['descent_cosine'] == pytest.approx(1.)
    none = direction_summary(gradient, torch.zeros(2))
    assert none['descent_cosine'] is None
    assert none['step_norm'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='checks CUDA linear batch-shape arithmetic')
def test_cuda_state_reconstruction_preserves_production_key_batch():
    torch.manual_seed(131)
    memory = ProgressiveDiagnosticMemory(64).cuda().requires_grad_(False)
    means = torch.randn(511, 23, device='cuda')
    complete = torch.ones_like(means, dtype=torch.bool)
    with torch.no_grad():
        memory.prior.normal_(std=.03)
        trace = memory.trace(means, complete, mode='memory')
        state = selected_states(memory, means, complete, [374])[374]
        assert torch.equal(state['key'] @ (state['after']-memory.prior), trace['delta'][374])
        changed = means.clone()
        changed[375:] += 100
        future = selected_states(memory, changed, complete, [374])[374]
        assert torch.equal(state['after'], future['after'])
        assert torch.equal(state['key'], future['key'])


def test_posthoc_measurement_sign_matches_actual_loss_without_model_updates():
    from types import SimpleNamespace
    from experiments.risk_aligned_online.update_direction_20261026.run import measure

    prior = torch.zeros(9, 92)
    model = SimpleNamespace(memory=SimpleNamespace(prior=prior),
                            risk=lambda h, v, delta: delta[:, 0])
    key = torch.zeros(9)
    key[0] = 1
    after = prior.clone()
    after[0, 0] = -.1
    state = dict(before=prior.clone(), after=after, key=key,
                 signal_residual=dict(before=1., after=.9), observed_target_scalars=1)
    delta = torch.zeros(32, 92)
    delta[5, 0] = -.1
    result = measure(model, torch.zeros(32, 2, 3), torch.ones(32, 2, dtype=torch.bool),
                     delta, 5, state, True, .5)
    assert result['first_order_loss_change'] == pytest.approx(.05)
    assert result['actual_loss_change'] > 0
    assert result['descent_cosine'] == pytest.approx(-1.)
    assert result['channel_removal_probability_effect'][0] < 0
    assert all(v == 0 for v in result['channel_removal_probability_effect'][1:])
    assert torch.equal(prior, torch.zeros(9, 92))
    assert not after.requires_grad and not state['before'].requires_grad
