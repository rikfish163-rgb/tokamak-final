"""Signal-only projection must preserve access boundaries and source predictions."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_gradient_projection import (
    EventProtectedFfnModel, make_directions, maturity_window, project_gradient,
)
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.redesign_20261004.test_intra import (
    MockContext, OpaqueOutcomeShot, PrefixData, shot,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_projection_is_orthogonal_nonexpansive_and_idempotent():
    g = torch.tensor([[2., 3., 4.]])
    j = torch.tensor([[1., 0., 0.], [2., 0., 0.]])
    projected, receipt = project_gradient(g, j)
    torch.testing.assert_close(projected, torch.tensor([[0., 3., 4.]]))
    again, _ = project_gradient(projected, j)
    torch.testing.assert_close(again, projected)
    assert receipt['rank'] == 1 and projected.norm() <= g.norm()
    torch.testing.assert_close(j @ projected[0], torch.zeros(2))


def test_zero_jacobian_is_identity_and_full_rank_is_zero_not_arbitrary_direction():
    g = torch.tensor([[2., -1.]])
    same, receipt = project_gradient(g, torch.zeros(2, 2))
    torch.testing.assert_close(same, g, rtol=0, atol=0)
    assert receipt['rank'] == 0
    zero, _ = project_gradient(g, torch.eye(2))
    assert not zero.any()


@pytest.mark.parametrize('g,j', [
    (torch.tensor([[float('nan'), 1.]]), torch.eye(2)),
    (torch.ones(1, 2), torch.ones(2, 3)),
    (torch.ones(2, 2), torch.eye(2)),
])
def test_invalid_projection_inputs_are_not_silently_replaced(g, j):
    with pytest.raises(ValueError):
        project_gradient(g, j)


def test_event_basis_spans_logit_contrasts_and_random_control_has_same_rank():
    torch.manual_seed(44)
    weight, scale = torch.randn(6, 192), torch.rand(192) + .5
    rng = torch.random.get_rng_state().clone()
    event = make_directions(weight, scale, mode='event', seed=131)
    random = make_directions(weight, scale, mode='random', seed=131)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert event.shape == random.shape == (5, 192)
    for basis in (event, random):
        torch.testing.assert_close(basis @ basis.T, torch.eye(5), atol=1e-6, rtol=1e-6)
    centered = (weight - weight.mean(0)) / scale
    torch.testing.assert_close(centered @ event.T @ event, centered, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(event, random)


def test_maturity_context_contains_only_last128_arrived_points():
    x = torch.arange(128.)[None, :, None].expand(1, 128, 23).clone()
    truth = torch.arange(128., 153.)[None, :, None].expand(1, 25, 23).clone()
    a, observed = torch.ones_like(x, dtype=torch.bool), torch.ones_like(truth, dtype=torch.bool)
    observed[:, -1, 0] = False
    values, mask = maturity_window(x, a, truth, observed)
    torch.testing.assert_close(values[0, :, 1], torch.arange(25., 153.))
    assert not mask[0, -1, 0] and mask[0, -1, 1]


def make_model():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), torch.zeros(23), torch.ones(23))
    basis = make_directions(torch.randn(6, 192), torch.ones(192), mode='event', seed=131)
    return EventProtectedFfnModel(base, basis, mode='event').eval()


def test_probe_hook_does_not_change_predictions_and_jacobian_is_real():
    m = make_model()
    control = DecoderFfnModel(deepcopy(m.base)).eval()
    x = torch.randn(1, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    state = {k: v.clone() for k, v in m.state_dict().items()}
    original = control(x, available, fast)
    coordinate, risk, jacobian, risk_grad = m.state_jacobian(x, available, fast.detach())
    actual = m(x, available, fast)
    for k in original:
        torch.testing.assert_close(actual[k], original[k], rtol=0, atol=0)
    assert jacobian.shape == (5, m.temporary_dim) and jacobian.norm() > 0
    assert risk_grad.shape == fast.shape and risk_grad.norm() > 0
    unit = jacobian[0] / jacobian[0].norm()
    epsilon = 1e-3
    plus, _ = m.state_values(x, available, fast + epsilon * unit)
    minus, _ = m.state_values(x, available, fast - epsilon * unit)
    difference = (plus[0, 0] - minus[0, 0]) / (2 * epsilon)
    torch.testing.assert_close(difference, jacobian[0] @ unit, atol=1e-3, rtol=2e-3)
    assert not m.base.forecast_head._forward_pre_hooks
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)


class RecordingProjection(MockContext):
    def __init__(self):
        super().__init__()
        self.projection_calls = []

    def project_intra_gradient(self, gradient, *, values, observed, truth, available, fast, step, radius):
        self.projection_calls.append(truth.detach().clone())
        return gradient, dict(rank=0, identity_test=True)


def test_runner_hook_sees_only_matured_truth_without_outcomes_or_rewriting_scores():
    m = RecordingProjection()
    stream = OpaqueOutcomeShot(shot(80))
    scores, audit = predict_shot(m, stream, intra=True, data_api=PrefixData())
    expected, _ = predict_shot(MockContext(), stream, intra=True, data_api=PrefixData())
    np.testing.assert_array_equal(scores, expected)
    assert len(m.projection_calls) == 1
    torch.testing.assert_close(m.projection_calls[0], torch.ones(1, 25, 23))
    row = audit['attempts'][0]
    assert row['gradient_projection']['identity_test']
    assert row['predictions_sealed_through_index'] == 25 and row['effective_index'] == 26


def test_projection_stream_is_prefix_causal_and_discards_fast_state():
    m = make_model()
    rng = np.random.default_rng(9)
    stream = OpaqueOutcomeShot(shot(80))
    stream['values'] = rng.normal(size=(80, 23)).astype('float32')
    changed = deepcopy(stream)
    changed['values'][50:] = 10000.
    scores, audit = predict_shot(m, stream, intra=True, data_api=PrefixData())
    other, _ = predict_shot(m, changed, intra=True, data_api=PrefixData())
    np.testing.assert_array_equal(scores[:50], other[:50])
    repeat, _ = predict_shot(m, stream, intra=True, data_api=PrefixData())
    np.testing.assert_array_equal(scores, repeat)
    assert audit['fast_discarded'] and not m.memory.any()
    assert audit['attempts'][0]['gradient_projection']['context'] == 'latest fully arrived window at maturity'
