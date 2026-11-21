import pytest
import torch

from experiments.risk_aligned_online.event_feedback_geometry import gradient_geometry, measure_geometry
from experiments.risk_aligned_online.tests.test_event_dynamics import make_model


def test_projection_known_rank_deficient_span():
    jacobian = torch.tensor([[1., 0., 0.], [2., 0., 0.]])
    row = gradient_geometry(jacobian, torch.tensor([3., 4., 0.]), rtol=1e-4)
    assert row['rank'] == 1
    assert row['projected_gradient_fraction'] == pytest.approx(.6)
    assert row['projected_gradient_energy_fraction'] == pytest.approx(.36)
    assert row['query_gradient_norm'] == pytest.approx(5.)


def test_empty_span_and_zero_query_are_distinguished():
    row = gradient_geometry(torch.zeros(2, 3), torch.ones(3), rtol=1e-4)
    assert row['rank'] == 0 and row['projected_gradient_fraction'] == 0
    flat = gradient_geometry(torch.eye(3), torch.zeros(3), rtol=1e-4)
    assert flat['projected_gradient_fraction'] is None and flat['query_gradient_norm'] == 0
    with pytest.raises(ValueError):
        gradient_geometry(torch.ones(2, 3), torch.ones(2), rtol=1e-4)
    with pytest.raises(ValueError):
        gradient_geometry(torch.full((2, 3), float('nan')), torch.ones(3), rtol=1e-4)


def test_actual_feedback_jacobian_chain_rule_and_frozen_model():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        m = make_model().requires_grad_(False)
        initial = {k: v.clone() for k, v in m.state_dict().items()}
        x, truth, qx = torch.randn(1, 128, 23), torch.randn(1, 25, 23), torch.randn(1, 128, 23)
        a, b = torch.ones_like(x, dtype=torch.bool), torch.ones_like(truth, dtype=torch.bool)
        result = measure_geometry(m, x, a, truth, b, qx, a, m.initial_fast(1))
        assert result['state_coordinates'] == 25 and result['temporary_parameters'] == m.temporary_dim
        assert result['chain_rule_relative_error'] < 1e-4
        assert result['state_gradient_norm'] > 0
        for r in result['rank_sensitivity']:
            assert 0 <= r['rank'] <= 25
            assert 0 <= r['projected_gradient_fraction'] <= 1.000001
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, initial[k], rtol=0, atol=0)
        assert all(p.grad is None for p in m.parameters())
        assert not m.teacher.base.forecast_head._forward_pre_hooks
    finally:
        torch.set_num_threads(old)
