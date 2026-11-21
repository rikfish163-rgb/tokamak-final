import torch

from experiments.risk_aligned_online.event_teaching_20261026.mechanism_audit import (
    reference_only_step, projection_summary,
)
from experiments.risk_aligned_online.event_teaching_20261026.reference_gradient_probe import gradient_geometry


def test_relative_objective_can_improve_by_worsening_reference_only():
    for label in (0., 1.):
        r = reference_only_step(label)
        assert r['after_loss_change'] == 0
        assert r['reference_loss_change'] > 0
        assert r['direction_loss_change'] < 0
        assert r['deployed_model_modified'] is False


def test_projection_summary_distinguishes_repeated_and_orthogonal_horizons():
    repeated = torch.eye(23, dtype=torch.double).repeat(1, 4)
    r = projection_summary(repeated)
    assert r['largest_energy_fraction'] < .05
    assert all(abs(v-1) < 1e-12 for v in r['horizon_cosines'])
    orthogonal = projection_summary(torch.eye(92, dtype=torch.double))
    assert all(abs(v) < 1e-12 for v in orthogonal['horizon_cosines'])


def test_reference_geometry_reports_cancellation_without_mutating_parameters():
    p = torch.tensor([.2, -.2], dtype=torch.double, requires_grad=True)
    a = torch.nn.functional.binary_cross_entropy_with_logits(p, torch.ones(2, dtype=torch.double), reduction='none')
    # At identical outputs, the relative objective cancels its own gradients.
    r = gradient_geometry(a, a, (p,))
    assert r['after_gradient_norm'] > 0
    assert r['reference_gradient_norm'] > 0
    assert r['net_gradient_norm'] == 0
    assert r['cancellation_ratio'] == 0
    assert r['net_dot_after_loss_gradient'] == 0
    assert p.grad is None
    assert torch.equal(p.detach(), torch.tensor([.2, -.2], dtype=torch.double))
