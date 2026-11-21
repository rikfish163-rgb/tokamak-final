import torch

from experiments.risk_aligned_online.predictive_risk_pilot_20261102.anchored import (
    AnchoredHead,
)
from experiments.risk_aligned_online.gate_loss_audit_20261101.core import centered_loss


def test_current_state_initialization_preserves_direct_temporal_shape_and_gradient():
    torch.manual_seed(131)
    head = torch.nn.Linear(3, 575, dtype=torch.float64)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    last = torch.randn(23, dtype=torch.float64)
    model = AnchoredHead(head, last)
    original = head(x).reshape(2, 25, 23)
    adjusted = model(x).reshape(2, 25, 23)
    torch.testing.assert_close(adjusted[:, 0], last.expand(2, 23), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        adjusted[:, 1:] - adjusted[:, :-1], original[:, 1:] - original[:, :-1]
    )
    target = torch.randn_like(original)
    mask = torch.ones_like(original, dtype=torch.bool)
    l0, l1 = (
        centered_loss(original, target, mask),
        centered_loss(adjusted, target, mask),
    )
    torch.testing.assert_close(l0, l1)
    g0 = torch.autograd.grad(l0, x, retain_graph=True)[0]
    g1 = torch.autograd.grad(l1, x)[0]
    torch.testing.assert_close(g0, g1, atol=1e-12, rtol=1e-12)
