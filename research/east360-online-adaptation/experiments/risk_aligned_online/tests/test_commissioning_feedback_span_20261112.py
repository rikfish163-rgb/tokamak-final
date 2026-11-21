import torch

from experiments.risk_aligned_online import commissioning_feedback_span_20261112 as span
from experiments.risk_aligned_online import commissioning_group_token_20261112 as group


def test_matrix_is_the_production_group_exchange_and_preserves_patch_sum():
    torch.manual_seed(131)
    edges = torch.randn(2, 3, 4, 5)
    coeff = torch.randn(2, 3, 4)
    basis = span.edge_matrix(edges)
    change = (basis @ coeff.flatten(1)[..., None]).reshape(2, 4, 5)
    torch.testing.assert_close(change, group.exchange(coeff, edges))
    torch.testing.assert_close(change.sum(1), torch.zeros(2, 5))


def test_write_range_can_have_a_risk_direction_absent_from_feedback_range():
    write = torch.eye(3)[None]
    # The first direction can be controlled by two linearly dependent
    # auxiliary outputs, the second is writable but not reached by them.
    feedback = write @ torch.tensor([[[1., 2.], [0., 0.], [0., 0.]]])
    risk = torch.tensor([[1., 3., 0.]])
    a, rank_a = span.project(risk, write)
    b, rank_b = span.project(risk, feedback)
    torch.testing.assert_close(a, risk)
    torch.testing.assert_close(b, torch.tensor([[1., 0., 0.]]))
    assert rank_a.item() == 3 and rank_b.item() == 1
    zero, rank = span.project(risk, torch.zeros_like(feedback))
    assert not zero.any() and rank.item() == 0


def test_auxiliary_jacobian_reproduces_true_smooth_l1_gradient():
    torch.manual_seed(131)
    state = torch.randn(2, 3, 4, requires_grad=True)
    pred = (state.flatten(1) @ torch.randn(12, 20)).tanh().reshape(2, 5, 4)
    error = torch.randn_like(pred)
    force = (error/group.BETA).clamp(-1., 1.)
    want, = torch.autograd.grad(pred, state, grad_outputs=force, retain_graph=True)
    jac = span.feedback_jacobian(pred, state)
    torch.testing.assert_close((jac @ force.flatten(1)[..., None]).reshape_as(state), want)


def test_parallel_orthogonal_split_preserves_real_update_including_zero_sensitivity():
    real = torch.tensor([[4., 3.], [2., 7.]])
    sensitivity = torch.tensor([[2., 0.], [0., 0.]])
    parallel, orthogonal = span.split_update(real, sensitivity)
    torch.testing.assert_close(parallel, torch.tensor([[4., 0.], [0., 0.]]))
    torch.testing.assert_close((orthogonal*sensitivity).sum(1), torch.zeros(2))
    torch.testing.assert_close(parallel+orthogonal, real)
