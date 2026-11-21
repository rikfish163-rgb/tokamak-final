import pytest
import torch

from experiments.risk_aligned_online.armijo_feedback_20261102.core import select_step


def test_oversized_first_step_is_reduced_using_the_same_matured_objective():
    state = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    def objective(x):
        return 50 * x.square().sum()

    before = objective(state)
    (gradient,) = torch.autograd.grad(before, state, create_graph=True)
    after, receipt = select_step(state, gradient, -gradient, before, objective)
    assert receipt["accepted"] and receipt["alpha"] == 1 / 64
    assert receipt["evaluations"] == 7
    assert receipt["loss_after"] < receipt["loss_before"]
    assert receipt["loss_after"] <= receipt["armijo_bound"]
    assert after.item() == -0.5625


def test_selected_step_keeps_the_auxiliary_parameter_gradient():
    state = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    def objective(x):
        return 0.5 * x.square().sum()

    before = objective(state)
    (gradient,) = torch.autograd.grad(before, state, create_graph=True)
    after, receipt = select_step(
        state, gradient, -coefficient * gradient, before, objective
    )
    assert receipt["accepted"] and receipt["alpha"] == 1
    (derivative,) = torch.autograd.grad(after.sum(), coefficient)
    assert derivative.item() == -1


def test_finite_online_budget_is_not_reported_as_stationarity():
    state = torch.tensor([1.0], requires_grad=True)
    gradient = torch.tensor([1.0])
    after, receipt = select_step(
        state, gradient, -gradient, torch.tensor(0.0), lambda x: torch.tensor(1.0)
    )
    assert torch.equal(after, state)
    assert not receipt["accepted"] and receipt["reason"] == "budget_exhausted"
    assert receipt["evaluations"] == 11 and receipt["alpha"] == 0


def test_zero_gradient_has_no_proposal_evaluation():
    state = torch.tensor([1.0], requires_grad=True)
    after, receipt = select_step(
        state,
        torch.zeros_like(state),
        torch.zeros_like(state),
        torch.tensor(0.0),
        lambda x: pytest.fail("no proposal"),
    )
    assert torch.equal(after, state) and receipt["evaluations"] == 0
    assert receipt["reason"] == "stationary"


@pytest.mark.parametrize("kind", ["nonfinite", "ascent"])
def test_invalid_descent_does_not_get_replaced_by_another_direction(kind):
    state = torch.tensor([1.0], requires_grad=True)
    gradient = torch.tensor([float("nan") if kind == "nonfinite" else 1.0])
    with pytest.raises((FloatingPointError, ValueError)):
        select_step(
            state,
            gradient,
            torch.ones_like(state),
            torch.tensor(1.0),
            lambda x: x.square().sum(),
        )


def test_nonfinite_proposal_loss_raises_instead_of_accepting_another_step():
    state = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(FloatingPointError):
        select_step(
            state,
            torch.ones_like(state),
            -torch.ones_like(state),
            torch.tensor(1.0),
            lambda x: torch.tensor(float("nan")),
        )
