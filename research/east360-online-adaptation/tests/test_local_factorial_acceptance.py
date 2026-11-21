"""Contract tests for the bounded, observed-loss Armijo search."""

import math

import pytest
import torch

from experiments.risk_aligned_online.local_factorial_20261029.acceptance import search


FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
RECEIPT_KEYS = {
    "trials",
    "factor",
    "reason",
    "loss_before",
    "loss_after",
    "proposed_norm",
    "accepted_norm",
    "directional_dot",
    "gradient_norm",
    "trial_losses",
    "proposed_directional_dot",
}


def assert_receipt_types(receipt):
    assert RECEIPT_KEYS <= receipt.keys()
    assert type(receipt["trials"]) is int
    assert isinstance(receipt["trial_losses"], list)
    assert len(receipt["trial_losses"]) == receipt["trials"]
    for key in RECEIPT_KEYS - {"trials", "trial_losses", "reason"}:
        assert type(receipt[key]) is float
        assert math.isfinite(receipt[key])
    assert all(
        type(loss) is float and math.isfinite(loss) for loss in receipt["trial_losses"]
    )


@pytest.mark.parametrize("power", [2, 4])
def test_overshoot_accepts_first_armijo_step_under_no_grad(power):
    proposal = torch.tensor([[-3.0, 0.0]], dtype=torch.float64, requires_grad=True)
    gradient = torch.tensor(
        [[float(power), 0.0]], dtype=torch.float64, requires_grad=True
    )
    calls = []

    def objective(trial):
        assert not torch.is_grad_enabled()
        assert not trial.requires_grad
        calls.append(trial.clone())
        return (1.0 + trial[0, 0]) ** power

    accepted, receipt = search(objective, proposal, gradient, torch.tensor(1.0))

    assert_receipt_types(receipt)
    assert receipt["reason"] == "accepted"
    assert receipt["trials"] == 2
    assert receipt["factor"] == 0.5
    assert torch.equal(calls[0], proposal.detach())
    assert torch.equal(calls[1], proposal.detach() * 0.5)
    assert torch.equal(accepted, proposal.detach() * 0.5)
    assert accepted.shape == proposal.shape and accepted.dtype == proposal.dtype
    assert accepted.device == proposal.device and not accepted.requires_grad
    assert receipt["trial_losses"] == [float(2**power), float(0.5**power)]
    assert receipt["loss_before"] == 1.0
    assert receipt["loss_after"] == float(0.5**power)
    assert receipt["proposed_norm"] == 3.0
    assert receipt["accepted_norm"] == 1.5
    assert receipt["gradient_norm"] == float(power)
    assert receipt["proposed_directional_dot"] == -3.0 * power
    assert receipt["directional_dot"] == -1.5 * power


def test_decrease_that_fails_armijo_is_rejected_before_later_acceptance():
    proposal = torch.tensor([-1.0], dtype=torch.float64)
    gradient = torch.tensor([100.0], dtype=torch.float64)

    def objective(trial):
        return 1.0 + 100.0 * trial[0] + 99.995 * trial[0] ** 2

    accepted, receipt = search(objective, proposal, gradient, 1.0)

    assert_receipt_types(receipt)
    assert receipt["trial_losses"][0] < 1.0
    assert receipt["trial_losses"][0] > 1.0 + 1e-4 * -100.0
    assert receipt["factor"] == 0.5
    assert receipt["trials"] == 2
    assert receipt["trial_losses"][1] <= 1.0 + 1e-4 * receipt["directional_dot"]
    assert torch.equal(accepted, proposal * 0.5)


@pytest.mark.parametrize(("index", "factor"), list(enumerate(FACTORS)))
def test_every_factor_can_be_the_first_accepted(index, factor):
    proposal = torch.tensor([-1.0], dtype=torch.float64)
    gradient = torch.tensor([1.0], dtype=torch.float64)
    curvature = 0.1 if index == 0 else 0.9999 / (1.5 * factor)

    def objective(trial):
        return 1.0 + trial[0] + curvature * trial[0] ** 2

    accepted, receipt = search(objective, proposal, gradient, 1.0)

    assert_receipt_types(receipt)
    assert receipt["reason"] == "accepted"
    assert receipt["trials"] == index + 1
    assert receipt["factor"] == factor
    assert torch.equal(accepted, proposal * factor)
    for rejected_factor, loss in zip(FACTORS, receipt["trial_losses"][:-1]):
        assert loss > 1.0 - 1e-4 * rejected_factor
    assert receipt["loss_after"] <= 1.0 + 1e-4 * receipt["directional_dot"]


def test_armijo_bound_is_inclusive_but_decrease_must_be_strict():
    proposal = torch.tensor([-1.0], dtype=torch.float64)
    gradient = torch.tensor([1.0], dtype=torch.float64)

    accepted, receipt = search(lambda trial: 1.0 - 1e-4, proposal, gradient, 1.0)

    assert_receipt_types(receipt)
    assert torch.equal(accepted, proposal)
    assert receipt["trials"] == 1
    assert receipt["factor"] == 1.0
    assert receipt["loss_after"] == 1.0 + 1e-4 * receipt["directional_dot"]


def test_eight_trial_budget_with_equal_loss_returns_exact_zero():
    proposal = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
    gradient = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    calls = []

    def objective(trial):
        calls.append(trial.clone())
        return 3.0

    accepted, receipt = search(objective, proposal, gradient, 3.0)

    assert_receipt_types(receipt)
    assert receipt["reason"] == "no_acceptable_step"
    assert receipt["trials"] == len(FACTORS)
    assert receipt["trial_losses"] == [3.0] * len(FACTORS)
    assert all(
        torch.equal(trial, proposal * factor) for trial, factor in zip(calls, FACTORS)
    )
    assert torch.equal(accepted, torch.zeros_like(proposal))
    assert receipt["factor"] == 0.0
    assert receipt["loss_before"] == receipt["loss_after"] == 3.0
    assert receipt["directional_dot"] == receipt["accepted_norm"] == 0.0
    assert receipt["proposed_directional_dot"] == -5.0


@pytest.mark.parametrize(
    ("proposal", "gradient", "reason", "proposed_dot"),
    [
        ([], [], "zero_direction", 0.0),
        ([0.0, 0.0], [1.0, 2.0], "zero_direction", 0.0),
        ([1.0, 2.0], [0.0, 0.0], "zero_direction", 0.0),
        ([0.0, 0.0], [0.0, 0.0], "zero_direction", 0.0),
        ([1.0, 2.0], [1.0, 2.0], "non_descent", 5.0),
        ([0.0, 1.0], [1.0, 0.0], "non_descent", 0.0),
    ],
)
def test_zero_or_non_descent_abstains_without_calling_objective(
    proposal, gradient, reason, proposed_dot
):
    proposal = torch.tensor([proposal], dtype=torch.float64, requires_grad=True)
    gradient = torch.tensor([gradient], dtype=torch.float64, requires_grad=True)

    def forbidden(_trial):
        pytest.fail("objective must not be called")

    accepted, receipt = search(forbidden, proposal, gradient, 4.0)

    assert_receipt_types(receipt)
    assert torch.equal(accepted, torch.zeros_like(proposal))
    assert accepted.shape == proposal.shape and not accepted.requires_grad
    assert receipt["reason"] == reason
    assert receipt["trials"] == 0 and receipt["trial_losses"] == []
    assert (
        receipt["factor"]
        == receipt["directional_dot"]
        == receipt["accepted_norm"]
        == 0.0
    )
    assert receipt["proposed_directional_dot"] == proposed_dot
    assert receipt["loss_before"] == receipt["loss_after"] == 4.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["proposal", "gradient", "loss_before"])
def test_nonfinite_inputs_raise_even_if_direction_would_otherwise_abstain(field, value):
    proposal = torch.tensor([0.0], dtype=torch.float64)
    gradient = torch.tensor([1.0], dtype=torch.float64)
    loss_before = 1.0
    if field == "proposal":
        proposal[0] = value
    elif field == "gradient":
        gradient[0] = value
    else:
        loss_before = torch.tensor(value)

    def forbidden(_trial):
        pytest.fail("objective must not be called on invalid inputs")

    with pytest.raises((ValueError, FloatingPointError)):
        search(forbidden, proposal, gradient, loss_before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("invalid_at", [1, 2])
def test_nonfinite_trial_loss_raises_immediately(value, invalid_at):
    calls = []

    def objective(trial):
        calls.append(trial.clone())
        return torch.tensor(value if len(calls) == invalid_at else 2.0)

    with pytest.raises((ValueError, FloatingPointError)):
        search(objective, torch.tensor([-1.0]), torch.tensor([1.0]), 1.0)
    assert len(calls) == invalid_at


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        search(lambda trial: 0.0, torch.ones(2), torch.ones(1), 1.0)


def test_subnormal_trials_are_still_recorded_in_factor_order():
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    proposal = -smallest.reshape(1)
    calls = []

    def objective(trial):
        calls.append(trial.clone())
        return 1.0

    accepted, receipt = search(objective, proposal, torch.ones_like(proposal), 1.0)

    assert_receipt_types(receipt)
    assert receipt["reason"] == "no_acceptable_step"
    assert receipt["trials"] == len(FACTORS)
    assert receipt["trial_losses"] == [1.0] * len(FACTORS)
    assert all(
        torch.equal(trial, proposal * factor) for trial, factor in zip(calls, FACTORS)
    )
    assert torch.equal(accepted, torch.zeros_like(proposal))
