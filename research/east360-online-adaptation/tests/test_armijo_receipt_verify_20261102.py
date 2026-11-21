"""Synthetic CPU receipts only; no experiment outputs or raw HDF5 are read."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.armijo_feedback_20261102 import core
from experiments.risk_aligned_online.armijo_feedback_20261102.receipt_verify import verify_receipt


def attach_vectors(before, after, gradient, receipt):
    receipt = dict(
        receipt,
        gradient_norm=float(gradient.double().norm()),
        step_norm=float((after - before).double().norm()),
        gradient256=gradient.flatten().tolist(),
        state256_after=after.flatten().tolist(),
    )
    return before.numpy().copy(), after.numpy().copy(), gradient.numpy().copy(), receipt, 550


def accepted():
    before = torch.zeros(256)
    before[0] = 1
    gradient = 100 * before
    after, receipt = core.select_step(before, gradient, -gradient, torch.tensor(50.0), lambda x: 50 * x.square().sum())
    return attach_vectors(before, after, gradient, receipt)


def rejected(reason="budget_exhausted"):
    before = torch.ones(256)
    gradient = torch.zeros_like(before) if reason == "stationary" else torch.ones_like(before)
    direction = -gradient if reason == "budget_exhausted" else torch.zeros_like(before)
    after, receipt = core.select_step(before, gradient, direction, torch.tensor(0.0), lambda x: torch.tensor(1.0))
    return attach_vectors(before, after, gradient, receipt)


def no_target(placeholder):
    state, truth = torch.ones(4, 64), torch.zeros(1, 25, 23)
    after, receipt = core.propose(
        None, {}, state, truth, torch.zeros_like(truth, dtype=torch.bool), torch.zeros(25, 23)
    )
    gradient = np.zeros(256, np.float32) if placeholder else None
    return state.flatten().numpy(), after.flatten().numpy(), gradient, receipt, 0


SUMMARY_KEYS = {
    "events",
    "observed",
    "observed_scalars",
    "evaluations",
    "accepted",
    "reason",
    "alpha",
    "gradient_computed",
    "gradient_placeholder",
    "nonzero_gradients",
    "gradient_norm",
    "step_norm",
    "rejected_trials",
    "neural_loss_recomputed",
    "hidden_rms_verified",
}


@pytest.mark.parametrize("exported", [False, True])
def test_real_accepted_receipt_and_exported_serialization_contract(exported):
    before, after, gradient, receipt, observed = accepted()
    if exported:
        receipt.pop("gradient256")
        receipt.pop("state256_after")
        receipt.update(gradient_computed=True, observed=observed)
    original = deepcopy(receipt)
    summary = verify_receipt(before, after, gradient, receipt, observed)
    assert set(summary) == SUMMARY_KEYS and receipt == original
    assert summary["accepted"] is True and summary["evaluations"] == 7
    assert summary["alpha"] == 1 / 64 and summary["rejected_trials"] == 6
    assert summary["events"] == summary["nonzero_gradients"] == 1
    assert summary["observed"] == summary["observed_scalars"] == 550
    assert summary["gradient_computed"] and not summary["gradient_placeholder"]
    assert not summary["neural_loss_recomputed"] and not summary["hidden_rms_verified"]


@pytest.mark.parametrize(
    "reason,evaluations", [("budget_exhausted", 11), ("stationary", 0), ("no_effective_direction", 0)]
)
def test_real_rejection_reasons_remain_distinct(reason, evaluations):
    summary = verify_receipt(*rejected(reason))
    assert not summary["accepted"] and summary["reason"] == reason
    assert summary["evaluations"] == summary["rejected_trials"] == evaluations
    assert summary["gradient_computed"] and not summary["gradient_placeholder"]
    assert summary["nonzero_gradients"] == int(reason != "stationary")
    assert summary["step_norm"] == summary["alpha"] == 0


@pytest.mark.parametrize("placeholder", [False, True])
def test_no_target_absent_and_zero_sentinel_are_not_measured_zero_gradients(placeholder):
    summary = verify_receipt(*no_target(placeholder))
    assert not summary["gradient_computed"] and summary["gradient_placeholder"] is placeholder
    assert summary["reason"] == "no_target" and summary["observed"] == 0
    assert summary["evaluations"] == summary["nonzero_gradients"] == 0


def test_float32_retained_state_roundoff_is_not_mistaken_for_fabricated_bound():
    before, gradient, direction = torch.zeros(256), torch.zeros(256), torch.zeros(256)
    before[0], gradient[0], direction[0] = 1000.123, 2.4e-4, -0.037
    after, receipt = core.select_step(before, gradient, direction, torch.tensor(1e-5), lambda x: torch.tensor(0.0))
    assert verify_receipt(*attach_vectors(before, after, gradient, receipt))["accepted"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("graph", [False, True])
def test_actual_propose_receipts_with_metric_steps_and_accumulated_states(monkeypatch, dtype, graph):
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23, dtype=dtype), scale=torch.ones(23, dtype=dtype))
    )
    phi = torch.arange(64, dtype=dtype).reshape(1, 16, 4) / 64
    pack = dict(phi=phi, valid=torch.ones(1, 17, dtype=torch.bool))
    t = torch.linspace(0, 6, 25, dtype=dtype).reshape(1, 25, 1)
    sine, cosine = t.sin().expand(1, 25, 23), t.cos().expand(1, 25, 23)

    def auxiliary(_model, _pack, state, b):
        return sine * (state[0, 0] + b[0, 0]) + cosine * (state[0, 1] + b[0, 1])

    monkeypatch.setattr(core, "auxiliary", auxiliary)
    b = torch.full((25, 23), 0.1, dtype=dtype, requires_grad=graph)
    mask = torch.ones(1, 25, 23, dtype=torch.bool)
    mask[:, :, 10] = False
    state = torch.zeros(4, 64, dtype=dtype)
    for index in range(8):
        truth = ((1.5 + index * 0.03) * sine + 0.4 * cosine).sinh()
        before = state.detach().clone()
        state, receipt = core.propose(model, pack, state, truth, mask, b, create_graph=graph)
        numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
        gradient = np.asarray(receipt["gradient256"], dtype=numpy_dtype)
        summary = verify_receipt(before.flatten().numpy(), state.detach().flatten().numpy(), gradient, receipt, 550)
        assert summary["evaluations"] <= 11 and summary["gradient_computed"]
        exported = {k: v for k, v in receipt.items() if k not in ("gradient256", "state256_after")}
        assert summary == verify_receipt(
            before.flatten().numpy(), state.detach().flatten().numpy(), gradient, exported, 550
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed", -1),
        ("observed", 576),
        ("evaluations", -1),
        ("evaluations", 12),
        ("alpha", "0.5"),
        ("loss_before", True),
    ],
)
def test_invalid_counts_and_non_numeric_scalars_rejected(field, value):
    before, after, gradient, receipt, observed = accepted()
    if field == "observed":
        observed = value
    else:
        receipt[field] = value
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)


@pytest.mark.parametrize(
    "change",
    [
        "evaluations",
        "alpha_sequence",
        "selected_alpha",
        "selected_loss",
        "selected_bound",
        "bound_schedule",
        "first_passing",
        "not_strict",
        "not_armijo",
        "reason",
        "gradient_norm",
        "step_norm",
        "gradient_record",
        "state_record",
        "gradient_flag",
        "observed_record",
        "legacy_loss",
        "ascent_state",
    ],
)
def test_accepted_receipt_tampering_is_rejected(change):
    before, after, gradient, receipt, observed = accepted()
    if change == "evaluations":
        receipt["evaluations"] += 1
    elif change == "alpha_sequence":
        receipt["trials"][1]["alpha"] = 0.3
    elif change == "selected_alpha":
        receipt["alpha"] *= 2
    elif change == "selected_loss":
        receipt["loss_after"] += 1
    elif change == "selected_bound":
        receipt["armijo_bound"] += 0.1
    elif change == "bound_schedule":
        receipt["trials"][2]["armijo_bound"] -= 0.1
    elif change == "first_passing":
        receipt["trials"][0]["loss"] = 0.0
    elif change == "not_strict":
        receipt["trials"][-1]["loss"] = receipt["loss_after"] = receipt["loss_before"]
    elif change == "not_armijo":
        receipt["trials"][-1]["loss"] = receipt["loss_after"] = np.nextafter(receipt["armijo_bound"], np.inf)
    elif change == "reason":
        receipt["reason"] = "improved"
    elif change == "gradient_norm":
        receipt["gradient_norm"] *= 1.01
    elif change == "step_norm":
        receipt["step_norm"] *= 1.01
    elif change == "gradient_record":
        receipt["gradient256"][1] = 1.0
    elif change == "state_record":
        receipt["state256_after"][1] = 1.0
    elif change == "gradient_flag":
        receipt["gradient_computed"] = False
    elif change == "observed_record":
        receipt["observed"] = observed - 1
    elif change == "legacy_loss":
        receipt["loss_proposal"] = receipt["loss_after"]
    else:
        after = before + (before - after)
        receipt["state256_after"] = after.tolist()
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)


@pytest.mark.parametrize(
    "change",
    [
        "short_budget",
        "passing_trial",
        "state",
        "negative_zero",
        "selected_bound",
        "loss_after",
        "stationary_nonzero",
        "zero_nonstationary",
    ],
)
def test_rejected_receipt_tampering_is_rejected(change):
    before, after, gradient, receipt, observed = rejected()
    if change == "short_budget":
        receipt["trials"].pop()
        receipt["evaluations"] -= 1
    elif change == "passing_trial":
        receipt["trials"][2]["loss"] = -1.0
    elif change == "state":
        after[0] += 0.01
        receipt.update(
            step_norm=float(np.linalg.norm((after - before).astype(np.float64))), state256_after=after.tolist()
        )
    elif change == "negative_zero":
        before[-1], after[-1] = 0.0, -0.0
        receipt["state256_after"] = after.tolist()
    elif change == "selected_bound":
        receipt["armijo_bound"] = receipt["trials"][-1]["armijo_bound"]
    elif change == "loss_after":
        receipt["loss_after"] = 1.0
    elif change == "stationary_nonzero":
        receipt.update(reason="stationary", evaluations=0, trials=[])
    else:
        gradient[:] = 0
        receipt.update(gradient_norm=0.0, gradient256=gradient.tolist())
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)


@pytest.mark.parametrize(
    "change", ["measured_gradient", "nonzero_placeholder", "loss", "state", "flag", "positive_observed"]
)
def test_no_target_never_becomes_a_measured_zero_gradient(change):
    before, after, gradient, receipt, observed = no_target(True)
    if change == "measured_gradient":
        receipt["gradient256"] = gradient.tolist()
    elif change == "nonzero_placeholder":
        gradient[0] = 1.0
    elif change == "loss":
        receipt["loss_before"] = 0.0
    elif change == "state":
        after[0] += 0.01
    elif change == "flag":
        receipt["gradient_computed"] = True
    else:
        observed = 1
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)


@pytest.mark.parametrize(
    "field", ["before", "after", "gradient", "loss_before", "loss_after", "bound", "norm", "trial_loss"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_values_rejected(field, bad):
    before, after, gradient, receipt, observed = accepted()
    if field in ("before", "after", "gradient"):
        {"before": before, "after": after, "gradient": gradient}[field][0] = bad
    elif field in ("loss_before", "loss_after"):
        receipt[field] = bad
    elif field == "bound":
        receipt["trials"][0]["armijo_bound"] = bad
    elif field == "norm":
        receipt["gradient_norm"] = bad
    else:
        receipt["trials"][0]["loss"] = bad
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)


@pytest.mark.parametrize(
    "field",
    [
        "before",
        "after",
        "gradient",
        "state_record",
        "gradient_record",
        "observed",
        "accepted",
        "evaluations",
        "trials",
        "missing_gradient",
    ],
)
def test_bad_shapes_and_types_rejected(field):
    before, after, gradient, receipt, observed = accepted()
    if field == "before":
        before = before.reshape(4, 64)
    elif field == "after":
        after = after[:-1]
    elif field == "gradient":
        gradient = gradient.reshape(16, 16)
    elif field == "state_record":
        receipt["state256_after"] = [[0.0] * 64] * 4
    elif field == "gradient_record":
        receipt["gradient256"] = [0.0] * 255
    elif field == "observed":
        observed = True
    elif field == "accepted":
        receipt["accepted"] = 1
    elif field == "evaluations":
        receipt["evaluations"] = True
    elif field == "trials":
        receipt["trials"] = {}
    else:
        gradient = None
    with pytest.raises(ValueError):
        verify_receipt(before, after, gradient, receipt, observed)
