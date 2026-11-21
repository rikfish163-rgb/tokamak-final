import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import (
    source_basis,
)
from experiments.risk_aligned_online.centered_gate_stream_20261101.model import (
    loss as shape_loss,
)
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.difference_feedback_probe_20261102.core import (
    components,
    propose,
)
from experiments.risk_aligned_online.lora_stream_20261102.core import encode, forward
from experiments.risk_aligned_online.simple23 import Config, Simple23

torch.set_num_threads(2)


@pytest.fixture
def case():
    torch.manual_seed(67)
    torch.backends.mha.set_fastpath_enabled(False)
    model = (
        ContextAdapter(
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                np.zeros(23),
                np.ones(23),
            )
        )
        .eval()
        .requires_grad_(False)
    )
    rng = np.random.default_rng(68)
    signal = dict(
        values=rng.normal(size=(192, 23)).astype(np.float32),
        available=rng.random((192, 23)) > 0.05,
    )
    pack = encode(model, signal, [134], source_basis(model))
    truth = torch.as_tensor(signal["values"][135:160][None])
    hold = torch.as_tensor(np.repeat(signal["values"][134:135], 25, axis=0)[None])
    mask = torch.as_tensor(signal["available"][135:160][None])
    return model, pack, truth, hold, mask


def test_objective_gradient_is_signal_minus_hold_plus_parameter_regularizer(case):
    model, pack, truth, hold, mask = case
    state = (torch.randn(4, 64) * 0.02).requires_grad_(True)
    predicted = forward(model, pack, state, forecast=True)["forecast"]
    gs = torch.autograd.grad(
        shape_loss(model, predicted, truth, mask), state, retain_graph=True
    )[0]
    gh = torch.autograd.grad(
        shape_loss(model, predicted, hold, mask), state, retain_graph=True
    )[0]
    terms = components(model, predicted, state, truth, hold, mask)
    actual = torch.autograd.grad(terms["objective"], state)[0]
    expected = gs - gh + (0.02 / 256) * state
    assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-5)


def test_pure_hold_from_zero_has_exactly_zero_gradient_and_no_update(case):
    model, pack, _, hold, mask = case
    state = torch.zeros(4, 64)
    after, receipt = propose(model, pack, state, hold, hold, mask)
    assert torch.equal(after, state)
    assert receipt["gradient_norm"] == 0
    assert not receipt["accepted"] and receipt["reason"] == "zero_gradient"
    assert receipt["loss_before"] == receipt["loss_proposal"] == 0


def test_update_only_keeps_strict_difference_objective_descent_and_source_unchanged(
    case,
):
    model, pack, truth, hold, mask = case
    original = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.zeros(4, 64)
    after, receipt = propose(model, pack, state, truth, hold, mask)
    assert receipt["gradient_norm"] > 0
    assert receipt["accepted"] == (receipt["loss_proposal"] < receipt["loss_before"])
    assert receipt["loss_retained"] <= receipt["loss_before"]
    assert torch.isfinite(after).all()
    assert len(receipt["gradient256"]) == len(receipt["state256_after"]) == 256
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert all(p.grad is None for p in model.parameters())


def test_no_observations_preserve_nonzero_state_without_fabricated_action_measure(case):
    model, pack, truth, hold, _ = case
    state = torch.ones(4, 64) * 0.02
    after, receipt = propose(
        model, pack, state, truth, hold, torch.zeros_like(truth, dtype=torch.bool)
    )
    assert torch.equal(after, state)
    assert receipt["reason"] == "no_target" and receipt["max_action_rms"] is None
    assert receipt["loss_before"] is None and receipt["gradient_norm"] == 0


def test_nonfinite_observed_target_fails(case):
    model, pack, truth, hold, mask = case
    truth = truth.clone()
    truth[mask] = float("nan")
    with pytest.raises(ValueError):
        propose(model, pack, torch.zeros(4, 64), truth, hold, mask)
