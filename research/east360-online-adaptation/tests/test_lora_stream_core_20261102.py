import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import (
    source_basis,
)
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.lora_stream_20261102.core import (
    cap_shift,
    direct,
    encode,
    forward,
    step,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23

torch.set_num_threads(2)


@pytest.fixture
def setup():
    torch.manual_seed(57)
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
    rng = np.random.default_rng(58)
    signal = dict(
        values=rng.normal(size=(384, 23)).astype(np.float32),
        available=rng.random((384, 23)) > 0.05,
    )
    return model, signal, source_basis(model)


def test_zero_state_is_exact_cold_and_exposes_original_ffn_addition(setup):
    model, signal, basis = setup
    pack = encode(model, signal, [0, 15, 127, 159, 160, 262], basis)
    assert set(pack) == {"h", "valid", "phi", "residual", "ffn"}
    assert torch.equal(pack["residual"] + pack["ffn"], pack["h"][:, :16])
    with torch.no_grad():
        expected = model.risk(pack["h"], pack["valid"], torch.zeros(6, 92))
        result = forward(model, pack, torch.zeros(4, 64), forecast=True)
    assert torch.equal(result["logit"], expected)
    assert torch.count_nonzero(result["shift"]) == 0
    assert result["forecast"].shape == (6, 25, 23)


def test_cache_matches_actual_ffn_insertion_and_has_real_gradients(setup):
    model, signal, basis = setup
    ends = [0, 159, 160, 262]
    pack = encode(model, signal, ends, basis)
    state = (torch.randn(4, 64) * 0.8).requires_grad_(True)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = forward(model, pack, state, forecast=True)
    actual = direct(model, signal, ends, basis, state, forecast=True)
    assert torch.equal(actual["h"][:, -1], pack["h"][:, -1])
    assert torch.equal(result["h"], actual["h"])
    assert torch.equal(result["logit"], actual["logit"])
    assert torch.equal(result["forecast"], actual["forecast"])
    result["logit"].sum().backward()
    assert (
        state.grad is not None
        and torch.isfinite(state.grad).all()
        and state.grad.norm() > 0
    )
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert not model.base.encoder.layers[0].linear2._forward_hooks
    assert not model.base.encoder.layers[0].norm2._forward_pre_hooks


def test_cap_is_per_current_endpoint_and_masks_padding():
    delta = torch.ones(3, 16, 64)
    delta[0] *= 0.1
    delta[1] *= 10
    valid = torch.ones(3, 16, dtype=torch.bool)
    valid[1, :8] = False
    valid[2] = False
    capped, rms = cap_shift(delta, valid)
    assert torch.equal(capped[0], delta[0])
    assert torch.count_nonzero(capped[1, :8]) == 0
    assert torch.count_nonzero(capped[2]) == 0
    assert torch.all(rms <= 0.250001)
    assert torch.allclose(capped[1, 8:], torch.full((8, 64), 0.25))
    singleton, _ = cap_shift(delta[:1], valid[:1])
    assert torch.equal(singleton[0], capped[0])


def test_cap_zero_gradient_is_finite():
    value = torch.zeros(2, 16, 64, requires_grad=True)
    capped, _ = cap_shift(value, torch.ones(2, 16, dtype=torch.bool))
    capped.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_forecast_update_is_finite_and_only_accepts_observed_loss_descent(setup):
    model, signal, basis = setup
    pack = encode(model, signal, [134], basis)
    truth = torch.as_tensor(signal["values"][135:160][None])
    mask = torch.as_tensor(signal["available"][135:160][None])
    before = torch.zeros(4, 64)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    after, receipt = step(model, pack, before, truth, mask)
    assert len(receipt["gradient256"]) == len(receipt["state256_after"]) == 256
    assert receipt["gradient_norm"] > 0
    assert receipt["loss_retained"] <= receipt["loss_before"]
    assert receipt["accepted"] == (receipt["loss_proposal"] < receipt["loss_before"])
    if not receipt["accepted"]:
        assert torch.equal(after, before)
    assert torch.isfinite(after).all()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_unobserved_target_is_explicit_no_update(setup):
    model, signal, basis = setup
    pack = encode(model, signal, [134], basis)
    state = torch.randn(4, 64) * 0.1
    after, receipt = step(
        model,
        pack,
        state,
        torch.full((1, 25, 23), float("nan")),
        torch.zeros(1, 25, 23, dtype=torch.bool),
    )
    assert torch.equal(after, state)
    assert not receipt["accepted"] and not receipt["gradient_computed"]
    assert receipt["reason"] == "no_target" and receipt["loss_before"] is None
    assert receipt["gradient_norm"] == receipt["observed"] == 0


def test_encoder_only_accepts_signals_and_prefix_features_do_not_read_future(setup):
    model, signal, basis = setup
    with pytest.raises(ValueError):
        encode(model, dict(signal, DownTime=1.0), [262], basis)
    ends = np.arange(256, 288)
    first = encode(model, signal, ends, basis)
    changed = {k: v.copy() for k, v in signal.items()}
    changed["values"][263:] += 100
    changed["available"][263:] = ~changed["available"][263:]
    second = encode(model, changed, ends, basis)
    assert all(torch.equal(first[k][6], second[k][6]) for k in first)
