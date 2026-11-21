import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.adaptivity_capacity_20261102.model import (
    cached_risk,
    direct_lora,
    encode,
    source_basis,
)
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23

torch.set_num_threads(2)


@pytest.fixture
def setup():
    torch.manual_seed(37)
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
    rng = np.random.default_rng(38)
    signal = dict(
        values=rng.normal(size=(384, 23)).astype(np.float32),
        available=rng.random((384, 23)) > 0.05,
    )
    return model, signal


def test_source_native_rank4_basis_is_orthonormal_and_deterministic(setup):
    model, _ = setup
    a = source_basis(model)
    assert a.shape == (4, 128) and not a.requires_grad
    assert torch.allclose(a @ a.T, torch.eye(4), atol=1e-6, rtol=0)
    assert torch.equal(a, source_basis(model))


def test_zero_cache_and_direct_ffn_intervention_exactly_preserve_cold(setup):
    model, signal = setup
    ends = [0, 15, 127, 159, 160, 254, 320]
    a = source_basis(model)
    pack = encode(model, signal, ends, a)
    assert set(pack) == {"h", "valid", "long", "snapshot", "phi"}
    assert pack["phi"].shape == (len(ends), 16, 4)
    with torch.no_grad():
        cold = model.risk(pack["h"], pack["valid"], torch.zeros(len(ends), 92))
        cached = cached_risk(model, pack, torch.zeros(len(ends), 16, 64))
        result = direct_lora(
            model,
            signal,
            ends,
            a,
            torch.zeros(4, 64),
            torch.ones(len(ends), dtype=torch.bool),
        )
    assert torch.equal(cold, cached) and torch.equal(cold, result["logit"])
    assert torch.equal(pack["h"], result["h"])
    assert not model.base.encoder.layers[0].linear2._forward_hooks


def test_first_ffn_lora_changes_data_not_first_block_cls_and_has_real_gradients(setup):
    model, signal = setup
    a, ends = source_basis(model), [262, 300]
    pack = encode(model, signal, ends, a)
    w = (torch.randn(4, 64) * 0.03).requires_grad_(True)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    direct = direct_lora(model, signal, ends, a, w, torch.ones(2, dtype=torch.bool))
    assert torch.equal(direct["h"][:, -1], pack["h"][:, -1])
    delta = pack["phi"] @ w
    assert torch.allclose(
        direct["h"][:, :16] - pack["h"][:, :16], delta, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        cached_risk(model, pack, delta), direct["logit"], atol=2e-6, rtol=0
    )
    direct["logit"].sum().backward()
    assert w.grad is not None and w.grad.norm() > 0 and torch.isfinite(w.grad).all()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    assert not model.base.encoder.layers[0].linear2._forward_hooks


def test_mutability_mask_cannot_change_immutable_rows(setup):
    model, signal = setup
    ends, a = [159, 160, 161], source_basis(model)
    pack = encode(model, signal, ends, a)
    result = direct_lora(
        model,
        signal,
        ends,
        a,
        torch.ones(4, 64) * 0.1,
        torch.tensor([False, True, True]),
    )
    cold = model.risk(pack["h"], pack["valid"], torch.zeros(3, 92))
    assert torch.equal(result["logit"][:1], cold[:1])
    assert torch.equal(result["h"][:1], pack["h"][:1])


def test_first_layer_features_do_not_read_later_samples_inside_same_batch(setup):
    model, signal = setup
    ends, a = np.arange(256, 288), source_basis(model)
    pack = encode(model, signal, ends, a)
    altered = {k: v.copy() for k, v in signal.items()}
    altered["values"][263:] += 1000
    altered["available"][263:] = ~altered["available"][263:]
    other = encode(model, altered, ends, a)
    assert all(torch.equal(pack[k][6], other[k][6]) for k in pack)
