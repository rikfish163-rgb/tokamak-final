import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.mask_separated_intra_20261103.model import (
    MaskSeparatedContext,
    mask_embedding_shift,
)


def example():
    g = torch.Generator(device="cpu").manual_seed(131)
    x = torch.randn(2, 16, 8, 46, generator=g, dtype=torch.float64)
    x[..., 23:] = 1
    w = torch.randn(64, 368, generator=g, dtype=torch.float64)
    return x, w


def test_source_coavailable_and_padding_patterns_have_exact_zero_shift():
    x, w = example()
    x[0, :4, :, 23:] = 0
    original = x.clone()
    shift = mask_embedding_shift(x.flatten(-2), w)
    assert not shift.any()
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_partial_mask_shift_equals_explicit_weight_redistribution():
    x, w = example()
    x[0, :, :, 24] = 0
    x[1, :, :, 25] = 0
    alt = w.clone()
    for p in range(8):
        bt, c, m = 46 * p + 23, 46 * p + 24, 46 * p + 25
        alt[:, bt] += w[:, c] + w[:, m]
        alt[:, c] = 0
        alt[:, m] = 0
    flat = x.flatten(-2)
    expected = F.linear(flat, alt) - F.linear(flat, w)
    torch.testing.assert_close(
        mask_embedding_shift(flat, w), expected, atol=1e-12, rtol=1e-12
    )


def test_shift_remains_differentiable_without_training_frozen_weight():
    x, w = example()
    x[..., 24] = 0
    x = x.flatten(-2).requires_grad_(True)
    mask_embedding_shift(x, w).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.any()
    assert w.grad is None


def base():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(131)
        return (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )


def test_context_source_identity_truthful_masks_and_no_new_parameters():
    b = base()
    original = ContextAdapter(b).eval().requires_grad_(False)
    model = MaskSeparatedContext(b, policy="reference_bt")
    assert set(model.state_dict()) == set(original.state_dict())
    assert not any(p.requires_grad for p in model.parameters())
    g = torch.Generator(device="cpu").manual_seed(17)
    x = torch.randn(2, 128, 23, generator=g)
    a = torch.ones_like(x, dtype=torch.bool)
    a[0, :32] = False
    saved = a.clone()
    before = {k: v.clone() for k, v in b.state_dict().items()}
    with torch.no_grad():
        h0, v0 = original.tokens(x, a)
        h1, v1 = model.tokens(x, a)
    torch.testing.assert_close(h0, h1, rtol=0, atol=0)
    assert torch.equal(a, saved) and torch.equal(v0, v1)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in before.items())
    assert len(b.embedding._forward_hooks) == 0


def test_scoped_hook_is_removed_on_error_and_does_not_remove_existing_hooks(
    monkeypatch,
):
    b = base()
    model = MaskSeparatedContext(b)
    old = b.embedding.register_forward_hook(lambda m, args, out: None)
    count = len(b.embedding._forward_hooks)

    def fail(*args, **kw):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(ContextAdapter, "tokens", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        model.tokens(torch.zeros(1, 128, 23), torch.ones(1, 128, 23, dtype=torch.bool))
    assert len(b.embedding._forward_hooks) == count
    old.remove()


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        MaskSeparatedContext(base(), policy="fake_observed")
