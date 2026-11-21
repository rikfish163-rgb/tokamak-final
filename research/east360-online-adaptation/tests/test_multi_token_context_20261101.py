import numpy as np
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import (
    ContextAdapter,
)
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import (
    patch_signal,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.simple_data import window
from experiments.risk_aligned_online.multi_token_context_20261101.model import (
    token_basis,
    context_at,
    context_trace,
    action,
    risk_with_shifts,
)


def model():
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(
        Config(fast_site="tokens", fast_rule="normalized"), np.zeros(23), np.ones(23)
    )
    return ContextAdapter(base).eval().requires_grad_(False)


def test_zero_state_matches_cold_and_does_not_write_cls():
    m = model()
    x, a = torch.randn(3, 128, 23), torch.ones(3, 128, 23, dtype=torch.bool)
    a[0, :120] = False
    with torch.no_grad():
        h, valid = m.tokens(x, a)
        b, _ = token_basis(m.memory, x, a, m.base.center, m.base.scale)
        shift, _ = action(
            m.memory.output.weight, b, torch.zeros(3, 8), valid, "per_token"
        )
        expected = m.risk(h, valid, torch.zeros(3, 92))
        assert torch.equal(risk_with_shifts(m, h, valid, shift), expected)
        assert not shift.any()


def test_zero_output_still_has_useful_gradient():
    m = model()
    x, a = torch.randn(2, 128, 23), torch.ones(2, 128, 23, dtype=torch.bool)
    h, valid = m.tokens(x, a)
    b, _ = token_basis(m.memory, x, a, m.base.center, m.base.scale)
    weight = torch.zeros(64, 92, requires_grad=True)
    shift, _ = action(weight, b, torch.ones(2, 8), valid, "per_token")
    risk_with_shifts(m, h, valid, shift).sum().backward()
    assert (
        weight.grad is not None
        and torch.isfinite(weight.grad).all()
        and weight.grad.norm() > 0
    )


def test_tiled_norm_and_unsupported_tokens():
    torch.manual_seed(7)
    basis, c, w = torch.randn(3, 16, 92, 8), torch.randn(3, 8), torch.randn(64, 92)
    valid = torch.ones(3, 17, dtype=torch.bool)
    valid[0, :8] = False
    real, _ = action(w, basis, c, valid, "per_token")
    tile, info = action(w, basis, c, valid, "tiled_norm")
    assert torch.allclose(real.norm(dim=(1, 2)), tile.norm(dim=(1, 2)), rtol=1e-6)
    assert not tile[0, :8].any() and not info["tiled_zero_direction"].any()
    zero, info = action(torch.zeros_like(w), basis, c, valid, "tiled_norm")
    assert not zero.any() and torch.isfinite(zero).all()


def test_no_complete_channel_does_not_create_bias_only_evidence():
    m = model()
    x = torch.randn(2, 128, 23)
    available = torch.zeros_like(x, dtype=torch.bool)
    available[:, ::8, 0] = True
    b, supported = token_basis(m.memory, x, available, m.base.center, m.base.scale)
    assert not supported.any() and not b.any()
    assert not context_at(torch.zeros(0, 8), [0, 7]).any()
    assert not context_at(torch.zeros(1, 8), [15]).any()


def test_all_clock_phases_are_prefix_invariant():
    m = model()
    rng = np.random.default_rng(131)
    x = rng.normal(size=(180, 23)).astype(np.float32)
    available = rng.random((180, 23)) > 0.01
    signal = dict(values=x, available=available)
    changed = dict(values=x.copy(), available=available.copy())
    center, scale = m.base.center, m.base.scale
    for end in [0, 15, 16, 23, 24, 126, 127, 128, *range(144, 152)]:
        changed["values"][:] = x
        changed["values"][end + 1 :] += 10000
        changed["available"][:] = available
        changed["available"][end + 1 :] = ~available[end + 1 :]
        contexts, bases = [], []
        for s in (signal, changed, {k: v[: end + 1] for k, v in signal.items()}):
            means, complete = patch_signal(s, center, scale)
            trace = context_trace(m.memory, means, complete)
            contexts.append(context_at(trace, [end]))
            xx, aa = window(s, end, 128)
            b, _ = token_basis(
                m.memory, torch.tensor(xx)[None], torch.tensor(aa)[None], center, scale
            )
            bases.append(b)
        assert torch.equal(contexts[0], contexts[1]) and torch.equal(
            contexts[0], contexts[2]
        )
        assert torch.equal(bases[0], bases[1]) and torch.equal(bases[0], bases[2])


def test_canonical32_trace_preserves_the_existing_update_algebra():
    m = model()
    torch.manual_seed(19)
    means = torch.randn(32, 23)
    complete = torch.rand(32, 23) > 0.05
    original = m.memory.trace(means, complete, mode="memory")
    actual = context_trace(m.memory, means, complete)
    assert torch.equal(actual, original["context"])
    norm = actual.norm(dim=1)
    for mode in ("reverse", "local"):
        expected = m.memory.trace(means, complete, mode=mode, reference_norm=norm)["context"]
        assert torch.equal(context_trace(m.memory, means, complete, mode=mode, reference_norm=norm), expected)
