import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from paired_diagnostic_step_20261106 import (
    Decoder,
    Mixer,
    actions,
    bank,
    effects,
    family_loss,
    mix_inputs,
    native,
    training_queries,
    write_slots,
)


def test_block_boundary_cannot_rewrite_its_maturity_sample():
    got = write_slots(110, np.array([63, 95], dtype=np.int64))
    assert (got[:64] == -1).all() and (got[64:96] == 0).all()
    assert (got[96:] == 1).all()
    with pytest.raises(ValueError, match="complete"):
        write_slots(110, np.array([62, 95], dtype=np.int64))


def test_two_families_have_equal_weight_even_with_different_channel_counts():
    d = torch.zeros((1, 11))
    d[0, :3], d[0, 3:7] = 2.0, 0.5
    valid = torch.ones_like(d, dtype=torch.bool)
    valid[0, 7:] = False
    loss, support = family_loss(d, torch.zeros_like(d), valid)
    torch.testing.assert_close(loss, torch.tensor([[1.5, 0.125, 0.0]]))
    torch.testing.assert_close(
        Mixer()(mix_inputs(d, valid), support), torch.tensor([[0.5, 0.5, 0.0]])
    )


def test_real_gradient_matches_independent_backward_without_changing_parent(monkeypatch):
    torch.manual_seed(2)
    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
    base.eval().requires_grad_(False)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    head = Decoder().eval().requires_grad_(False)
    x = torch.randn(2, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    target = torch.linspace(-0.6, 0.8, 22).reshape(2, 11)
    valid = torch.ones_like(target, dtype=torch.bool)
    w = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    p, g, present = bank(base, head, x, mask, target, valid)
    u = torch.zeros((2, 128), requires_grad=True)
    h, _ = native(base, x, mask, u)
    observed = head(h)
    losses = torch.nn.functional.smooth_l1_loss(observed, target, reduction="none")
    loss = sum(
        (losses[:, j:k].mean(1) * w[:, i]).sum()
        for i, (j, k) in enumerate(((0, 3), (3, 7), (7, 11)))
    )
    loss.backward()
    torch.testing.assert_close((g * w[..., None]).sum(1), u.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p, observed.detach())
    assert present.all() and g.norm() > 0
    assert not base.risk_head._forward_pre_hooks
    assert all(torch.equal(base.state_dict()[k], v) for k, v in state.items())


def test_directions_match_norm_and_respect_common_support():
    real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reverse, unsigned, zero = -real, real.flip(1), 2 * real
    mix, valid = actions(
        (real, reverse, unsigned, zero), torch.tensor([True, False]), 0.2
    )
    norms = torch.stack([a.norm(dim=1) for a in mix])
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(norms, norms[:1].expand_as(norms))
    assert 0 < norms[0, 0] < 0.5 and (norms[:, 1] == 0).all()
    assert (mix[0][0] * mix[1][0]).sum() < 0
    torch.testing.assert_close(effects(real, 0.2)[0], mix[0][0])


def test_offline_query_cannot_precede_release_or_relabel_late_source():
    times = np.arange(130) / 1000
    z = np.arange(130, dtype=float)
    slots, ends, group = training_queries(
        times,
        z,
        np.array([63, 95, 127]),
        np.array([True, True, False]),
        True,
        0.122,
        25,
    )
    assert slots.tolist() == [0, 1, 1]
    assert ends.tolist() == [95, 111, 96]
    assert group.tolist() == [2, 0, 2]
