"""Causal clock and gradient geometry checks for local LayerNorm interventions."""

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from local_affine_reach_20261105 import (
    arrival_for,
    local_weights,
    routed_logit,
    value_and_gradient,
)
from timed_token_norm_20261105 import support_weights


def test_clock_uses_only_already_arrived_complete_patches():
    q = [135, 136, 143, 158, 159, 166, 167, 168]
    m = arrival_for(q)
    assert m.tolist() == [135] * 6 + [167] * 2
    a = torch.ones(len(q), 128, 23, dtype=torch.bool)
    recent = local_weights(a, q, m, "recent")
    assert recent.count_nonzero(1).tolist() == [4, 3, 4, 3, 4, 3, 4, 3]
    assert torch.allclose(recent.square().sum(1), torch.full((len(q),), 4.0))
    for i, (query, clock) in enumerate(zip(q, m, strict=True)):
        for token in torch.nonzero(recent[i], as_tuple=True)[0].tolist():
            end = query - 8 * (15 - token)
            assert int(clock) - 31 <= end - 7 <= end <= clock
    assert torch.equal(recent[4:5], support_weights(a[4:5], [159], [135], "recent"))
    with pytest.raises(ValueError, match="arrived"):
        local_weights(a, q, m + 32, "recent")
    with pytest.raises(ValueError, match="arrived"):
        arrival_for([134])


def test_recent_mask_does_not_depend_on_post_arrival_availability():
    a = torch.ones(1, 128, 23, dtype=torch.bool)
    old = local_weights(a, [159], [135], "recent")
    a[:, -24:] = False
    assert torch.equal(local_weights(a, [159], [135], "recent"), old)
    a[:, 72:80] = False
    assert not local_weights(a, [159], [135], "recent").any()
    assert not local_weights(a, [159], [135], "older").any()


def test_gradient_is_per_row_exact_zero_parent_and_route_hook_is_removed(request):
    torch.set_num_threads(1)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    request.addfinalizer(lambda: torch.backends.mha.set_fastpath_enabled(fastpath))
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(7)
    base = (
        Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23),
            torch.ones(23),
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(2, 128, 23, dtype=torch.float64)
    a = torch.ones_like(x, dtype=torch.bool)
    w = local_weights(a, [159, 160], [135, 135], "recent")
    with torch.no_grad():
        expected = base(x, a)["logit"]
    for layer in (0, 1):
        z, g = value_and_gradient(base, x, a, w, layer)
        assert torch.equal(z, expected)
        assert g.shape == (2, 128)
        d = g / g.norm(dim=1, keepdim=True)
        with torch.no_grad():
            up, energy = routed_logit(base, x, a, w, d * 1e-4, layer)
            dn, _ = routed_logit(base, x, a, w, -d * 1e-4, layer)
        np.testing.assert_allclose((up - dn) / 2e-4, g.norm(dim=1), rtol=2e-6)
        assert bool((energy > 0).all())
        assert not base.encoder.layers[layer].norm1._forward_hooks
        assert all(p.grad is None for p in base.parameters())
