import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_intra_20261110 import (
    ARMS, feedback_blocks, replay_shot,
)
from experiments.risk_aligned_online.paired_motion_20261110 import motion_views
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.commissioning_transfer_20261108 import predict


def test_forecast_effect_begins_only_after_the_entire_future_arrives():
    assert list(feedback_blocks(153)) == []
    assert list(feedback_blocks(154)) == [(127, 153, 154)]
    assert list(feedback_blocks(230)) == [(127, 153, 185), (159, 185, 217), (191, 217, 230)]


@pytest.mark.parametrize("variant", ["native", "anchored", "banded"])
def test_replay_is_prefix_causal_updates_internal_state_and_resets_between_shots(variant):
    torch.manual_seed(131)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(dim=8, heads=2, layers=1, fast_site="tokens",
                          fast_rule="normalized"), np.zeros(23), np.ones(23))
    base.eval().requires_grad_(False)
    rng = np.random.default_rng(131)
    values = rng.normal(size=(231, 23)).cumsum(0).astype(np.float32)
    available = np.ones_like(values, bool)
    available[137:140, 1] = False
    signal = dict(values=values, available=available)
    coeff = torch.randn(8 + 23 + 1) / 5

    def simulate(n):
        selected = {k: v[:n] for k, v in signal.items()}
        static = predict(base, selected, {"cold": coeff}, "cpu")["cold"]
        options = {} if variant == "native" else {variant: True}
        result = replay_shot(base, coeff, selected, static, **options)
        return static, result

    static, (curves, details) = simulate(231)
    expected = {"native": set(ARMS), "anchored": {"anchored_motion", "anchored_reversed"},
                "banded": {"banded_motion", "banded_reversed"}}
    assert set(curves) == expected[variant]
    for arm in curves:
        np.testing.assert_array_equal(curves[arm][:153], static[:153])
        assert not np.array_equal(curves[arm][153:], static[153:])
        assert details[arm]["changed_steps"] == 3
        assert details[arm]["max_state_norm"] <= 1.000001
    _, (early, _) = simulate(173)
    _, (again, _) = simulate(231)
    for arm in curves:
        np.testing.assert_allclose(early[arm], curves[arm][:173], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(again[arm], curves[arm])
    assert all(p.grad is None for p in base.parameters())


def test_persistence_anchor_removes_origin_bias_but_keeps_forecast_derivative():
    rng = torch.Generator().manual_seed(131)
    past = torch.randn(1, 63, 23, generator=rng, dtype=torch.float64)
    truth = torch.randn(1, 25, 23, generator=rng, dtype=torch.float64)
    forecast = (torch.randn(truth.shape, generator=rng, dtype=torch.float64) + 2).requires_grad_()
    p, _, hold, valid = motion_views(forecast, truth, torch.ones_like(truth, dtype=torch.bool),
                                   past, torch.ones_like(past, dtype=torch.bool),
                                   torch.zeros(23, dtype=torch.float64),
                                   torch.ones(23, dtype=torch.float64),
                                   frozen_forecast=forecast.detach().clone())
    torch.testing.assert_close(p, hold, atol=1e-12, rtol=0)
    (grad,) = torch.autograd.grad(p[valid].sum(), forecast)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_per_band_budget_does_not_double_update_on_any_token():
    from experiments.risk_aligned_online.commissioning_intra_20261110 import fast_step

    torch.manual_seed(131)
    zero = torch.zeros(3, 16)
    grad = torch.randn_like(zero)
    single = fast_step(zero, zero, grad)
    both = fast_step(zero[:, None].expand(-1, 2, -1), zero[:, None].expand(-1, 2, -1),
                     grad[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(both, single[:, None].expand(-1, 2, -1), atol=0, rtol=0)
    assert both.norm(dim=-1).max() <= .050001
