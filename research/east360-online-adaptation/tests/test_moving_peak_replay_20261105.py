import numpy as np
import pytest
import torch

from moving_peak_replay_20261105 import (
    BUDGET,
    EPOCHS_PER_ROUND,
    ROUNDS,
    mining_queries,
    shared_orders,
)


def test_moving_negative_peak_not_native_peak_is_mined_with_equal_count():
    times = np.arange(215) / 1000
    native = np.zeros(len(times))
    adapted = native.copy()
    native[0], adapted[0] = 1.0, 1.0
    native[139], adapted[141] = 0.85, 0.90
    fixed = mining_queries(times, native, adapted, False, None, "fixed_peak")
    moving = mining_queries(times, native, adapted, False, None, "moving_peak")
    assert len(fixed) == len(moving) == 8
    assert min(fixed) == 139 and min(moving) == 141
    np.testing.assert_array_equal(fixed[1:], moving[1:])


def test_positive_windows_never_use_current_peak_or_tardy_leads():
    t = np.arange(1600) / 1000
    p = np.random.default_rng(17).uniform(size=len(t))
    fixed = mining_queries(t, p, p[::-1], True, 1.0, "fixed_peak")
    moving = mining_queries(t, p, p[::-1], True, 1.0, "moving_peak")
    assert len(fixed) == 8 and fixed[0] == 875 and fixed[-1] == 969
    np.testing.assert_array_equal(fixed, moving)


def test_shared_order_has_fixed_budget_and_identical_seeded_batches():
    n = BUDGET * 8
    first, second = shared_orders(n), shared_orders(n)
    assert len(first) == len(second) == ROUNDS
    for a, b in zip(first, second, strict=True):
        assert len(a) == len(b) == EPOCHS_PER_ROUND
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y)
            torch.testing.assert_close(x.sort().values, torch.arange(n))


def test_malformed_curves_are_not_used_for_mining():
    t = np.arange(1600) / 1000
    p = np.zeros(len(t))
    p[144] = np.nan
    with pytest.raises(ValueError):
        mining_queries(t, p, p[::-1], False, None, "moving_peak")
    with pytest.raises(ValueError):
        mining_queries(t, p[1:], p, False, None, "fixed_peak")


def test_runtime_uses_same_mature_history_on_truncated_prefix():
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    import latent_trajectory_adapt_20261105 as latent
    from moving_peak_replay_20261105 import actions, forecast

    was_fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            base = (
                Simple23(
                    Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23),
                    torch.ones(23),
                )
                .eval()
                .requires_grad_(False)
            )
            rule = latent.Generator(torch.zeros(194), torch.ones(194)).eval()
            with torch.no_grad():
                rule.layers[-1].weight.normal_(0, 0.03)
            rule.requires_grad_(False)
            x = torch.randn(200, 23).numpy()
            available = np.ones_like(x, dtype=bool)
            saved = []
            for n in (200, 170):
                ctx = latent.native_context(
                    base, {"values": x[:n], "available": available[:n]}
                )
                real, stale = actions(rule, ctx, stale=True)
                curves, updates = forecast(base, ctx, {"real": real, "stale": stale})
                assert updates["real"]["nonzero"] == (n - 136) // 32 + 1
                np.testing.assert_array_equal(curves["real"][:135], ctx[2][:135])
                saved.append(curves)
            for arm in saved[0]:
                np.testing.assert_allclose(
                    saved[0][arm][:170], saved[1][arm], rtol=0, atol=4e-6
                )
    finally:
        torch.backends.mha.set_fastpath_enabled(was_fast)
