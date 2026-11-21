import torch

from experiments.risk_aligned_online.event_temporal_teacher_20261103.features import (
    representation,
    order_step,
)


def packs():
    g = torch.Generator(device="cpu").manual_seed(131)

    def one():
        return dict(
            residual=torch.randn(3, 16, 64, generator=g, dtype=torch.float64),
            ffn=torch.randn(3, 16, 64, generator=g, dtype=torch.float64),
            phi=torch.randn(3, 16, 4, generator=g, dtype=torch.float64) * 0.2,
            valid=torch.ones(3, 17, dtype=torch.bool),
        )

    return one(), one()


def test_pool_mean_last_valid_and_no_valid_is_explicit_zero():
    a, _ = packs()
    a["valid"][0, 14:16] = False
    a["valid"][1, :16] = False
    z, ok, rms = representation(a, torch.zeros(3, 4, 64, dtype=torch.float64))
    data = a["residual"] + a["ffn"]
    torch.testing.assert_close(z[0, :64], data[0, :14].mean(0))
    torch.testing.assert_close(z[0, 64:], data[0, 13])
    assert ok.tolist() == [True, False, True]
    assert not z[1].any() and not rms.any()


def test_normalized_order_gradient_has_known_rank_two_and_reverse_is_control():
    a, b = packs()
    w = torch.linspace(-0.2, 0.3, 128, dtype=torch.float64)
    std = torch.linspace(0.7, 1.4, 128, dtype=torch.float64)
    state, receipt = order_step(a, b, w, std)
    reverse, _ = order_step(a, b, w, std, reverse=True)
    doubled, _ = order_step(a, b, 2 * w, std)
    torch.testing.assert_close(reverse, -state, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(doubled, state, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        state.flatten(1).norm(dim=1), torch.full((3,), 0.1, dtype=torch.float64)
    )
    u = w / std
    K = (b["phi"].mean(1) - a["phi"].mean(1))[:, :, None] * u[:64][None, None, :]
    K += (b["phi"][:, -1] - a["phi"][:, -1])[:, :, None] * u[64:][None, None, :]
    expected = 0.1 * K / K.flatten(1).norm(dim=1)[:, None, None]
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=1e-12)
    assert all(torch.linalg.matrix_rank(v, atol=1e-10).item() <= 2 for v in state)
    assert all(v > 0 for v in receipt["gradient_norm"])


def test_batch_rows_are_independent_and_no_token_does_not_invent_update():
    a, b = packs()
    w = torch.ones(128, dtype=torch.float64)
    std = w.clone()
    a["valid"][1, :16] = False
    state, r = order_step(a, b, w, std)
    assert not state[1].any() and r["eligible"][1] is False
    a["residual"][2] = 1e3
    changed, _ = order_step(a, b, w, std)
    torch.testing.assert_close(changed[0], state[0])
