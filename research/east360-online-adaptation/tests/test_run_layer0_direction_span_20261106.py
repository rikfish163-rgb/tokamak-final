import numpy as np
import pytest
import torch

from run_layer0_direction_span_20261106 import (
    anchor_gradient,
    basis_from_columns,
    direction_targets,
    random_anchors,
)


def test_basis_keeps_exact_range_including_a_new_bias_direction():
    eye = np.eye(128, dtype=float)
    columns = np.c_[eye[:, :2], eye[:, 0], eye[:, 2]]
    q, sv = basis_from_columns(columns)
    assert q.shape == (128, 3) and sv.shape == (3,)
    np.testing.assert_allclose(q @ q.T, eye[:, :3] @ eye[:, :3].T, atol=1e-12)


def test_exact_opposite_same_norm_spaces_and_zero_action():
    g = torch.zeros(4, 128)
    g[0, :2] = torch.tensor([3.0, 4.0])
    g[1, 0] = 1.0  # exactly in span: no arbitrary complement
    g[3, 1] = 1.0  # original zero: no action in any family
    u = torch.zeros_like(g)
    u[:3, 2] = 0.9
    q = torch.eye(128, dtype=torch.float64)[:, :1]
    a, evidence = direction_targets(u, g, q, shot=317)
    torch.testing.assert_close(a["in_span_pos"][0, :2], torch.tensor([0.9, 0.0]))
    torch.testing.assert_close(a["out_span_pos"][0, :2], torch.tensor([0.0, 0.9]))
    np.testing.assert_array_equal(
        evidence["anchor_valid"],
        [[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
    )
    for family in ("full", "in_span", "out_span", "random"):
        torch.testing.assert_close(
            a[f"{family}_pos"], -a[f"{family}_neg"], atol=0, rtol=0
        )
        assert not a[f"{family}_pos"][:, 128:].any()
    np.testing.assert_allclose(evidence["anchor_gradient_norms"][0], [5.0, 3.0, 4.0])


def test_random_control_does_not_depend_on_unseen_shot_length():
    np.testing.assert_array_equal(random_anchors(2, 613), random_anchors(7, 613)[:2])
    assert not np.array_equal(random_anchors(2, 613), random_anchors(2, 614))


def test_invalid_inputs_fail_explicitly():
    u = torch.zeros(1, 128)
    q = torch.eye(128, dtype=torch.float64)[:, :1]
    with pytest.raises(ValueError):
        direction_targets(u, torch.full_like(u, float("nan")), q, shot=3)


def test_real_gradient_uses_only_anchor_window_and_does_not_change_frozen_base():
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )
        windows = torch.randn(168, 128, 23)
        available = torch.ones_like(windows, dtype=torch.bool)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            native = model(windows, available)["logit"].sigmoid().numpy()
        full = anchor_gradient(model, (windows, available, native))
        short = anchor_gradient(model, (windows[:136], available[:136], native[:136]))
        torch.testing.assert_close(full[:1], short, atol=1e-6, rtol=1e-5)
        assert anchor_gradient(
            model, (windows[:12], available[:12], native[:12])
        ).shape == (0, 128)
        assert (full.norm(dim=1) > 0).all()
        assert all(v.grad is None for v in model.parameters())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
