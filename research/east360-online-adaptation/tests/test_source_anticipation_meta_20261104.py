"""CPU-only teacher-objective tests: trusted source and synthetic inputs only."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
    sha256,
)
from experiments.risk_aligned_online.source_anticipation_20261104 import meta


@pytest.fixture(scope="module")
def source():
    assert sha256(DEFAULT_PARENT) == PARENT_SHA256
    torch.set_num_threads(1)
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = load_parent(DEFAULT_PARENT)
    assert base.center.device.type == "cpu"
    yield base
    torch.backends.mha.set_fastpath_enabled(old_fastpath)


@pytest.fixture(autouse=True)
def source_is_immutable(source):
    snapshot = {k: v.detach().clone() for k, v in source.state_dict().items()}
    hooks = {
        n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for n, m in source.named_modules()
    }
    yield
    assert all(not m.training for m in source.modules())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in snapshot.items())
    assert hooks == {
        n: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for n, m in source.named_modules()
    }


@pytest.fixture
def sample(source):
    g = torch.Generator(device="cpu").manual_seed(20261104)
    origin = source.center + source.scale * 0.3 * torch.randn(2, 128, 23, generator=g)
    query = source.center + source.scale * 0.3 * torch.randn(2, 128, 23, generator=g)
    truth = source.center + source.scale * 0.7 * torch.randn(2, 25, 23, generator=g)
    return dict(
        origin_x=origin,
        origin_mask=torch.rand(2, 128, 23, generator=g) > 0.1,
        query_x=query,
        query_mask=torch.rand(2, 128, 23, generator=g) > 0.1,
        truth25=truth,
        target_mask=torch.rand(2, 25, 23, generator=g) > 0.2,
        labels=torch.tensor([0.0, 1.0]),
    )


@pytest.fixture
def teacher(source):
    head = deepcopy(source.forecast_head).requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    return head, initial


def test_zero_offset_matches_native_forward_and_keeps_feature_graph(
    source, sample, teacher
):
    head, _ = teacher
    x, mask = sample["origin_x"], sample["origin_mask"]
    captured = []
    hook = source.forecast_head.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0])
    )
    try:
        with torch.no_grad():
            native = source(x, mask)
    finally:
        hook.remove()
    delta = torch.zeros(2, 128, requires_grad=True)
    actual = meta.features(source, x, mask, delta)
    assert torch.equal(actual["logit"], native["logit"])
    assert torch.equal(actual["features"], captured[0])
    assert torch.equal(head(actual["features"]).reshape(2, 25, 23), native["forecast"])
    gradient = torch.autograd.grad(head(actual["features"]).square().mean(), delta)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_objective_teacher_sign_and_alignment_head_gradient(source, sample, teacher):
    head, initial = teacher
    loss, info = meta.objective(source, head, **sample, initial_head_state=initial)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.allclose(
        loss, info["alignment"] + 0.20 * info["absolute_proxy"] + 1e-4 * info["anchor"]
    )
    assert info["observed_rows"] == info["alignment_rows"] == 2
    assert info["unobserved_rows"] == info["zero_desired_rows"] == 0
    assert info["anchor"] == 0
    direction = torch.autograd.grad(info["alignment"], head.weight, retain_graph=True)[
        0
    ]
    assert torch.isfinite(direction).all() and direction.norm() > 0
    assert not info["desired_gradient"].requires_grad
    delta = torch.zeros(2, 128, requires_grad=True)
    z = meta.features(source, sample["query_x"], sample["query_mask"], delta)["logit"]
    risk_gradient = torch.autograd.grad(z.sum(), delta)[0]
    expected = (1 - 2 * sample["labels"][:, None]) * risk_gradient
    torch.testing.assert_close(info["desired_gradient"], expected, rtol=0, atol=0)
    g = info["proxy_gradient"]
    d = info["desired_gradient"]
    cosine = (g * d).sum(1) / (
        (g.square().sum(1) + 1e-12).sqrt() * (d.square().sum(1) + 1e-12).sqrt()
    )
    torch.testing.assert_close(info["alignment"], (1 - cosine).mean())


def test_proxy_is_native_absolute_loss_and_backward_only_touches_head(
    source, sample, teacher
):
    head, initial = teacher
    loss, info = meta.objective(source, head, **sample, initial_head_state=initial)
    delta = torch.zeros(2, 128, requires_grad=True)
    origin = meta.features(source, sample["origin_x"], sample["origin_mask"], delta)
    prediction = head(origin["features"]).reshape(2, 25, 23)
    expected = source.forecast_loss(
        prediction, sample["truth25"], sample["target_mask"]
    )
    expected_gradient = torch.autograd.grad(expected, delta)[0]
    torch.testing.assert_close(info["absolute_proxy"], expected, atol=0, rtol=0)
    torch.testing.assert_close(
        info["proxy_gradient"], expected_gradient, atol=0, rtol=0
    )
    loss.backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.norm() > 0
    assert head.bias.grad is not None and torch.isfinite(head.bias.grad).all()
    assert all(p.grad is None for p in source.parameters())


@pytest.mark.parametrize(
    ("dtype", "epsilon", "relative_tolerance", "absolute_tolerance"),
    [(torch.float32, 1e-3, 5e-4, 1e-5), (torch.float64, 2e-5, 3e-4, 2e-7)],
)
def test_source_alignment_directional_derivative(
    source, sample, dtype, epsilon, relative_tolerance, absolute_tolerance
):
    base = deepcopy(source).to(dtype=dtype)
    values = {
        k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in sample.items()
    }
    head = deepcopy(base.forecast_head).requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    _, info = meta.objective(base, head, **values, initial_head_state=initial)
    gradient = torch.autograd.grad(info["alignment"], head.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    direction = (gradient / gradient.norm()).detach()
    analytic = float((gradient * direction).sum())
    with torch.no_grad():
        original = head.weight.detach().clone()
    try:
        with torch.no_grad():
            head.weight.copy_(original + epsilon * direction)
        _, plus = meta.objective(base, head, **values, initial_head_state=initial)
        with torch.no_grad():
            head.weight.copy_(original - epsilon * direction)
        _, minus = meta.objective(base, head, **values, initial_head_state=initial)
        numerical = float((plus["alignment"] - minus["alignment"]).detach()) / (
            2 * epsilon
        )
    finally:
        with torch.no_grad():
            head.weight.copy_(original)
    assert numerical == pytest.approx(
        analytic, rel=relative_tolerance, abs=absolute_tolerance
    )
    assert all(p.grad is None for p in base.parameters())


def test_inner_state_matches_native_rule_and_optionally_preserves_outer_graph(
    source, sample, teacher
):
    head, initial = teacher
    _, info = meta.objective(source, head, **sample, initial_head_state=initial)
    delta, gradient = info["delta"], info["proxy_gradient"]
    origin = source.initial_fast(2)
    expected = (
        normalized_step(origin + delta, origin, gradient, step=0.05, radius=1.0)
        - origin
    )
    with torch.no_grad():
        graph_state = meta.inner_state(source, delta, gradient, create_graph=True)
    detached_state = meta.inner_state(source, delta, gradient, create_graph=False)
    torch.testing.assert_close(graph_state, expected, atol=0, rtol=0)
    torch.testing.assert_close(detached_state, expected.detach(), atol=0, rtol=0)
    assert graph_state.requires_grad and not detached_state.requires_grad
    assert (detached_state.norm(dim=1) <= 1.0 + 1e-6).all()
    query = meta.features(source, sample["query_x"], sample["query_mask"], graph_state)[
        "logit"
    ]
    outer = torch.autograd.grad(query.sum(), head.weight)[0]
    assert torch.isfinite(outer).all() and outer.norm() > 0


def test_anchor_is_mean_over_all_head_parameters_and_snapshot_is_unchanged(
    source, sample, teacher
):
    head, initial = teacher
    snapshots = {k: v.clone() for k, v in initial.items()}
    with torch.no_grad():
        head.bias[0] += 0.25
        head.weight[0, 0] -= 0.5
    loss, info = meta.objective(source, head, **sample, initial_head_state=initial)
    expected = (
        (head.weight - initial["weight"]).square().sum()
        + (head.bias - initial["bias"]).square().sum()
    ) / sum(p.numel() for p in head.parameters())
    torch.testing.assert_close(info["anchor"], expected, atol=0, rtol=0)
    torch.testing.assert_close(
        loss, info["alignment"] + 0.20 * info["absolute_proxy"] + 1e-4 * expected
    )
    assert all(torch.equal(initial[k], snapshots[k]) for k in snapshots)


@pytest.mark.parametrize("bad", ["shape", "gradient_nan", "nonboolean_graph"])
def test_inner_state_rejects_invalid_arguments(source, bad):
    delta = torch.zeros(2, 128)
    gradient = torch.zeros_like(delta)
    options = {}
    if bad == "shape":
        gradient = gradient[:, :-1]
    if bad == "gradient_nan":
        gradient[0, 0] = float("nan")
    if bad == "nonboolean_graph":
        options["create_graph"] = "false"
    with pytest.raises((ValueError, TypeError)):
        meta.inner_state(source, delta, gradient, **options)


def test_mixed_support_and_zero_proxy_are_reported(source, sample, teacher):
    head, initial = teacher
    sample["target_mask"][1] = False
    sample["truth25"][1] = float("nan")
    _, info = meta.objective(source, head, **sample, initial_head_state=initial)
    assert (info["observed_rows"], info["unobserved_rows"], info["alignment_rows"]) == (
        1,
        1,
        1,
    )
    assert not info["proxy_gradient"][1].any()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    _, info = meta.objective(source, head, **sample, initial_head_state=initial)
    assert info["zero_proxy_rows"] == 1 and info["alignment_rows"] == 1
    assert info["alignment"] == 1


def test_all_unobserved_or_zero_teacher_is_explicit(
    source, sample, teacher, monkeypatch
):
    head, initial = teacher
    no_target = dict(sample, target_mask=torch.zeros_like(sample["target_mask"]))
    with monkeypatch.context() as m:
        m.setattr(
            source,
            "forward",
            lambda *_a, **_k: pytest.fail("no-target must not forward"),
        )
        with pytest.raises(ValueError, match="no_target"):
            meta.objective(source, head, **no_target, initial_head_state=initial)
    no_teacher = dict(sample, query_mask=torch.zeros_like(sample["query_mask"]))
    with pytest.raises(ValueError, match="no_teacher_direction"):
        meta.objective(source, head, **no_teacher, initial_head_state=initial)


def test_mixed_zero_teacher_is_excluded_but_reported(source, sample, teacher):
    head, initial = teacher
    sample["query_mask"][1] = False
    _, info = meta.objective(source, head, **sample, initial_head_state=initial)
    assert info["observed_rows"] == 2 and info["zero_desired_rows"] == 1
    assert info["alignment_rows"] == 1 and torch.isfinite(info["alignment"])


@pytest.mark.parametrize(
    "bad",
    [
        "labels_shape",
        "labels_value",
        "labels_nan",
        "truth_shape",
        "truth_nan",
        "mask_dtype",
        "query_shape",
    ],
)
def test_invalid_targets_labels_or_inputs_raise(source, sample, teacher, bad):
    head, initial = teacher
    values = dict(sample)
    if bad == "labels_shape":
        values["labels"] = torch.ones(2, 1)
    if bad == "labels_value":
        values["labels"] = torch.tensor([0.0, 0.5])
    if bad == "labels_nan":
        values["labels"] = torch.tensor([0.0, float("nan")])
    if bad == "truth_shape":
        values["truth25"] = values["truth25"][:, :-1]
    if bad == "truth_nan":
        values["truth25"] = values["truth25"].clone()
        values["target_mask"] = torch.ones_like(values["target_mask"])
        values["truth25"][0, 0, 0] = float("nan")
    if bad == "mask_dtype":
        values["target_mask"] = values["target_mask"].float()
    if bad == "query_shape":
        values["query_x"] = values["query_x"][:, :-1]
    with pytest.raises((ValueError, TypeError)):
        meta.objective(source, head, **values, initial_head_state=initial)


def test_trainable_head_independent_snapshot_and_frozen_base_required(
    source, sample, teacher
):
    head, initial = teacher
    shared = nn.Linear(1088, 575)
    shared.weight = nn.Parameter(source.forecast_head.weight.detach())
    with pytest.raises(ValueError, match="storage"):
        meta.objective(source, shared, **sample, initial_head_state=initial)
    with pytest.raises(ValueError, match="snapshot"):
        meta.objective(source, head, **sample, initial_head_state=head.state_dict())
    frozen = deepcopy(head).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        meta.objective(source, frozen, **sample, initial_head_state=initial)
    unfrozen = deepcopy(source).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        meta.objective(unfrozen, head, **sample, initial_head_state=initial)


def test_hook_cleanup_on_forward_exception_preserves_other_hooks(
    source, sample, monkeypatch
):
    sentinel = source.forecast_head.register_forward_pre_hook(lambda *_args: None)
    existing = tuple(source.forecast_head._forward_pre_hooks)
    try:
        with monkeypatch.context() as m:

            def fail(*_args, **_kwargs):
                raise RuntimeError("injected failure")

            m.setattr(source, "forward", fail)
            with pytest.raises(RuntimeError, match="injected failure"):
                meta.features(
                    source,
                    sample["origin_x"],
                    sample["origin_mask"],
                    torch.zeros(2, 128),
                )
        assert tuple(source.forecast_head._forward_pre_hooks) == existing
    finally:
        sentinel.remove()
