"""CPU-only actual-update utility tests with a hash-locked source and synthetic data."""

from copy import deepcopy
import math

import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
    sha256,
)
from experiments.risk_aligned_online.source_anticipation_20261104 import meta, utility


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
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    hooks = {
        name: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for name, m in source.named_modules()
    }
    yield
    assert all(not m.training for m in source.modules())
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in state.items())
    assert hooks == {
        name: (tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
        for name, m in source.named_modules()
    }


@pytest.fixture
def sample(source):
    rng = torch.Generator(device="cpu").manual_seed(20261104)
    origin = source.center + source.scale * 0.3 * torch.randn(2, 128, 23, generator=rng)
    query = source.center + source.scale * 0.3 * torch.randn(2, 128, 23, generator=rng)
    truth = source.center + source.scale * 0.7 * torch.randn(2, 25, 23, generator=rng)
    return dict(
        origin_x=origin,
        origin_mask=torch.rand(2, 128, 23, generator=rng) > 0.1,
        query_x=query,
        query_mask=torch.rand(2, 128, 23, generator=rng) > 0.1,
        truth25=truth,
        target_mask=torch.rand(2, 25, 23, generator=rng) > 0.2,
        labels=torch.tensor([0.0, 1.0]),
    )


@pytest.fixture
def teacher(source):
    head = deepcopy(source.forecast_head).requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    return head, initial


def test_actual_step_double_bce_and_detached_scale_match_independent_formula(
    source, sample, teacher
):
    head, initial = teacher
    threshold = 0.7258946895599365
    with torch.no_grad():
        loss, info = utility.objective(
            source, head, **sample, initial_head_state=initial, threshold=threshold
        )
    assert loss.requires_grad and loss.dtype == torch.float64
    _, reference = meta.objective(source, head, **sample, initial_head_state=initial)
    torch.testing.assert_close(
        info["alignment"], reference["alignment"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        info["absolute_proxy"], reference["absolute_proxy"], atol=0, rtol=0
    )
    torch.testing.assert_close(info["anchor"], reference["anchor"], atol=0, rtol=0)
    origin = source.initial_fast(2)
    expected_state = (
        normalized_step(
            origin + info["delta"],
            origin,
            info["proxy_gradient"],
            step=0.05,
            radius=1.0,
        )
        - origin
    )
    torch.testing.assert_close(info["updated_state"], expected_state, atol=0, rtol=0)
    assert info["updated_state"].requires_grad
    assert (info["updated_state"].detach().norm(dim=1) <= 1.0 + 1e-6).all()
    with torch.no_grad():
        before = meta.features(
            source, sample["query_x"], sample["query_mask"], info["delta"].detach()
        )["logit"]
    after = meta.features(
        source, sample["query_x"], sample["query_mask"], info["updated_state"]
    )["logit"]
    cut = math.log(threshold / (1 - threshold))
    target = sample["labels"].double()
    old_bce = F.binary_cross_entropy_with_logits(
        before.double() - cut, target, reduction="none"
    )
    new_bce = F.binary_cross_entropy_with_logits(
        after.double() - cut, target, reduction="none"
    )
    expected_delta = new_bce - old_bce
    expected_sensitivity = (
        info["desired_gradient"].double().square().sum(1) + 1e-12
    ).sqrt()
    expected_scale = 0.05 * expected_sensitivity
    torch.testing.assert_close(info["perrowrisk_delta"], expected_delta, atol=0, rtol=0)
    torch.testing.assert_close(
        info["sensitivity"], expected_sensitivity, atol=0, rtol=0
    )
    torch.testing.assert_close(
        info["normalization_scale"], expected_scale, atol=0, rtol=0
    )
    torch.testing.assert_close(
        info["risk_utility"], (expected_delta / expected_scale).mean(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        loss,
        info["risk_utility"] + 0.20 * info["absolute_proxy"] + 1e-4 * info["anchor"],
    )
    assert info["risk_utility_rows"] == 2
    assert not info["baseline_logit"].requires_grad
    assert (
        not info["sensitivity"].requires_grad
        and not info["normalization_scale"].requires_grad
    )
    assert info["perrowrisk_delta"].dtype == torch.float64
    gradient = torch.autograd.grad(
        info["risk_utility"], head.weight, retain_graph=True
    )[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    loss.backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.norm() > 0 and all(
        p.grad is None for p in source.parameters()
    )


def test_bce_delta_keeps_confidence_under_identical_logit_move():
    before = torch.tensor(
        [12.0, -12.0, 12.0, -12.0], dtype=torch.float64, requires_grad=True
    )
    after = (before.detach() + 0.2).requires_grad_(True)
    labels = torch.tensor([1, 0, 0, 1])
    delta = utility._margin_delta(before, after, labels, 0.0)
    expected = torch.logaddexp(
        torch.zeros(4), (1 - 2 * labels) * after
    ) - torch.logaddexp(torch.zeros(4), (1 - 2 * labels) * before.detach())
    torch.testing.assert_close(delta, expected, rtol=1e-10, atol=2e-15)
    assert abs(delta[0]) < 1e-4 * abs(delta[2])
    assert abs(delta[1]) < 1e-4 * abs(delta[3])
    gb, ga = torch.autograd.grad(delta.sum(), (before, after), allow_unused=True)
    assert gb is None and ga is not None and torch.isfinite(ga).all()


def test_full_utility_preserves_confidence_weighting(source, teacher, monkeypatch):
    head, initial = teacher
    before = torch.tensor([12.0, -12.0, 12.0, -12.0], dtype=torch.float64)
    labels = torch.tensor([1, 0, 0, 1])
    zero = torch.zeros(4, 128, dtype=torch.float64, requires_grad=True)
    desired = torch.ones(4, 128, dtype=torch.float64)
    diagnostic_alignment = torch.tensor(0.75, dtype=torch.float64)

    def teacher_objective(*_args, **_kwargs):
        return torch.tensor(0.0), dict(
            delta=zero,
            proxy_gradient=zero,
            desired_gradient=desired,
            alignment=diagnostic_alignment,
            absolute_proxy=torch.tensor(0.0, dtype=torch.float64),
            anchor=torch.tensor(0.0, dtype=torch.float64),
        )

    def inner(*_args, **_kwargs):
        return torch.ones(4, 128, dtype=torch.float64, requires_grad=True)

    def fake_features(_base, _x, _mask, state):
        return {"logit": before + 0.2 * state[:, 0]}

    with monkeypatch.context() as m:
        m.setattr(meta, "objective", teacher_objective)
        m.setattr(meta, "inner_state", inner)
        m.setattr(meta, "features", fake_features)
        loss, info = utility.objective(
            source,
            head,
            torch.zeros(4, 128, 23),
            torch.ones(4, 128, 23, dtype=torch.bool),
            torch.zeros(4, 128, 23),
            torch.ones(4, 128, 23, dtype=torch.bool),
            torch.zeros(4, 25, 23),
            torch.ones(4, 25, 23, dtype=torch.bool),
            labels,
            initial_head_state=initial,
            threshold=0.5,
        )
    per_row = info["perrowrisk_delta"] / info["normalization_scale"]
    assert abs(per_row[0]) < 1e-4 * abs(per_row[2])
    assert abs(per_row[1]) < 1e-4 * abs(per_row[3])
    torch.testing.assert_close(loss, per_row.mean())
    assert info["alignment"] is diagnostic_alignment and info["risk_utility_rows"] == 4


@pytest.mark.parametrize("unsupported", ["unobserved", "zero_sensitivity"])
def test_mixed_support_averages_only_observed_sensitive_rows(
    source, sample, teacher, unsupported
):
    head, initial = teacher
    if unsupported == "unobserved":
        sample["target_mask"][1] = False
        sample["truth25"][1] = float("nan")
    else:
        sample["query_mask"][1] = False
    loss, info = utility.objective(
        source, head, **sample, initial_head_state=initial, threshold=0.5
    )
    assert torch.isfinite(loss) and info["risk_utility_rows"] == 1
    if unsupported == "unobserved":
        assert info["unobserved_rows"] == 1 and not info["updated_state"][1].any()
    else:
        assert info["zero_desired_rows"] == 1 and not info["desired_gradient"][1].any()
    torch.testing.assert_close(
        info["risk_utility"],
        info["perrowrisk_delta"][0] / info["normalization_scale"][0],
    )


@pytest.mark.parametrize("unsupported", ["unobserved", "zero_sensitivity"])
def test_no_supported_rows_raise_explicitly(source, sample, teacher, unsupported):
    head, initial = teacher
    if unsupported == "unobserved":
        sample["target_mask"][:] = False
        expected = "no_target"
    else:
        sample["query_mask"][:] = False
        expected = "no_teacher_direction"
    with pytest.raises(ValueError, match=expected):
        utility.objective(
            source, head, **sample, initial_head_state=initial, threshold=0.5
        )


@pytest.mark.parametrize(
    "threshold", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"]
)
def test_invalid_probability_threshold_rejected(source, sample, teacher, threshold):
    head, initial = teacher
    with pytest.raises((TypeError, ValueError), match="threshold"):
        utility.objective(
            source, head, **sample, initial_head_state=initial, threshold=threshold
        )


def test_invalid_meta_input_and_aliased_head_are_rejected(source, sample, teacher):
    head, initial = teacher
    with pytest.raises(ValueError, match="binary"):
        utility.objective(
            source,
            head,
            **dict(sample, labels=torch.tensor([0.0, 0.5])),
            initial_head_state=initial,
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="snapshot"):
        utility.objective(
            source, head, **sample, initial_head_state=head.state_dict(), threshold=0.5
        )


def test_hooks_cleanup_if_baseline_forward_raises(source, sample, teacher, monkeypatch):
    head, initial = teacher
    sentinel = source.forecast_head.register_forward_pre_hook(lambda *_args: None)
    hooks = tuple(source.forecast_head._forward_pre_hooks)
    old_forward = source.forward
    calls = []

    def fail_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("baseline failure")
        return old_forward(*args, **kwargs)

    try:
        with monkeypatch.context() as m:
            m.setattr(source, "forward", fail_third)
            with pytest.raises(RuntimeError, match="baseline failure"):
                utility.objective(
                    source, head, **sample, initial_head_state=initial, threshold=0.5
                )
        assert tuple(source.forecast_head._forward_pre_hooks) == hooks
    finally:
        sentinel.remove()


def test_float64_actual_utility_directional_derivative(source, sample):
    base = deepcopy(source).double()
    values = {k: v.double() if v.is_floating_point() else v for k, v in sample.items()}
    head = deepcopy(base.forecast_head).requires_grad_(True)
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    _, info = utility.objective(
        base, head, **values, initial_head_state=initial, threshold=0.7258946895599365
    )
    gradient = torch.autograd.grad(info["risk_utility"], head.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    direction = (gradient / gradient.norm()).detach()
    analytic = float((gradient * direction).sum())
    epsilon = 2e-5
    original = head.weight.detach().clone()
    try:
        with torch.no_grad():
            head.weight.copy_(original + epsilon * direction)
        _, plus = utility.objective(
            base,
            head,
            **values,
            initial_head_state=initial,
            threshold=0.7258946895599365,
        )
        with torch.no_grad():
            head.weight.copy_(original - epsilon * direction)
        _, minus = utility.objective(
            base,
            head,
            **values,
            initial_head_state=initial,
            threshold=0.7258946895599365,
        )
        numerical = float((plus["risk_utility"] - minus["risk_utility"]).detach()) / (
            2 * epsilon
        )
    finally:
        with torch.no_grad():
            head.weight.copy_(original)
    assert numerical == pytest.approx(analytic, rel=5e-4, abs=2e-7)
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize("fastpath", [False, True])
def test_zero_update_has_exactly_zero_utility_on_same_execution_path(
    source, sample, teacher, fastpath
):
    head, _ = teacher
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        complete = torch.ones_like(sample["origin_mask"])
        _, info = utility.objective(
            source,
            head,
            **dict(
                sample,
                origin_mask=complete,
                query_x=sample["origin_x"],
                query_mask=complete,
            ),
            initial_head_state=initial,
            threshold=0.7258946895599365,
        )
        assert not info["updated_state"].detach().any()
        assert not info["perrowrisk_delta"].detach().any()
        assert info["risk_utility"].detach().item() == 0.0
        assert not info["baseline_logit"].requires_grad
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
