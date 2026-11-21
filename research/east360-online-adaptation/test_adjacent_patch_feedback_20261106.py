"""Independent, CPU-only synthetic checks for adjacent-patch feedback."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import adjacent_patch_feedback_20261106 as feedback
from paired_diagnostic_features_20261106 import GROUPS


SEGMENTS = (range(0, 6), range(6, 12), range(12, 17))


def _levels(raw, available):
    """Reference the arrived-H25 eligibility by explicit channel enumeration."""
    result = torch.zeros_like(available)
    for row in range(len(raw)):
        for _, channels in GROUPS:
            for channel in channels:
                observed = available[row, :, channel] & torch.isfinite(
                    raw[row, :, channel]
                )
                if observed.sum() >= 13:
                    result[row, :, channel] = observed
    return result


def _reverse(raw, levels):
    result = raw.clone()
    for row in range(len(raw)):
        for _, channels in GROUPS:
            for channel in channels:
                positions = torch.nonzero(levels[row, :, channel], as_tuple=True)[0]
                result[row, positions, channel] = raw[row, positions.flip(0), channel]
    return result


def _reference(forecast, raw, available, center, scale):
    """Enumerate pairs and terms, with no vectorized production loss code."""
    levels = _levels(raw, available)
    pairs = levels[:, :-8] & levels[:, 8:]
    reverse = _reverse(raw, levels)
    all_rows, all_present = [], []
    for row in range(len(raw)):
        arms = []
        for target in (raw, reverse):
            terms = []
            for _, channels in GROUPS:
                for segment in SEGMENTS:
                    predicted, observed = [], []
                    for channel in channels:
                        for left in segment:
                            if pairs[row, left, channel]:
                                right = left + 8
                                predicted.append(
                                    forecast[row, right, channel]
                                    - forecast[row, left, channel]
                                )
                                hi = torch.asinh(
                                    (target[row, right, channel] - center[channel])
                                    / scale[channel]
                                ).to(forecast.dtype)
                                lo = torch.asinh(
                                    (target[row, left, channel] - center[channel])
                                    / scale[channel]
                                ).to(forecast.dtype)
                                observed.append(hi - lo)
                    terms.append(
                        F.smooth_l1_loss(
                            torch.stack(predicted), torch.stack(observed), beta=1
                        )
                        if predicted
                        else forecast.new_zeros(())
                    )
            arms.append(torch.stack(terms))
        all_rows.append(torch.stack(arms))
        all_present.append(
            torch.tensor(
                [
                    any(
                        pairs[row, left, channel]
                        for channel in channels
                        for left in segment
                    )
                    for _, channels in GROUPS
                    for segment in SEGMENTS
                ],
                dtype=torch.bool,
                device=forecast.device,
            )
        )
    return torch.stack(all_rows), torch.stack(all_present), pairs


@pytest.fixture
def synthetic():
    generator = torch.Generator(device="cpu").manual_seed(6106)
    center = torch.linspace(-0.8, 0.7, 23, dtype=torch.double)
    scale = torch.linspace(0.5, 1.8, 23, dtype=torch.double)
    time = torch.linspace(-1.0, 1.2, 25, dtype=torch.double)[None, :, None]
    raw = center + scale * torch.sinh(
        time + 0.2 * torch.randn((2, 25, 23), generator=generator, dtype=torch.double)
    )
    available = torch.rand((2, 25, 23), generator=generator) > 0.15
    available[0, :, 3] = False
    available[0, :12, 3] = True  # Only 12 levels: excluded completely.
    available[0, :, 4] = False
    irregular = [0, 1, 3, 5, 6, 8, 9, 12, 14, 15, 19, 21, 24]
    available[0, irregular, 4] = True  # Exactly 13, with an irregular mask.
    raw[1, 2, 14] = float("nan")
    raw[1, 8, 15] = float("inf")
    available[1, 2, 14] = available[1, 8, 15] = True
    model = nn.Linear(64, 25 * 23, dtype=torch.double).requires_grad_(False).eval()
    with torch.no_grad():
        model.weight.copy_(
            torch.randn(model.weight.shape, generator=generator, dtype=torch.double)
            * 0.03
        )
        model.bias.copy_(
            torch.randn(model.bias.shape, generator=generator, dtype=torch.double) * 0.2
        )
    return model, raw, available, center, scale


def test_exact_losses_masks_reversal_and_vjps_match_independent_reference(synthetic):
    model, raw, available, center, scale = synthetic
    width = torch.randn(2, 64, dtype=torch.double, requires_grad=True)
    width.grad = torch.full_like(width, 7)
    model.weight.grad = torch.full_like(model.weight, 0.25)
    before_grad = model.weight.grad.clone()
    calls = []

    def forward(state):
        calls.append(1)
        assert torch.is_grad_enabled() and state.is_leaf and state.requires_grad
        assert state.shape == (2, 64) and state.dtype == torch.double
        assert torch.equal(state, torch.zeros_like(state))
        return model(state).reshape(2, 25, 23)

    with torch.no_grad():
        result = feedback.vjps(forward, width, raw, available, center, scale)
        assert not torch.is_grad_enabled()
    assert calls == [1]
    assert set(result) == {"true", "reverse", "terms", "loss", "mask"}
    assert result["true"].shape == result["reverse"].shape == (2, 9, 64)
    assert result["terms"].shape == (2, 9) and result["terms"].dtype == torch.bool
    assert result["mask"].shape == (2, 17, 23) and result["mask"].dtype == torch.bool
    assert result["loss"].shape == (2, 2, 9)
    assert all(not value.requires_grad for value in result.values())
    assert all(value.device.type == "cpu" for value in result.values())
    assert torch.equal(width.grad, torch.full_like(width, 7))
    assert torch.equal(model.weight.grad, before_grad)
    assert all(not parameter.requires_grad for parameter in model.parameters())

    state = torch.zeros(2, 64, dtype=torch.double, requires_grad=True)
    expected, terms, pairs = _reference(
        model(state).reshape(2, 25, 23), raw, available, center, scale
    )
    assert not pairs[0, :, 3].any()
    assert _levels(raw, available)[0, :, 4].sum() == 13
    assert not pairs[:, :, 0].any()
    assert torch.equal(result["mask"], pairs)
    assert torch.equal(result["terms"], terms)
    torch.testing.assert_close(
        result["loss"], expected.detach(), rtol=1e-12, atol=1e-12
    )
    for arm, key in enumerate(("true", "reverse")):
        for row in range(2):
            for term in range(9):
                if terms[row, term]:
                    grad = torch.autograd.grad(
                        expected[row, arm, term], state, retain_graph=True
                    )[0]
                    torch.testing.assert_close(
                        result[key][row, term], grad[row], rtol=1e-11, atol=1e-12
                    )
                else:
                    assert not result[key][row, term].any()
    assert not torch.allclose(result["true"], result["reverse"])


def test_family_major_thirds_have_correct_sign_and_no_cross_normalization():
    dtype = torch.double
    center = torch.linspace(-0.4, 0.8, 23, dtype=dtype)
    scale = torch.linspace(0.7, 1.7, 23, dtype=dtype)
    basis = torch.zeros(25, 23, 64, dtype=dtype)
    raw = center.expand(1, 25, 23).clone()
    increments = (0.2, 0.6, 1.2)
    for family, (_, channels) in enumerate(GROUPS):
        for channel in channels:
            raw[0, :, channel] = center[channel] + scale[channel] * torch.sinh(
                (torch.arange(25, dtype=dtype) - 12) * increments[family] / 8
            )
            for segment, starts in enumerate(SEGMENTS):
                for t in range(8, 25):
                    basis[t, channel, 3 * family + segment] = basis[
                        t - 8, channel, 3 * family + segment
                    ] + (t - 8 in starts)
    width = torch.full((1, 64), float("nan"), dtype=dtype)

    def forward(state):
        return torch.einsum("bw,tcw->btc", state, basis)

    result = feedback.vjps(
        forward, width, raw, torch.ones_like(raw, dtype=torch.bool), center, scale
    )
    assert result["terms"].all()
    for family, increment in enumerate(increments):
        for segment in range(3):
            term = family * 3 + segment
            expected = torch.zeros(64, dtype=dtype)
            expected[term] = -min(increment, 1.0)
            torch.testing.assert_close(
                result["true"][0, term], expected, atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                result["reverse"][0, term], -expected, atol=1e-12, rtol=1e-12
            )
            loss = 0.5 * increment**2 if increment < 1 else increment - 0.5
            torch.testing.assert_close(
                result["loss"][0, :, term], torch.full((2,), loss, dtype=dtype)
            )


@pytest.mark.parametrize("arm", [0, 1])
def test_directional_finite_difference_for_each_term(synthetic, arm):
    model, raw, available, center, scale = synthetic
    result = feedback.vjps(
        lambda state: model(state).reshape(2, 25, 23),
        torch.zeros(2, 64, dtype=torch.double),
        raw,
        available,
        center,
        scale,
    )
    direction = torch.linspace(-0.7, 0.8, 64, dtype=torch.double)
    epsilon = 1e-5
    for row in range(2):
        delta = torch.zeros(2, 64, dtype=torch.double)
        delta[row] = epsilon * direction
        with torch.no_grad():
            plus = _reference(
                model(delta).reshape(2, 25, 23), raw, available, center, scale
            )[0]
            minus = _reference(
                model(-delta).reshape(2, 25, 23), raw, available, center, scale
            )[0]
        numerical = (plus[row, arm] - minus[row, arm]) / (2 * epsilon)
        analytical = (result[("true", "reverse")[arm]][row] * direction).sum(dim=1)
        torch.testing.assert_close(analytical, numerical, rtol=4e-6, atol=3e-9)


def test_missing_terms_and_irregular_reversal_are_exact():
    raw = torch.zeros(2, 25, 23, dtype=torch.double)
    available = torch.zeros_like(raw, dtype=torch.bool)
    raw[0, :, 3] = torch.arange(25, dtype=torch.double) / 30
    available[0, :12, 3] = True
    available[0, :13, 14] = True  # Pairs only in first third.
    raw[0, :, 14] = torch.arange(25, dtype=torch.double) / 7
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20]
    available[0, positions, 19] = True  # 13 levels, no lag-8 pair.
    available[1, 12:, 3] = True  # Pairs only in last third.
    raw[1, :, 3] = torch.arange(25, dtype=torch.double) / 11
    raw[1, 0, 14] = float("inf")
    available[1, 0, 14] = True
    center, scale = (
        torch.zeros(23, dtype=torch.double),
        torch.ones(23, dtype=torch.double),
    )

    def forward(state):
        return (
            state[:, :1, None].expand(-1, 25, 23)
            * torch.arange(25, dtype=torch.double)[None, :, None]
        )

    first = feedback.vjps(
        forward, torch.zeros(2, 64, dtype=torch.double), raw, available, center, scale
    )
    expected_terms = torch.tensor(
        [
            [False, False, False, True, False, False, False, False, False],
            [False, False, True, False, False, False, False, False, False],
        ]
    )
    assert torch.equal(first["terms"], expected_terms)
    for key in ("true", "reverse"):
        assert not first[key][~expected_terms].any()
    assert not first["loss"].transpose(1, 2)[~expected_terms].any()
    assert not first["mask"][0, :, 19].any()
    reversed_raw = _reverse(raw, _levels(raw, available))
    second = feedback.vjps(
        forward,
        torch.zeros(2, 64, dtype=torch.double),
        reversed_raw,
        available,
        center,
        scale,
    )
    torch.testing.assert_close(first["reverse"], second["true"], rtol=0, atol=0)
    torch.testing.assert_close(
        first["loss"][:, 1], second["loss"][:, 0], rtol=0, atol=0
    )
    assert torch.equal(first["mask"], second["mask"])


def test_reversal_matches_independent_enumeration_for_varied_masks():
    generator = torch.Generator(device="cpu").manual_seed(8345)
    raw = torch.randn(7, 25, 23, dtype=torch.double, generator=generator)
    probabilities = torch.linspace(0.35, 0.95, 7)[:, None, None]
    available = torch.rand((7, 25, 23), generator=generator) < probabilities
    raw[0, 0, 3] = float("nan")
    raw[1, 4, 14] = float("inf")
    available[6] = False
    levels = _levels(raw, available)
    actual = feedback._reverse_observed(raw, levels)
    expected = _reverse(raw, levels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        feedback._reverse_observed(actual, levels), raw, rtol=0, atol=0, equal_nan=True
    )


def test_cross_batch_coupling_and_masked_poison_do_not_contaminate():
    dtype = torch.double
    width = torch.zeros(3, 64, dtype=dtype)
    raw = torch.zeros(3, 25, 23, dtype=dtype)
    available = torch.zeros_like(raw, dtype=torch.bool)
    for row in (0, 1):
        available[row, :, 3] = True
        raw[row, :, 3] = torch.sinh(torch.arange(25, dtype=dtype) * (row + 1) / 80)
    raw[~available] = float("nan")
    center, scale = torch.zeros(23, dtype=dtype), torch.ones(23, dtype=dtype)
    output_valid = available.clone()

    def forward(state):
        mixed = state[:, 0] + 20 * state[:, 0].roll(1)
        prediction = (
            mixed[:, None, None] * torch.arange(25, dtype=dtype)[None, :, None] / 8
        )
        prediction = prediction.expand(-1, 25, 23)
        return torch.where(
            output_valid, prediction, torch.full_like(prediction, float("nan"))
        )

    result = feedback.vjps(forward, width, raw, available, center, scale)
    assert result["terms"][:2, :3].all() and not result["terms"][2].any()
    torch.testing.assert_close(
        result["true"][0, :3, 0], torch.full((3,), -0.1, dtype=dtype)
    )
    torch.testing.assert_close(
        result["true"][1, :3, 0], torch.full((3,), -0.2, dtype=dtype)
    )
    assert not result["true"][2].any() and not result["reverse"][2].any()
    poisoned = raw.clone()
    poisoned[~available] = -1e200
    repeated = feedback.vjps(forward, width, poisoned, available, center, scale)
    for key in result:
        torch.testing.assert_close(result[key], repeated[key], rtol=0, atol=0)


def test_masked_forecast_positions_have_no_gradient():
    dtype = torch.double
    raw = torch.zeros(1, 25, 23, dtype=dtype)
    available = torch.zeros_like(raw, dtype=torch.bool)
    available[0, :13, 3] = True
    raw[0, :, 3] = torch.sinh(torch.arange(25, dtype=dtype) / 80)
    raw[0, 13:, 3] = float("inf")

    def forward(state):
        return state[:, :25, None].expand(-1, -1, 23)

    result = feedback.vjps(
        forward,
        torch.zeros(1, 64, dtype=dtype),
        raw,
        available,
        torch.zeros(23, dtype=dtype),
        torch.ones(23, dtype=dtype),
    )
    assert result["terms"].tolist() == [[True] + [False] * 8]
    assert result["mask"][0, :, 3].tolist() == [True] * 5 + [False] * 12
    expected = torch.zeros(64, dtype=dtype)
    expected[:5] = 0.02
    expected[8:13] = -0.02
    torch.testing.assert_close(result["true"][0, 0], expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        result["reverse"][0, 0], -expected, atol=1e-14, rtol=1e-14
    )
    assert not result["true"][0, 1:].any()


def test_independent_rows_fast_path_matches_exact_and_uses_18_backward_calls(
    synthetic, monkeypatch
):
    model, raw, available, center, scale = synthetic
    available[0, :, list(GROUPS[0][1])] = False
    available[1, :, list(GROUPS[2][1])] = False
    width = torch.randn(2, 64, dtype=torch.double)
    callbacks = []

    def forward(state):
        callbacks.append(1)
        return model(state).reshape(2, 25, 23)

    exact = feedback.vjps(forward, width, raw, available, center, scale)
    calls = []
    original = torch.autograd.grad

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.autograd, "grad", counted)
        with torch.no_grad():
            fast = feedback.vjps(
                forward, width, raw, available, center, scale, independent_rows=True
            )
            assert not torch.is_grad_enabled()
    assert callbacks == [1, 1]
    assert len(calls) == 18
    assert not fast["terms"][0, :3].any() and not fast["terms"][1, 6:].any()
    for key in exact:
        torch.testing.assert_close(fast[key], exact[key], rtol=1e-11, atol=1e-12)
    for key in ("true", "reverse"):
        assert not fast[key][~fast["terms"]].any()


@pytest.mark.parametrize("independent_rows", [False, True])
def test_caller_gradient_flags_and_existing_gradients_are_untouched(
    synthetic, independent_rows
):
    model, raw, available, center, scale = synthetic
    model.weight.requires_grad_(True)
    model.weight.grad = torch.full_like(model.weight, 0.42)
    before = model.weight.grad.clone()
    width = torch.full((2, 64), 19.0, dtype=torch.double, requires_grad=True)
    width.grad = torch.full_like(width, -3)
    outer_flags = []

    def forward(state):
        outer_flags.append(torch.is_grad_enabled())
        return model(state).reshape(2, 25, 23)

    with torch.no_grad():
        result = feedback.vjps(
            forward,
            width,
            raw,
            available,
            center,
            scale,
            independent_rows=independent_rows,
        )
        assert not torch.is_grad_enabled()
    assert outer_flags == [True] and model.weight.requires_grad
    assert not model.bias.requires_grad
    assert torch.equal(model.weight.grad, before)
    assert torch.equal(width.grad, torch.full_like(width, -3))
    assert result["true"].abs().sum() > 0


def test_numpy_float64_targets_with_float32_width_and_forecast(synthetic):
    model, raw, available, center, scale = synthetic
    model = model.float()
    width = torch.zeros(2, 64, dtype=torch.float32)

    def forward(state):
        return model(state).reshape(2, 25, 23)

    result = feedback.vjps(
        forward,
        width,
        raw.numpy(),
        available.numpy(),
        center.numpy(),
        scale.numpy(),
        independent_rows=True,
    )
    assert all(
        result[key].dtype == torch.float32 for key in ("true", "reverse", "loss")
    )
    state = torch.zeros_like(width, requires_grad=True)
    expected, terms, pairs = _reference(forward(state), raw, available, center, scale)
    assert torch.equal(result["terms"], terms)
    assert torch.equal(result["mask"], pairs)
    torch.testing.assert_close(result["loss"], expected.detach(), rtol=1e-6, atol=1e-7)
    for arm, key in enumerate(("true", "reverse")):
        for term in range(9):
            gradient = torch.autograd.grad(
                expected[:, arm, term].sum(), state, retain_graph=True
            )[0]
            torch.testing.assert_close(
                result[key][:, term], gradient, rtol=2e-5, atol=1e-7
            )


@pytest.mark.parametrize("independent_rows", [False, True])
def test_constant_forecast_produces_zero_gradients_and_empty_batch_short_circuits(
    independent_rows,
):
    width = torch.zeros(1, 64)
    raw = torch.ones(1, 25, 23)
    available = torch.ones_like(raw, dtype=torch.bool)
    center, scale = torch.zeros(23), torch.ones(23)
    result = feedback.vjps(
        lambda _: torch.zeros_like(raw),
        width,
        raw,
        available,
        center,
        scale,
        independent_rows=independent_rows,
    )
    assert result["terms"].all()
    assert not result["true"].any() and not result["reverse"].any()

    def must_not_run(_):
        raise AssertionError("empty batches must not call forward")

    empty = feedback.vjps(
        must_not_run,
        width[:0],
        raw[:0],
        available[:0],
        center,
        scale,
        independent_rows=independent_rows,
    )
    assert empty["true"].shape == empty["reverse"].shape == (0, 9, 64)
    assert empty["terms"].shape == (0, 9)
    assert empty["loss"].shape == (0, 2, 9)
    assert empty["mask"].shape == (0, 17, 23)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], torch.tensor(True)])
def test_independent_rows_must_be_a_keyword_boolean(invalid):
    width = torch.zeros(1, 64)
    raw = torch.ones(1, 25, 23)
    available = torch.ones_like(raw, dtype=torch.bool)
    center, scale = torch.zeros(23), torch.ones(23)

    def forward(state):
        return state[:, :1, None].expand(-1, 25, 23)

    with pytest.raises(ValueError, match="independent_rows"):
        feedback.vjps(
            forward, width, raw, available, center, scale, independent_rows=invalid
        )
    with pytest.raises(TypeError):
        feedback.vjps(forward, width, raw, available, center, scale, True)


def test_input_validation_and_nonfinite_observed_forecast():
    width, raw = torch.zeros(1, 64), torch.ones(1, 25, 23)
    available = torch.ones_like(raw, dtype=torch.bool)
    center, scale = torch.zeros(23), torch.ones(23)

    def forward(state):
        return state[:, :1, None].expand(-1, 25, 23)

    bad_cases = [
        (forward, width[:, :63], raw, available, center, scale),
        (forward, width.long(), raw, available, center, scale),
        (forward, width, raw[:, :24], available[:, :24], center, scale),
        (forward, width, raw, available.float(), center, scale),
        (forward, width, raw, available, center[:22], scale),
        (forward, width, raw, available, center, torch.zeros(23)),
        (lambda state: state, width, raw, available, center, scale),
    ]
    for args in bad_cases:
        with pytest.raises(ValueError):
            feedback.vjps(*args)
    with pytest.raises(FloatingPointError):
        feedback.vjps(
            lambda state: torch.full_like(raw, float("nan")),
            width,
            raw,
            available,
            center,
            scale,
        )
