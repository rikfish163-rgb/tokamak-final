import pytest
import torch
from types import SimpleNamespace

from experiments.risk_aligned_online.east_decoder_calibration_20261102 import (
    functional_update as engine,
)
from experiments.risk_aligned_online.east_decoder_calibration_20261102.functional_stream import (
    predict,
    choices,
)
from experiments.risk_aligned_online.east_decoder_calibration_20261102.functional_update import (
    protection,
    first_improvement,
)


def test_hidden_protection_uses_function_change_not_arbitrary_factor_scale():
    phi = torch.arange(64, dtype=torch.float32).reshape(1, 16, 4) / 100
    pack = dict(phi=phi, valid=torch.ones(1, 17, dtype=torch.bool))
    state = torch.ones(4, 64)
    original = protection(pack, state, "hidden")
    rescaled = protection(dict(phi=phi * 10, valid=pack["valid"]), state / 10, "hidden")
    torch.testing.assert_close(original, rescaled)
    assert float(protection(pack, state / 10, "parameter")) < float(
        protection(pack, state, "parameter")
    )
    with pytest.raises(ValueError):
        protection(pack, state, "unknown")


def test_missing_tokens_do_not_contribute_and_functional_penalty_has_gradient():
    pack = dict(phi=torch.ones(1, 16, 4), valid=torch.zeros(1, 17, dtype=torch.bool))
    pack["valid"][:, 0] = True
    pack["phi"][:, 1:] = 1000
    state = torch.ones(4, 64, requires_grad=True)
    loss = protection(pack, state, "hidden")
    assert float(loss.detach()) == pytest.approx(0.16)
    gradient = torch.autograd.grad(loss, state)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_first_strict_improvement_is_prespecified_not_best_risk_selection():
    assert first_improvement(1.0, [1.0, 0.9, 0.7]) == 1
    assert first_improvement(1.0, [1.0, 1.1]) is None
    with pytest.raises(FloatingPointError):
        first_improvement(1.0, [float("nan")])


def test_stream_controls_preserve_the_same_engine_and_refuse_label_input():
    assert choices()["hidden_real"] == ("hidden", "real")
    assert choices()["hidden_reverse"] == ("hidden", "reverse")
    assert choices()["hidden_hold"] == ("hidden", "hold")
    assert choices()["parameter_real"] == ("parameter", "real")
    with pytest.raises(ValueError):
        predict(None, dict(values=None, available=None, outcome=1), None, None)


@pytest.mark.parametrize("kind", ["parameter", "hidden"])
def test_actual_step_uses_only_proxy_losses_and_handles_no_target(monkeypatch, kind):
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    wave = torch.linspace(-1, 1, 25)[None, :, None].expand(1, 25, 23)
    pack = dict(phi=torch.ones(1, 16, 4), valid=torch.ones(1, 17, dtype=torch.bool))
    monkeypatch.setattr(
        engine, "prediction", lambda model, pack, state, decoder: state.mean() * wave
    )
    truth = torch.sinh(wave * 0.2)
    state = torch.zeros(4, 64)
    mask = torch.ones_like(truth, dtype=torch.bool)
    after, receipt = engine.advance(model, pack, state, truth, mask, None, kind=kind)
    assert receipt["accepted"]
    assert receipt["loss_proposal"] < receipt["loss_before"]
    assert receipt["chosen_fraction"] in engine.FRACTIONS
    assert receipt["gradient_norm"] > 0 and receipt["step_norm"] > 0
    assert torch.equal(state, torch.zeros_like(state))
    assert not after.requires_grad
    unchanged, empty = engine.advance(
        model, pack, after, truth, torch.zeros_like(mask), None, kind=kind
    )
    assert torch.equal(unchanged, after) and empty["reason"] == "no_target"
