from types import SimpleNamespace

import pytest
import torch

from experiments.risk_aligned_online.innovation_metric_20261102 import core


def test_exact_matured_prediction_has_zero_data_gradient():
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    truth = torch.linspace(-0.5, 0.8, 25 * 23).reshape(1, 25, 23)
    prediction = truth.asinh().clone().requires_grad_()
    mask = torch.ones_like(truth, dtype=torch.bool)
    loss = core.innovation_loss(model, prediction, truth, mask)
    (g,) = torch.autograd.grad(loss, prediction)
    assert float(loss.detach()) == 0
    assert torch.equal(g, torch.zeros_like(g))


def make_pack(dtype=torch.float64):
    phi = torch.arange(64, dtype=dtype).reshape(1, 16, 4) / 64
    return dict(phi=phi, valid=torch.ones(1, 17, dtype=torch.bool))


def test_metric_step_is_descent_and_has_actual_hidden_rms_bound():
    pack = make_pack()
    gradient = torch.arange(256, dtype=torch.float64).reshape(4, 64) + 1
    delta = core.metric_direction(pack, gradient)
    assert torch.sum(delta * gradient) < 0
    rms = ((pack["phi"] @ delta).square().mean()).sqrt()
    assert float(rms) <= 0.05 + 1e-12
    torch.testing.assert_close(rms, torch.tensor(0.05, dtype=rms.dtype))


def test_metric_step_vanishes_continuously_with_prediction_gradient():
    pack = make_pack()
    gradient = torch.ones(4, 64, dtype=torch.float64)
    d1 = core.metric_direction(pack, gradient * 1e-10)
    d2 = core.metric_direction(pack, gradient * 1e-11)
    torch.testing.assert_close(d2, d1 * 0.1, rtol=1e-10, atol=1e-15)
    assert torch.equal(core.metric_direction(pack, gradient * 0), gradient * 0)


def test_metric_uses_only_observed_tokens():
    pack = make_pack()
    pack["valid"][:, 8:16] = False
    g = torch.randn(4, 64, dtype=torch.float64)
    first = core.metric_direction(pack, g)
    pack["phi"][:, 8:] = 1e10
    torch.testing.assert_close(core.metric_direction(pack, g), first, rtol=0, atol=0)


def test_metric_remains_differentiable_for_outer_meta_training():
    pack = make_pack()
    g = (torch.arange(256, dtype=torch.float64).reshape(4, 64) * 1e-9).requires_grad_()
    (derivative,) = torch.autograd.grad(
        core.metric_direction(pack, g).square().sum(), g
    )
    assert torch.isfinite(derivative).all()
    assert derivative.abs().max() > 0


def test_zero_metric_gradient_has_finite_outer_derivative():
    g = torch.zeros(4, 64, dtype=torch.float64, requires_grad=True)
    (derivative,) = torch.autograd.grad(core.metric_direction(make_pack(), g).sum(), g)
    assert torch.isfinite(derivative).all()


def test_no_valid_token_has_no_functional_step():
    pack = make_pack()
    pack["valid"][:] = False
    g = torch.ones(4, 64, dtype=torch.float64)
    assert torch.equal(core.metric_direction(pack, g), torch.zeros_like(g))


def test_empty_observation_has_no_gradient_or_step():
    state = torch.randn(4, 64)
    truth = torch.zeros(1, 25, 23)
    after, receipt = core.propose(
        None,
        {},
        state,
        truth,
        truth,
        torch.zeros_like(truth, dtype=torch.bool),
        torch.zeros(25, 23),
        geometry="metric",
    )
    assert torch.equal(after, state)
    assert not receipt["accepted"]
    assert "gradient256" not in receipt


@pytest.mark.parametrize("geometry", ["unit", "metric"])
def test_rejected_exact_fit_has_no_auxiliary_head_risk_path(monkeypatch, geometry):
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    truth = torch.zeros(1, 25, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    coefficient = torch.zeros(25, 23, requires_grad=True)

    def auxiliary(_model, _pack, state, table):
        ramp = torch.arange(25, dtype=torch.float32).reshape(1, 25, 1)
        return ramp * (state[0, 0] + table[0, 0]) + truth

    monkeypatch.setattr(core, "auxiliary", auxiliary)
    after, receipt = core.propose(
        model,
        make_pack(torch.float32),
        torch.zeros(4, 64),
        truth,
        truth,
        mask,
        coefficient,
        geometry=geometry,
        create_graph=True,
    )
    assert not receipt["accepted"]
    assert torch.equal(after, torch.zeros_like(after))
    # This assertion tests graph disconnection, not a zero-filled measured grad.
    (derivative,) = torch.autograd.grad(after.sum(), coefficient, allow_unused=True)
    assert derivative is None


@pytest.mark.parametrize("geometry", ["unit", "metric"])
def test_accepted_update_reduces_actual_diagnostic_error(monkeypatch, geometry):
    model = SimpleNamespace(
        base=SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    )
    pattern = torch.linspace(-1, 1, 25).reshape(1, 25, 1).expand(1, 25, 23)
    truth = (0.5 * pattern).sinh()
    mask = torch.ones_like(truth, dtype=torch.bool)
    pack = dict(
        phi=torch.eye(4).repeat(4, 1)[None], valid=torch.ones(1, 17, dtype=torch.bool)
    )

    def auxiliary(_model, _pack, state, table):
        return pattern * (state[0, 0] + table[0, 0])

    monkeypatch.setattr(core, "auxiliary", auxiliary)
    state = torch.zeros(4, 64)
    table = torch.zeros(25, 23, requires_grad=True)
    after, receipt = core.propose(
        model,
        pack,
        state,
        truth,
        truth,
        mask,
        table,
        geometry=geometry,
        create_graph=True,
    )
    assert receipt["accepted"]
    assert receipt["loss_proposal"] < receipt["loss_before"]
    assert after[0, 0] > 0
    (head_gradient,) = torch.autograd.grad(after.square().sum(), table)
    assert torch.isfinite(head_gradient).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradient_rejected(bad):
    g = torch.ones(4, 64, dtype=torch.float64)
    g[0, 0] = bad
    with pytest.raises(FloatingPointError):
        core.metric_direction(make_pack(), g)
