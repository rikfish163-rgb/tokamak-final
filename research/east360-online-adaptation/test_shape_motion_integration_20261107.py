from copy import deepcopy

import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
import motion_metric_model_20261107 as model
from shape_motion_kernel_20261107 import ShapeMetric
from run_shape_motion_20261107 import calibration, state


@pytest.fixture
def setup():
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    valid = torch.ones_like(x, dtype=torch.bool)
    pack = model.encode_windows(base, x, valid, model.basis_for(base))
    query = model.encode_windows(base, x.roll(7, 1), valid, model.basis_for(base))
    data = dict(packs={"0": pack, "q": query}, context=torch.randn(3, 69),
                motion=torch.randn(3, 25, 23), mask=torch.ones(3, 25, 23, dtype=torch.bool),
                labels=torch.tensor([0., 1., 0.]))
    yield base, data
    torch.backends.mha.set_fastpath_enabled(old)


def test_initial_shape_rule_has_identical_state_to_level_rule(setup):
    base, data = setup
    metric = ShapeMetric()
    a, _ = state(base, data, metric, kind="shape", shape_scale=3., update_scale=.05)
    b, _ = state(base, data, metric, kind="level", shape_scale=3., update_scale=.05)
    torch.testing.assert_close(a, b, atol=0., rtol=0.)


def test_risk_gradient_teaches_shape_weights_without_changing_source(setup):
    base, data = setup
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    metric = ShapeMetric()
    updated, _ = state(base, data, metric, kind="shape", shape_scale=3., update_scale=.05,
                       create_graph=True)
    z = model.forward_state(base, data["packs"]["q"], updated)["logit"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, data["labels"])
    grad, = torch.autograd.grad(loss, metric.net[-1].bias)
    assert torch.isfinite(grad).all() and grad[23:].abs().sum() > 0
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())


def test_constant_motion_is_an_exact_noop_with_both_terms(setup):
    base, data = setup
    data["motion"].zero_()
    metric = ShapeMetric()
    with torch.no_grad():
        metric.net[-1].bias[23:].fill_(.4)
    updated, _ = state(base, data, metric, kind="shape", shape_scale=3., update_scale=.05)
    assert torch.count_nonzero(updated) == 0
    got = model.forward_state(base, data["packs"]["q"], updated)["logit"]
    torch.testing.assert_close(got, data["packs"]["q"]["native"], atol=2e-6, rtol=1e-5)


def test_gradient_scale_does_not_read_labels_or_future_query(setup):
    base, data = setup
    expected = calibration(base, data)
    changed = deepcopy(data)
    changed["labels"].fill_(float("nan"))
    changed["packs"]["q"] = {k: torch.full_like(v, float("nan")) if v.is_floating_point() else v
                                 for k, v in changed["packs"]["q"].items()}
    assert calibration(base, changed) == expected
    assert expected["level_reference"] > 0 and expected["shape_reference"] > 0
    assert expected["shape_scale"] == expected["level_reference"] / expected["shape_reference"]
