from copy import deepcopy

import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
import motion_metric_model_20261107 as native
import plasticity_basis_model_20261107 as model
from run_plasticity_basis_20261107 import objective, state


@pytest.fixture
def setup():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    p = native.basis_for(base)
    packs = {k: model.encode_windows(base, v, a, p) for k, v in
             (("0", x), ("q", x.roll(7, 1)), ("r", x.roll(4, 1)))}
    data = dict(packs=packs, motion=torch.randn(3, 25, 23),
                mask=torch.ones(3, 25, 23, dtype=torch.bool), context=torch.zeros(3, 69),
                labels=torch.tensor([0., 1., 0.]), scale=torch.ones(3))
    scales = dict(level_reference=.05, shape_scale=3.5)
    yield base, data, p, scales
    torch.backends.mha.set_fastpath_enabled(previous)


def test_state_never_reads_later_query_or_outcome(setup):
    base, data, p, scales = setup
    expected = state(base, data, p, scales)
    altered = deepcopy(data)
    altered["labels"].fill_(float("nan"))
    altered["packs"]["q"] = {k: torch.full_like(v, float("nan")) if v.is_floating_point() else v
                                 for k, v in altered["packs"]["q"].items()}
    torch.testing.assert_close(state(base, altered, p, scales), expected, atol=0., rtol=0.)


@pytest.mark.parametrize("kind", ["risk", "feedback", "wrong_feedback"])
def test_outer_objective_reaches_basis_without_source_update(setup, kind):
    base, data, p, scales = setup
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    basis = p.clone().requires_grad_()
    loss, info = objective(base, data, basis, scales, kind=kind, tau=0.)
    grad, = torch.autograd.grad(loss, basis)
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
    assert info["state_norm_max"] <= .350001
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())
    assert all(v.grad is None for v in base.parameters())


def test_zero_feedback_gives_zero_state_for_any_basis(setup):
    base, data, p, scales = setup
    data["motion"].zero_()
    assert torch.count_nonzero(state(base, data, model.random_basis(p), scales)) == 0
