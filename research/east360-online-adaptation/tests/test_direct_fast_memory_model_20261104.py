from dataclasses import replace

import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import (
    Injection,
    levels,
    predict,
)


def fixture():
    torch.manual_seed(13)
    torch.backends.mha.set_fastpath_enabled(False)
    model = Simple23(
        replace(Config(), fast_site="tokens"), torch.zeros(23), torch.ones(23)
    )
    model.eval().requires_grad_(False)
    x = torch.randn(2, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    return model, x, a


def test_zero_innovation_is_exact_original_with_nonzero_projection():
    base, x, a = fixture()
    projection = Injection()
    torch.nn.init.normal_(projection.weight)
    assert torch.equal(
        predict(base, projection, x, a, torch.zeros(2, 168)), base(x, a)["logit"]
    )


def test_projection_is_zero_initial_but_can_learn_internal_effect():
    base, x, a = fixture()
    projection = Injection()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    innov = torch.randn(2, 168)
    z = predict(base, projection, x, a, innov)
    assert torch.equal(z, base(x, a)["logit"])
    z.sum().backward()
    assert projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())
    assert all(torch.equal(v, state[k]) for k, v in base.state_dict().items())


def test_levels_hide_unavailable_values_and_have_no_target_statistics():
    base, x, a = fixture()
    a[:, :, 2] = False
    x[:, :, 2] = float("nan")
    result = levels(base, x, a)
    assert result.isfinite().all() and not result[:, :, 2].any()
    assert torch.equal(result[:, :, 3], x[:, :, 3].asinh())


def test_output_update_site_is_not_silently_accepted():
    base, x, a = fixture()
    base.cfg = replace(base.cfg, fast_site="output")
    with pytest.raises(ValueError, match="token"):
        predict(base, Injection(), x, a, torch.zeros(2, 168))
