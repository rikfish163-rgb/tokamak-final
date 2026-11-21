import copy
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.tests.test_increment_forecast import model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.increment_forecast_20261023.learning import update
from experiments.risk_aligned_online.credit_horizon_20261023.learning import observed_step, episode


def fixture():
    _, m = model('increment')
    with torch.no_grad(): m.base.forecast_head.weight.normal_(0, .03)
    m.base.forecast_head.requires_grad_(True)
    rng = np.random.default_rng(43)
    s = dict(values=rng.normal(size=(680, 23)).astype('float32'), available=np.ones((680, 23), bool))
    return m, s


def test_observed_step_matches_deployment_and_has_no_future_access():
    m, s = fixture(); count = 256
    fast = initial_encoder(m); prefix = {k:v[:count] for k,v in s.items()}
    expected, _ = update(m, prefix, count, fast, device='cpu')
    actual, _ = observed_step(m, s, count, fast, create_graph=False, device='cpu')
    for k in fast: torch.testing.assert_close(actual[k], expected[k], atol=0, rtol=0)
    s['values'][count:] = 1e10
    changed, _ = observed_step(m, s, count, fast, create_graph=False, device='cpu')
    for k in fast: torch.testing.assert_close(actual[k], changed[k], atol=0, rtol=0)
    with pytest.raises(ValueError): observed_step(m, dict(s,y=1), count, fast, create_graph=False, device='cpu')


def test_truncation_changes_derivative_not_forward_state():
    m, s = fixture(); outputs = []
    for tail in (1, 8):
        r = episode(m, s, 512, [512, 520], True, boundary=.4, tail=tail, device='cpu')
        g = torch.autograd.grad(r['risk'], tuple(m.base.forecast_head.parameters()))
        outputs.append((r, torch.cat([v.flatten() for v in g])))
    left, right = outputs
    torch.testing.assert_close(left[0]['risk'], right[0]['risk'], atol=0, rtol=0)
    for k in left[0]['fast']: torch.testing.assert_close(left[0]['fast'][k], right[0]['fast'][k], atol=0, rtol=0)
    assert torch.isfinite(right[1]).all() and right[1].norm() > 0
    assert not torch.allclose(left[1], right[1], atol=1e-10, rtol=1e-3)
    assert left[0]['differentiated'] == [512] and right[0]['differentiated'] == [256,384,512]


def test_full_short_unroll_derivative_matches_finite_difference():
    m, s = fixture(); m = m.double(); s['values'] = s['values'].astype('float64')
    def value(): return episode(m, s, 384, [384,392], True, boundary=.4, tail=8, device='cpu')['risk']
    params = tuple(m.base.forecast_head.parameters()); value0 = value()
    grad = torch.autograd.grad(value0, params)
    direction = [g / torch.sqrt(sum(v.square().sum() for v in grad)) for g in grad]
    saved = [p.detach().clone() for p in params]; vals = []
    eps = 1e-4
    for sign in (1, -1):
        with torch.no_grad():
            for p, old, d in zip(params, saved, direction): p.copy_(old+sign*eps*d)
        vals.append(float(value().detach()))
    with torch.no_grad():
        for p, old in zip(params, saved): p.copy_(old)
    numeric = (vals[0]-vals[1])/(2*eps)
    expected = float(sum((g*d).sum() for g,d in zip(grad,direction)))
    assert numeric == pytest.approx(expected, rel=2e-3, abs=1e-8)
