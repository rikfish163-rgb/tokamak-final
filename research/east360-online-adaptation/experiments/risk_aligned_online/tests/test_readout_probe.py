import numpy as np
import pytest
from scipy.optimize import check_grad

from experiments.risk_aligned_online.readout_probe_20261023.fit import (
    objective, fit_readout, endpoint_manifest,
)


def test_analytic_gradient_and_strictly_convex_fit():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(40, 4)); y = (x[:, 0] > 0).astype(float)
    weights = np.where(y == 1, .5 / y.sum(), .5 / (1-y).sum())
    origin = np.zeros(5); theta = rng.normal(size=5)
    f = lambda p: objective(p, x, y, weights, origin)[0]
    g = lambda p: objective(p, x, y, weights, origin)[1]
    assert check_grad(f, g, theta) < 1e-6
    fitted, info = fit_readout(x, y, weights, origin)
    assert info['converged'] and info['gradient_inf'] < 1e-6
    assert f(fitted) < f(origin)
    assert np.mean((x @ fitted[:-1] + fitted[-1] > 0) == y) > .9


def test_endpoint_weights_balance_class_then_shot_without_duplicates():
    def shot(sid, y):
        return dict(shot=sid, valid=np.ones(len(y), bool), y=np.array(y))
    rows = endpoint_manifest([shot(1, [0]*200), shot(2, [0]*3+[1]*5)])
    assert len({(r['shot'], r['end']) for r in rows}) == len(rows)
    assert sum(r['weight'] for r in rows if r['label'] == 0) == pytest.approx(.5)
    assert sum(r['weight'] for r in rows if r['label'] == 1) == pytest.approx(.5)
    assert sum(r['weight'] for r in rows if r['shot'] == 1) == pytest.approx(.25)
    assert max(r['end'] for r in rows if r['shot'] == 2) == 7
    with pytest.raises(ValueError):
        endpoint_manifest([shot(1, [0]*10)])


def test_nonfinite_or_malformed_inputs_rejected():
    x = np.ones((2, 3)); y = np.array([0., 1.]); w = np.ones(2)/2
    with pytest.raises(ValueError):
        fit_readout(x * np.nan, y, w, np.zeros(4))
    with pytest.raises(ValueError):
        fit_readout(x, y, w*2, np.zeros(4))
