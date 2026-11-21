from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
import motion_metric_model_20261107 as native
import plasticity_basis_model_20261107 as model
from shape_motion_kernel_20261107 import ShapeMetric
from run_shape_motion_stream_20261107 import FIRST, predict_shot as previous
from run_plasticity_basis_stream_20261107 import predict_shot


@pytest.fixture
def setup():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    initial = native.basis_for(base)
    bases = dict(svd=initial, random=model.random_basis(initial), risk=initial.clone(),
                 feedback=initial.clone(), wrong_feedback=initial.clone())
    rng = np.random.default_rng(131)
    signal = dict(values=rng.normal(size=(220, 23)).astype(np.float32),
                  available=rng.random((220, 23)) > .03)
    signal["available"][:, 2] = False
    scales = dict(level_reference=.05, shape_scale=3.5)
    yield base, bases, signal, scales
    torch.backends.mha.set_fastpath_enabled(previous)


def test_svd_explicit_basis_reproduces_previous_actual_updates(setup):
    base, bases, signal, scales = setup
    got, info = predict_shot(base, signal, bases, scales)
    old, _ = previous(base, signal, ShapeMetric().eval().requires_grad_(False), scales)
    for current, former in (("svd", "fixed_shape"), ("static", "static"),
                            ("time", "time"), ("channel", "channel"), ("scalar", "scalar")):
        np.testing.assert_allclose(got[current], old[former], atol=3e-6, rtol=1e-5)
    assert info["control_norm_error"] < 1e-6


def test_new_direction_stream_uses_only_arrived_prefix(setup):
    base, bases, signal, scales = setup
    bases["risk"] = bases["random"]
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    with patch("moving_peak_replay_20261105.io23.read_outcome", side_effect=AssertionError("unrevealed")):
        full, _ = predict_shot(base, signal, bases, scales)
        prefix, _ = predict_shot(base, {k: v[:187] for k, v in signal.items()}, bases, scales)
    for name, curve in prefix.items():
        np.testing.assert_allclose(curve, full[name][:187], atol=3e-6, rtol=1e-5)
        np.testing.assert_array_equal(curve[:FIRST], prefix["static"][:FIRST])
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())


def test_cold_segment_never_performs_basis_update(setup):
    base, bases, signal, scales = setup
    with patch("run_plasticity_basis_stream_20261107.state", side_effect=AssertionError("cold")):
        curves, info = predict_shot(base, signal, bases, scales, warm=True)
    for curve in curves.values():
        np.testing.assert_array_equal(curve, curves["static"])
    assert info["releases"] == 0
