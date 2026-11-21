from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from shape_motion_kernel_20261107 import ShapeMetric
from run_shape_motion_stream_20261107 import FIRST, STRIDE, predict_shot, schedule


def test_update_clock_uses_only_matured_targets_and_strictly_later_risk():
    for n in (1, 127, 153, 154, 155, 186, 187, 260):
        origins, releases = schedule(n)
        np.testing.assert_array_equal(releases, origins+26)
        assert np.all(origins+25 < releases)
        assert np.all(releases+1 < n)
        np.testing.assert_array_equal(origins, 127+STRIDE*np.arange(len(origins)))
    assert FIRST == 154
    np.testing.assert_array_equal(schedule(187)[0], [127, 159])


@pytest.fixture
def setup():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens", fast_rule="normalized"),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    rng = np.random.default_rng(131)
    signal = dict(values=rng.normal(size=(220, 23)).astype(np.float32),
                  available=rng.random((220, 23)) > .03)
    signal["available"][:, 2] = False
    scales = dict(level_reference=.05, shape_reference=.014, shape_scale=.05/.014)
    yield base, ShapeMetric().eval().requires_grad_(False), signal, scales
    torch.backends.mha.set_fastpath_enabled(previous)


def test_stream_prefix_is_invariant_to_future_signal_and_has_no_outcome_access(setup):
    base, rule, signal, scales = setup
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    with patch("moving_peak_replay_20261105.io23.read_outcome", side_effect=AssertionError("unrevealed outcome")):
        full, info = predict_shot(base, signal, rule, scales)
        prefix = {k: v[:187] for k, v in signal.items()}
        shorter, _ = predict_shot(base, prefix, rule, scales)
    for name, curve in full.items():
        np.testing.assert_allclose(curve[:187], shorter[name], atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(curve[:FIRST], full["static"][:FIRST], atol=0., rtol=0.)
    assert info["releases"] == len(schedule(220)[0])
    assert info["control_norm_error"] < 1e-6
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())


def test_prepare_segment_does_not_adapt_or_use_learned_rule(setup):
    base, rule, signal, scales = setup
    with patch.object(rule, "forward", side_effect=AssertionError("cold segment must stay source-only")):
        curves, info = predict_shot(base, signal, rule, scales, warm=True)
    for curve in curves.values():
        np.testing.assert_array_equal(curve, curves["static"])
    assert info["releases"] == 0


def test_short_shot_is_native_without_invented_feedback(setup):
    base, rule, signal, scales = setup
    short = {k: v[:FIRST] for k, v in signal.items()}
    curves, info = predict_shot(base, short, rule, scales)
    for curve in curves.values():
        np.testing.assert_array_equal(curve, curves["static"])
    assert info["releases"] == 0
