import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_weak_shape_data_20261111 as shape
from experiments.risk_aligned_online.commissioning_prefix_learning_20261111 import (
    at_width, history_weight, latent,
)


def test_short_view_matches_isolated_observed_suffix_without_cross_record_padding():
    rng = np.random.default_rng(12)
    z = rng.standard_normal((211, 10))
    c = rng.standard_normal((211, 2))
    m = rng.random(z.shape) > .2
    cm = rng.random(c.shape) > .2
    ends = np.array([1, 20, 64, 83, 199])
    actual, context = at_width(z, m, c, cm, ends, 64)
    for i, e in enumerate(ends):
        start = max(0, e - 63)
        end = slice(start, e+1)
        x, ctx = shape.inputs(z[end], m[end], c[end], cm[end], np.array([e-start]))
        np.testing.assert_array_equal(actual[i], x[0])
        np.testing.assert_array_equal(context[i], ctx[0])


def test_startup_weight_is_gradual_and_fully_supported_ramp_has_no_late_effect():
    q = np.array([0, 7, 8, 126, 127, 158, 159, 160, 256])
    a = history_weight(q+1)
    assert a[0] == a[1] == 0.
    assert 0 < a[2] < a[3] < a[4] < a[5] < 1.
    assert a[4] - a[3] < .02
    np.testing.assert_array_equal(a[6:], 1.)
    p = torch.linspace(-2, 2, len(a)*24).reshape(len(a), 24)
    actual = latent(p, torch.from_numpy(a), torch.zeros_like(p))
    torch.testing.assert_close(actual[6:], p[6:].tanh(), rtol=0, atol=0)
    torch.testing.assert_close(actual[:2], torch.zeros_like(p[:2]), rtol=0, atol=0)
