import numpy as np
import pytest
import torch

from coherence_token_write_20261106 import offsets, scalar_control, shifted_logits
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_effective_internal_bias_matches_the_exact_normalized_value_intervention():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23))
    base.eval().requires_grad_(False)
    x = torch.randn(3, 128, 23)
    a = torch.rand(3, 128, 23) > 0.1
    shift = torch.randn_like(x) * 0.1
    shift[:, :-32] = 0.0
    initial = {k: v.clone() for k, v in base.state_dict().items()}
    hooked = shifted_logits(base, x, a, shift)
    with torch.no_grad():
        expected = base(torch.sinh(torch.asinh(x) + shift), a)["logit"]
        unmodified = base(x, a)["logit"]
    torch.testing.assert_close(hooked, expected, atol=3e-6, rtol=1e-5)
    assert not torch.equal(hooked, unmodified)
    torch.testing.assert_close(
        shifted_logits(base, x, a, torch.zeros_like(shift)), unmodified
    )
    assert not base.embedding._forward_hooks
    assert all(torch.equal(v, initial[k]) for k, v in base.state_dict().items())


def test_offsets_keep_perturbation_in_old_tokens_but_correction_in_recent_four():
    d = np.zeros((640, 23), dtype=np.float32)
    d[256:512, 3] = 0.6
    a = np.zeros_like(d)
    a[320:512, 3] = 0.4
    t = np.array([1, 319, 320, 511])
    out = offsets(t, d, a)
    assert not out[0].any()
    assert out[1, -32:, 3].min() == pytest.approx(0.6)
    assert out[2, -32:, 3].min() == pytest.approx(0.2)
    assert out[2, -33, 3] == pytest.approx(0.6)
    assert out[2, 0, 3] == 0.0
    limited = offsets(t, d, None, oracle=True)
    assert not limited[2, -32:].any() and limited[2, -33, 3] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="clock"):
        offsets(np.array([640]), d, a)


def test_scalar_matches_each_anchor_and_is_unchanged_by_future_coherent_values():
    s = np.arange(96, dtype=np.float32) / 100.0
    c = s + np.arange(96, dtype=np.float32) / 200.0
    p = scalar_control(s, c)
    np.testing.assert_array_equal(p[::32], c[::32])
    assert (p - s)[38] == pytest.approx((c - s)[32])
    c[60:] = -10
    np.testing.assert_array_equal(scalar_control(s, c)[:64], p[:64])
