"""Synthetic checks for actual positions and shared two-sided action sizes."""

import numpy as np
import torch

import run_patch_utility_site_20261106 as s
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_common_scale_covers_both_sites_signs_and_anchors():
    shift = np.zeros((2, 2, 2, 5, 3), dtype=np.float32)
    shift[1, 0, 1, 0, 2] = .11
    shift[0, 1, 0, :, 1] = .15
    a = np.ones((2, 2), dtype=bool)
    radius = s.shared_radius(shift, a)
    assert radius.tolist() == [[.5, .5], [0., 0.]]


def test_zero_writes_equal_original_and_middle_matches_native_hook():
    torch.manual_seed(105)
    base = Simple23(Config(fast_site="tokens"), torch.zeros(23), torch.ones(23)).eval()
    base.requires_grad_(False)
    x = torch.randn(2, 128, 23)
    observed = torch.ones(x.shape, dtype=torch.bool)
    observed[:, :24] = False
    p = s.b.op.encode(base, x, observed, s.b.op.basis(base))
    z = torch.zeros(2, 64)
    u = torch.randn_like(z) * .015
    state = s.b.op.delta(p, u, 0)
    original = base(x, observed)["logit"]
    for site in s.SITES:
        torch.testing.assert_close(s.forward(base, p, z, site), original, rtol=1e-5, atol=5e-6)
    torch.testing.assert_close(s.forward(base, p, u, "input"), s.b.op.forward(base, p, u, 0)["logit"])

    def hook(_, __, output):
        return torch.cat((output[:, :-1] + state, output[:, -1:]), dim=1)

    h = base.encoder.layers[0].register_forward_hook(hook)
    try:
        witness = base(x, observed)["logit"]
    finally:
        h.remove()
    torch.testing.assert_close(s.forward(base, p, u, "middle"), witness, rtol=1e-5, atol=5e-6)
