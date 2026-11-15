import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_lag_mixer_20261111 import (
    CHANNELS, CROSS, _features, delayed_slots, matched_steps, motion, ordered_signature, patch_input,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_patch_mix_preserves_latest_level_mean_and_missingness():
    g = torch.Generator().manual_seed(31)
    x = torch.randn((2, 128, 23), generator=g)
    m = torch.rand((2, 128, 23), generator=g) > .11
    x = x.masked_fill(~m, 0.)
    src = torch.cat((x, m.float()), -1).reshape(2, 16, -1)
    w = .1 * torch.randn((2, len(CHANNELS), len(CHANNELS)), generator=g)
    new = patch_input(src, w).reshape(2, 128, 46)
    torch.testing.assert_close(new[..., 23:], m.float(), rtol=0., atol=0.)
    torch.testing.assert_close(new[:, -1, :23], x[:, -1], rtol=0., atol=0.)
    torch.testing.assert_close(new[..., :23].sum(1), x.sum(1), rtol=0., atol=4e-6)
    assert not new[..., :23][~m].any()
    assert not torch.equal(new[..., :23], x)
    assert torch.equal(patch_input(src, torch.zeros_like(w)), src)


def test_only_cross_group_coefficients_affect_embeddings():
    g = torch.Generator().manual_seed(7)
    source = torch.cat((torch.randn((1, 128, 23), generator=g), torch.ones(1, 128, 23)),
                       dim=-1).reshape(1, 16, -1)
    x = torch.ones(1, 10, 10) * (~CROSS)
    assert torch.equal(patch_input(source, x), source)


def test_phase_control_keeps_motion_distribution_per_channel():
    t = torch.arange(25).float()
    y = (t[:, None]/(5. + torch.arange(10))[None, :]).sin()[None]
    genuine, shifted = motion(y), motion(y, shift=True)
    torch.testing.assert_close(genuine.sort(dim=1).values, shifted.sort(dim=1).values)
    assert not torch.allclose(ordered_signature(genuine), ordered_signature(shifted))


def test_slot_starts_only_after_complete_feedback_and_shadow_holds_previous():
    q = np.array([127, 159, 191])
    ends = np.array([127, 152, 153, 184, 185, 217])
    new, skip = delayed_slots(q, ends)
    np.testing.assert_array_equal(new, [0, 0, 1, 1, 2, 3])
    np.testing.assert_array_equal(skip, [0, 0, 0, 0, 1, 2])


def test_matched_steps_do_not_invent_a_direction_for_no_support():
    a, b = torch.zeros(2, 10, 10), torch.zeros(2, 10, 10)
    a[:, 0, 3] = 1.
    b[0, 0, 4] = -2.
    true, shifted = matched_steps(a, b)
    assert true[0, 0, 3] < 0 < shifted[0, 0, 4]
    torch.testing.assert_close(true.norm(dim=(1, 2)), shifted.norm(dim=(1, 2)))
    assert not true[1].any()
    assert not shifted[1].any()


def test_gradient_reaches_lag_connections_through_actual_frozen_encoder():
    torch.manual_seed(24)
    base = Simple23(Config(dim=8, heads=2), torch.zeros(23), torch.ones(23))
    base.eval().requires_grad_(False)
    x = torch.randn(2, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    t = torch.zeros(2, 10, 10, requires_grad=True)
    reference = base(x, mask)['logit']
    result = base.risk_head(_features(base, x, mask, t)).flatten()
    # CPU can select a different attention kernel for the gradient-carrying
    # pass; exact identity of the zero-matrix embedding input is tested above.
    torch.testing.assert_close(result, reference, rtol=0., atol=5e-7)
    gradient, = torch.autograd.grad(result.sum(), t)
    assert torch.isfinite(gradient).all() and gradient[:, CROSS].abs().max() > 0.
    assert not gradient[:, ~CROSS].any()
    assert not any(p.grad is not None for p in base.parameters())
