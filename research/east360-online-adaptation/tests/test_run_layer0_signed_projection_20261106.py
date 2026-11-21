from types import SimpleNamespace

import numpy as np
import torch

from run_layer0_signed_projection_20261106 import (
    action_targets,
    following_upshifts,
)


def test_supported_and_downward_moves_untouched_with_matched_norm_controls():
    u = torch.zeros(3, 128)
    u[:, 0] = 0.5
    w = u.clone()
    w[1, 0], w[1, 2] = 0.0, 0.1
    w[0] = 0
    receipt = SimpleNamespace(values=w, restricted=torch.tensor([True, True, False]))
    arms = action_targets(
        u, u.roll(1, dims=1), torch.tensor([True, False, False]), receipt
    )
    torch.testing.assert_close(arms["signed"][0, :128], u[0])
    torch.testing.assert_close(arms["signed"][1, :128], w[1])
    torch.testing.assert_close(arms["signed"][2, :128], u[2])
    assert torch.count_nonzero(arms["blind_signed"][0]) == 0
    for name in ("shrink_match", "lagged_match"):
        torch.testing.assert_close(arms["signed"].norm(dim=1), arms[name].norm(dim=1))
        assert torch.count_nonzero(arms[name][:, 128:]) == 0


def test_following_points_do_not_treat_an_anchor_as_evidence_about_future():
    native = np.zeros(181)
    signed = native.copy()
    signed[135], signed[136], signed[168] = 0.7, 0.1, 0.2
    out = following_upshifts(native, signed, np.array([True, False]))
    assert out.tolist() == [True, False]
    signed[136] = 0.0
    assert not following_upshifts(native, signed, np.array([True, False])).any()
