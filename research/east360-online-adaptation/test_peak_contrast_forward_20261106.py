import numpy as np
import torch

from peak_contrast_forward_20261106 import (
    ARMS,
    SITE_NORM,
    episode_windows,
    fixed_states,
    routes,
)


def test_positive_teaching_does_not_discard_an_already_correct_shot():
    time = np.arange(110) / 1000
    z = np.full(110, -1.0)
    z[3] = 1.0
    out = episode_windows(time, z, 63, True, 0.100, 22.0, 0.5)
    positive, protect, timely = out
    assert np.flatnonzero(positive).tolist() == list(range(14, 26))
    assert not protect.any() and timely


def test_protect_mask_is_a_real_first_alarm_prefix_even_outside_canonical():
    time = np.arange(120) / 1000
    z = np.full(120, -2.0)
    z[[66, 70, 110]] = 1
    positive, protect, timely = episode_windows(time, z, 63, True, 0.110, 30, 0.5)
    assert np.flatnonzero(positive).tolist() == list(range(16, 32))
    assert np.flatnonzero(protect).tolist() == [0, 1, 2]
    assert timely


def test_safe_requires_full_observed_next_block_and_late_is_not_timely():
    time = np.arange(90) / 1000
    z = np.full(90, -1.0)
    z[86] = 1.0
    assert episode_windows(time, z, 63, False, None, 125, 0.5) is None
    positive, protect, timely = episode_windows(time, z, 63, True, 0.091, 25, 0.5)
    assert np.flatnonzero(positive).tolist() == list(range(2, 17))
    assert not protect.any() and not timely


def test_route_touches_only_valid_last_four_data_patches_not_cls():
    mask = torch.ones(2, 128, 23, dtype=torch.bool)
    mask[1, 112:120] = False
    a = routes(mask, "all")
    b = routes(mask, "recent4")
    assert a.shape == b.shape == (2, 2, 17)
    assert a[0].all() and a[1, 0, 16]
    assert torch.where(b[0, 0])[0].tolist() == [12, 13, 14, 15]
    assert torch.where(b[1, 0])[0].tolist() == [12, 13, 15]


def test_site_and_sign_controls_have_same_norm_and_zero_common_unavailable():
    a, b = torch.zeros(2, 7, 128), torch.zeros(2, 7, 128)
    a[:, 0, 1] = 2
    b[0, 0, 0] = 1
    states, available = fixed_states(a, b)
    j = ARMS.index("signed_mir_plus")
    assert ARMS[j + 1] == "signed_mir_minus"
    assert tuple(states.shape) == (2, 2, len(ARMS), 128)
    assert available[:, j].tolist() == [True, False]
    torch.testing.assert_close(
        states[0, :, j].norm(dim=-1), torch.full((2,), SITE_NORM)
    )
    torch.testing.assert_close(states[:, :, j + 1], -states[:, :, j])
    assert states[1, :, j].count_nonzero() == 0
