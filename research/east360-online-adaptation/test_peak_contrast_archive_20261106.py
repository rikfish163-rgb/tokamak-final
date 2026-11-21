import numpy as np
import torch

import run_peak_contrast_capacity_20261106 as run


def test_dense_archive_keeps_aligned_offsets_sites_and_zero_actions(monkeypatch):
    class Base:
        center = torch.zeros(1)

    def scalar_forward(_base, _tape, endpoints, states, site):
        return endpoints[:, None] * 0.01 + states[..., 0] * (1 if site == "all" else 2)

    monkeypatch.setattr(run, "routed_batch", scalar_forward)
    rows = []
    for m, n in ((0, 20), (50, 32)):
        ix = np.arange(m + 1, m + 33)
        present = np.arange(32) < n
        rows.append(
            dict(
                future_ix=ix,
                present=present,
                anchor=m * 0.01,
                mx=m,
                native=np.where(present, ix * 0.01, 0.0).astype(np.float32),
            )
        )
    a = len(run.forward.ARMS)
    states = torch.zeros(2, 2, 3, a, 128)
    common = torch.ones(1, 3, a, dtype=torch.bool)
    common[0, 1, 2:4] = False
    states[..., ::2, 0] = 0.35
    states[..., 1::2, 0] = -0.35
    states.masked_fill_(~common[:, None, :, :, None], 0.0)
    out = run.effects(Base(), {}, rows, states, common, torch.zeros(2, 3, 35))
    assert out["future"].shape == (1, 2, 2, 3, a, 32)
    assert out["future"][0, 0, :, :, :, 20:].sum() == 0
    np.testing.assert_allclose(out["future"][0, 1, :, 1, 2:4, 0], 0.51)
    np.testing.assert_allclose(out["future"][0, 1, :, 0, 0, 0], [0.86, 1.21])
    np.testing.assert_allclose(out["anchor_delta"][0, 0, :, 0, 0], [0.35, 0.70])
    np.testing.assert_allclose(out["anchor_delta"][0, 1, :, 0, 1], [-0.35, -0.70])
