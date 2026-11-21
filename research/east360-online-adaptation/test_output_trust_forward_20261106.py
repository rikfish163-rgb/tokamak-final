import numpy as np
import pytest
import torch

import paired_diagnostic_step_20261106 as old
from experiments.risk_aligned_online.simple23 import Config, Simple23
from output_trust_forward_20261106 import (
    batched_risk,
    encode,
    packed_risk,
    teaching_masks,
)


def test_cached_risk_runs_the_same_real_transformer_layer_and_readout():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.manual_seed(131)
        base = (
            Simple23(
                Config(fast_site="tokens", fast_rule="normalized"),
                torch.zeros(23),
                torch.ones(23),
            )
            .eval()
            .requires_grad_(False)
        )
        with torch.no_grad():
            base.fast_origin.copy_(torch.randn_like(base.fast_origin) * 0.2)
        x = torch.randn(5, 128, 23)
        a = torch.rand_like(x) > 0.1
        a[0, :96] = False
        x[2, 5, 4] = float("nan")
        a[2, 5, 4] = False
        state = torch.randn(5, 128) * 0.05
        p = encode(base, x, a)
        with torch.no_grad():
            torch.testing.assert_close(
                packed_risk(base, p, state),
                old.risk(base, x, a, state),
                atol=3e-6,
                rtol=0,
            )
            exact = torch.stack(
                (old.risk(base, x, a, state), old.risk(base, x, a, -state)), 1
            )
            torch.testing.assert_close(
                batched_risk(base, p, torch.stack((state, -state), 1), batch_size=6),
                exact,
                atol=3e-6,
                rtol=0,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_guard_first_and_positive_masks_are_distinct_and_keep_late_strict():
    t = np.arange(140, dtype=np.float64) / 1000
    native = np.full(140, -1.0)
    native[68] = 0.6
    pos, protect, first = teaching_masks(
        t, native, np.array([63, 95]), True, 0.113, 29.0, 0.5
    )
    assert np.flatnonzero(protect[0]).tolist() == [0, 1, 2, 3, 4]
    assert not protect[1].any()
    assert np.flatnonzero(pos[0]).tolist() == list(range(20, 32))
    assert np.flatnonzero(pos[1]).tolist() == list(range(7))
    assert first == 68


def test_safe_complete_padded_block_and_wrong_clock():
    t = np.arange(80) / 1000
    z = np.full(80, -1.0)
    p, g, k = teaching_masks(t, z, np.array([63]), False, None, 25, 0.5)
    assert not p.any() and not g.any() and k == -1
    with pytest.raises(ValueError, match="matur"):
        teaching_masks(t, z, np.array([62]), False, None, 25, 0.5)
