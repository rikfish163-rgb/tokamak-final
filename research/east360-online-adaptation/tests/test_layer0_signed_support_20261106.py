import numpy as np
import torch

from layer0_signed_support_20261106 import (
    causal_phase_support,
    paired_clock_indices,
)


def test_support_uses_early_bank_of_nearest_precursor_not_an_unrelated_shot():
    keys = torch.tensor([0.0, 1.0, 10.0])[:, None].expand(3, 192)
    precursor = torch.stack((keys[0], keys[2]))[:, None, :]
    early = torch.stack((keys[1], keys[0]))[:, None, :]
    out = causal_phase_support(
        keys, [(p, e) for p, e in zip(precursor, early, strict=True)]
    )
    assert out["nearest_bank"].tolist() == [0, 0, 1]
    assert out["allow"].tolist() == [True, False, True]
    torch.testing.assert_close(
        out["early_rms"], torch.tensor([1.0, 0.0, 10.0], dtype=torch.float64)
    )


def test_exact_phase_tie_abstains():
    keys = torch.zeros(1, 192)
    p, e = torch.ones(1, 192), -torch.ones(1, 192)
    assert causal_phase_support(keys, [(p, e)])["allow"].tolist() == [False]


def test_banks_matured_on_real_clock_and_sampled_same_count():
    times = np.arange(1000, dtype=np.float64) / 1000.0
    p, e = paired_clock_indices(times, 0.8)
    assert p.tolist() == [679, 711, 743]
    assert e.tolist() == [135, 199, 295]
    assert len(p) == len(e)
    p2, e2 = paired_clock_indices(times, 0.3)
    assert not len(p2) and not len(e2)
