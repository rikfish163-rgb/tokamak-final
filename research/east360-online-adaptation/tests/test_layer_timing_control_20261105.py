import pytest
import torch

from layer_timing_control_20261105 import timing_targets


def test_comparison_changes_only_the_direction_within_the_same_site():
    real = torch.arange(1, 769, dtype=torch.float32).reshape(3, 256) / 1e4
    stale = -real
    prev = torch.flip(real[:2], dims=(1,))
    targets, n_tail = timing_targets(real, stale, prev)
    assert n_tail == 1
    for i in range(2):
        arm = f"layer{i}"
        section = slice(i * 128, (i + 1) * 128)
        other = slice((1 - i) * 128, (2 - i) * 128)
        orig = targets[arm]
        torch.testing.assert_close(orig[:, section], real[:, section])
        torch.testing.assert_close(targets[arm + "_stale"], -orig)
        for suffix in ("stale", "initial", "donor"):
            v = targets[f"{arm}_{suffix}"]
            assert torch.count_nonzero(v[:, other]) == 0
            torch.testing.assert_close(v.norm(dim=1), orig.norm(dim=1))
        init = torch.nn.functional.normalize(targets[arm + "_initial"])
        torch.testing.assert_close(init[1:], init[:1].expand(2, -1))
        donor = torch.nn.functional.normalize(targets[arm + "_donor"])
        torch.testing.assert_close(donor[1], donor[2])
        torch.testing.assert_close(
            donor[:2, section], torch.nn.functional.normalize(prev[:, section])
        )


def test_missing_donor_or_zero_direction_cannot_masquerade_as_matched():
    real = torch.full((2, 256), 0.02)
    with pytest.raises(ValueError, match="nonempty"):
        timing_targets(real, real, real[:0])
    with pytest.raises(ValueError, match="same norm"):
        timing_targets(real, real, torch.zeros_like(real))
