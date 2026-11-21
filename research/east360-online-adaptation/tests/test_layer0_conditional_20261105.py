import torch

from run_layer0_conditional_20261105 import same_site_targets


def test_conditional_controls_keep_norm_only_at_same_site():
    real = torch.zeros(3, 256)
    real[0, 0], real[1, 1], real[2, 2] = 0.4, 0.7, 0.8
    real[:, 128:] = 0.03
    stale = -2 * real.roll(1, dims=1)
    out = same_site_targets(real, stale)
    torch.testing.assert_close(out["intra"][:, :128], real[:, :128])
    torch.testing.assert_close(
        out["first_direction"][:, 0], torch.tensor([0.4, 0.7, 0.8])
    )
    for target in out.values():
        assert torch.count_nonzero(target[:, 128:]) == 0
        torch.testing.assert_close(target.norm(dim=1), real[:, :128].norm(dim=1))


def test_no_mature_anchor_has_no_action():
    values = torch.zeros(0, 256)
    assert all(v.shape == (0, 256) for v in same_site_targets(values, values).values())
