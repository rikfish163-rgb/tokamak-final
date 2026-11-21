import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_radius_feedback_20261111 as radius


def test_radius_is_relative_to_the_preissued_reference_not_absolute_level():
    mean = torch.arange(10, dtype=torch.float32)[None]
    std = torch.arange(1, 11, dtype=torch.float32)[None]
    z = (mean + 2*std)[:, None].repeat(1, 25, 1)
    got = radius.normalized_radius(z, mean, std)
    assert got.shape == (1, 5, 4)
    torch.testing.assert_close(got, torch.full_like(got, float(np.log1p(2.))))
    torch.testing.assert_close(got, radius.normalized_radius(2*mean[:,None]-z, mean, std))


def test_control_relabels_errors_without_changing_their_size_or_time():
    pred = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4).requires_grad_()
    truth = pred.detach()+torch.linspace(-.2, .2, 20).reshape(1,5,4)
    real, wrong = radius.residual_targets(pred, truth)
    assert not real.requires_grad and not wrong.requires_grad
    torch.testing.assert_close(real, truth-pred.detach())
    torch.testing.assert_close(wrong, real.roll(1, dims=-1))
    torch.testing.assert_close(real.abs().sum(-1),wrong.abs().sum(-1))


def test_origin_requires_a_known_reference_and_a_complete_target_before_effect():
    q = np.array([127, 159, 191, 223])
    visible = np.ones((4, 25, 10), dtype=bool)
    visible[2, 4, 8] = False
    known = np.ones((4, 10), dtype=bool)
    known[0, 0] = False
    assert radius.mature(q, visible, known, 249).tolist() == [False, True, False, False]
    assert radius.mature(q, visible, known, 250).tolist() == [False, True, False, True]
