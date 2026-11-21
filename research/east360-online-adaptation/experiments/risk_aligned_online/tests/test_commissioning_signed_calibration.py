import torch

from experiments.risk_aligned_online.commissioning_signed_calibration_20261111 import (
    alarm_loss, mirror_loss,
)


def test_mirror_loss_prefers_separation_not_a_global_risk_lift():
    mirror = torch.tensor([[.2, .4], [.3, .5]])
    real = torch.tensor([[.5, .2], [.6, .3]])
    common = mirror+2.
    assert mirror_loss(real, mirror) < mirror_loss(common, mirror)
    assert mirror_loss(real, mirror) < mirror_loss(mirror, real)


def test_alarm_fit_penalizes_one_bad_safe_peak_even_when_average_looks_safe():
    d = torch.tensor([True, False])
    benign = torch.tensor([0., 2., -3., -4., -2.])
    unsafe = torch.tensor([0., 2., -3., 1., -2.])
    assert alarm_loss(benign, [2, 3], d, .0) < alarm_loss(unsafe, [2, 3], d, .0)
