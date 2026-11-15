import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_hidden_background_20261113 as experiment


def test_background_uses_only_earlier_nonoverlapping_encoder_windows():
    h = np.arange(660,dtype=np.float32)[:,None]
    full = experiment.background_features(h)
    assert full[0][experiment.FIRST-1,0]==0.
    expected = h[experiment.FIRST,0] - h[127:255,0].mean()
    assert full[1][experiment.FIRST,0]==expected
    altered = h.copy()
    altered[430:] += 1000.
    for a,b in zip(full,experiment.background_features(altered),strict=True):
        np.testing.assert_array_equal(a[:430],b[:430])
    for a,b in zip(full,experiment.background_features(h[:405]),strict=True):
        np.testing.assert_array_equal(a[:405],b)


def test_safety_loss_penalizes_moved_peak_and_keeps_point_labels():
    base = torch.zeros(5)
    training = torch.tensor([0,1,1,0,0])
    on_time = torch.tensor([False,True,False,False,False])
    pos = torch.tensor([0.,1.,3.,0.,0.])
    a = torch.tensor([1.,-.2,-.4,-.4,-.4])
    b = torch.tensor([.9,2.,-.4,-.4,-.4])
    def sig(z):
        return experiment.shot_loss([pos,z],[training,base.long()],[on_time,torch.zeros_like(on_time)],1)
    assert sig(b)>sig(a)
