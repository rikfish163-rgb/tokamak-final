import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_field_teaching_20261113 as experiment


def test_point_weights_balance_endpoint_labels_without_length_dominance():
    shot = np.repeat([0,1,2,3],[6,3,1,2])
    disruptive = np.array([True,True,False,False])
    label = np.array([0,0,0,0,0,1,0,1,1,0,0,0],dtype=bool)
    old,balanced = experiment.point_weights(shot,label,disruptive)
    np.testing.assert_allclose([old[shot==i].sum() for i in range(4)],[.25]*4)
    np.testing.assert_allclose(balanced[label].sum(),.5)
    np.testing.assert_allclose([balanced[label & (shot==i)].sum() for i in (0,1)],[.25,.25])
    np.testing.assert_allclose([balanced[~label & (shot==i)].sum() for i in range(4)],[.125]*4)


def test_safety_peak_penalizes_a_peak_that_moves_after_a_readout_change():
    g = torch.tensor([0,0,1,1,1,2,2])
    y = torch.tensor([False,True,False,False,False,False,False])
    f = torch.tensor([-.1,1.,-.1,-.2,-.3,-1.,-.5])[None]
    later = f.clone()
    later[0,2:5] = torch.tensor([-.4,2.,-.4])
    pos,neg = torch.tensor([0]),torch.tensor([1,2])
    assert experiment.warning(later,g,y,pos,neg,3) > experiment.warning(f,g,y,pos,neg,3)
