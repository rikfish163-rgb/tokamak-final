import torch

from experiments.risk_aligned_online.commissioning_receiver_reach_20261111 import project_write


def test_receiver_projection_preserves_mask_and_actual_shot_dose():
    value = torch.tensor([[[3.,4.],[7.,2.]], [[1.,0.],[4.,5.]], [[.2,.0],[.0,.0]]])
    valid = torch.tensor([[True,False],[False,False],[True,True]])
    radius = torch.tensor([2., 1., 1.])[:,None,None]
    out = project_write(value,valid,radius)
    torch.testing.assert_close(out[0,0],torch.tensor([1.2,1.6]))
    assert out[0,1].count_nonzero() == out[1].count_nonzero() == 0
    torch.testing.assert_close(out[2],value[2])
