import torch

from experiments.risk_aligned_online.persistence_control_20261023.control import persistence_target


def test_sham_uses_origin_observation_not_any_future_target_value():
    x=torch.arange(2*4*23).reshape(2,4,23).float(); m=torch.ones_like(x,dtype=torch.bool)
    truth=torch.randn(2,25,23);mask=torch.ones_like(truth,dtype=torch.bool);mask[0,:,5]=False
    ends=[128,153]; original=truth.clone()
    a=persistence_target((x,m,truth,mask,ends))
    b=persistence_target((x,m,truth+1000,mask,ends))
    torch.testing.assert_close(a[2],x[:,-1:].expand_as(truth))
    torch.testing.assert_close(a[2],b[2]);torch.testing.assert_close(truth,original)
    assert a[0] is x and a[1] is m and a[3] is mask and a[4] is ends
    assert a[2].data_ptr()!=truth.data_ptr()
