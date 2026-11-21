import torch
from experiments.risk_aligned_online.anchored_fast_window_20261023.constant_probe import constant_observed_window


def test_preserves_observed_transformed_mean_and_availability():
    x=torch.arange(2*8*23,dtype=torch.float32).reshape(2,8,23)/100
    mask=torch.ones_like(x,dtype=torch.bool);mask[:,0,0]=False;mask[:,:,2]=False
    center=torch.arange(23,dtype=torch.float32)/10;scale=torch.ones(23)*2
    old=x.clone();c=constant_observed_window(x,mask,center,scale)
    a=torch.asinh((torch.where(mask,x,center)-center)/scale)
    b=torch.asinh((torch.where(mask,c,center)-center)/scale)
    torch.testing.assert_close(a.sum(1)/mask.sum(1).clamp_min(1),b.sum(1)/mask.sum(1).clamp_min(1),atol=1e-6,rtol=1e-5)
    for i in range(2):
        for j in range(23):
            selected=c[i,:,j][mask[i,:,j]]
            if len(selected): assert torch.all(selected==selected[0])
    torch.testing.assert_close(x,old,atol=0,rtol=0)
    assert torch.isfinite(c).all()
