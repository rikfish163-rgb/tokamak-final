import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_record_dynamics_20261112 as model


def test_nonpositive_record_does_not_provide_continuous_motion_target():
    n,c = 256,len(model.lag.CHANNELS)
    data = np.tile(np.arange(n,dtype=np.float32)[:,None],(1,c))
    mask = np.ones_like(data,dtype=bool)
    known = np.ones(n,dtype=bool)
    state = known.copy()
    q = np.array([127])
    state[q[0]+5] = False
    targets = model.block_targets(data,mask,state,known,q,np.ones(c,np.float32))
    assert targets['status'][0,4]==0. and targets['status_observed'][0,4]
    for k in model.SPECTRAL:
        assert not targets['motion_observed'][0,:5,k].any()
        assert targets['motion_observed'][0,5:,k].all()
    assert targets['motion_observed'][0,:,model.N1].all()
    assert targets['motion'][0,:,model.N1].min()>0.


def test_completed_clock_and_prefix_context_do_not_use_current_target():
    assert len(model.origins(model.lag.FIRST+model.lag.H+1))==0
    assert model.origins(model.lag.FIRST+model.lag.H+2).tolist()==[model.lag.FIRST]
    a = np.ones(256,dtype=bool)
    known = a.copy()
    q = np.array([127,159])
    b = a.copy()
    b[128:] = False
    h = model.record_context(a,known,q)
    k = model.record_context(b,known,q)
    np.testing.assert_array_equal(h[0],k[0])
    assert not np.array_equal(h[1],k[1])


def test_shift_keeps_conditional_membership_and_observed_values():
    y = torch.arange(32,dtype=torch.float32).reshape(2,8,2)
    mask = torch.ones_like(y,dtype=torch.bool)
    mask[0,[2,5,6],0] = False
    mask[1,:,1] = False
    shifted = model.shift_supported(y,mask)
    torch.testing.assert_close(shifted[~mask],y[~mask])
    for i in range(len(y)):
        for j in range(y.shape[-1]):
            keep = mask[i,:,j]
            torch.testing.assert_close(shifted[i,keep,j].sort().values,y[i,keep,j].sort().values)
    assert not torch.equal(shifted[mask],y[mask])


def test_isolation_removes_only_the_first_order_record_loss_component():
    a = torch.tensor([[[2.,3.],[1.,4.]],[[1.,-2.],[3.,.5]]])
    b = torch.tensor([[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]])
    r = model.isolate(a,b)
    torch.testing.assert_close(r[0],torch.tensor([[0.,3.],[1.,4.]]))
    torch.testing.assert_close((r*b).sum((1,2)),torch.zeros(2))
    torch.testing.assert_close(r[1],a[1])


def test_all_rotation_arms_get_the_same_actual_norm_and_common_eligibility():
    old = torch.zeros((2,4,2))
    old[:,0,0] = .6
    candidate = torch.zeros((3,2,4,2))
    candidate[0,:,1,0] = .1
    candidate[1,:,2,0] = .8
    candidate[2,:,3,0] = .3
    candidate[2,1] = old[1]
    out,used = model.equal_rotation(old,candidate)
    assert used.tolist()==[True,False]
    torch.testing.assert_close(out.flatten(2).norm(dim=2),old.flatten(1).norm(dim=1).expand(3,-1))
    torch.testing.assert_close(out[:,1],old[1].expand_as(out[:,1]))
    assert (out[:,0,0,0]<old[0,0,0]).all()
