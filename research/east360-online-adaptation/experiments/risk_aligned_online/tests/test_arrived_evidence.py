import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.arrived_evidence_20261024.model import EvidenceProbe, feedback_arrays, rotate_within_shot


def test_no_new_variation_has_exactly_zero_increment():
    torch.manual_seed(1)
    net = EvidenceProbe()
    h = torch.randn(8,64); d = torch.zeros(8,5,23); m = torch.rand_like(d) > .3
    a,b = net(h,d,m)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_actual_feedback_changes_score_and_trains_parameters():
    torch.manual_seed(1); net = EvidenceProbe()
    h=torch.randn(8,64);d=torch.randn(8,5,23);m=torch.ones_like(d,dtype=torch.bool)
    a,b=net(h,d,m)
    assert (a-b).abs().max()>0
    b.square().mean().backward()
    assert net.update[0].weight.grad[:,64:179].abs().sum()>0


def test_unobserved_values_never_change_increment():
    net=EvidenceProbe(); h=torch.randn(2,64);d=torch.randn(2,5,23);m=torch.rand_like(d)>.3
    altered=d.clone();altered[~m]=100000
    for a,b in zip(net(h,d,m),net(h,altered,m)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_feedback_only_reads_arrived_prefix_and_has_zero_constant_control():
    x=np.arange(40*23,dtype=float).reshape(40,23)/100
    m=np.ones_like(x,bool);m[11,2]=False
    s=dict(values=x,available=m);ends=np.array([10,20])
    c=np.zeros(23);scale=np.ones(23)
    d,v,old,ov=feedback_arrays(s,ends,c,scale)
    xx=x.copy();xx[26:]=1e6
    other=feedback_arrays(dict(values=xx,available=m),ends,c,scale)
    for a,b in zip((d,v,old,ov),other): np.testing.assert_array_equal(a,b)
    assert not v[0,0,2]
    q=np.arcsinh(x)
    np.testing.assert_allclose(d[0,:,0],q[11:16,0]-q[10,0])
    np.testing.assert_allclose(old[0,:,0],q[6:11,0]-q[5,0])
    with pytest.raises(ValueError): feedback_arrays(s,np.array([35]),c,scale)


def test_temporal_control_never_crosses_shots():
    ids=np.array([1,1,1,1,2,2]);d=np.arange(12).reshape(6,2);m=d%2==0
    dd,mm=rotate_within_shot(d,m,ids)
    np.testing.assert_array_equal(dd[:4],np.roll(d[:4],2,axis=0))
    np.testing.assert_array_equal(dd[4:],d[4:][::-1])
    np.testing.assert_array_equal(mm,dd%2==0)
