import numpy as np

from experiments.risk_aligned_online.commissioning_lag_memory_20261111 import memory_views


def test_aligned_mean_uses_same_release_as_shifted_and_is_not_a_risk_offset():
    q=np.array([127,159])
    past=np.tile([1.,0.],(211,1)).astype(np.float32)
    past[159]=0.
    aligned=np.array([[0.,1.],[1.,0.]],np.float32)
    shifted=np.array([[0.,-1.],[0.,1.]],np.float32)
    origin=np.array([.1,0.],np.float32)
    present=np.array([True,False])
    out,stats=memory_views(past,q,aligned,shifted,present,origin)
    short,_=memory_views(past[:177],q[:1],aligned[:1],shifted[:1],present[:1],origin)
    for k in out:
        np.testing.assert_array_equal(out[k][:153],out['fixed'][:153])
        np.testing.assert_array_equal(out[k][:177],short[k])
        np.testing.assert_array_equal(out[k][:,:2],past)
        np.testing.assert_array_equal(out[k][159],0.)
    assert stats['updates']=={'real':1,'shifted':1}
    assert out['real'][153,2]<out['fixed'][153,2]


def test_true_and_shifted_memory_differ_when_current_edge_is_present():
    x=np.tile([0.,1.],(185,1)).astype(np.float32)
    aligned=np.array([[0.,1.]],np.float32)
    shifted=-aligned
    m,_=memory_views(x,np.array([127]),aligned,shifted,np.array([True]),np.array([.1,0.],np.float32))
    assert m['real'][153,3]>m['fixed'][153,3]>m['shifted'][153,3]
