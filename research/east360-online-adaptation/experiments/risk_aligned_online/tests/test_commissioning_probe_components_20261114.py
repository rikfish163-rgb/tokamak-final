import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_weak_signal_probe_20261114 as probe
from experiments.risk_aligned_online import commissioning_probe_components_20261114 as pieces


def test_level_and_shape_views_separate_a_constant_offset():
    ramp=np.arange(128,dtype=np.float32)[:,None]*.01+np.zeros((1,8),np.float32)
    same=np.ones_like(ramp,dtype=bool)
    ends=np.array([127])
    before,_=probe.trajectory(ramp,same,ends)
    after,_=probe.trajectory(ramp+1.,same,ends)
    np.testing.assert_allclose(pieces.component(after,'level')-pieces.component(before,'level'),1.)
    np.testing.assert_allclose(pieces.component(before,'shape'),pieces.component(after,'shape'),atol=1e-6)


def test_value_subspace_can_be_full_rank_without_isolating_other_lanes():
    # Pure bin-constant signals in the selected eight lanes invert exactly.
    # An unselected lane aligned to one of the same directions confounds the readout.
    w=torch.zeros((64,8,46))
    for i,k in enumerate(probe.COLUMNS):
        w[i,:,k]=1./8
    j=next(i for i in range(23) if i not in probe.COLUMNS)
    w[0,:,j]=1./8
    b=torch.zeros(64)
    x=torch.zeros((1,16,8,46))
    x[:,:,:,list(probe.COLUMNS)]=torch.arange(1.,9.)
    subspace,inverse=pieces.value_subspace(w.flatten(1))
    assert torch.linalg.matrix_rank(subspace)==8
    one=pieces.patch_views(x.flatten(2),w.flatten(1),b,inverse)
    np.testing.assert_allclose(pieces.component(one['lane'],'level'),np.arange(1.,9.)[None,:])
    x[:,:,:,j]=2.
    other=pieces.patch_views(x.flatten(2),w.flatten(1),b,inverse)
    np.testing.assert_allclose(other['lane'][0,0],3.)
    np.testing.assert_array_equal(one['raw'],other['raw'])


def test_after_deadline_pairs_are_not_counted_as_usable_feedback():
    actual=np.array([True,True,False,True])
    allowed=np.array([True,False,True,True])
    assert pieces.later_subset(actual,allowed).tolist()==[0,3]
