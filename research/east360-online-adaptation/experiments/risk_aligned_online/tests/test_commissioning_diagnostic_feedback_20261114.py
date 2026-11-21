import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_diagnostic_feedback_20261114 as run
from experiments.risk_aligned_online import commissioning_latent_teacher_20261114 as prior
from experiments.risk_aligned_online.io23 import RAW23


def test_replay_compares_same_support_for_older_packet_and_validated_record():
    x = np.ones((98,23),np.float32)
    m = np.ones_like(x,dtype=bool)
    a,f = run.SPECTRUM
    ciii = RAW23.index('CIII')
    x[41:44,[a,f]] = 0.
    m[8,ciii] = False
    signal = dict(values=x,available=m)
    # Each index holds a full [q, q+25] packet; compare q=32 to q=0.
    s = run.available_groups(signal,np.array([0,32,64]))
    assert not s[0,run.GROUPS.index((ciii,))]
    assert not s[1,run.GROUPS.index(run.SPECTRUM)]
    assert s[2,run.GROUPS.index(run.SPECTRUM)]
    assert s[1,run.GROUPS.index((RAW23.index('Mir_VV'),))]


def test_new_teacher_views_preserve_real_core_mask_and_diagnostic_marginals():
    x = torch.arange(128*23,dtype=torch.float32).reshape(1,128,23)
    m = torch.ones_like(x,dtype=torch.bool)
    m[:,-26,run.GROUPS[0][0]] = False
    views,valid = run.diagnostic_views(x,m)
    core,_ = prior.teacher_views(x,m)
    torch.testing.assert_close(views[0],core[0])
    for j,cols in enumerate(run.GROUPS):
        rev,true = views[2+2*j],views[1+2*j]
        move = list(cols)
        stay = [c for c in range(23) if c not in move]
        torch.testing.assert_close(true[:,-25:,move],x[:,-25:,move])
        torch.testing.assert_close(rev[:,-25:,move],x[:,-25:,move].flip(1))
        torch.testing.assert_close(true[:,-25:,stay],core[0,:,-25:,stay])
        torch.testing.assert_close(rev[:,-25:,stay],core[0,:,-25:,stay])
        torch.testing.assert_close(true[:,:-25],x[:,:-25])
        torch.testing.assert_close(rev[:,:-25],x[:,:-25])
    assert not valid[:,-25:,run.GROUPS[0][0]].any()


def test_absent_extra_information_is_exactly_the_core_direction():
    h = torch.tensor([[1.,2.,3.],[3.,2.,1.]])
    extra = torch.arange(2*4*3).reshape(2,4,3).float()
    gate = torch.zeros(2,4,dtype=torch.bool)
    torch.testing.assert_close(run.fuse(h,extra,gate),h,rtol=0.,atol=0.)
    gate[0,0] = True
    extra[0,0] *= 10.
    updated = run.fuse(h,extra,gate)
    torch.testing.assert_close(updated[1],h[1],rtol=0.,atol=0.)
    torch.testing.assert_close((updated-h)[0].norm(),h[0].norm())


def test_tiny_view_difference_cannot_turn_rounding_into_an_equal_length_step():
    core = torch.tensor([[1.,2.],[1.,2.]])
    extra = torch.tensor([[[1.e-7,-1.e-7]],[[.001,-.001]]])
    out = run.fuse(core,extra,torch.ones(2,1,dtype=torch.bool))
    torch.testing.assert_close(out[0],core[0],atol=2.e-7,rtol=0.)
    assert not torch.equal(out[1],core[1])
