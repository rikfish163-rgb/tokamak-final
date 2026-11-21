import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_async_observation_20261114 as trial
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_group_releases_follow_physical_packet_maturity_not_next_issue():
    q = np.array([127,159,191])
    async_rel = trial.release_schedule(q,8,asynchronous=True)
    sync_rel = trial.release_schedule(q,8,asynchronous=False)
    np.testing.assert_array_equal(async_rel- q[:,None], [[26,26,34,26]]*3)
    np.testing.assert_array_equal(sync_rel- q[:,None], [[34,34,34,34]]*3)
    assert async_rel[0,0] < q[1] < async_rel[0,2]
    np.testing.assert_array_equal(trial.release_schedule(q,0,asynchronous=True),
                                  trial.release_schedule(q,0,asynchronous=False))


def test_late_radiation_group_cannot_erase_on_time_eligible_groups():
    q = np.array([127,159,191])
    m = np.ones((228,len(trial.lag.CHANNELS)),bool)
    # Remove all four radiation outcomes in the first physical target window.
    m[128:153, trial.clock.FEEDBACK] = False
    ctx_ok = np.ones((len(q), m.shape[1]),bool)
    a = trial.release_schedule(q,8,asynchronous=True)
    b = trial.release_schedule(q,8,asynchronous=False)
    diff = ~np.array([False,False,True,False])
    v = trial.packet_support(q,m,ctx_ok,a)
    w = trial.packet_support(q,m,ctx_ok,b)
    np.testing.assert_array_equal(v[0],diff)
    np.testing.assert_array_equal(v,w)
    # Truncating at q+34 (non-inclusive) excludes the slow final release,
    # without withdrawing the active physical prefix of the other groups.
    v = trial.packet_support(q,m[:225],ctx_ok,a)
    w = trial.packet_support(q,m[:225],ctx_ok,b)
    np.testing.assert_array_equal(v[-1],diff)
    assert not w[-1].any()


def test_asynchronous_handoff_preserves_earlier_direction_until_its_own_clock():
    starts = np.array([[10,10,18,10],[42,42,50,42]])
    table = torch.zeros((2,3,1,4))
    table[0,1:] = torch.tensor([[2.,3.,5.,7.],[11.,13.,17.,19.]])[:,None,:]
    table[1] = -table[0]
    t = np.array([9,10,14,18,22,42,46,50,54,58])
    out = trial.blended_state(starts,table,t,8)[0,:,0]
    torch.testing.assert_close(out[1],torch.zeros(4))
    torch.testing.assert_close(out[2],torch.tensor([1.,1.5,0.,3.5]))
    torch.testing.assert_close(out[3],torch.tensor([2.,3.,0.,7.]))
    torch.testing.assert_close(out[4],torch.tensor([2.,3.,2.5,7.]))
    torch.testing.assert_close(out[5],torch.tensor([2.,3.,5.,7.]))
    torch.testing.assert_close(out[6],torch.tensor([6.5,8.,5.,13.]))
    torch.testing.assert_close(out[-1],torch.tensor([11.,13.,17.,19.]))


def test_real_encoder_holds_estimated_values_out_of_the_token_write_basis():
    torch.manual_seed(13)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2,input_view='source'),
                    torch.zeros(23),torch.ones(23)).eval().requires_grad_(False)
    raw = torch.randn(3,16,23)
    mask = torch.ones_like(raw,dtype=torch.bool)
    mask[:, :, trial.clock.RADIATION] = False
    ends = np.array([25,26,27])
    val, expected, center, carry = trial.risk_inputs(base,raw,mask,ends,8)
    assert not expected[:, :, trial.clock.RADIATION].any()
    # Artificially optimistic risk view has a very variable radiation lane.
    # The actual write basis has none, so the radiation-only states cannot write.
    augmented = torch.ones_like(mask)
    carry[:, :, trial.clock.RADIATION] = torch.randn_like(carry[:, :, trial.clock.RADIATION])
    state = torch.zeros((2,3,3,4))
    state[...,2] = 1.
    out, dose, err = trial.forward_views(base,val,mask,(carry,),(augmented,),
                                        lambda v: v,((1,state),),.8)
    torch.testing.assert_close(out[1],out[2],rtol=0.,atol=1e-6)
    torch.testing.assert_close(out[1],out[3],rtol=0.,atol=1e-6)
    assert not dose.any() and err<1e-6


def test_real_encoder_realizes_matched_nonzero_dose_without_touching_parent():
    torch.manual_seed(17)
    base = Simple23(Config(window=16,patch=4,dim=8,heads=2,layers=2),torch.zeros(23),
                    torch.ones(23)).eval().requires_grad_(False)
    v = torch.randn(4,16,23)
    m = torch.ones_like(v,dtype=torch.bool)
    state = torch.randn(2,4,3,4)
    out,dose,err = trial.forward_views(base,v,m,(),(),lambda x:x,((0,state),),.33)
    plain,_,_ = trial.forward_views(base,v,m,(),(),lambda x:x,(),.33)
    torch.testing.assert_close(plain[0],out[0],rtol=0.,atol=1e-6)
    torch.testing.assert_close(dose,torch.full((4,),.33),rtol=0.,atol=1e-6)
    assert err<1e-6 and not torch.allclose(out[1],out[2])


def test_nearly_cancelled_family_gradient_is_not_amplified_to_a_full_step():
    g = torch.tensor([[0.,0.],[1e-6,-1e-6],[3.,4.]])
    d = trial.normalize_direction(g)
    torch.testing.assert_close(d[0],g[0])
    torch.testing.assert_close(d[1],torch.tensor([.0001,-.0001]))
    torch.testing.assert_close(d[2],torch.tensor([.6,.8]))
