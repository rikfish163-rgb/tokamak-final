import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_embedding_time_scope_20261114 as run


def test_level_and_motion_split_only_changes_measured_correction():
    v = torch.arange(128, dtype=torch.float32)[None, :, None].expand(1, 128, 2).clone()
    m = torch.ones_like(v, dtype=torch.bool)
    m[:, 13:21, 1] = False
    v.masked_fill_(~m, 0.)
    old = torch.ones_like(v)*20.
    terms = run.counterfactual_values(v, m, old, torch.ones_like(m), torch.tensor([9., 10.]))
    assert not any(t[~m].any() for t in terms.values())
    torch.testing.assert_close(terms['past_level']+terms['current_departure'], v)
    torch.testing.assert_close(terms['early96']+terms['latest32'], v)
    assert (terms['past_level'][m]==20.).all()
    assert torch.count_nonzero(terms['first_level'][m])==m.sum()


def test_reverse_respects_missing_lanes_and_keeps_value_distribution():
    v = torch.randn(2, 128, 3, generator=torch.Generator().manual_seed(1))
    available = torch.ones_like(v, dtype=torch.bool)
    available[0, 64:, 1] = False
    v.masked_fill_(~available, 0.)
    old = v+torch.arange(3.)
    terms = run.counterfactual_values(v, available, old, available, torch.ones(3))
    p = terms['wrong_patch_order']
    torch.testing.assert_close(v.sort(1).values, p.sort(1).values, atol=1e-6, rtol=0.)
    torch.testing.assert_close(v[0, :, 1], p[0, :, 1])
    assert not torch.equal(v[1], p[1])
    torch.testing.assert_close(v[:,-32:],terms['reverse_early96'][:,-32:])
    torch.testing.assert_close(v[:,:96],terms['reverse_latest32'][:,:96])
    assert not torch.equal(v[1,-32:],terms['reverse_latest32'][1,-32:])


def test_current_level_keeps_current_measured_channel_mean_not_the_past_mean():
    v = torch.tensor([1.,2.]).expand(1,128,2).clone()
    v[:,64:,0] = 3.
    m = torch.ones_like(v,dtype=torch.bool)
    m[:,64:,1] = False
    v.masked_fill_(~m,0.)
    terms = run.counterfactual_values(v,m,v*4.,m,torch.zeros(2))
    z = terms['current_level']
    torch.testing.assert_close(z[0,:64], torch.full((64,2),2.))
    assert not z[~m].any()
    assert not torch.equal(terms['past_level'],z)


def test_reference_samples_are_first_or_disjoint_earlier_than_current_window():
    v = np.arange(600, dtype=np.float32)[:, None]+np.zeros((600, 23), np.float32)
    t = np.array([256, 272, 320])
    x, m = run.past_windows(v, np.ones_like(v, dtype=bool), t)
    np.testing.assert_array_equal(x[:, 0, 0], t-255)
    np.testing.assert_array_equal(x[:, -1, 0], t-128)
    assert m.all()
    v[321:] = 9999.
    np.testing.assert_array_equal(x, run.past_windows(v[:321], np.ones_like(v[:321], dtype=bool), t)[0])
