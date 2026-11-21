import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_ordinal_pairs_20261111 as pairs


def test_positive_window_excluded_and_leads_recomputed_on_nonzero_clock():
    t = 4. + np.arange(240)/1000.
    down = 4.2
    take = pairs.endpoints(t,np.ones(len(t),bool),True,down,60.,first_effect=3)
    lead = np.round((down-t[take[:,1]])*1000.,9)
    np.testing.assert_array_equal(lead,[75.,105.])
    assert (t[take[:,1]]-t[take[:,0]]>=.032-1e-9).all()
    assert take.min()>=3


def test_strict_30_ms_and_inclusive_125_ms_on_sparse_valid_clock():
    t = 4. + np.arange(240)/1000.
    valid = np.zeros(len(t),bool)
    valid[[43,75,138,170]]=True
    take = pairs.endpoints(t,valid,True,4.2,25.,first_effect=3)
    np.testing.assert_array_equal(take,[[43,75]])


def test_safe_pair_uses_elapsed_time_on_gapped_grid_and_includes_last_valid():
    t = np.delete(np.arange(130),[15,16,17,18,40,48,99]).astype(float)/1000.
    valid = np.ones(len(t),bool)
    valid[-1]=False
    take = pairs.endpoints(t,valid,False,None,None,first_effect=5)
    assert take[-1,1]==len(t)-2
    assert (t[take[:,1]]-t[take[:,0]]>=.032-1e-9).all()
    assert valid[take].all() and (take[:,0]>=5).all()


def test_match_uses_both_endpoints_and_deterministic_ties_in_each_stratum():
    x=np.zeros((6,2,7))
    x[1,0]=2.
    x[2:4]=.2
    x[4,0]=.7
    x[5]=.2
    d=np.array([True,False,False,False,False,False])
    tail=np.array([False,True,True,True,False,False])
    a,b,c,r=pairs.match(x,d,tail,np.ones(7))
    np.testing.assert_array_equal(a,[0,0])
    np.testing.assert_array_equal(b,[2,5])
    np.testing.assert_array_equal(c,[True,False])
    np.testing.assert_allclose(r,[.2,.2])
    with pytest.raises(ValueError, match='stratum'):
        pairs.match(x,d,np.zeros(6,bool),np.ones(7))
