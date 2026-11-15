import numpy as np

from experiments.risk_aligned_online import commissioning_field_scale_20261113 as audit


def test_fixed_prefix_selects_both_directions_and_changes_only_declared_values():
    split, gain = audit.reference_levels([.025,.027,.029,2.3,2.5,2.7])
    assert gain > 90 and .029 < split < 2.3
    for v,sign in ((.027,1),(2.5,-1)):
        x = np.ones((160,23),np.float32)
        x[:,audit.BT] = -v
        x[100:,audit.BT] = -1.5*v
        x[:,audit.QA] = np.arange(160,dtype=np.float32)
        m = np.ones_like(x,dtype=bool)
        m[-1,audit.QA] = False
        s = dict(values=x,available=m)
        got = audit.rescale(s,(audit.BT,audit.QA),gain**sign)
        expected = x.copy()
        for c in (audit.BT,audit.QA):
            expected[m[:,c],c] *= gain**sign
        np.testing.assert_allclose(got['values'],expected)
        np.testing.assert_array_equal(got['available'],m)
        assert not np.shares_memory(got['values'],x)
        assert audit.direction(x[:,audit.BT],split)==sign
        assert audit.direction(x[:99,audit.BT],split)==0
        assert audit.direction(x[:100,audit.BT],split)==sign
