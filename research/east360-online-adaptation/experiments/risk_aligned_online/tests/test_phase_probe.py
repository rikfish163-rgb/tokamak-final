import numpy as np


def test_probe_closed_form_satisfies_primal_normal_equation_and_swap_reverses_direction():
    from experiments.risk_aligned_online.phase_memory_20261024.probe import fit_direction
    rng=np.random.default_rng(2);e=rng.normal(size=(12,5));l=e+rng.normal(size=(12,5))
    shots=np.repeat([1,2,3],4)
    w,receipt=fit_direction(e,l,shots,'ordered')
    a=(l-e)/np.sqrt(12);b=np.ones(12)/np.sqrt(12)
    expected=np.linalg.solve(a.T@a+receipt['ridge']*np.eye(5),a.T@b)
    np.testing.assert_allclose(w,expected,rtol=1e-12,atol=1e-12)
    swapped,_=fit_direction(l,e,shots,'ordered')
    np.testing.assert_allclose(swapped,-w,rtol=1e-12,atol=1e-12)


def test_order_controls_preserve_all_windows_and_do_not_use_query():
    from experiments.risk_aligned_online.phase_memory_20261024.probe import fit_direction
    rng=np.random.default_rng(12);e=rng.normal(size=(12,5));l=rng.normal(size=(12,5));shots=np.repeat([1,2,3],4)
    _,u=fit_direction(e,l,shots,'unpaired')
    ix=np.array(u['early_indexes'])
    np.testing.assert_array_equal(np.sort(ix),np.arange(12))
    assert (shots[ix]!=shots).all()
    w,s=fit_direction(e,l,shots,'shuffled_order')
    y=np.array(s['targets'])
    assert all(y[shots==sid].sum()==0 for sid in (1,2,3))
    repeated,_=fit_direction(e,l,shots,'shuffled_order')
    np.testing.assert_array_equal(w,repeated)
