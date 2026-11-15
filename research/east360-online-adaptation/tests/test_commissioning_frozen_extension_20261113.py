import numpy as np

from experiments.risk_aligned_online import commissioning_frozen_extension_20261113 as extension


def test_balanced_replay_keeps_both_strata_equal_and_preserves_original_order():
    y = [True, True, False, False, True, False, False, True, False, False, False]
    prior = [True, True, True, True, True, True, False, False, False, False, False]
    i = extension.balanced_indices(y, prior)
    y, prior = np.asarray(y), np.asarray(prior)
    assert len(i) == 8
    assert np.all(np.diff(i)>0)
    assert all((y[i] & (prior[i]==g)).sum() == (~y[i] & (prior[i]==g)).sum() for g in (True,False))
    assert np.flatnonzero(~prior & y)[0] in i
    np.testing.assert_array_equal(i, extension.balanced_indices(y, prior))


def test_inventory_only_excludes_the_original_ids_not_signal_or_outcome(tmp_path):
    for s in (61324,61329,61998,62122):
        path = tmp_path / str(s//100*100) / f'{s}.hdf5'
        path.parent.mkdir(exist_ok=True)
        path.touch()
    rows = extension.inventory(tmp_path, [61324,62122], [61998])
    assert [(r['shot'],r['old835']) for r in rows] == [(61329,False),(61998,True)]
