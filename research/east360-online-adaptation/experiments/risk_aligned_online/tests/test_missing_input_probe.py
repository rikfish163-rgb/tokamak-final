import numpy as np

from experiments.risk_aligned_online.missing_input_probe import mask_inputs
from experiments.risk_aligned_online.io23 import RAW23


def test_mask_does_not_modify_original_or_other_channels():
    shot = dict(values=np.ones((10, 23)), available=np.ones((10, 23), bool), y=np.arange(10))
    changed, = mask_inputs([shot], ['MNM'])
    idx = RAW23.index('MNM')
    assert shot['available'].all() and (shot['values'] == 1).all()
    assert not changed['available'][:, idx].any()
    assert (changed['values'][:, idx] == 0).all()
    assert np.delete(changed['available'], idx, axis=1).all()
    assert (np.delete(changed['values'], idx, axis=1) == 1).all()
    assert np.array_equal(changed['y'], shot['y'])
