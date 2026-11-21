import json

import numpy as np
import pytest

from experiments.risk_aligned_online.matched_forecast_meta_20261022.resume import completed_prefix


def save_pair(root, sid, *, arms=('static', 'original', 'forecast', 'meta')):
    np.savez_compressed(root / f'{sid}.npz', times=np.arange(3.),
                        **{arm: np.array([.1, .2, .3]) for arm in arms})
    (root / f'{sid}_updates.json').write_text('{}')


def test_complete_pairs_are_recognized_without_outcomes(tmp_path):
    save_pair(tmp_path, 11)
    save_pair(tmp_path, 12)
    assert completed_prefix(tmp_path, [11, 12, 13]) == [11, 12]


def test_missing_update_or_noncontiguous_shot_is_rejected(tmp_path):
    save_pair(tmp_path, 11)
    (tmp_path / '11_updates.json').unlink()
    with pytest.raises(ValueError, match='paired'):
        completed_prefix(tmp_path, [11, 12])
    (tmp_path / '11_updates.json').write_text('{}')
    save_pair(tmp_path, 13)
    with pytest.raises(ValueError, match='contiguous'):
        completed_prefix(tmp_path, [11, 12, 13])


def test_corrupt_or_wrong_trace_is_rejected(tmp_path):
    save_pair(tmp_path, 11, arms=('static',))
    with pytest.raises(ValueError, match='trace'):
        completed_prefix(tmp_path, [11, 12])
    save_pair(tmp_path, 11)
    (tmp_path / '11_updates.json').write_text('[]')
    with pytest.raises(ValueError, match='updates'):
        completed_prefix(tmp_path, [11, 12])


def test_nonfinite_or_invalid_probability_is_rejected(tmp_path):
    save_pair(tmp_path, 11)
    for bad in (np.nan, 1.1):
        np.savez_compressed(tmp_path / '11.npz', times=np.arange(3.),
                            **{arm: np.array([.1, bad, .3])
                               for arm in ('static', 'original', 'forecast', 'meta')})
        with pytest.raises(ValueError, match='trace'):
            completed_prefix(tmp_path, [11, 12])
