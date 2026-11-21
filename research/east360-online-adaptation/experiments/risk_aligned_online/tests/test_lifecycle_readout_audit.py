import pytest

from experiments.risk_aligned_online.lifecycle_stream_20261025.compare_readout import validate_protocols


def test_only_readout_may_differ_in_paired_protocol():
    common = dict(threshold=.7, seed=131, source_sha256='a', source_statistics_sha256='b',
                  contract={'shot_ids': [1, 2]}, warmup=20, target_loss='disruptive_positive_only',
                  arms=['source', 'cold', 'intra'], torch='same', gpu='same')
    # Archived shape runs predate the explicit readout field; this is their
    # documented implementation, not an inferred general compatibility rule.
    validate_protocols(common, dict(common, readout='full'))
    validate_protocols(dict(common, readout='shape'), dict(common, readout='full'))
    with pytest.raises(ValueError, match='target_loss'):
        validate_protocols(common, dict(common, readout='full', target_loss='all_points'))
    with pytest.raises(ValueError, match='readout'):
        validate_protocols(dict(common, readout='level'), dict(common, readout='full'))
    with pytest.raises(ValueError, match='threshold'):
        validate_protocols(common, dict(common, readout='full', threshold=.8))
