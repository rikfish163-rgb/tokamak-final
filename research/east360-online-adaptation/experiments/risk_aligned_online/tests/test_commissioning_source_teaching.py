import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_ordinal_pairs_20261111 import endpoints
from experiments.risk_aligned_online.commissioning_source_teaching_20261111 import source_endpoints
from experiments.risk_aligned_online.commissioning_source_teaching_east_20261111 import match_magnitude


def test_relaxed_context_retains_existing_pairs_and_recovers_empty_bin():
    times = np.arange(330) * .001
    down = .320
    valid = times <= down
    official = valid & (times >= down-.025)
    result = source_endpoints(times, valid, official, down, 25.)
    original = endpoints(times, valid, True, down, 0., first_effect=217)
    np.testing.assert_array_equal(result['legacy'], original)
    np.testing.assert_array_equal(result['coverage'][:len(original)], original)
    assert len(original) == 2
    far, near = result['coverage'][2]
    assert 127 <= far < 217 <= near
    assert 95 < round((down-times[near])*1000, 9) <= 125


def test_short_source_shot_can_teach_after_release_using_only_old_context():
    t = np.arange(256)*.001
    down = .250
    valid = t <= down
    labels = valid & (t >= down-.025)
    r = source_endpoints(t, valid, labels, down, 25.)
    assert len(r['legacy']) == 0 and len(r['coverage']) == 1 and len(r['native']) == 3
    assert np.all(r['coverage'][:,1] >= 217)
    lead = np.round((down-t[r['native'][:,1]])*1000., 9)
    assert np.all((lead > 10.) & (lead <= 25.))
    assert np.all(labels[r['native'][:,1]])
    assert np.all(t[r['native'][:,0]] <= t[r['native'][:,1]]-.032+1e-9)


def test_native_uses_actual_official_mask_and_excludes_ten_ms_boundary():
    t = np.arange(256)*.001
    valid = t <= .250
    label = np.zeros(len(t), bool)
    label[[238, 240]] = True
    r = source_endpoints(t, valid, label, .250, 25.)
    np.testing.assert_array_equal(r['native'], [[206, 238]])


def test_norm_control_preserves_direction_and_uses_the_same_per_sample_magnitude():
    candidate = torch.tensor([[3., 4.], [-4., 0.], [0., 0.]])
    reference = torch.tensor([[0., 2.], [0., 0.], [0., 0.]])
    y = match_magnitude(candidate, reference)
    torch.testing.assert_close(y, torch.tensor([[1.2, 1.6], [0., 0.], [0., 0.]]))
    torch.testing.assert_close(y.norm(dim=1), reference.norm(dim=1))
    with pytest.raises(ValueError, match='direction'):
        match_magnitude(torch.zeros_like(reference), reference)
