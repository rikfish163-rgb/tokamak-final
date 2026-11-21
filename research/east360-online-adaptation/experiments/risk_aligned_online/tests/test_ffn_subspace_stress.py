import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ffn_subspace_stress import (
    sample_convention, map_values, distort_chain, distort_source_record)
from experiments.risk_aligned_online.tests.test_ffn_subspace import model, example


def test_same_numerical_convention_in_all_past_truth_and_later_windows(model):
    e = example()
    e['qx'] = e['x'].clone()
    e['qa'] = e['a'].clone()
    e['truth'] = e['x'][:, :25].clone()
    e['mask'] = e['a'][:, :25].clone()
    e['segment_x'] = e['x'].expand(3, -1, -1).clone()
    e['segment_a'] = e['a'].expand(3, -1, -1).clone()
    e['segment_y'] = torch.zeros(3)
    e['segment_lead'] = torch.ones(3)
    chain = dict(device='J-TEXT', steps=[e, e], is_disrupt=False)
    before = e['x'].clone()
    moved, receipt = distort_chain(model, chain, np.random.default_rng(131))
    out = moved['steps'][0]
    assert receipt['scope'] == 'SOURCE_NUMERICAL_STRESS_NOT_PHYSICS'
    assert len(receipt['gain']) == len(receipt['bias']) == 23
    assert not torch.equal(out['x'], before)
    assert torch.equal(out['x'], out['qx'])
    assert torch.equal(out['truth'], out['x'][:, :25])
    assert torch.equal(out['segment_x'], out['x'].expand(3, -1, -1))
    assert torch.equal(out['x'], moved['steps'][1]['x'])
    assert torch.equal(out['segment_y'], e['segment_y'])
    assert torch.equal(e['x'], before)


def test_identity_is_exact_and_missing_values_do_not_change(model):
    e = example(); x = e['x'].clone(); mask = e['a'].clone()
    mask[:, :, 2] = False
    convention = dict(gain=[1.] * 23, bias=[0.] * 23)
    assert torch.equal(map_values(model, x, mask, convention), x)
    convention['bias'][2] = 1.
    assert torch.equal(map_values(model, x, mask, convention), x)
    convention['bias'][1] = 1.
    out = map_values(model, x, mask, convention)
    assert torch.equal(out[:, :, 2], x[:, :, 2])
    assert not torch.equal(out[:, :, 1], x[:, :, 1])


def test_source_record_keeps_time_labels_and_availability(model):
    e = example()
    s = dict(device='J-TEXT', values=e['x'][0].numpy(), available=e['a'][0].numpy(),
             times=np.arange(128) / 1000., y=np.zeros(128), down=.127, is_disrupt=True)
    moved, receipt = distort_source_record(model, s, np.random.default_rng(131))
    assert not np.array_equal(moved['values'], s['values'])
    for name in ('times', 'y', 'available'):
        assert np.array_equal(moved[name], s[name])
    assert moved['down'] == s['down'] and moved['is_disrupt'] == s['is_disrupt']
    with pytest.raises(ValueError, match='SOURCE'):
        distort_source_record(model, dict(s, device='EAST'), np.random.default_rng(131))
    with pytest.raises(ValueError, match='SOURCE'):
        distort_chain(model, dict(device='EAST', steps=[e]), np.random.default_rng(131))


def test_sampling_reproducible_without_target_statistics():
    left = sample_convention(np.random.default_rng(13))
    right = sample_convention(np.random.default_rng(13))
    assert left == right
    assert np.all(np.asarray(left['gain']) > 0)
    assert np.max(np.abs(left['bias'])) <= 1.
    assert np.max(np.abs(np.log(left['gain']))) <= .35
