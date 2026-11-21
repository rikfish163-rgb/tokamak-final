import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import feedback
from experiments.risk_aligned_online.shadow_commit_20261104.onset_attribution import (
    first_update_from_tensors,
    integrated_gradients,
    read_authorized_prefix,
    read_prefix_group,
)


def test_integrated_gradients_linear_and_quadratic_completeness():
    x = torch.tensor([1.25, -2.0, 0.75], dtype=torch.float64)
    coeff = torch.tensor([2.0, -0.5, 3.0], dtype=torch.float64)

    def function(batch):
        return (batch * coeff + batch.square()).sum(dim=1)

    attribution = integrated_gradients(function, x, steps=8)
    torch.testing.assert_close(attribution, coeff * x + x.square())
    torch.testing.assert_close(attribution.sum(), function(x[None])[0] - function(torch.zeros_like(x)[None])[0])


def test_integrated_gradients_refinement_and_zero_input():
    x = torch.tensor([2.0, -1.0], dtype=torch.float64)

    def function(batch):
        return batch.pow(3).sum(dim=1)

    exact = x.pow(3)
    coarse = integrated_gradients(function, x, steps=4)
    fine = integrated_gradients(function, x, steps=32)
    assert torch.linalg.vector_norm(fine - exact) < torch.linalg.vector_norm(coarse - exact)
    assert torch.equal(integrated_gradients(function, torch.zeros_like(x), steps=4), torch.zeros_like(x))


@pytest.mark.parametrize('steps', [0, -1, True, 1.5])
def test_integrated_gradients_rejects_invalid_steps(steps):
    with pytest.raises(ValueError):
        integrated_gradients(lambda x: x.sum(dim=1), torch.ones(2), steps=steps)


def test_first_ridge_step_matches_closed_form_and_respects_mask():
    initial = torch.zeros(168, 70, dtype=torch.float64)
    key_origin = torch.zeros(70, dtype=torch.float64)
    key_origin[:2] = torch.tensor([1.0, 2.0])
    key_current = torch.zeros(70, dtype=torch.float64)
    key_current[:2] = torch.tensor([3.0, -1.0])
    target = torch.linspace(-0.02, 0.02, 168, dtype=torch.float64)
    mask = torch.ones(168, dtype=torch.bool)
    mask[::3] = False
    actual, weight, _ = first_update_from_tensors(initial, key_origin, target, mask, key_current)
    expected_weight = torch.where(mask[:, None], target[:, None] * key_origin / 6.0, 0.0)
    torch.testing.assert_close(weight, expected_weight)
    torch.testing.assert_close(actual, expected_weight @ key_current)
    assert torch.equal(initial, torch.zeros_like(initial))


def test_first_ridge_step_projects_at_radius():
    initial = torch.zeros(168, 70, dtype=torch.float64)
    key = torch.zeros(70, dtype=torch.float64)
    key[0] = 1.0
    _, weight, _ = first_update_from_tensors(initial, key, torch.full((168,), 5.0, dtype=torch.float64), torch.ones(168, dtype=torch.bool), key)
    assert math.isclose(weight.norm().item(), 2.0, rel_tol=1e-12)


class _PrefixOnlyDataset:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.attrs = {'SampleRate': np.asarray([1000.0]), 'StartTime': np.asarray([1.25])}
        self.reads = []

    def __getitem__(self, item):
        assert isinstance(item, slice)
        assert item.start is None and item.stop == 136 and item.step is None
        self.reads.append(item)
        return self.values[item]


def test_only_136_raw_points_per_channel_and_actual_masks():
    datasets = {name: _PrefixOnlyDataset(np.full(200, 1.0, np.float32)) for name in io23.RAW23}
    datasets['CIII'].values[135] = -10.0
    datasets['Bt'].values[134] = np.nan
    raw = read_prefix_group(datasets, shot=87689)
    assert all(len(ds.reads) == 1 for ds in datasets.values())
    assert raw['values'].shape == raw['available'].shape == (136, 23)
    assert not raw['available'][:, 2].any()
    assert not raw['available'][134, 0]
    prepared = apply_input_contract(raw)
    assert not prepared['available'][135, 1]
    assert prepared['available'][134, 1]
    np.testing.assert_allclose(raw['times'][[0, 135]], [1.25, 1.385], rtol=0, atol=1e-14)


@pytest.mark.parametrize('shot', [0, 87688, 87689.0, True])
def test_other_shots_are_rejected_before_opening_a_raw_file(shot):
    with pytest.raises(ValueError):
        read_authorized_prefix(shot)


def test_feedback_origin_127_uses_exactly_points_128_to_135():
    values = np.broadcast_to(np.arange(136, dtype=np.float32)[:, None] / 10., (136, 23)).copy()
    available = np.ones((136, 23), dtype=bool)
    available[:, 2] = False
    available[135, 1] = False
    base = SimpleNamespace(center=torch.zeros(23), scale=torch.ones(23))
    _, target, observed = feedback(base, dict(values=values, available=available), 127, 135, device='cpu')
    target = target.reshape(8, 21)
    observed = observed.reshape(8, 21)
    expected = torch.asinh(torch.arange(128, 136, dtype=torch.float32) / 10.) - torch.asinh(torch.tensor(12.7))
    torch.testing.assert_close(target[:, 0], expected)
    assert observed[:, 0].all() and observed[:7, 1].all() and not observed[7, 1]
    assert target[7, 1] == 0
    assert int(observed.sum()) == 167
