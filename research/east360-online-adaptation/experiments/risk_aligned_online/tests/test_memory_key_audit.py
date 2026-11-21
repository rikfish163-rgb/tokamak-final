import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.memory_key_audit_20261027 import (
    key_statistics, summarize,
)


def test_constant_keys_do_not_acquire_artificial_rank():
    result = key_statistics(torch.ones(5, 8))
    assert result['within_trace'] == 0
    assert result['saturated_fraction'] == 1
    assert result['step_rms'] == 0
    overall = summarize([dict(shot=1, **result)])
    assert overall['within_effective_rank'] is None
    assert overall['total_effective_rank'] is None


def test_domain_moments_weight_shots_not_duration():
    a = torch.zeros(2, 8)
    a[:, 0] = torch.tensor([-1., 1.])
    b = torch.zeros(20, 8)
    b[:, 0] = .5
    rows = [dict(shot=1, **key_statistics(a)), dict(shot=2, **key_statistics(b))]
    stats = summarize(rows)
    assert stats['mean'][0] == pytest.approx(.25)
    assert stats['within_trace'] == pytest.approx(.5)
    assert stats['between_trace'] == pytest.approx(.0625)
    assert stats['total_trace'] == pytest.approx(.5625)
    assert stats['total_effective_rank'] == pytest.approx(1.)
    assert stats['saturated_fraction'] == pytest.approx(1/16)
    assert np.allclose(stats['total_covariance'],
                       np.diag([.5625, 0, 0, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize('x', [torch.empty(0, 8), torch.ones(2, 9),
                              torch.full((2, 8), float('nan')), torch.full((2, 8), 2.)])
def test_invalid_or_non_tanh_keys_rejected(x):
    with pytest.raises(ValueError):
        key_statistics(x)


def test_one_patch_has_no_temporal_difference():
    assert key_statistics(torch.zeros(1, 8))['step_rms'] is None
    with pytest.raises(ValueError):
        summarize([])
    r = dict(shot=1, **key_statistics(torch.zeros(1, 8)))
    with pytest.raises(ValueError):
        summarize([r, r])
