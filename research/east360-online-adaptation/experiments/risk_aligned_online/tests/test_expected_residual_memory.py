import copy

import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.context_memory_20261028.scaled import ScaledContextMemory
from experiments.risk_aligned_online.expected_dynamics_20261028.forecast import DirectForecast
from experiments.risk_aligned_online.expected_dynamics_20261028.memory import ExpectedMemory


class KnownIncrement(nn.Module):
    def forward(self, features):
        return torch.arange(1, 5, device=features.device, dtype=features.dtype)[None, :, None].expand(len(features), -1, 23)


def test_zero_expectation_preserves_existing_context_algorithm():
    torch.manual_seed(131)
    old = ScaledContextMemory(64, torch.ones(4, 23))
    new = ExpectedMemory(64, torch.ones(4, 23), DirectForecast(), use_expected=True)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(k.startswith('predictor.') for k in missing) and not unexpected
    x = torch.randn(14, 23); a = torch.rand_like(x) > .15
    norm = old.trace(x, a, mode='memory')['norm']
    for mode in ('off', 'memory', 'hold', 'reverse', 'local'):
        left = old.trace(x, a, mode=mode, reference_norm=norm, return_before=True)
        right = new.trace(x, a, mode=mode, reference_norm=norm, return_before=True)
        for key in ('delta', 'before_delta', 'forecast', 'forecast_loss', 'context', 'norm'):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_expected_change_is_not_misinterpreted_as_shot_local_error():
    m = ExpectedMemory(64, torch.ones(4, 23), KnownIncrement(), use_expected=True)
    x = torch.arange(12.).unsqueeze(-1).expand(-1, 23).clone(); a = torch.ones_like(x, dtype=torch.bool)
    trace = m.trace(x, a, mode='memory')
    assert trace['context'].count_nonzero() == 0
    assert trace['delta'].count_nonzero() == 0
    assert trace['forecast_loss'] == 0
    x[6:] += 2
    shifted = m.trace(x, a, mode='memory')
    assert shifted['context'][:6].count_nonzero() == 0
    assert shifted['context'][6:].norm() > 0
    off = m.trace(x, a, mode='off')
    assert off['delta'].count_nonzero() == 0 and off['forecast'].norm() > 0


def test_expected_feedback_is_prefix_causal_and_handles_missing_channels():
    f = DirectForecast()
    with torch.no_grad():
        f.net[-1].weight.normal_(std=.02)
    m = ExpectedMemory(64, torch.ones(4, 23), f, use_expected=True)
    x = torch.randn(17, 23); a = torch.rand_like(x) > .1
    x[~a] = float('nan')
    full = m.trace(x, a, mode='memory', return_before=True)
    prefix = m.trace(x[:9], a[:9], mode='memory', return_before=True)
    for key in ('delta', 'context', 'before_delta', 'expected_forecast'):
        torch.testing.assert_close(full[key][:9], prefix[key], atol=1e-7, rtol=1e-6)
    clean = torch.where(a, x, 0.)
    torch.testing.assert_close(full['delta'], m.trace(clean, a, mode='memory')['delta'])
    assert torch.isfinite(full['forecast_loss'])
    assert m.trace(x, torch.zeros_like(a), mode='memory')['delta'].count_nonzero() == 0


def test_forecast_is_frozen_and_only_residual_has_fast_training_gradient():
    f = DirectForecast()
    m = ExpectedMemory(64, torch.ones(4, 23), f, use_expected=True)
    initial = copy.deepcopy(m.predictor.state_dict())
    x = torch.randn(12, 23); a = torch.ones_like(x, dtype=torch.bool)
    m.trace(x, a, mode='memory')['forecast_loss'].backward()
    assert m.kernel.grad.norm() > 0
    assert all(not p.requires_grad and p.grad is None for p in m.predictor.parameters())
    for name, value in m.predictor.state_dict().items():
        assert torch.equal(value, initial[name])
    with pytest.raises(ValueError):
        ExpectedMemory(64, torch.ones(4, 23), f, use_expected='yes')
