import copy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import (
    DiagnosticMemory, endpoint_corrections,
)
from experiments.risk_aligned_online.progressive_feedback_20261026.model import (
    ProgressiveDiagnosticMemory,
)
from experiments.risk_aligned_online.lifecycle_stream_20261025.model import LifecycleAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23


def inputs(n=12):
    torch.manual_seed(131)
    torch.set_num_threads(1)
    means = torch.randn(n, 23) * .02
    return means, torch.ones_like(means, dtype=torch.bool)


def test_progressive_first_update_uses_only_first_matured_horizon():
    means, valid = inputs()
    memory = ProgressiveDiagnosticMemory(64)
    trace = memory.trace(means, valid, mode='memory')
    assert trace['forecast'].shape == (12, 4, 23)
    assert not trace['delta'][0].any() and trace['delta'][1].any()
    assert trace['horizon_targets'] == [11, 10, 9, 8]
    assert trace['updates'] == 11
    assert trace['target_scalars'] == 23 * (11 + 10 + 9 + 8)
    assert trace['first_mask'][0].all() and not trace['first_mask'][1:].any()
    torch.testing.assert_close(trace['first_target'][0], means[1] - means[0])
    ends = endpoint_corrections(trace['delta'], torch.arange(80))
    assert not ends[:16].any() and ends[16].any()


def test_each_target_is_consumed_once_with_the_correct_origin_key():
    means, valid = inputs(8)
    memory = ProgressiveDiagnosticMemory(64).double()
    means = means.double()
    features = torch.cat((means, valid.to(means)), -1)
    keys = torch.cat((memory.key(features).tanh(), torch.ones(8, 1, dtype=means.dtype)), -1)
    w = memory.prior
    expected = []
    # Independent autograd reference: sum of four disjoint column losses.
    for j in range(8):
        if j:
            loss = means.sum() * 0
            for h in range(1, min(j, 4) + 1):
                key = keys[j-h]
                error = key @ w[:, (h-1)*23:h*23] - (means[j]-means[j-h])
                loss = loss + .5 * error.square().sum() / (1 + key.square().sum())
            grad = torch.autograd.grad(loss, w, create_graph=True)[0]
            candidate = w-memory.eta*grad
            delta = candidate-memory.prior
            w = memory.prior+delta/torch.linalg.vector_norm(delta).clamp_min(1.)
        expected.append(keys[j] @ (w-memory.prior))
    actual = memory.trace(means, valid, mode='memory')
    torch.testing.assert_close(actual['delta'], torch.stack(expected), atol=1e-12, rtol=1e-12)


def test_prefix_invariance_reset_and_finite_training_gradients():
    means, valid = inputs(18)
    memory = ProgressiveDiagnosticMemory(64)
    before = copy.deepcopy(memory.state_dict())
    main = memory.trace(means, valid, mode='memory')
    later = means.clone(); later[7:] += 10
    mask_later = valid.clone(); mask_later[9:] = False
    changed = memory.trace(later, mask_later, mode='memory')
    for key in ('delta', 'forecast', 'norm'):
        torch.testing.assert_close(main[key][:7], changed[key][:7], atol=0, rtol=0)
    repeated = memory.trace(means, valid, mode='memory')
    torch.testing.assert_close(main['delta'], repeated['delta'], atol=0, rtol=0)
    assert all(torch.equal(v, memory.state_dict()[k]) for k, v in before.items())
    main['delta'].square().mean().backward()
    for param in (memory.key.weight, memory.prior, memory.rate):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_missing_channels_off_and_norm_matched_controls():
    means, valid = inputs()
    memory = ProgressiveDiagnosticMemory(64)
    valid[:, 3] = False
    main = memory.trace(means, valid, mode='memory')
    assert not main['delta'].reshape(12, 4, 23)[:, :, 3].any()
    assert main['target_scalars'] == 22 * (11 + 10 + 9 + 8)
    bad = means.clone(); bad[:, 3] = 10000
    torch.testing.assert_close(main['delta'], memory.trace(bad, valid, mode='memory')['delta'], atol=0, rtol=0)
    assert not memory.trace(means, valid, mode='off')['delta'].any()
    empty = memory.trace(means, torch.zeros_like(valid), mode='memory')
    assert empty['updates'] == 0 and not empty['delta'].any()
    for mode in ('local', 'hold', 'reverse'):
        control = memory.trace(means, valid, mode=mode, reference_norm=main['norm'])
        ok = ~control['zero_direction']
        torch.testing.assert_close(control['norm'][ok], main['norm'][ok], atol=1e-6, rtol=1e-6)
    assert main['norm'].max() <= 1.000001
    with pytest.raises(ValueError): memory.trace(means, valid, mode='reverse')


def test_legacy_default_and_checkpoint_layout_unchanged():
    means, valid = inputs()
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(fast_site='tokens'), np.zeros(23), np.ones(23)).eval()
    torch.manual_seed(4); default = LifecycleAdapter(base, readout='full').eval()
    torch.manual_seed(4); declared = LifecycleAdapter(base, readout='full', feedback='delayed').eval()
    torch.manual_seed(4); progressive = LifecycleAdapter(base, readout='full', feedback='progressive').eval()
    assert isinstance(default.memory, DiagnosticMemory)
    for k, v in default.state_dict().items():
        assert torch.equal(v, declared.state_dict()[k])
        assert torch.equal(v, progressive.state_dict()[k])
    x = torch.randn(2, 128, 23); available = torch.ones_like(x, dtype=torch.bool)
    h, v = progressive.tokens(x, available)
    with torch.no_grad():
        torch.testing.assert_close(progressive.risk(h, v, torch.zeros(2, 92)),
                                   base(x, available)['logit'], atol=0, rtol=0)
    signal = dict(values=means.numpy().repeat(8, axis=0), available=np.ones((96, 23), bool))
    delta, info = progressive.memory_delta(signal)
    assert info['first_affected_index'] == 16 and info['feedback'] == 'progressive'
    assert info['horizon_targets'] == [11, 10, 9, 8]
    assert not delta[:16].any() and delta[16].any()
    with pytest.raises(ValueError): LifecycleAdapter(base, feedback='future')


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
def test_short_streams_never_require_unobserved_horizons(n):
    means, valid = inputs(n)
    memory = ProgressiveDiagnosticMemory(64)
    main = memory.trace(means, valid, mode='memory')
    assert main['forecast'].shape == (n, 4, 23)
    assert main['updates'] == max(n-1, 0)
    assert main['horizon_targets'] == [max(n-h, 0) for h in range(1, 5)]
    assert torch.isfinite(main['forecast_loss'])
    reverse = memory.trace(means, valid, mode='reverse', reference_norm=main['norm'])
    assert torch.isfinite(reverse['delta']).all()


def test_reverse_control_does_not_read_future_targets():
    means, valid = inputs(12)
    memory = ProgressiveDiagnosticMemory(64)
    main = memory.trace(means, valid, mode='memory')
    reverse = memory.trace(means, valid, mode='reverse', reference_norm=main['norm'])
    means[5:] += 100
    other = memory.trace(means, valid, mode='memory')
    reversed_other = memory.trace(means, valid, mode='reverse', reference_norm=other['norm'])
    torch.testing.assert_close(reverse['delta'][:5], reversed_other['delta'][:5], atol=0, rtol=0)
