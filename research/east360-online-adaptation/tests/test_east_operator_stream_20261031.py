import copy
import math

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.learning import batch
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import patch_signal
from experiments.risk_aligned_online.east_context_operator_20261031.stream import ARMS, predict
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture(scope='module', autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def case():
    torch.manual_seed(131)
    base = Simple23(Config(window=128, dim=16, fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    reference = ContextAdapter(base).eval()
    with torch.no_grad():
        reference.slow.weight.normal_(0, .03)
        reference.memory.output.weight.normal_(0, .3)
    learned = copy.deepcopy(reference).eval()
    with torch.no_grad():
        learned.memory.key.weight.add_(.13)
        learned.memory.key.bias.add_(.2)
        learned.memory.kernel.mul_(1.3)
        learned.memory.rate.add_(.4)
    learned.memory.key.weight.requires_grad_(True)
    rng = np.random.default_rng(131)
    values = rng.normal(size=(73, 23)).astype(np.float32)
    values += np.arange(len(values), dtype=np.float32)[:, None] * .025
    available = np.ones_like(values, dtype=bool)
    available[40:48, 3] = False
    return reference, learned, dict(values=values, available=available)


def test_six_finite_curves_and_per_shot_trace_receipts(case):
    reference, learned, signal = case
    curves, info = predict(reference, learned, signal)

    assert ARMS == ('cold', 'source_rule', 'learned_rule', 'reverse_feedback',
                    'negative_shift', 'zero_feedback')
    assert tuple(curves) == ARMS
    assert set(info) == set(ARMS)
    for arm in ARMS:
        assert isinstance(curves[arm], np.ndarray)
        assert curves[arm].shape == (len(signal['values']),)
        assert np.isfinite(curves[arm]).all()
        assert ((0 <= curves[arm]) & (curves[arm] <= 1)).all()
        receipt = info[arm]
        assert receipt['first_affected_index'] == 16  # earliest possible, not guaranteed
        assert receipt['reset_each_shot'] is True
        assert isinstance(receipt['updates'], int) and receipt['updates'] >= 0
        assert np.isfinite(receipt['norm']).all()
        assert math.isfinite(receipt['forecast_loss'])
        assert len(receipt['zero_direction']) == len(receipt['norm']) == 9
        assert all(isinstance(value, bool) for value in receipt['zero_direction'])
    np.testing.assert_array_equal(curves['cold'], curves['zero_feedback'])
    for arm in ARMS:
        np.testing.assert_array_equal(curves[arm][:16], curves['cold'][:16])
    for arm in ('source_rule', 'learned_rule', 'reverse_feedback', 'negative_shift'):
        assert np.any(curves[arm][16:] != curves['cold'][16:]), arm
    for arm in ('source_rule', 'reverse_feedback', 'negative_shift'):
        assert np.any(curves[arm] != curves['learned_rule']), arm
    assert info['learned_rule']['updates'] > 0
    assert info['zero_feedback']['updates'] == 0
    assert not any(info['zero_feedback']['norm'])
    assert info['negative_shift']['delta_sign'] == -1
    assert info['negative_shift']['trace_source'] == 'learned_rule'
    assert info['cold']['trace_source'] == 'zero_feedback'
    matched = ~np.asarray(info['reverse_feedback']['zero_direction'])
    np.testing.assert_allclose(np.asarray(info['reverse_feedback']['norm'])[matched],
                               np.asarray(info['learned_rule']['norm'])[matched], atol=1e-7)


def test_manual_endpoint_replay_uses_each_actual_internal_delta(case):
    reference, learned, signal = case
    curves, _ = predict(reference, learned, signal)
    with torch.no_grad():
        means, complete = patch_signal(signal, reference.base.center, reference.base.scale)
        source = reference.memory.trace(means, complete, mode='memory')
        main = learned.memory.trace(means, complete, mode='memory')
        reverse = learned.memory.trace(means, complete, mode='reverse',
                                       reference_norm=main['norm'])
        off = learned.memory.trace(means, complete, mode='off')
        assert torch.count_nonzero(off['delta']) == 0
        assert torch.count_nonzero(main['delta'][1]) > 0
        assert not torch.equal(source['delta'], main['delta'])
        expected = {arm: [] for arm in ARMS}
        for end in range(len(signal['values'])):
            x, mask = batch(signal, [end], reference.base.center.device)
            h, valid = reference.tokens(x, mask)
            # Hand-index the last permitted complete patch. In particular,
            # patch 1 can affect index 16 but not index 15.
            index = end // 8 - 1
            def correction(trace):
                return trace['delta'][index:index + 1] if index >= 0 else h.new_zeros(1, 92)
            deltas = {
                'cold': h.new_zeros(1, 92),
                'source_rule': correction(source),
                'learned_rule': correction(main),
                'reverse_feedback': correction(reverse),
                'negative_shift': -correction(main),
                'zero_feedback': correction(off),
            }
            for arm, delta in deltas.items():
                model = reference if arm in ('cold', 'source_rule') else learned
                expected[arm].append(model.risk(h, valid, delta).sigmoid().item())
    for arm in ARMS:
        # Single-window and batch-of-32 kernels can round differently.
        np.testing.assert_allclose(curves[arm], expected[arm], rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize('cutoff', [7, 15, 16, 31, 45, 63])
def test_future_values_and_availability_cannot_change_prefix(case, cutoff):
    reference, learned, signal = case
    changed = {key: value.copy() for key, value in signal.items()}
    changed['values'][cutoff + 1:] = -1000
    changed['available'][cutoff + 1:] = False
    original, _ = predict(reference, learned, signal)
    altered, _ = predict(reference, learned, changed)
    for arm in ARMS:
        np.testing.assert_array_equal(original[arm][:cutoff + 1], altered[arm][:cutoff + 1])


def test_no_grad_no_model_mutation_reuse_tokens_and_only_declared_traces(case, monkeypatch):
    reference, learned, signal = case
    models = (reference, learned)
    learned.memory.key.weight.grad = torch.full_like(learned.memory.key.weight, .125)
    before = [{name: value.clone() for name, value in model.state_dict().items()}
              for model in models]
    gradients = [[None if p.grad is None else p.grad.clone() for p in model.parameters()]
                 for model in models]
    flags = [[(module.training, tuple(p.requires_grad for p in module.parameters(recurse=False)))
              for module in model.modules()] for model in models]
    signal_before = {key: value.copy() for key, value in signal.items()}
    calls, token_sizes, risk_sizes, observed = [], [], [], {}
    for label, model in zip(('reference', 'learned'), models):
        trace = model.memory.trace
        def traced(*args, _trace=trace, _label=label, **kwargs):
            assert not torch.is_grad_enabled()
            calls.append((_label, kwargs['mode'], kwargs.get('reference_norm')))
            result = _trace(*args, **kwargs)
            observed[(_label, kwargs['mode'])] = result
            return result
        monkeypatch.setattr(model.memory, 'trace', traced)
        tokens = model.tokens
        def tokenized(x, mask, _tokens=tokens, _label=label):
            assert not torch.is_grad_enabled()
            token_sizes.append((_label, len(x)))
            return _tokens(x, mask)
        monkeypatch.setattr(model, 'tokens', tokenized)
        risk = model.risk
        def risked(h, valid, delta, _risk=risk, _label=label):
            assert not torch.is_grad_enabled()
            risk_sizes.append((_label, len(h)))
            return _risk(h, valid, delta)
        monkeypatch.setattr(model, 'risk', risked)

    with torch.enable_grad():
        first, _ = predict(reference, learned, signal)
    assert torch.is_grad_enabled()
    assert [(label, mode) for label, mode, _ in calls] == [
        ('reference', 'memory'), ('learned', 'memory'),
        ('learned', 'reverse'), ('learned', 'off')]
    assert calls[2][2] is observed[('learned', 'memory')]['norm']
    assert token_sizes == [('reference', 32), ('reference', 32), ('reference', 9)]
    assert len(risk_sizes) == 18 and max(size for _, size in risk_sizes) == 32
    for model, original, original_flags, original_gradients in zip(models, before, flags, gradients):
        for name, value in model.state_dict().items():
            assert torch.equal(value, original[name]), name
        assert [(module.training, tuple(p.requires_grad for p in module.parameters(recurse=False)))
                for module in model.modules()] == original_flags
        for parameter, gradient in zip(model.parameters(), original_gradients):
            assert (parameter.grad is None if gradient is None
                    else torch.equal(parameter.grad, gradient))
    for key in signal:
        np.testing.assert_array_equal(signal[key], signal_before[key])
    other = {key: value[::-1].copy() for key, value in signal.items()}
    predict(reference, learned, other)
    second, _ = predict(reference, learned, signal)
    for arm in ARMS:
        np.testing.assert_array_equal(first[arm], second[arm])


@pytest.mark.parametrize('length', [0, 1, 7, 8, 15, 16])
def test_short_or_empty_shot_has_finite_cold_baselines(case, length):
    reference, learned, signal = case
    short = {key: value[:length] for key, value in signal.items()}
    curves, info = predict(reference, learned, short)
    for arm in ARMS:
        assert curves[arm].shape == (length,)
        assert np.isfinite(curves[arm]).all()
        # At length 16 the second patch has arrived, so a write is recorded,
        # but it cannot affect any prediction until the next endpoint (16).
        assert info[arm]['updates'] == (1 if length == 16 and arm not in
                                        ('cold', 'zero_feedback') else 0)
        assert math.isfinite(info[arm]['forecast_loss'])
        np.testing.assert_array_equal(curves[arm], curves['cold'])


def test_nonfinite_unavailable_samples_are_treated_as_missing(case):
    reference, learned, signal = case
    missing = {key: value.copy() for key, value in signal.items()}
    missing['available'][3:6, 0] = False
    missing['values'][3:6, 0] = [np.nan, np.inf, -np.inf]
    clean = {key: value.copy() for key, value in missing.items()}
    clean['values'][3:6, 0] = 0
    actual, _ = predict(reference, learned, missing)
    expected, _ = predict(reference, learned, clean)
    for arm in ARMS:
        assert np.isfinite(actual[arm]).all()
        np.testing.assert_array_equal(actual[arm], expected[arm])


def test_all_missing_shot_never_writes(case):
    reference, learned, signal = case
    missing = dict(signal, available=np.zeros_like(signal['available']))
    curves, info = predict(reference, learned, missing)
    for arm in ARMS:
        assert info[arm]['updates'] == 0
        assert np.isfinite(curves[arm]).all()
        np.testing.assert_array_equal(curves[arm], curves['cold'])


@pytest.mark.parametrize('bad', [
    lambda s: dict(s, labels=np.zeros(len(s['values']))),
    lambda s: dict(values=s['values']),
    lambda s: dict(s, values=s['values'][:, :22]),
    lambda s: dict(s, available=s['available'].astype(np.float32)),
    lambda s: dict(s, available=s['available'][:-1]),
])
def test_invalid_signal_rejected(case, bad):
    reference, learned, signal = case
    with pytest.raises((ValueError, TypeError)):
        predict(reference, learned, bad(signal))


@pytest.mark.parametrize('part', ['base', 'slow', 'output', 'readout', 'window'])
def test_incompatible_frozen_parts_or_configuration_are_rejected(case, part):
    reference, learned, signal = case
    with torch.no_grad():
        if part == 'base':
            learned.base.risk_head.bias.add_(1)
        elif part == 'slow':
            learned.slow.weight.add_(1)
        elif part == 'output':
            learned.memory.output.weight.add_(1)
        elif part == 'readout':
            learned.readout = 'shape'
        else:
            reference.cfg = Config(window=16, dim=16, fast_site='tokens')
    with pytest.raises(ValueError):
        predict(reference, learned, signal)


@pytest.mark.parametrize('field', ['delta', 'norm', 'forecast_loss'])
def test_nonfinite_trace_is_rejected(case, monkeypatch, field):
    reference, learned, signal = case
    original = learned.memory.trace
    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs['mode'] == 'memory':
            result[field] = torch.full_like(result[field], float('nan'))
        return result
    monkeypatch.setattr(learned.memory, 'trace', corrupted)
    with pytest.raises((ValueError, FloatingPointError)):
        predict(reference, learned, signal)


def test_nonzero_off_trace_is_rejected(case, monkeypatch):
    reference, learned, signal = case
    original = learned.memory.trace
    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs['mode'] == 'off':
            result['delta'] = result['delta'] + 1
        return result
    monkeypatch.setattr(learned.memory, 'trace', corrupted)
    with pytest.raises((ValueError, AssertionError)):
        predict(reference, learned, signal)


def test_nonfinite_risk_is_rejected(case, monkeypatch):
    reference, learned, signal = case
    monkeypatch.setattr(reference, 'risk',
                        lambda h, valid, delta: h.new_full((len(h),), float('inf')))
    with pytest.raises(FloatingPointError, match='finite risk'):
        predict(reference, learned, signal)


def test_training_mode_is_rejected_without_changing_it(case):
    reference, learned, signal = case
    learned.train()
    with pytest.raises(ValueError, match='evaluation'):
        predict(reference, learned, signal)
    assert learned.training
