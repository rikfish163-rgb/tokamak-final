"""CPU contracts for end-of-shot diagnostic decoder teaching."""

from copy import deepcopy
import math

import h5py
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_risk_meta_sequence_20261108 import episode_sequence
from test_risk_meta_step_20261107 import teacher_case


@pytest.fixture(autouse=True)
def forbid_hdf(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("EOS teaching must not open HDF files")

    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(h5py.h5f, "open", forbidden)


@pytest.fixture
def case():
    model, _ = teacher_case()
    slow, decoder = model.risk, model.decoder
    decoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=0., foreach=False)
    episodes, query, query_available = episode_sequence()
    return slow, decoder, optimizer, episodes, query, query_available


def teach(slow, decoder, optimizer, episodes, query, query_available, labels, *, credit):
    from risk_eos_teacher_20261108 import teach_decoder

    with sdpa_kernel(SDPBackend.MATH):
        return teach_decoder(slow, decoder, optimizer, episodes, query, query_available,
                             labels, credit=credit)


def call(case, labels, credit):
    return teach(*case, labels, credit=credit)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_state_equal(module, expected):
    assert module.state_dict().keys() == expected.keys()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def assert_frozen_clean(decoder):
    assert all(not p.requires_grad and p.grad is None for p in decoder.parameters())


def test_updates_persistent_decoder_without_changing_or_promoting_slow(case, monkeypatch):
    import risk_eos_teacher_20261108 as eos

    slow, decoder, optimizer, episodes, query, query_available = case
    with torch.no_grad():
        slow.encoder.layers[0].linear2.weight.add_(0.037)
    before_slow, before_decoder = snapshot(slow), snapshot(decoder)
    decoder.train()
    original_sequence = eos.teacher_sequence
    original_clip = torch.nn.utils.clip_grad_norm_
    observed = {}

    def traced_sequence(model, *args, **kwargs):
        assert not model.training and not model.risk.training and not model.decoder.training
        assert model.risk is not slow and model.decoder is decoder
        assert model.weight is not slow.encoder.layers[0].linear2.weight
        assert model.weight.requires_grad
        assert sum(p.requires_grad for p in model.risk.parameters()) == 1
        assert all(p.requires_grad for p in decoder.parameters())
        torch.testing.assert_close(model.initial_weight, slow.encoder.layers[0].linear2.weight,
                                   rtol=0, atol=0)
        result = original_sequence(model, *args, **kwargs)
        observed['signal_loss'] = float(result['signal_loss'].detach())
        observed['risk_loss'] = float(result['risk_loss'].detach())
        observed['logit'] = float(result['logit'].detach().mean())
        observed['fast'] = result['steps'][-1]['weight'].detach().clone()
        return result

    def traced_clip(parameters, max_norm, **kwargs):
        assert max_norm == 1. and kwargs['error_if_nonfinite'] is True
        norm = original_clip(parameters, max_norm, **kwargs)
        observed['gradient_norm'] = float(norm)
        return norm

    monkeypatch.setattr(eos, 'teacher_sequence', traced_sequence)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', traced_clip)
    info = teach(slow, decoder, optimizer, episodes, query, query_available,
                 torch.ones(1), credit='last')

    assert any(not torch.equal(value, before_decoder[name])
               for name, value in decoder.state_dict().items())
    assert_state_equal(slow, before_slow)
    assert not torch.equal(observed['fast'], slow.encoder.layers[0].linear2.weight)
    assert not decoder.training
    assert_frozen_clean(decoder)
    assert set(info) >= {'loss', 'signal_loss', 'risk_loss', 'gradient_norm', 'logit'}
    assert all(isinstance(info[key], float) and math.isfinite(info[key])
               for key in ('loss', 'signal_loss', 'risk_loss', 'gradient_norm', 'logit'))
    assert info['gradient_norm'] > 0
    for key in ('signal_loss', 'risk_loss', 'gradient_norm', 'logit'):
        assert info[key] == observed[key]
    assert info['loss'] == pytest.approx(info['signal_loss'] + info['risk_loss'], abs=1e-6)


def test_control_and_guided_have_identical_forward_losses_but_different_updates(case):
    slow, decoder, _, episodes, query, query_available = case
    control, guided = deepcopy(decoder), deepcopy(decoder)
    opt_control = torch.optim.AdamW(control.parameters(), lr=3e-4, weight_decay=0.)
    opt_guided = torch.optim.AdamW(guided.parameters(), lr=3e-4, weight_decay=0.)
    args = (episodes, query, query_available, torch.ones(1))
    a = teach(slow, control, opt_control, *args, credit='none')
    b = teach(slow, guided, opt_guided, *args, credit='last')
    for key in ('loss', 'signal_loss', 'risk_loss', 'logit'):
        assert a[key] == b[key]
    assert any(not torch.equal(x, y) for x, y in zip(control.parameters(), guided.parameters(), strict=True))
    assert_frozen_clean(control)
    assert_frozen_clean(guided)


def test_labels_do_not_change_control_update_but_do_change_guided_update(case):
    slow, decoder, _, episodes, query, query_available = case
    results = {}
    for credit in ('none', 'last'):
        for label in (0., 1.):
            current = deepcopy(decoder)
            optimizer = torch.optim.AdamW(current.parameters(), lr=3e-4, weight_decay=0.)
            info = teach(slow, current, optimizer, episodes, query, query_available,
                         torch.tensor([label]), credit=credit)
            results[credit, label] = (snapshot(current), info)
    for credit in ('none', 'last'):
        a, b = results[credit, 0.], results[credit, 1.]
        assert a[1]['signal_loss'] == b[1]['signal_loss']
        assert a[1]['logit'] == b[1]['logit']
        assert a[1]['risk_loss'] != b[1]['risk_loss']
        changes = [not torch.equal(a[0][name], b[0][name]) for name in a[0]]
        assert any(changes) if credit == 'last' else not any(changes)


def test_optimizer_state_accumulates_across_two_eos_calls(case):
    _, decoder, optimizer, _, _, _ = case
    call(case, torch.ones(1), 'last')
    assert optimizer.state
    first = {p: (state, state['step'].item(), state['exp_avg'].detach().clone())
             for p, state in optimizer.state.items()}
    assert all(step == 1 for _, step, _ in first.values())
    assert any(avg.count_nonzero() for _, _, avg in first.values())
    call(case, torch.ones(1), 'last')
    for p, (state, _, avg) in first.items():
        assert optimizer.state[p] is state
        assert state['step'].item() == 2
        assert not torch.equal(state['exp_avg'], avg)
    assert_frozen_clean(decoder)


def test_batched_query_reports_the_mean_logit_as_a_scalar(case, monkeypatch):
    import risk_eos_teacher_20261108 as eos

    slow, decoder, optimizer, episodes, query, query_available = case
    original_sequence = eos.teacher_sequence
    observed = {}

    def traced_sequence(*args, **kwargs):
        result = original_sequence(*args, **kwargs)
        observed['logit'] = float(result['logit'].detach().mean())
        return result

    monkeypatch.setattr(eos, 'teacher_sequence', traced_sequence)
    info = teach(slow, decoder, optimizer, episodes, query.repeat(2, 1, 1),
                 query_available.repeat(2, 1, 1), torch.tensor([0., 1.]), credit='last')
    assert isinstance(info['logit'], float) and math.isfinite(info['logit'])
    assert info['logit'] == observed['logit']


def test_eos_temporarily_enables_grad_inside_caller_no_grad(case):
    _, decoder, _, _, _, _ = case
    before = snapshot(decoder)
    with torch.no_grad():
        info = call(case, torch.ones(1), 'last')
        assert not torch.is_grad_enabled()
    assert info['gradient_norm'] > 0
    assert any(not torch.equal(value, before[name]) for name, value in decoder.state_dict().items())
    assert_frozen_clean(decoder)


@pytest.mark.parametrize('mismatch', ['missing', 'extra'])
def test_optimizer_parameter_mismatch_is_rejected_without_mutation(case, mismatch):
    slow, decoder, _, episodes, query, query_available = case
    params = list(decoder.parameters())
    if mismatch == 'missing':
        params = params[:-1]
    else:
        params.append(slow.encoder.layers[0].linear2.weight)
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.)
    before_slow, before_decoder = snapshot(slow), snapshot(decoder)
    next(decoder.parameters()).grad = torch.ones_like(next(decoder.parameters()))
    with pytest.raises(ValueError, match='optimizer'):
        teach(slow, decoder, optimizer, episodes, query, query_available,
              torch.ones(1), credit='last')
    assert_state_equal(slow, before_slow)
    assert_state_equal(decoder, before_decoder)
    assert not optimizer.state
    assert_frozen_clean(decoder)


def test_invalid_credit_rejected_and_decoder_cleaned(case):
    _, decoder, optimizer, _, _, _ = case
    decoder.requires_grad_(True)
    next(decoder.parameters()).grad = torch.ones_like(next(decoder.parameters()))
    with pytest.raises(ValueError, match='credit'):
        call(case, torch.ones(1), 'full')
    assert not optimizer.state
    assert_frozen_clean(decoder)


def test_exception_during_sequence_freezes_decoder_and_clears_grad(case, monkeypatch):
    import risk_eos_teacher_20261108 as eos

    slow, decoder, optimizer, _, _, _ = case
    before_slow, before_decoder = snapshot(slow), snapshot(decoder)

    def fail(model, *args, **kwargs):
        assert model.decoder is decoder and all(p.requires_grad for p in decoder.parameters())
        p = next(decoder.parameters())
        p.grad = torch.ones_like(p)
        raise RuntimeError('sequence failed')

    monkeypatch.setattr(eos, 'teacher_sequence', fail)
    with pytest.raises(RuntimeError, match='sequence failed'):
        call(case, torch.ones(1), 'last')
    assert_state_equal(slow, before_slow)
    assert_state_equal(decoder, before_decoder)
    assert not optimizer.state
    assert_frozen_clean(decoder)


def test_nonfinite_decoder_gradient_fails_before_optimizer_step(case):
    slow, decoder, optimizer, _, _, _ = case
    before_slow, before_decoder = snapshot(slow), snapshot(decoder)
    with pytest.raises(RuntimeError, match='non-finite'):
        call(case, torch.tensor([float('nan')]), 'last')
    assert_state_equal(slow, before_slow)
    assert_state_equal(decoder, before_decoder)
    assert not optimizer.state
    assert_frozen_clean(decoder)
