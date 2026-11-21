import inspect

import torch
from torch.nn import functional as F

torch.set_num_threads(2)

from experiments.risk_aligned_online.ttt_state import (  # noqa: E402
    TTTConfig,
    TTTRiskState,
)


def make_model(chunk_size=16):
    torch.manual_seed(231)
    return TTTRiskState(
        TTTConfig(dim=32, state_dim=16, chunk_size=chunk_size),
        torch.zeros(23),
        torch.ones(23),
    )


def make_inputs(batch=2, length=32):
    torch.manual_seed(232)
    values = torch.randn(batch, length, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    return values, available


def test_contract_and_adapt_false_control_have_same_structure():
    model = make_model()
    values, available = make_inputs()
    adapted = model(values, available, adapt=True)
    static = model(values, available, adapt=False)

    assert model.cfg.dim == 32 and model.cfg.state_dim == 16
    assert model.cfg.chunk_size == 16
    assert model.W0.shape == (16, 16)
    assert adapted.keys() == static.keys() == {'logits', 'state'}
    assert adapted['logits'].shape == (2, 32)
    assert adapted['state'].W.shape == (2, 16, 16)
    assert static['state'].W.shape == adapted['state'].W.shape
    torch.testing.assert_close(static['state'].W, model.initial_state(2).W)
    assert not hasattr(model, 'forecast_head')
    assert not any('film' in name.lower() for name, _ in model.named_parameters())


def test_module_docstring_marks_icml_ttt_as_inspiration_not_tokamak_evidence():
    from experiments.risk_aligned_online import ttt_state

    doc = ttt_state.__doc__ or ''
    assert 'ICML 2025' in doc
    assert 'tokamak' in doc.lower()
    assert 'not' in doc.lower()


def test_future_input_cannot_change_earlier_predictions():
    model = make_model().eval()
    values, available = make_inputs(batch=1)
    changed = values.clone()
    changed[:, 5:] += 9.

    before = model(values, available)['logits']
    after = model(changed, available)['logits']
    torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)


def test_inner_update_is_unlabeled_and_does_not_mutate_parameters():
    model = make_model()
    values, available = make_inputs()
    available[:, 3, 4] = False
    values[:, 3, 4] = float('nan')
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    signature = inspect.signature(model.forward)
    assert 'labels' not in signature.parameters
    assert 'outcome' not in signature.parameters

    output = model(values, available, adapt=True)

    assert torch.isfinite(output['state'].W).all()
    assert not torch.equal(output['state'].W, model.initial_state(2).W)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_each_call_resets_to_w0_unless_explicit_state_is_supplied():
    model = make_model()
    values, available = make_inputs(batch=1)

    first = model(values, available, adapt=True)
    second = model(values, available, adapt=True)
    torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
    torch.testing.assert_close(first['state'].W, second['state'].W, rtol=0, atol=0)

    continued = model(values[:, :4], available[:, :4],
                      state=first['state'], adapt=True)
    assert not torch.equal(continued['state'].W, model.initial_state(1).W)


def test_outer_unroll_keeps_gradients_to_qkv_eta_w0_and_risk_readout():
    model = make_model()
    values, available = make_inputs(length=32)
    labels = torch.randint(0, 2, (2, 32), dtype=torch.float32)
    output = model(values, available, adapt=True)
    loss = .5 * F.binary_cross_entropy_with_logits(output['logits'], labels)
    loss.backward()

    for name in ('key.weight', 'value.weight', 'query.weight',
                 'eta', 'W0', 'risk_readout.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_arbitrary_stream_splits_match_full_unroll_and_do_not_update_tail():
    model = make_model(chunk_size=16).eval()
    values, available = make_inputs(batch=2, length=37)
    full = model(values, available, adapt=True)

    state = None
    logits = []
    boundaries = (3, 11, 18, 29, 37)
    start = 0
    for end in boundaries:
        output = model(values[:, start:end], available[:, start:end],
                       state=state, adapt=True)
        logits.append(output['logits'])
        state = output['state']
        start = end
    streamed = torch.cat(logits, dim=1)

    torch.testing.assert_close(streamed, full['logits'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state.W, full['state'].W, rtol=1e-6, atol=1e-6)
    assert state.pending_k.shape[1] == 5
    assert state.pending_v.shape[1] == 5
    assert state.pending_observed.shape[1] == 5


def test_long_sequence_stays_finite_with_missing_values():
    model = make_model(chunk_size=8).eval()
    values, available = make_inputs(batch=2, length=512)
    available[:, ::7, 2] = False
    values[:, ::7, 2] = float('nan')

    output = model(values, available, adapt=True)

    assert torch.isfinite(output['logits']).all()
    assert torch.isfinite(output['state'].W).all()
