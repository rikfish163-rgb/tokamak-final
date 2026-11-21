"""CPU-only credit-window contracts for offline source replay."""
import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from .credit_source import credit_causal_logit
from .model import ClsModel
from .source import causal_logit, enable_decoder_only


def fixture(*, missing=False):
    old, shot, _ = case()
    model = ClsModel(old.base).double()
    enable_decoder_only(model)
    if missing:
        # Eight legal slots at q=104, but only six executed events. Include
        # both finite-but-masked targets and nonfinite "available" targets.
        shot['values'][27:32] = np.nan
        shot['available'][84:104] = False
    return model, shot


@pytest.mark.parametrize('use_task', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_default_four_exactly_matches_old_fp64_forward_and_gradient(use_task, missing):
    model, shot = fixture(missing=missing)
    old, old_row = causal_logit(model, shot, 104, use_task=use_task, create_graph=True)
    new, row = credit_causal_logit(model, shot, 104, use_task=use_task, create_graph=True)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    assert row == old_row
    old_grad = torch.autograd.grad(old, tuple(model.decoder.parameters()))
    new_grad = torch.autograd.grad(new, tuple(model.decoder.parameters()))
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert row['executed_updates'] == (6 if missing else 8)
    assert row['differentiated_updates'] == 4
    assert row['burn_in_updates'] == (2 if missing else 4)


@pytest.mark.parametrize('use_task', [False, True])
def test_full_credit_same_forward_and_full_finite_difference(use_task):
    model, shot = fixture(missing=True)
    full, row = credit_causal_logit(model, shot, 104, use_task=use_task,
                                  credit_steps=None, create_graph=True)
    four, _ = credit_causal_logit(model, shot, 104, use_task=use_task,
                                 credit_steps=4, create_graph=True)
    torch.testing.assert_close(full, four, rtol=0, atol=0)
    assert row['executed_updates'] == row['differentiated_updates'] == 6
    assert row['missing_target_events'] == 2
    assert row['burn_in_updates'] == 0
    gradient, = torch.autograd.grad(full, model.decoder.weight)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    index = np.unravel_index(int(gradient.abs().argmax()), gradient.shape)
    original = model.decoder.weight[index].detach().clone()
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                model.decoder.weight[index] = original + sign * 1e-5
            value, _ = credit_causal_logit(model, shot, 104, use_task=use_task,
                                          credit_steps=None, create_graph=False)
            values.append(float(value.detach()))
    finally:
        with torch.no_grad():
            model.decoder.weight[index] = original
    assert float(gradient[index]) == pytest.approx((values[0] - values[1]) / 2e-5,
                                                  rel=.004, abs=2e-7)


@pytest.mark.parametrize('credit_steps', [1, 4, None])
def test_credit_changes_only_backward_window(credit_steps):
    model, shot = fixture(missing=True)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    reference, _ = causal_logit(model, shot, 104, use_task=False)
    value, row = credit_causal_logit(model, shot, 104, use_task=False,
                                    credit_steps=credit_steps, create_graph=True)
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    credited = 6 if credit_steps is None else credit_steps
    assert row['burn_in_updates'] == 6 - credited
    assert row['differentiated_updates'] == credited
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@pytest.mark.parametrize('credit_steps', [0, -1, True, False, 1.5, '4'])
def test_invalid_credit_rejected(credit_steps):
    model, shot = fixture()
    with pytest.raises(ValueError, match='credit_steps'):
        credit_causal_logit(model, shot, 104, use_task=False, credit_steps=credit_steps)


@pytest.mark.parametrize('query', [0, 5, 25, 26])
def test_legal_arrival_boundary_and_zero_events(query):
    model, shot = fixture()
    value, row = credit_causal_logit(model, shot, query, use_task=True,
                                    credit_steps=None, create_graph=True)
    old, old_row = causal_logit(model, shot, query, use_task=True, create_graph=True)
    torch.testing.assert_close(value, old, rtol=0, atol=0)
    assert row == old_row
    assert row['executed_updates'] == (0 if query < 6 else 1 if query < 26 else 2)


def test_zero_gradient_events_are_executed_not_discarded():
    model, shot = fixture(missing=True)
    with torch.no_grad():
        model.decoder.weight.zero_()
    value, row = credit_causal_logit(model, shot, 104, use_task=False,
                                    credit_steps=1, create_graph=True)
    assert torch.isfinite(value)
    assert row['delta_norm'] == 0
    assert row['executed_updates'] == 6
    assert row['burn_in_updates'] == 5
    assert row['differentiated_updates'] == 1


@pytest.mark.parametrize('credit_steps', [1, 4, None])
def test_no_graph_mode_detaches_every_executed_update(credit_steps):
    model, shot = fixture(missing=True)
    graph_inputs = []
    # Record only booleans: the observer itself must not retain any graphs.
    hook = model.register_forward_pre_hook(
        lambda module, args: graph_inputs.append(args[2].grad_fn is not None))
    try:
        value, row = credit_causal_logit(model, shot, 104, use_task=False,
                                        credit_steps=credit_steps, create_graph=False)
    finally:
        hook.remove()
    assert graph_inputs == [False] * 7  # six updates and the endpoint readout
    assert row['differentiated_updates'] == 0
    assert torch.isfinite(value)


@pytest.mark.parametrize('query', [-1, 160, True, 1.0])
def test_invalid_source_endpoint_rejected(query):
    model, shot = fixture()
    with pytest.raises(ValueError, match='offline source endpoint'):
        credit_causal_logit(model, shot, query, use_task=False)


def test_target_device_rejected():
    model, shot = fixture()
    shot['device'] = 'EAST'
    with pytest.raises(ValueError, match='offline source endpoint'):
        credit_causal_logit(model, shot, 104, use_task=False)


@pytest.mark.parametrize('credit_steps', [1, 99, None])
def test_all_missing_targets_do_not_consume_credit(credit_steps):
    model, shot = fixture()
    shot['available'][1:105] = False
    value, row = credit_causal_logit(model, shot, 104, use_task=True,
                                    credit_steps=credit_steps, create_graph=True)
    assert torch.isfinite(value)
    assert row['executed_updates'] == row['differentiated_updates'] == 0
    assert row['burn_in_updates'] == 0
    assert row['missing_target_events'] == 8


def test_future_signals_and_source_outcome_do_not_affect_replay():
    model, shot = fixture()
    before, before_row = credit_causal_logit(model, shot, 104, use_task=True,
                                            credit_steps=None, create_graph=True)
    before_grad, = torch.autograd.grad(before, model.decoder.weight)
    shot['values'][105:] = 1e6
    shot['available'][105:] = False
    shot['is_disrupt'] = False
    shot['down'] = 0.
    after, after_row = credit_causal_logit(model, shot, 104, use_task=True,
                                          credit_steps=None, create_graph=True)
    after_grad, = torch.autograd.grad(after, model.decoder.weight)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(after_grad, before_grad, rtol=0, atol=0)
    assert after_row == before_row
