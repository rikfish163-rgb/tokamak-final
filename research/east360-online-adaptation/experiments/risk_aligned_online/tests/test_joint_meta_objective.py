from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.fast_meta import temporary_step_components
from experiments.risk_aligned_online.joint_meta import alignment_loss, train_step
from experiments.risk_aligned_online.tests.test_joint_meta import setup


def _fixture():
    model, episode = setup()
    # Double precision makes the small difference between two full gradients
    # measurable without changing the real temporary-update implementation.
    model.double()
    episode = {
        key: value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in episode.items()
    }
    return model, episode


def _reference_risks_and_gradient(model, episode):
    reference = deepcopy(model)
    reference.eval()
    reference.fast_origin.requires_grad_(False)
    updated, signal, fast, signal_gradient = temporary_step_components(
        reference, episode, create_graph=True
    )
    losses = F.binary_cross_entropy_with_logits(
        reference(episode['qx'], episode['qa'], fast)['logit'], episode['y'], reduction='none'
    )
    before = losses.mean()
    after = F.binary_cross_entropy_with_logits(
        reference(episode['qx'], episode['qa'], updated)['logit'], episode['y']
    )
    risk_gradient, = torch.autograd.grad(losses.sum(), fast, retain_graph=True)
    alignment = alignment_loss(signal_gradient, risk_gradient)
    components = {
        'risk_before': before.item(),
        'risk_after': after.item(),
        'signal_loss': signal.item(),
        'alignment_loss': alignment.item(),
    }
    parameters = list(reference.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    grads = iter(torch.autograd.grad(after - before, trainable, allow_unused=True))
    flattened = []
    for parameter in parameters:
        gradient = next(grads) if parameter.requires_grad else None
        flattened.append((gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1))
    return components, torch.cat(flattened).detach()


def test_aligned_weight_changes_real_backward_and_logged_objective(monkeypatch):
    model, episode = _fixture()
    components, risk_difference_gradient = _reference_risks_and_gradient(model, episode)
    expected_difference = .5 * risk_difference_gradient
    assert expected_difference.norm().item() > 1e-9

    backwards = []
    unclipped = []
    original_backward = torch.Tensor.backward
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_backward(tensor, *args, **kwargs):
        backwards.append(tensor.detach().item())
        return original_backward(tensor, *args, **kwargs)

    def capture_before_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        unclipped.append(torch.cat([
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in parameters
        ]))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'backward', capture_backward)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture_before_clip)

    for weight in (0., .5):
        paired_model = deepcopy(model)
        optimizer = torch.optim.SGD(paired_model.parameters(), lr=0.)
        log = train_step(
            paired_model, optimizer, episode, mode='aligned', signal_weight=.2,
            risk_after_weight=weight,
        )
        assert log['risk_after_weight'] == weight
        for key, value in components.items():
            assert log[key] == pytest.approx(value, rel=1e-12, abs=1e-12)
        expected_objective = (
            (1. - weight) * components['risk_before'] + weight * components['risk_after']
            + .2 * components['signal_loss'] + .1 * components['alignment_loss']
        )
        assert log['objective'] == pytest.approx(expected_objective, rel=1e-12, abs=1e-12)
        assert log['objective'] == backwards[-1]

    assert len(backwards) == len(unclipped) == 2
    actual_difference = unclipped[1] - unclipped[0]
    relative_error = (actual_difference - expected_difference).norm() / expected_difference.norm()
    assert relative_error.item() < 1e-4


@pytest.mark.parametrize(('mode', 'expected_weight'), [('meta', .5), ('joint', 0.), ('aligned', .5)])
def test_default_weight_and_frozen_source_state(mode, expected_weight):
    model, episode = _fixture()
    initial = deepcopy(model.state_dict())
    options = [{}, {'risk_after_weight': None}]
    if mode == 'aligned':
        options.extend([{'risk_after_weight': .5}, {'risk_after_weight': 0.}])

    results = []
    for option in options:
        paired_model = deepcopy(model)
        optimizer = torch.optim.SGD(paired_model.parameters(), lr=1e-3)
        log = train_step(paired_model, optimizer, episode, mode=mode, **option)
        state = paired_model.state_dict()
        for key in ('fast_origin', 'center', 'scale'):
            assert torch.equal(state[key], initial[key])
        assert paired_model.fast_origin.requires_grad == model.fast_origin.requires_grad
        assert any(not torch.equal(state[key], initial[key]) for key in state)
        results.append((log, state))

    default_log, default_state = results[0]
    assert default_log['risk_after_weight'] == expected_weight
    # Omission, explicit None, and (for aligned) explicit 0.5 are equivalent.
    equivalent = results[1:3] if mode == 'aligned' else results[1:]
    for log, state in equivalent:
        assert log == default_log
        for key in state:
            torch.testing.assert_close(state[key], default_state[key], rtol=0, atol=0)
    if mode == 'aligned':
        zero_log, _ = results[-1]
        assert zero_log['risk_after_weight'] == 0.
        for key in ('risk_before', 'risk_after', 'signal_loss', 'alignment_loss'):
            assert zero_log[key] == default_log[key]


@pytest.mark.parametrize('weight', [float('nan'), -1e-6, 1.000001])
def test_invalid_after_weights_are_rejected(weight):
    model, episode = _fixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    with pytest.raises(ValueError):
        train_step(model, optimizer, episode, mode='aligned', risk_after_weight=weight)
