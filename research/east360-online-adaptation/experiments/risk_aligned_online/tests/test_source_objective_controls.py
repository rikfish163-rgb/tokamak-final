from copy import deepcopy

import pytest
import torch

from experiments.risk_aligned_online.aligned_encoder import encoder_parameters, source_step
from experiments.risk_aligned_online.tests.test_cross_channel_stable import model


@pytest.fixture(autouse=True)
def thread_limit():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def candidate(teacher):
    student = deepcopy(teacher)
    params = list(encoder_parameters(student).values()) + list(student.decoder.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    return student, torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)


@pytest.mark.parametrize('objective,supervised,signal', [
    ('joint', 1., .2), ('no_signal', 1., 0.), ('no_supervised', 0., .2),
])
def test_loss_controls_remove_only_declared_term(objective, supervised, signal):
    teacher = model()
    student, optimizer = candidate(teacher)
    old = {k: v.clone() for k, v in teacher.state_dict().items()}
    x = torch.randn(3, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    row = source_step(student, teacher, optimizer, x, mask, torch.tensor([0., 1., 0.]),
                      phase=0, objective=objective)
    expected = supervised * row['supervised'] + row['alignment'] + signal * row['signal']
    assert row['total'] == pytest.approx(expected, abs=1e-6)
    changes = [k for k, v in student.state_dict().items() if not torch.equal(v, old[k])]
    assert any(k.startswith('base.encoder.') for k in changes)
    assert all(k.startswith(('base.embedding.', 'base.encoder.', 'base.norm.', 'decoder.'))
               for k in changes)
    if not signal:
        assert not any(k.startswith('decoder.') for k in changes)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, old[key], rtol=0, atol=0)


def test_removing_classification_makes_update_independent_of_endpoint_label():
    teacher = model()
    first, opt_first = candidate(teacher)
    second, opt_second = candidate(teacher)
    x = torch.randn(3, 128, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    for student, optimizer, label in ((first, opt_first, 0.), (second, opt_second, 1.)):
        source_step(student, teacher, optimizer, x, mask, torch.full((3,), label),
                    phase=1, objective='no_supervised')
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_invalid_objective_does_not_mutate_candidate():
    teacher = model()
    student, optimizer = candidate(teacher)
    old = {k: v.clone() for k, v in student.state_dict().items()}
    x = torch.randn(2, 128, 23)
    with pytest.raises(ValueError, match='objective'):
        source_step(student, teacher, optimizer, x, torch.ones_like(x, dtype=torch.bool),
                    torch.ones(2), phase=0, objective='risk_tuned_on_east')
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, old[key], rtol=0, atol=0)


def test_history_controls_only_remove_observations_and_leave_original_mask_intact():
    from experiments.risk_aligned_online.source_objective_controls import controlled_mask, LENGTHS
    available = torch.ones(2, 128, 23, dtype=torch.bool)
    available[:, :, 4] = False
    old = available.clone()
    for step in range(16):
        shortened, length = controlled_mask(available, step, short_history=True)
        full, full_length = controlled_mask(available, step, short_history=False)
        assert length == LENGTHS[step % 8] and full_length == 128
        assert not shortened[:, :128-length].any()
        torch.testing.assert_close(shortened[:, 128-length:], old[:, 128-length:])
        torch.testing.assert_close(full, old)
        torch.testing.assert_close(available, old)
    with pytest.raises(ValueError):
        controlled_mask(available, -1, short_history=True)
    with pytest.raises(ValueError):
        controlled_mask(available, 1, short_history='yes')
