import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ffn_subspace_sequence import (
    operational_loss, chain_origins, meta_sequence_step)
from experiments.risk_aligned_online.tests.test_ffn_subspace import model, example
from experiments.risk_aligned_online.ffn_subspace import one_step
from experiments.risk_aligned_online.ffn_subspace_sequence_train import source_chain_spec


def test_safe_loss_targets_worst_risk_not_average_easy_negative():
    logits = torch.tensor([-5., .2], requires_grad=True)
    loss = operational_loss(logits, False, torch.zeros(2), torch.zeros(2), threshold_logit=0.)
    gradient, = torch.autograd.grad(loss, logits)
    assert gradient[0] == 0 and gradient[1] > 0


def test_disruptive_loss_uses_true_positive_window_and_strict_ten_ms():
    logits = torch.tensor([8., -.2, 9.], requires_grad=True)
    loss = operational_loss(logits, True, torch.tensor([0., 1., 1.]),
                            torch.tensor([100., 11., 10.]), threshold_logit=0.)
    gradient, = torch.autograd.grad(loss, logits)
    assert gradient[0] == 0 and gradient[1] < 0 and gradient[2] == 0
    with pytest.raises(ValueError, match='timely'):
        operational_loss(logits, True, torch.zeros(3), torch.ones(3), threshold_logit=0.)


def test_source_chain_matches_actual_maturity_and_accumulation():
    assert chain_origins(335) == [0, 100, 200, 300]
    assert chain_origins(315) == [0, 100, 200]
    assert chain_origins(735) == [400, 500, 600, 700]
    with pytest.raises(ValueError, match='26'):
        chain_origins(25)


def test_chain_selection_is_source_only_and_frozen_risk_peak_is_an_anchor():
    shot = dict(device='J-TEXT', shot=1, times=np.arange(450) / 1000.,
                valid=np.ones(450, bool), y=np.zeros(450), is_disrupt=False, down=None)
    scores = np.zeros(450); scores[335] = .72
    spec = source_chain_spec(shot, scores)
    assert spec['origins'] == [0, 100, 200, 300]
    assert 335 in spec['anchors'][-1]
    assert all(o + 26 <= t <= o + 125 for o, ts in zip(spec['origins'], spec['anchors']) for t in ts)
    with pytest.raises(ValueError, match='SOURCE'):
        source_chain_spec(dict(shot, device='EAST'), scores)


def test_signal_step_preserves_cumulative_input_state(model):
    e = example()
    current = model.initial_fast(1).requires_grad_()
    once, _ = one_step(model, e, fast=current, create_graph=True)
    twice, _ = one_step(model, e, fast=once, create_graph=True)
    assert not torch.equal(once, twice)
    assert twice.norm() <= 1.00001
    assert torch.autograd.grad(twice.sum(), model.basis)[0].abs().sum() > 0


@pytest.mark.parametrize('mode', ['risk', 'signal'])
def test_chain_meta_changes_geometry_not_source_weights(model, mode):
    steps = []
    for _ in range(2):
        e = example(); e['segment_x'] = e['qx'].expand(3, -1, -1).clone()
        e['segment_a'] = e['qa'].expand(3, -1, -1).clone()
        e['segment_y'] = torch.zeros(3); e['segment_lead'] = torch.zeros(3)
        steps.append(e)
    state = {k: v.clone() for k, v in model.base.state_dict().items()}
    basis = model.basis.detach().clone()
    optimizer = torch.optim.Adam([model.basis], lr=.001)
    row = meta_sequence_step(model, optimizer, [dict(steps=steps, is_disrupt=False)],
                             mode=mode, threshold_logit=0.)
    assert row['basis_grad_norm'] > 0
    assert not torch.equal(model.basis, basis)
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in state.items())
