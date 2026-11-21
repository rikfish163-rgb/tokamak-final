"""Tests of the updater's optional source-memory transaction, not efficacy."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.redesign_20261004 import source_replay
from .test_inter import ContextStub, shot, update


def install_bank(model, monkeypatch, *, reject=False):
    model.source_bank = {'bank_sha256': 'fixed_test_bank'}
    model.source_replay_weight = 1.
    calls = []
    def sample(bank, rng, batch_size=64):
        assert bank is model.source_bank and batch_size == 64
        return torch.arange(64)
    def loss(current, bank, device, rng, batch_size=64, *, indices=None):
        calls.append(indices.clone())
        result = current.memory.square().mean()
        return result, dict(indices=indices.tolist(), bce=float(result.detach()),
                            positive_count=32, safe_count=32)
    def guard(current, bank, device, threshold, **kwargs):
        assert threshold == .5
        return dict(passed=not reject, reason='source_threshold_crossing' if reject else
                    'retained_original_correct_points', protected_tp=32, protected_tn=32,
                    lost_tp=int(reject), lost_tn=0, checked_points=128)
    monkeypatch.setattr(source_replay, 'sample_source', sample)
    monkeypatch.setattr(source_replay, 'source_loss', loss)
    monkeypatch.setattr(source_replay, 'evaluate_source_guard', guard)
    return calls


def test_source_batch_is_fixed_across_candidate_steps_and_audited(monkeypatch):
    model = ContextStub()
    calls = install_bank(model, monkeypatch)
    result = update(model, shot(1), lr=.03)
    assert result['accepted']
    assert len(calls) == 6  # before + four gradients + candidate
    assert all(torch.equal(i, calls[0]) for i in calls)
    source = result['source_replay']
    assert source['weight'] == 1. and source['bank_sha256'] == 'fixed_test_bank'
    assert source['before']['positive_count'] == source['before']['safe_count'] == 32
    assert source['candidate']['bce'] > 0 and source['guard']['passed']


def test_source_crossing_failure_rolls_back_target_update(monkeypatch):
    model = ContextStub()
    install_bank(model, monkeypatch, reject=True)
    result = update(model, shot(1))
    assert not result['accepted'] and 'source_threshold_crossing' in result['reasons']
    assert result['loss']['candidate_bce'] < result['loss']['before_bce']
    assert result['drift']['candidate_step_l2'] > 0 and result['drift']['step_l2'] == 0
    assert torch.equal(model.memory, torch.zeros_like(model.memory))


def test_correct_experience_only_does_not_run_a_source_gradient(monkeypatch):
    model = ContextStub()
    model.update_policy = 'error_only'
    calls = install_bank(model, monkeypatch)
    result = update(model, shot(1, safe=True, sealed=.1))
    assert result['reason'] == 'correct_shot_experience_only' and not calls
    assert result['source_replay']['before'] is None


@pytest.mark.parametrize('weight', [-1., 0., float('nan')])
def test_invalid_source_weight_rejected(weight, monkeypatch):
    model = ContextStub()
    install_bank(model, monkeypatch)
    model.source_replay_weight = weight
    with pytest.raises(ValueError, match='source'):
        update(model, shot(1))


def test_absent_source_bank_keeps_legacy_objective():
    result = update(ContextStub(), shot(1))
    assert result['source_replay'] is None and result['accepted']


def test_source_crossing_can_reduce_step_until_existing_decisions_are_preserved(monkeypatch):
    model = ContextStub()
    install_bank(model, monkeypatch)
    model.source_backtracking = 8
    def bounded_guard(current, bank, device, threshold, **kwargs):
        failed = bool(current.memory.norm() > .01)
        return dict(passed=not failed, reason='source_threshold_crossing' if failed else
                    'retained_original_correct_points', protected_tp=32, protected_tn=32,
                    lost_tp=int(failed), lost_tn=0, checked_points=128)
    monkeypatch.setattr(source_replay, 'evaluate_source_guard', bounded_guard)
    result = update(model, shot(1), lr=.4)
    assert result['accepted'] and 0 < model.memory.norm() <= .01
    assert result['proposal_scale'] < 1. and len(result['backtracking']) > 1
    assert result['backtracking'][0]['reasons'] == ['source_threshold_crossing']
    assert result['backtracking'][-1]['reasons'] == []
    assert result['loss']['candidate_bce'] < result['loss']['before_bce']


def test_no_feasible_reduced_candidate_still_rolls_back(monkeypatch):
    model = ContextStub()
    install_bank(model, monkeypatch, reject=True)
    model.source_backtracking = 3
    result = update(model, shot(1))
    assert not result['accepted'] and len(result['backtracking']) == 4
    assert torch.equal(model.memory, torch.zeros_like(model.memory))


def test_explicit_diagnostic_source_guard_records_crossings_but_keeps_source_loss(monkeypatch):
    model = ContextStub()
    calls = install_bank(model, monkeypatch, reject=True)
    model.source_guard_policy = 'diagnostic'
    result = update(model, shot(1))
    assert result['accepted'] and calls
    assert not result['source_replay']['guard']['passed']
    assert result['source_replay']['guard_policy'] == 'diagnostic'
    assert result['source_replay']['candidate']['bce'] > 0
    assert model.memory.norm() > 0


def test_source_guard_policy_must_be_explicitly_known(monkeypatch):
    model = ContextStub()
    install_bank(model, monkeypatch)
    model.source_guard_policy = 'invented'
    with pytest.raises(ValueError, match='source guard'):
        update(model, shot(1))
