import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.progressive_feedback_20261026.model import ProgressiveDiagnosticMemory
from experiments.risk_aligned_online.update_direction_20261026.analysis import selected_states
from experiments.risk_aligned_online.direction_teaching_20261026.learning import local_targets, objective


def test_optional_before_read_matches_same_historical_state():
    torch.manual_seed(131)
    m = ProgressiveDiagnosticMemory(64).double()
    means = torch.randn(16, 23, dtype=torch.double)
    mask = torch.ones_like(means, dtype=torch.bool)
    plain = m.trace(means, mask, mode='memory')
    paired = m.trace(means, mask, mode='memory', return_before=True)
    assert torch.equal(plain['delta'], paired['delta'])
    states = selected_states(m, means, mask, [0, 1, 5, 15])
    for j, s in states.items():
        assert torch.equal(paired['before_delta'][j], s['key'] @ (s['before']-m.prior))
    changed = means.clone()
    changed[8:] += 100
    future = m.trace(changed, mask, mode='memory', return_before=True)
    assert torch.equal(paired['before_delta'][:8], future['before_delta'][:8])


def test_local_supervision_uses_jtext_window_not_east_or_early_negative():
    times = np.arange(200)*.001
    shot = dict(device='J-TEXT', times=times, down=.199, is_disrupt=True,
                y=(times >= .174).astype('float32'), valid=np.ones(200, bool))
    result = local_targets(shot, np.array([0, 150, 174, 188, 189, 199]))
    assert result['local_mask'].tolist() == [False, False, True, True, False, False]
    assert result['direction_mask'].tolist() == [False, False, True, True, False, False]
    safe = dict(shot, is_disrupt=False, down=None, y=np.zeros(200, 'float32'))
    result = local_targets(safe, np.array([7, 15, 16, 199]))
    assert result['local_mask'].all()
    assert result['direction_mask'].tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match='source'):
        local_targets(dict(shot, device='EAST'), np.array([174]))


def test_direction_loss_penalizes_harm_and_backpropagates_both_sides():
    after = torch.tensor([-.2, .2], requires_grad=True)
    before = torch.tensor([0., 0.], requires_grad=True)
    e = dict(flag=True, labels=torch.tensor([1., 0.]), local_mask=torch.ones(2, dtype=torch.bool),
             direction_mask=torch.ones(2, dtype=torch.bool))
    _, components = objective(after, before, e, .5, 'direction')
    components['direction'].backward()
    assert after.grad[0] < 0 and after.grad[1] > 0
    assert before.grad[0] > 0 and before.grad[1] < 0
    _, improved = objective(-after.detach(), before.detach(), e, .5, 'direction')
    assert improved['direction'] < components['direction']


@pytest.mark.parametrize('kind', ['direction', 'detached'])
def test_empty_local_supervision_is_explicit_zero_not_fabricated_target(kind):
    z = torch.tensor([0.], requires_grad=True)
    e = dict(flag=True, labels=torch.ones(1), local_mask=torch.zeros(1, dtype=torch.bool),
             direction_mask=torch.zeros(1, dtype=torch.bool))
    total, detail = objective(z, z, e, .5, kind)
    assert detail['local'] == 0 and detail['direction'] == 0
    assert torch.isfinite(total)


def test_training_audit_requires_both_full_epochs_and_real_rule_gradients():
    from experiments.risk_aligned_online.direction_teaching_20261026.audit import validate_training
    rows = [dict(epoch=epoch, ids=[1, 2], arms={k: dict(gradients={
        'key.weight': .1, 'prior': .2, 'rate': .3, 'output.weight': .4})
        for k in ('bag', 'direction')}) for epoch in (1, 2)]
    assert validate_training(rows, [1, 2])['complete_epochs'] == 2
    with pytest.raises(ValueError, match='coverage'):
        validate_training(rows[:1], [1, 2])
    rows[1]['arms']['direction']['gradients']['rate'] = float('nan')
    with pytest.raises(ValueError, match='gradient'):
        validate_training(rows, [1, 2])


def test_checkpoint_cannot_silently_deploy_different_feedback():
    from types import SimpleNamespace
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    from experiments.risk_aligned_online.direction_teaching_20261026.checkpoint import validate_metadata
    roles = dict(train=list(range(1353)), val=list(range(2000, 2160)), test=list(range(3000, 3220)))
    a = dict(profile='direction_teaching_v1', kind='direction', feedback='progressive',
             readout='full', epoch=2, source_sha256=PARENT_SHA256,
             train_ids=roles['train'], val_ids=roles['val'])
    model = SimpleNamespace(feedback='progressive', readout='full')
    validate_metadata(a, model, roles)
    with pytest.raises(ValueError, match='feedback'):
        validate_metadata(a, SimpleNamespace(feedback='delayed', readout='full'), roles)
    with pytest.raises(ValueError, match='epoch'):
        validate_metadata(dict(a, epoch=1), model, roles)


@pytest.mark.parametrize('kind', ['direction', 'detached'])
def test_candidate_gradient_traverses_actual_writes_not_source_parameters(kind):
    from experiments.risk_aligned_online.tests.test_source_task_memory import setup
    from experiments.risk_aligned_online.direction_teaching_20261026.learning import make_adapter, loss_from_example
    old, e = setup()
    model = make_adapter(old.base)
    model.memory.load_state_dict(old.memory.state_dict())
    e.update(labels=torch.ones(4), local_mask=torch.ones(4, dtype=torch.bool),
             direction_mask=torch.ones(4, dtype=torch.bool))
    before = {k: v.clone() for k, v in model.base.state_dict().items()}
    total, _ = loss_from_example(model, e, .7, kind)
    total.backward()
    for name in ('key.weight', 'prior', 'rate', 'output.weight'):
        g = dict(model.memory.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0
    assert model.slow.weight.grad is None
    assert all(p.grad is None for p in model.base.parameters())
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in before.items())


def test_full_stream_pair_rejects_changed_alarm_or_feedback():
    from experiments.risk_aligned_online.direction_teaching_20261026.compare import validate_pair
    a = dict(threshold=.7, feedback='progressive', source_deployment_feedback_match=True,
             source_memory=dict(profile='direction_teaching_v1', kind='bag', epoch=2,
                                source_protocol_sha256='a'*64, audit_sha256='b'*64))
    b = dict(a, source_memory=dict(a['source_memory'], kind='direction'))
    validate_pair(a, b)
    with pytest.raises(ValueError, match='paired'):
        validate_pair(a, dict(b, threshold=.8))
    with pytest.raises(ValueError, match='feedback'):
        validate_pair(dict(a, source_deployment_feedback_match=False),
                      dict(b, source_deployment_feedback_match=False))


def test_detached_reference_changes_gradient_only_not_forward_objective():
    after = torch.tensor([-.2, .2], requires_grad=True)
    before = torch.zeros(2, requires_grad=True)
    e = dict(flag=True, labels=torch.tensor([1., 0.]),
             local_mask=torch.ones(2, dtype=torch.bool), direction_mask=torch.ones(2, dtype=torch.bool))
    joint, joint_parts = objective(after, before, e, .5, 'direction')
    detached, detached_parts = objective(after, before, e, .5, 'detached')
    assert torch.equal(joint, detached)
    assert torch.equal(joint_parts['direction'], detached_parts['direction'])
    joint_grads = torch.autograd.grad(joint, (after, before), retain_graph=True)
    detached_grads = torch.autograd.grad(detached, (after, before), allow_unused=True)
    assert torch.equal(joint_grads[0], detached_grads[0])
    assert joint_grads[1].abs().sum() > 0
    assert detached_grads[1] is None


def test_reference_ablation_has_explicit_audited_pair():
    from experiments.risk_aligned_online.direction_teaching_20261026.audit import validate_training
    rows = [dict(epoch=epoch, ids=[1, 2], arms={k: dict(gradients={
        'key.weight': .1, 'prior': .2, 'rate': .3, 'output.weight': .4})
        for k in ('direction', 'detached')}) for epoch in (1, 2)]
    assert validate_training(rows, [1, 2], ('direction', 'detached'))['complete_epochs'] == 2
    with pytest.raises(ValueError, match='arms'):
        validate_training(rows, [1, 2])


def test_reference_checkpoint_rejects_ambiguous_or_mislabelled_policy():
    from types import SimpleNamespace
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    from experiments.risk_aligned_online.direction_teaching_20261026.checkpoint import validate_metadata
    roles = dict(train=list(range(1353)), val=list(range(2000, 2160)), test=list(range(3000, 3220)))
    a = dict(profile='reference_gradient_ablation_v2', kind='detached', reference_gradient='stopped',
             feedback='progressive', readout='full', epoch=2, source_sha256=PARENT_SHA256,
             train_ids=roles['train'], val_ids=roles['val'])
    model = SimpleNamespace(feedback='progressive', readout='full')
    validate_metadata(a, model, roles)
    for bad in (dict(a, reference_gradient='joint'), dict(a, profile='direction_teaching_v1'),
                dict(a, kind='bag'), {k: v for k, v in a.items() if k != 'reference_gradient'}):
        with pytest.raises(ValueError):
            validate_metadata(bad, model, roles)


def test_reference_full_stream_pair_requires_same_source_trial_and_policy():
    from experiments.risk_aligned_online.direction_teaching_20261026.compare import validate_pair
    a = dict(threshold=.7, feedback='progressive', source_deployment_feedback_match=True,
             source_memory=dict(profile='reference_gradient_ablation_v2', kind='direction', epoch=2,
                                reference_gradient='joint',
                                source_protocol_sha256='a'*64, audit_sha256='b'*64))
    b = dict(a, source_memory=dict(a['source_memory'], kind='detached', reference_gradient='stopped'))
    validate_pair(a, b)
    with pytest.raises(ValueError, match='artifacts'):
        validate_pair(a, dict(b, source_memory=dict(b['source_memory'], reference_gradient='joint')))
    with pytest.raises(ValueError, match='artifacts'):
        validate_pair(a, dict(b, source_memory=dict(b['source_memory'], source_protocol_sha256='c'*64)))
