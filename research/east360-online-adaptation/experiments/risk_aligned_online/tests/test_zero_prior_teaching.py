from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.direction_teaching_20261026.learning import make_adapter, objective
from experiments.risk_aligned_online.direction_teaching_20261026.audit import validate_training
from experiments.risk_aligned_online.direction_teaching_20261026.checkpoint import validate_metadata
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256


def test_zero_prior_prevents_adaptation_without_signal_change():
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), np.zeros(23), np.ones(23))
    model = make_adapter(base, zero_prior=True)
    assert not model.memory.prior.requires_grad and not model.memory.prior.count_nonzero()
    means = torch.ones(16, 23)
    trace = model.memory.trace(means, torch.ones_like(means, dtype=torch.bool), mode='memory', return_before=True)
    assert not trace['delta'].count_nonzero()
    assert not trace['before_delta'].count_nonzero()
    assert trace['updates'] == 15
    changed = means.clone(); changed[8:, 0] += 1.
    trace2 = model.memory.trace(changed, torch.ones_like(means, dtype=torch.bool), mode='memory')
    assert not trace2['delta'][:8].count_nonzero()
    assert trace2['delta'][8:].abs().max() > 0


def test_zero_prior_loss_matches_detached_reference_without_new_loss_terms():
    z = torch.tensor([-.2, .3], requires_grad=True)
    before = torch.zeros(2, requires_grad=True)
    e = dict(flag=True, labels=torch.tensor([1., 0.]), local_mask=torch.ones(2, dtype=torch.bool),
             direction_mask=torch.ones(2, dtype=torch.bool))
    a, _ = objective(z, before, e, .7, 'detached')
    b, _ = objective(z, before, e, .7, 'zero_prior')
    assert torch.equal(a, b)
    assert torch.autograd.grad(b, before, allow_unused=True)[0] is None


def test_audit_checks_expected_frozen_prior_not_silently_ignoring_missing_gradients():
    rows = [dict(epoch=epoch, ids=[1, 2], arms={kind: dict(gradients={
        'key.weight': .1, 'prior': None if kind == 'zero_prior' else .2,
        'rate': .3, 'output.weight': .4}) for kind in ('detached', 'zero_prior')}) for epoch in (1, 2)]
    a = validate_training(rows, [1, 2], ('detached', 'zero_prior'))
    assert a['frozen_parameters'] == {'zero_prior': ['prior']}
    rows[0]['arms']['zero_prior']['gradients']['prior'] = 0.
    with pytest.raises(ValueError, match='frozen prior'):
        validate_training(rows, [1, 2], ('detached', 'zero_prior'))
    rows[0]['arms']['zero_prior']['gradients']['prior'] = None
    rows[0]['arms']['zero_prior']['gradients']['key.weight'] = None
    with pytest.raises(ValueError, match='gradient'):
        validate_training(rows, [1, 2], ('detached', 'zero_prior'))


def test_zero_prior_checkpoint_rejects_drift_or_wrong_policy():
    roles = dict(train=list(range(1353)), val=list(range(2000, 2160)), test=list(range(3000, 3220)))
    a = dict(profile='zero_prior_ablation_v3', kind='zero_prior', prior_policy='fixed_zero',
             reference_gradient='stopped', feedback='progressive', readout='full', epoch=2,
             source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'],
             memory={'prior': torch.zeros(9, 92)})
    model = SimpleNamespace(feedback='progressive', readout='full')
    validate_metadata(a, model, roles)
    for bad in (dict(a, prior_policy='learned'), dict(a, reference_gradient='joint'),
                dict(a, memory={'prior': torch.ones(9, 92)})):
        with pytest.raises(ValueError):
            validate_metadata(bad, model, roles)


def test_zero_prior_pair_cannot_mix_reference_objectives_or_prior_policies():
    from experiments.risk_aligned_online.direction_teaching_20261026.compare import validate_pair
    a = dict(threshold=.7, feedback='progressive', source_deployment_feedback_match=True,
        source_memory=dict(profile='zero_prior_ablation_v3', kind='detached', epoch=2,
             reference_gradient='stopped', prior_policy='learned',
             source_protocol_sha256='a'*64, audit_sha256='b'*64))
    b = dict(a, source_memory=dict(a['source_memory'], kind='zero_prior', prior_policy='fixed_zero'))
    validate_pair(a, b)
    with pytest.raises(ValueError):
        validate_pair(a, dict(b, source_memory=dict(b['source_memory'], prior_policy='learned')))
