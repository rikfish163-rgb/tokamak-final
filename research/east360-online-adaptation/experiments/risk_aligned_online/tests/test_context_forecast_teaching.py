import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.direction_teaching_20261026.learning import loss_from_example
from experiments.risk_aligned_online.tests.test_source_task_memory import setup


def test_forecast_objective_adds_real_gradient_without_changing_old_default():
    old, e = setup()
    model = ContextAdapter(old.base).eval()
    model.slow.requires_grad_(False)
    model.memory.requires_grad_(True)
    e.update(labels=torch.ones(4), local_mask=torch.ones(4, dtype=torch.bool),
             direction_mask=torch.ones(4, dtype=torch.bool))
    default, info = loss_from_example(model, e, .7, 'detached')
    zero, _ = loss_from_example(model, e, .7, 'detached', forecast_weight=0.)
    trained, _ = loss_from_example(model, e, .7, 'detached', forecast_weight=1.)
    assert torch.equal(default, zero)
    assert float((trained-default).detach()) == pytest.approx(info['forecast_loss'], abs=2e-7)
    # Zero output initially blocks risk gradients to the forecast basis. The
    # explicit prediction target must teach it immediately, not only log a loss.
    g0, = torch.autograd.grad(default, model.memory.kernel)
    g1, = torch.autograd.grad(trained, model.memory.kernel)
    assert not g0.count_nonzero()
    assert torch.isfinite(g1).all() and g1.norm() > 0
    assert all(p.grad is None for p in model.base.parameters())
    with pytest.raises(ValueError):
        loss_from_example(model, e, .7, 'detached', forecast_weight=.5)


def test_forecast_profile_cannot_be_mislabelled_as_risk_only():
    from experiments.risk_aligned_online.context_memory_20261028.checkpoint import objective_weight, validate_metadata
    from experiments.risk_aligned_online.context_memory_20261028.model import ContextMemory
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    assert objective_weight(dict(profile='context8_v1', objective='detached')) == 0
    candidate = dict(profile='context8_forecast_v2', objective='detached_forecast', forecast_weight=1.)
    assert objective_weight(candidate) == 1
    roles = dict(train=list(range(1353)), val=list(range(2000,2160)), test=list(range(3000,3220)))
    checkpoint = dict(candidate, kind='context8', context_dim=8, epoch=2, feedback='progressive',
        readout='full', source_sha256=PARENT_SHA256, train_ids=roles['train'], val_ids=roles['val'],
        memory=ContextMemory(64).state_dict())
    validate_metadata(checkpoint, roles)
    for bad in (dict(candidate, forecast_weight=0), dict(candidate, profile='context8_v1'),
                dict(candidate, objective='detached'), {k:v for k,v in candidate.items() if k != 'forecast_weight'}):
        with pytest.raises(ValueError):
            objective_weight(bad)


def test_source_gate_requires_forecast_skill_and_risk_preservation():
    from experiments.risk_aligned_online.context_memory_20261028.forecast_gate import decide
    before = dict(bag=.3, local=.4)
    after = dict(bag=.29, local=.39)
    forecast = dict(shots=160, persistence_mean=1., memory_mean=.98,
                    reverse_mean=.99, better_shots=81)
    assert decide(before, after, forecast)['pass']
    for change in (dict(memory_mean=.995), dict(reverse_mean=.97), dict(better_shots=80)):
        assert not decide(before, after, dict(forecast, **change))['pass']
    assert not decide(before, dict(after, bag=.31), forecast)['pass']
    assert not decide(before, dict(after, local=.41), forecast)['pass']
