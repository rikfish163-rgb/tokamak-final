import math

import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.delayed_recurrent_memory_20261025.model import (
    endpoint_corrections,
)
from experiments.risk_aligned_online.east_context_operator_20261031.learning import (
    configure_operator_training,
    objective,
    risk_logits,
)
from experiments.risk_aligned_online.lifecycle_stream_20261025.learning import window_loss
from experiments.risk_aligned_online.simple23 import Config, Simple23


TRAINABLE = {
    'memory.key.weight',
    'memory.key.bias',
    'memory.kernel',
    'memory.rate',
}


@pytest.fixture
def case():
    torch.manual_seed(131)
    base = Simple23(Config(window=16, dim=64, fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23)).eval()
    model = ContextAdapter(base).eval()
    # The default output is zero; use a nonzero frozen projection for this test.
    with torch.no_grad():
        model.memory.output.weight.normal_(0, .2)
        model.slow.weight.normal_(0, .01)
        windows = torch.randn(4, 16, 23)
        h, valid = model.tokens(windows, torch.ones_like(windows, dtype=torch.bool))
    means = torch.randn(12, 23)
    example = dict(means=means, complete=torch.ones_like(means, dtype=torch.bool),
                   h=h, valid=valid, ends=torch.tensor([16, 40, 64, 88]),
                   y=torch.tensor([0., 1., 1., 1.]), flag=True)
    return model, example


def test_configure_freezes_everything_but_the_four_operator_groups(case):
    model, _ = case
    model.requires_grad_(True)
    before = {name: p.clone() for name, p in model.named_parameters()}
    selected = configure_operator_training(model)
    parameters = dict(model.named_parameters())
    assert {name for name, p in parameters.items() if p.requires_grad} == TRAINABLE
    assert {id(p) for p in selected} == {id(parameters[name]) for name in TRAINABLE}
    assert len(selected) == 4 and sum(p.numel() for p in selected) == 7001
    assert all(torch.equal(p, before[name]) for name, p in parameters.items())


def test_risk_loss_backpropagates_through_all_four_operator_groups(case):
    model, example = case
    configure_operator_training(model)
    loss, detail = objective(model, example)
    z = risk_logits(model, example)
    torch.testing.assert_close(loss, window_loss(z, example['y'], True,
                                                'disruptive_positive_only'))
    assert detail['writes'] > 0
    assert math.isfinite(detail['max_state_norm']) and detail['max_state_norm'] > 0
    assert math.isfinite(detail['forecast_loss']) and detail['forecast_loss'] >= 0
    loss.backward()
    for name, parameter in model.named_parameters():
        if name in TRAINABLE:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name
        else:
            assert parameter.grad is None, name


def test_rate_gradient_matches_finite_difference_through_real_writes(case):
    model, example = case
    model.double()
    example = {key: value.double() if isinstance(value, torch.Tensor)
               and value.is_floating_point() else value for key, value in example.items()}
    configure_operator_training(model)
    gradient, = torch.autograd.grad(objective(model, example)[0], model.memory.rate)
    epsilon = 1e-4
    with torch.no_grad():
        model.memory.rate.add_(epsilon)
    plus = objective(model, example)[0].item()
    with torch.no_grad():
        model.memory.rate.sub_(2 * epsilon)
    minus = objective(model, example)[0].item()
    assert gradient.item() == pytest.approx((plus - minus) / (2 * epsilon),
                                             rel=1e-4, abs=1e-8)


def test_adam_updates_only_operator_and_zero_delta_off_risk_is_unchanged(case):
    model, example = case
    selected = configure_operator_training(model)
    before = {name: p.clone() for name, p in model.named_parameters()}
    off_before = risk_logits(model, example, mode='off')
    direct_zero = model.risk(example['h'], example['valid'], torch.zeros(4, 92))
    # The frozen no-grad and differentiable Transformer paths can differ by
    # float32 roundoff even though the supplied correction is identically zero.
    torch.testing.assert_close(off_before, direct_zero, atol=1e-6, rtol=1e-6)
    memory_before = risk_logits(model, example)
    optimizer = torch.optim.Adam(selected, lr=.01)
    optimizer.zero_grad(set_to_none=True)
    objective(model, example)[0].backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in TRAINABLE:
            assert not torch.equal(parameter, before[name]), name
        else:
            assert torch.equal(parameter, before[name]), name
    torch.testing.assert_close(risk_logits(model, example, mode='off'), off_before,
                               atol=0, rtol=0)
    assert not torch.equal(risk_logits(model, example), memory_before)


def test_future_means_cannot_change_earlier_endpoint_corrections_or_logits(case):
    model, example = case
    configure_operator_training(model)
    early = dict(example, h=example['h'][:2], valid=example['valid'][:2],
                 ends=example['ends'][:2], y=example['y'][:2])
    changed = dict(early, means=early['means'].clone())
    changed['means'][5:] += 1000
    original_trace = model.memory.trace(early['means'], early['complete'], mode='memory')
    changed_trace = model.memory.trace(changed['means'], changed['complete'], mode='memory')
    before = endpoint_corrections(original_trace['delta'], early['ends'])
    after = endpoint_corrections(changed_trace['delta'], early['ends'])
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.testing.assert_close(risk_logits(model, early), risk_logits(model, changed),
                               atol=0, rtol=0)


def test_logits_use_at_most_32_endpoints_per_risk_call(case, monkeypatch):
    model, example = case
    configure_operator_training(model)
    example = dict(example, h=example['h'].repeat(9, 1, 1)[:35],
                   valid=example['valid'].repeat(9, 1)[:35],
                   ends=example['ends'].repeat(9)[:35])
    calls = []
    actual_risk = model.risk

    def counted_risk(h, valid, delta):
        calls.append(len(h))
        return actual_risk(h, valid, delta)

    monkeypatch.setattr(model, 'risk', counted_risk)
    z = risk_logits(model, example)
    assert z.shape == (35,) and calls == [32, 3]
    assert torch.isfinite(z).all()
    with pytest.raises(ValueError, match='mode'):
        risk_logits(model, example, mode='reverse')


def test_logits_accept_existing_list_endpoint_contract_without_labels(case):
    model, example = case
    configure_operator_training(model)
    unlabeled = {key: value for key, value in example.items() if key not in ('y', 'flag')}
    unlabeled['ends'] = example['ends'].tolist()
    torch.testing.assert_close(risk_logits(model, unlabeled), risk_logits(model, example))


def test_zero_frozen_output_blocks_operator_risk_gradients(case):
    model, example = case
    with torch.no_grad():
        model.memory.output.weight.zero_()
    selected = configure_operator_training(model)
    objective(model, example)[0].backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in selected)


def test_no_observed_patches_gives_zero_writes_and_finite_off_identity(case):
    model, example = case
    configure_operator_training(model)
    empty = dict(example, means=example['means'][:0], complete=example['complete'][:0],
                 ends=torch.tensor([0, 1, 2, 7]))
    loss, detail = objective(model, empty)
    assert torch.isfinite(loss)
    assert detail == dict(writes=0, max_state_norm=0., forecast_loss=0.)
    torch.testing.assert_close(risk_logits(model, empty), risk_logits(model, empty, mode='off'),
                               atol=0, rtol=0)


def test_unobserved_nonfinite_means_are_ignored(case):
    model, example = case
    configure_operator_training(model)
    missing = dict(example, means=example['means'].clone(), complete=example['complete'].clone())
    missing['complete'][0, 0] = False
    missing['means'][0, 0] = float('nan')
    assert torch.isfinite(objective(model, missing)[0])
    missing['complete'][0, 0] = True
    with pytest.raises(ValueError, match='nonfinite'):
        objective(model, missing)


@pytest.mark.parametrize('field,value', [
    ('y', None),
    ('y', torch.ones(4, 1)),
    ('y', torch.tensor([0., 1., float('nan'), 1.])),
    ('y', torch.tensor([0., 1., .5, 1.])),
    ('flag', None),
    ('flag', 'False'),
    ('flag', False),  # a safe shot cannot contain positive window labels
    ('ends', torch.tensor([16, 40, 64])),
    ('ends', torch.tensor([[16, 40, 64, 88]])),
    ('ends', torch.tensor([16., 40., 64., 88.])),
    ('ends', torch.tensor([-1, 40, 64, 88])),
    ('ends', torch.tensor([16, 40, 64, 104])),
    ('valid', torch.ones(4, 3)),
    ('valid', torch.ones(4, 2, dtype=torch.bool)),
    ('h', torch.zeros(4, 3, 63)),
    ('h', torch.full((4, 3, 64), float('inf'))),
    ('means', torch.zeros(12, 22)),
    ('complete', torch.ones(12, 23)),
])
def test_invalid_example_is_rejected(case, field, value):
    model, example = case
    configure_operator_training(model)
    bad = dict(example, **{field: value})
    with pytest.raises(ValueError):
        objective(model, bad)


def test_missing_labels_and_disruptive_shot_without_positives_are_rejected(case):
    model, example = case
    configure_operator_training(model)
    for field in ('y', 'flag'):
        bad = dict(example)
        del bad[field]
        with pytest.raises(ValueError):
            objective(model, bad)
    with pytest.raises(ValueError, match='positive'):
        objective(model, dict(example, y=torch.zeros_like(example['y'])))


def test_safe_shot_uses_the_existing_safe_window_policy(case):
    model, example = case
    configure_operator_training(model)
    safe = dict(example, flag=False, y=torch.zeros_like(example['y']))
    loss, _ = objective(model, safe)
    expected = window_loss(risk_logits(model, safe), safe['y'], False,
                           'disruptive_positive_only')
    torch.testing.assert_close(loss, expected)
