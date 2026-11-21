import io

import pytest
import torch
import torch.nn.functional as F

from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.state_capacity_20261031.optimize import (
    probe,
    risk_with_shift,
)


@pytest.fixture
def case():
    torch.manual_seed(61031)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23))
    model = ContextAdapter(base).eval()
    with torch.no_grad():
        model.slow.weight.normal_(0, .04)
        model.memory.output.weight.normal_(0, .03)
    model.requires_grad_(False)
    h = torch.randn(3, 17, 64)
    valid = torch.ones(3, 17, dtype=torch.bool)
    valid[0, 2] = False
    valid[1, 15] = False
    basis = torch.randn(3, 92, 8) * .2
    actual_context = torch.randn(3, 8) * .1
    y = torch.tensor([0., 1., 1.])
    enabled = torch.tensor([True, True, False])
    return model, h, valid, basis, actual_context, y, enabled


def test_shift_matches_original_context_and_zero_risk(case):
    model, h, valid, basis, context, _, _ = case
    delta = torch.einsum('nij,nj->ni', basis, context)
    shift = h.new_zeros(3, 16, 64)
    shift[:, -1] = model.memory.output(delta)
    torch.testing.assert_close(risk_with_shift(model, h, valid, shift),
                               model.risk(h, valid, delta), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(risk_with_shift(model, h, valid, torch.zeros_like(shift)),
                               model.risk(h, valid, torch.zeros_like(delta)),
                               atol=1e-6, rtol=1e-6)


def test_spectral_radius_is_computed_in_float64_before_cast(case, monkeypatch):
    model, h, valid, basis, actual, y, enabled = case
    original = torch.linalg.svdvals
    seen = []

    def checked(matrix, *args, **kwargs):
        seen.append(matrix.dtype)
        assert matrix.dtype == torch.float64
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'svdvals', checked)
    result = probe(model, h, valid, basis, actual, y, enabled=enabled,
                   threshold=.5, steps=1)
    assert seen == [torch.float64] and result['radius'].dtype == h.dtype


def test_shift_masks_padding_protects_cls_and_has_finite_gradient(case):
    model, h, valid, _, _, _, _ = case
    h = h.requires_grad_()
    shift = torch.randn(3, 16, 64, requires_grad=True)
    seen = []
    hook = model.base.encoder.layers[1].register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(args[0].detach().clone()),
        with_kwargs=True)
    try:
        z = risk_with_shift(model, h, valid, shift)
        grad, h_grad = torch.autograd.grad(z.sum(), (shift, h))
    finally:
        hook.remove()
    expected = h.clone()
    expected[:, -2] += model.slow(h[:, -2]) * valid[:, -2, None]
    expected[:, :-1] += shift.detach() * valid[:, :-1, None]
    torch.testing.assert_close(seen[0], expected)
    assert torch.equal(seen[0][:, -1], h[:, -1])
    assert torch.isfinite(grad).all()
    assert torch.isfinite(h_grad).all()
    assert torch.equal(grad[~valid[:, :-1]], torch.zeros_like(grad[~valid[:, :-1]]))


def test_probe_replay_constraints_best_so_far_and_model_unchanged(case):
    model, h, valid, basis, actual, y, enabled = case
    # Even a pre-existing model gradient belongs to the caller.
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, .125)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [(p.requires_grad, p.grad) for p in model.parameters()]
    grad_before = first_parameter.grad.clone()
    modes = [m.training for m in model.modules()]
    result = probe(model, h, valid, basis, actual, y, enabled=enabled,
                   threshold=.5, steps=3, lr=.1)
    expected_radius = torch.linalg.svdvals(model.memory.output.weight @ basis)[:, 0]
    expected_radius = expected_radius * enabled
    torch.testing.assert_close(result['radius'], expected_radius)
    effective_actual = actual * enabled[:, None]
    actual_delta = torch.einsum('nij,nj->ni', basis, effective_actual)
    torch.testing.assert_close(result['actual_logits'], model.risk(h, valid, actual_delta),
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result['actual_context_norm'], actual.norm(dim=-1))
    expected_zero = model.risk(h, valid, torch.zeros_like(actual_delta))
    torch.testing.assert_close(result['zero_logits'], expected_zero, atol=1e-6, rtol=1e-6)
    zero_loss = F.binary_cross_entropy_with_logits(expected_zero, y, reduction='none')
    prior_loss = F.binary_cross_entropy_with_logits(result['actual_logits'], y, reduction='none')
    previous = None
    for name in ('context8', 'free_last', 'free_sequence'):
        row = result['paths'][name]
        p, shift = row['best_parameter'], row['best_shift']
        assert shift.shape == (3, 16, 64)
        assert torch.equal(shift[~valid[:, :-1]], torch.zeros_like(shift[~valid[:, :-1]]))
        assert not shift[~enabled].count_nonzero()
        assert not p[~enabled].count_nonzero()
        assert torch.isfinite(p).all() and torch.isfinite(shift).all()
        assert (shift.flatten(1).norm(dim=-1) <= expected_radius + 1e-5).all()
        assert (row['best_step'] >= 0).all() and (row['best_step'] <= 3).all()
        assert ((row['best_init'] == 0) | (row['best_init'] == 1)).all()
        replay = risk_with_shift(model, h, valid, shift)
        torch.testing.assert_close(row['best_logits'], replay, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(row['best_loss'], F.binary_cross_entropy_with_logits(
            replay, y, reduction='none'), atol=1e-6, rtol=1e-6)
        assert (row['best_loss'] <= zero_loss + 1e-6).all()
        if name == 'context8':
            assert (p.norm(dim=-1) <= 1 + 1e-6).all()
            assert (row['best_loss'] <= prior_loss + 1e-6).all()
            delta = torch.einsum('nij,nj->ni', basis, p)
            torch.testing.assert_close(row['best_logits'], model.risk(h, valid, delta),
                                       atol=1e-6, rtol=1e-6)
        elif name == 'free_last':
            assert not shift[:, :-1].count_nonzero()
        if previous is not None:
            assert (row['best_loss'] <= previous + 1e-6).all()
        previous = row['best_loss']
        assert row['summary']['count'] == len(y)
        assert row['summary']['mean_loss'] == pytest.approx(float(row['best_loss'].mean()))
        target = y.bool()
        positive_prediction = row['best_logits'].sigmoid() >= .5
        summary = row['summary']
        assert summary['selected_endpoint_positive_targets'] == int(target.sum())
        assert summary['selected_endpoint_negative_targets'] == int((~target).sum())
        assert summary['selected_endpoint_positive_met'] == int((target & positive_prediction).sum())
        assert summary['selected_endpoint_negative_met'] == int((~target & ~positive_prediction).sum())
        assert not {'true_positive', 'false_positive', 'true_negative', 'false_negative'} & summary.keys()
    assert 'oracle' in result['diagnostic'].lower()
    buffer = io.BytesIO()
    torch.save(result, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    torch.testing.assert_close(loaded['paths']['free_sequence']['best_shift'],
                               result['paths']['free_sequence']['best_shift'])
    for k, v in model.state_dict().items():
        assert torch.equal(v, state[k])
    assert flags == [(p.requires_grad, p.grad) for p in model.parameters()]
    assert torch.equal(first_parameter.grad, grad_before)
    assert modes == [m.training for m in model.modules()]


def test_invalid_inputs_are_rejected_without_changing_model(case):
    model, h, valid, basis, actual, y, enabled = case
    with pytest.raises(ValueError):
        probe(model, h, valid, basis, actual, torch.full_like(y, .5),
              enabled=enabled, threshold=.5, steps=2)
    bad = basis.clone()
    bad[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        probe(model, h, valid, bad, actual, y, enabled=enabled, threshold=.5, steps=2)
    with pytest.raises(ValueError):
        risk_with_shift(model, h, valid.float(), torch.zeros(3, 16, 64))
    with pytest.raises(ValueError):
        probe(model, h, valid, basis, actual, y, enabled=enabled.float(),
              threshold=.5, steps=2)
    model.memory.output.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        probe(model, h, valid, basis, actual, y, enabled=enabled, threshold=.5, steps=2)
    assert model.memory.output.weight.requires_grad
    model.memory.output.weight.requires_grad_(False)
    model.train()
    with pytest.raises(ValueError):
        probe(model, h, valid, basis, actual, y, enabled=enabled, threshold=.5, steps=2)
    assert model.training


def test_zero_radius_and_no_grad_caller_preserve_zero_shifts(case):
    model, h, valid, basis, actual, y, enabled = case
    with torch.no_grad():
        model.memory.output.weight.zero_()
        result = probe(model, h, valid, basis, actual, y, enabled=enabled,
                       threshold=.5, steps=2)
    assert not result['radius'].count_nonzero()
    for row in result['paths'].values():
        assert not row['best_shift'].count_nonzero()
        torch.testing.assert_close(row['best_logits'], result['zero_logits'], atol=1e-6, rtol=1e-6)


def test_replay_is_consistent_across_batch_slices(case):
    model, h, valid, basis, actual, y, enabled = case
    result = probe(model, h, valid, basis, actual, y, enabled=enabled,
                   threshold=.4, steps=2)
    for row in result['paths'].values():
        shift = row['best_shift']
        with torch.no_grad():
            replay = torch.cat([risk_with_shift(model, h[i:i+1], valid[i:i+1], shift[i:i+1])
                                for i in range(len(h))])
        torch.testing.assert_close(replay, row['best_logits'], atol=1e-6, rtol=1e-6)
        h33, valid33, shift33 = h.repeat(11, 1, 1), valid.repeat(11, 1), shift.repeat(11, 1, 1)
        with torch.no_grad():
            all_at_once = risk_with_shift(model, h33, valid33, shift33)
            batch32 = torch.cat([risk_with_shift(model, h33[i:i+32], valid33[i:i+32],
                                                 shift33[i:i+32]) for i in (0, 32)])
        torch.testing.assert_close(all_at_once, batch32, atol=1e-6, rtol=1e-6)


def test_exactly_two_starts_and_requested_adam_steps(case, monkeypatch):
    model, h, valid, basis, actual, y, enabled = case
    initial, updates = [], []
    original_adam = torch.optim.Adam

    class CountingAdam(original_adam):
        def __init__(self, params, **kwargs):
            params = list(params)
            initial.append(params[0].detach().clone())
            updates.append(0)
            self.slot = len(updates) - 1
            super().__init__(params, **kwargs)

        def step(self, *args, **kwargs):
            updates[self.slot] += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, 'Adam', CountingAdam)
    result = probe(model, h, valid, basis, actual, y, enabled=enabled,
                   threshold=.5, steps=2, lr=.1)
    assert updates == [2] * 6
    for index in (0, 2, 4):
        assert not initial[index].count_nonzero()
    torch.testing.assert_close(initial[1], actual * enabled[:, None])
    torch.testing.assert_close(initial[3], result['paths']['context8']['best_shift'][:, -1],
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(initial[5], result['paths']['free_last']['best_shift'],
                               atol=1e-6, rtol=1e-6)


def test_float64_and_integer_binary_labels(case):
    model, h, valid, basis, actual, y, enabled = case
    model = model.double()
    h, basis, actual = h.double(), basis.double(), actual.double()
    result = probe(model, h, valid, basis, actual, y.long(), enabled=enabled,
                   threshold=.5, steps=2)
    for row in result['paths'].values():
        assert row['best_shift'].dtype == torch.float64
        assert row['best_logits'].dtype == torch.float64
        assert torch.isfinite(row['best_loss']).all()


def test_progress_write_failure_propagates(case):
    model, h, valid, basis, actual, y, enabled = case
    events = []

    def failed_writer(event):
        events.append(event)
        raise OSError('evidence write failed')

    with pytest.raises(OSError, match='evidence write failed'):
        probe(model, h[:1], valid[:1], basis[:1], actual[:1], y[:1],
              enabled=enabled[:1], threshold=.5, steps=20, progress=failed_writer)
    assert len(events) == 1
    assert (events[0]['path'], events[0]['init'], events[0]['step']) == ('context8', 0, 20)
