"""A source-only feedback task must not see its hidden values or future data."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.masked_prefix_feedback import (
    MaskedPrefixModel, patch_mask, predict_shot,
)
from experiments.risk_aligned_online.fast_update import normalized_step


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_model():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23))
    return MaskedPrefixModel(base).eval().requires_grad_(False)


def signal(count=300):
    values = np.random.default_rng(131).normal(size=(count, 23)).astype(np.float32)
    return dict(values=values, available=np.ones_like(values, dtype=bool),
                times=np.arange(count)/1000., shot=1, device='EAST')


def test_four_phases_partition_the_observed_window():
    a = torch.ones(2, 128, 23, dtype=torch.bool)
    a[:, 0, 0] = False
    masks = [patch_mask(a, phase=p) for p in range(4)]
    assert all(m.dtype == torch.bool for m in masks)
    assert torch.equal(torch.stack(masks).sum(0), a.int())
    assert masks[0].sum() == 2*(4*8*23-1)
    with pytest.raises(ValueError): patch_mask(a, phase=4)
    with pytest.raises(ValueError): patch_mask(a[:, :120], phase=0)


def test_hidden_values_cannot_enter_reconstruction_prediction():
    m = make_model()
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    hidden = patch_mask(a, phase=2)
    altered = x.clone(); altered[hidden] += 100.
    p1, target1, mask1 = m.reconstruction(x, a, m.initial_fast(1), phase=2)
    p2, target2, mask2 = m.reconstruction(altered, a, m.initial_fast(1), phase=2)
    torch.testing.assert_close(p1, p2, atol=0, rtol=0)
    assert not torch.equal(target1, target2) and torch.equal(mask1, mask2)
    assert p1.shape == x.shape and mask1.sum() == 4*8*23


def test_target_detached_but_fast_gradient_and_meta_gradient_exist():
    m = make_model()
    m.decoder.requires_grad_(True)
    x = torch.randn(1, 128, 23, requires_grad=True)
    a = torch.ones_like(x, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    _, target, _ = m.reconstruction(x, a, fast, phase=0)
    assert not target.requires_grad
    loss = m.reconstruction_loss(x, a, fast, phase=0)
    gradient, = torch.autograd.grad(loss, fast, create_graph=True)
    assert gradient.norm() > 0
    adapted = normalized_step(fast, fast.detach(), gradient)
    risk = F.binary_cross_entropy_with_logits(m(x.detach(), a, adapted)['logit'], torch.ones(1))
    head_gradients = torch.autograd.grad(risk, tuple(m.decoder.parameters()))
    assert all(torch.isfinite(g).all() for g in head_gradients)
    assert sum(float(g.norm()) for g in head_gradients) > 0
    assert all(p.grad is None for p in m.base.parameters())


def test_missing_targets_do_not_generate_learning_signal():
    m = make_model()
    x = torch.randn(1, 128, 23)
    a = torch.zeros_like(x, dtype=torch.bool)
    fast = m.initial_fast(1).requires_grad_()
    loss = m.reconstruction_loss(x, a, fast, phase=0)
    gradient, = torch.autograd.grad(loss, fast)
    assert float(loss.detach()) == 0 and torch.equal(gradient, torch.zeros_like(gradient))


def test_decoder_never_changes_the_unadapted_risk_path():
    m = make_model()
    x = torch.randn(1, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    fixed_fast = m.base.initial_fast(1).detach().requires_grad_()
    expected = m.base(x, a, fixed_fast)['logit']
    torch.testing.assert_close(m(x, a)['logit'], expected, atol=0, rtol=0)
    with torch.no_grad(): m.decoder.weight.add_(100.)
    torch.testing.assert_close(m(x, a)['logit'], expected, atol=0, rtol=0)


def test_matched_clock_seals_prediction_then_updates_and_resets_each_shot():
    m = make_model()
    initial = {k: v.clone() for k, v in m.state_dict().items()}
    fresh, af = predict_shot(m, signal(), intra=True, lag=0)
    stale, ast = predict_shot(m, signal(), intra=True, lag=25)
    repeat, ar = predict_shot(m, signal(), intra=True, lag=0)
    np.testing.assert_array_equal(fresh, repeat)
    assert af['fast_initial']['l2'] == ast['fast_initial']['l2'] == ar['fast_initial']['l2'] == 0
    assert af['fast_discarded'] and ast['fast_discarded']
    assert af['attempted_updates'] == ast['attempted_updates'] == 2
    for k, audit in ((0, af), (25, ast)):
        assert [r['update_index'] for r in audit['attempts']] == [152, 252]
        for row in audit['attempts']:
            assert row['feedback_end_index'] == row['update_index'] - k
            assert row['predictions_sealed_through_index'] == row['update_index']
            assert row['effective_index'] == row['update_index'] + 1
            if row['accepted']:
                assert row['loss_after'] < row['loss_before']
                assert row['fast_after']['l2'] <= 1.00001
    np.testing.assert_array_equal(fresh[:153], stale[:153])
    for k, v in m.state_dict().items(): torch.testing.assert_close(v, initial[k], atol=0, rtol=0)


def test_future_tail_cannot_change_earlier_outputs_or_feedback():
    m = make_model()
    s = signal()
    changed = deepcopy(s); changed['values'][200:] += 100.
    p1, a1 = predict_shot(m, s, intra=True, lag=0)
    p2, a2 = predict_shot(m, changed, intra=True, lag=0)
    np.testing.assert_array_equal(p1[:200], p2[:200])
    assert a1['attempts'][0] == a2['attempts'][0]


def test_label_access_is_rejected_and_short_shots_are_not_silently_discarded():
    m = make_model()
    s = signal(90)
    scores, audit = predict_shot(m, s, intra=True, lag=0)
    assert scores.shape == (90,) and audit['attempted_updates'] == 0
    with pytest.raises(ValueError, match='outcome'):
        predict_shot(m, dict(s, is_disrupt=True), intra=True, lag=0)
    with pytest.raises(ValueError): predict_shot(m, s, intra=True, lag=26)
