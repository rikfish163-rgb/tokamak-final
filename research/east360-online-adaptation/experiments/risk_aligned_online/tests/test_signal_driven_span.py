import copy

import numpy as np
import pytest
import torch


def test_history_is_equal_shot_and_contrast_removes_class_means():
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import history_spaces
    torch.manual_seed(31)
    g = torch.randn(12, 15, dtype=torch.float64)
    shots = np.repeat(np.arange(6), 2)
    labels = np.repeat([0, 0, 0, 1, 1, 1], 2)
    spaces, receipt = history_spaces(g, shots, labels)
    means = torch.stack([g[shots == i].mean(0) for i in range(6)])
    means /= means.norm(dim=1, keepdim=True)
    centered = means.clone()
    for ids in (slice(0, 3), slice(3, 6)):
        centered[ids] -= centered[ids].mean(0)
    q = spaces['contrast']
    assert len(q) == 4 and receipt['shots'] == list(range(6))
    torch.testing.assert_close(q @ q.T, torch.eye(4, dtype=q.dtype))
    torch.testing.assert_close(centered @ q.T @ q, centered)
    duplicated, _ = history_spaces(g.repeat_interleave(2, 0), shots.repeat(2), labels.repeat(2))
    torch.testing.assert_close(duplicated['contrast'].T @ duplicated['contrast'], q.T @ q)


def test_projected_direction_is_descent_and_zero_has_no_substitute():
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import projected_step
    q = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    g = torch.tensor([2., 3., 8.])
    step, r = projected_step(g, q)
    assert g @ step > 0 and abs(float(step.norm()) - .01) < 1e-8
    assert step[-1] == 0 and r['step_length'] == .01
    zero, r = projected_step(torch.tensor([0., 0., 1.]), q)
    assert not zero.any() and r['step_length'] == 0
    with pytest.raises(ValueError):
        projected_step(torch.tensor([float('nan'), 1., 2.]), q)


def test_future_cannot_change_predictions_and_parent_is_immutable():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    from experiments.risk_aligned_online.history_tangent_20261023.learning import basis_from_rows, permute_basis
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import predict, ARMS
    torch.set_num_threads(2); torch.manual_seed(8)
    model = small_model(); old = copy.deepcopy(model.state_dict()); params = initial_encoder(model)
    q = basis_from_rows(torch.randn(3, sum(v.numel() for v in params.values())))
    spaces = {'global': q, 'contrast': q[:2], 'permuted': permute_basis(q[:2], params)}
    rng = np.random.default_rng(19)
    signal = {'values': rng.normal(size=(330, 23)).astype('float32'), 'available': np.ones((330, 23), bool)}
    first, receipts = predict(model, signal, spaces, device='cpu')
    changed = copy.deepcopy(signal); changed['values'][300:] += 100
    second, _ = predict(model, changed, spaces, device='cpu')
    repeated, _ = predict(model, signal, spaces, device='cpu')
    for a in ('off',) + ARMS:
        np.testing.assert_array_equal(first[a][:300], second[a][:300])
        np.testing.assert_array_equal(first[a], repeated[a])
        np.testing.assert_array_equal(first[a][:256], first['off'][:256])
    for arm in ARMS:
        assert len(receipts[arm]) == 1
        assert receipts[arm][0]['last_observation'] == 255
        assert receipts[arm][0]['first_affected_prediction'] == 256
        assert receipts[arm][0]['gradient_steps'] == 1
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, old[k], rtol=0, atol=0)
    with pytest.raises(ValueError):
        predict(model, dict(signal, outcome=True), spaces, device='cpu')


def test_missing_targets_do_not_become_zero_labels():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import update_all, ARMS
    model = small_model(); params = initial_encoder(model)
    d = sum(v.numel() for v in params.values())
    q = torch.zeros(1, d); q[0, 0] = 1
    signal = {'values': np.zeros((256, 23), 'float32'), 'available': np.zeros((256, 23), bool)}
    states, receipts = update_all(model, signal, 256, {'global': q, 'contrast': q, 'permuted': q}, 'cpu')
    for a in ARMS:
        assert receipts[a]['gradient_steps'] == 0
        for k, v in params.items():
            torch.testing.assert_close(states[a][k], v)


def test_screen_gate_requires_both_groups_and_real_feedback_controls():
    from experiments.risk_aligned_online.signal_driven_span_20261024.run import eligible
    arms = ('off', 'raw', 'global', 'contrast', 'permuted', 'hold', 'reverse')
    r = {a: dict(tp=2, fp=0, n_disrupt=2, n_safe=8,
                 danger_loss=2., safe_loss=.1, balanced_loss=1.05) for a in arms}
    r['contrast'].update(danger_loss=1.8, safe_loss=.09, balanced_loss=.945)
    assert eligible(r)
    bad = copy.deepcopy(r); bad['contrast']['tp'] = 1
    assert not eligible(bad)
    bad = copy.deepcopy(r); bad['contrast']['safe_loss'] = .11
    assert not eligible(bad)
    bad = copy.deepcopy(r); bad['hold']['balanced_loss'] = .945
    assert not eligible(bad)
    bad = copy.deepcopy(r); bad['contrast']['balanced_loss'] = float('nan')
    assert not eligible(bad)


def test_diagnostic_losses_are_equal_shot_and_exclude_tardy_endpoints():
    from experiments.risk_aligned_online.signal_driven_span_20261024.run import diagnostic_losses
    rows = [dict(is_disrupt=True, down=1., times=np.array([.8, .9, .98]), scores=np.array([.2,.5,.99])),
            dict(is_disrupt=True, down=1., times=np.array([.9]), scores=np.array([.25])),
            dict(is_disrupt=False, down=None, times=np.arange(3), scores=np.array([.1,.2,.1]))]
    r = diagnostic_losses(rows)
    assert r['danger_loss'] == pytest.approx((np.log(2)+np.log(4))/2)
    assert r['safe_loss'] == pytest.approx(-np.log(.8))
    assert r['balanced_loss'] == pytest.approx((r['danger_loss']+r['safe_loss'])/2)


def test_long_float32_basis_validation_accumulates_in_double():
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import validate_spaces
    q = torch.full((1, 58240), 58240 ** -.5, dtype=torch.float32)
    spaces = {a:q for a in ('global', 'contrast', 'permuted')}
    validate_spaces(spaces)
    bad = dict(spaces, contrast=q*1.01)
    with pytest.raises(AssertionError):
        validate_spaces(bad)


def test_static_matches_separate_parent_inference_with_same_dispatch():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    from experiments.risk_aligned_online.signal_driven_span_20261024.learning import predict
    from experiments.risk_aligned_online.warmstart_alignment_20261022.run import predict_many
    model = small_model(); n = sum(p.numel() for p in initial_encoder(model).values())
    q = torch.zeros(1, n); q[0, 0] = 1
    signal = dict(values=np.random.default_rng(25).normal(size=(300,23)).astype('float32'),
                  available=np.ones((300,23), bool))
    curves, _ = predict(model, signal, {k:q for k in ('global','contrast','permuted')}, device='cpu')
    with torch.no_grad():
        independent = predict_many(model, {'parent':signal}, device='cpu')['parent']
    np.testing.assert_array_equal(curves['off'], independent)
