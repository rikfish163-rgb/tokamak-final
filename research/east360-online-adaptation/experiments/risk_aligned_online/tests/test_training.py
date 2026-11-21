import numpy as np
import torch
import pytest

from experiments.risk_aligned_online import train_source as T
from experiments.risk_aligned_online.model import ModelConfig, RiskAlignedModel


def shot(positive):
    n = 220
    values = np.random.default_rng(3).normal(size=(n, 23)).astype('float32')
    available = np.ones_like(values, bool)
    return dict(shot=int(positive), device='J-TEXT', values=values, available=available,
                times=np.arange(n)/1000, rate=1000, valid=np.ones(n, bool),
                y=np.full(n, float(positive), np.float32),
                y125=np.full(n, float(positive), np.float32), is_disrupt=positive,
                down=.3 if positive else None)


def test_episode_query_follows_support_target_and_masks_dropout():
    rng = np.random.default_rng(4)
    ep = T.sample_episode([shot(True), shot(False)], rng, horizon=4, window_length=16,
                          mode='meta', force_index=0, force_origin=50, dropout_probability=1.)
    assert ep['query_end'] > ep['support_end'] + 4
    assert ep['support_target'].shape == (4, 23)
    assert ep['support_mask'].any()
    assert ep['level_drop'].all()


def test_meta_training_step_has_finite_second_order_gradients():
    torch.manual_seed(2)
    cfg = ModelConfig(window=16, patch=4, dim=16, heads=4, rank=4, horizon=4)
    model = RiskAlignedModel(cfg, np.zeros(23), np.ones(23))
    ep = T.sample_episode([shot(True), shot(False)], np.random.default_rng(5),
                          horizon=4, window_length=16, mode='meta')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    result = T.training_step(model, optimizer, [ep], mode='meta', device='cpu')
    assert np.isfinite(result['loss'])
    assert np.isfinite(result['gate'])
    assert np.isfinite(result['meta_improvement'])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_joint_training_step_is_finite():
    cfg = ModelConfig(window=16, patch=4, dim=16, heads=4, rank=4, horizon=4)
    model = RiskAlignedModel(cfg, np.zeros(23), np.ones(23))
    ep = T.sample_episode([shot(True), shot(False)], np.random.default_rng(5),
                          horizon=4, window_length=16, mode='joint')
    result = T.training_step(model, torch.optim.AdamW(model.parameters(), lr=1e-4),
                             [ep], mode='joint', device='cpu')
    assert np.isfinite(result['loss'])


def test_joint_and_meta_use_identical_query_time():
    cases = [shot(True), shot(False)]
    joint = T.sample_episode(cases, np.random.default_rng(1), horizon=4,
                             window_length=16, mode='joint', force_index=0, force_origin=50)
    meta = T.sample_episode(cases, np.random.default_rng(1), horizon=4,
                            window_length=16, mode='meta', force_index=0, force_origin=50)
    assert joint['query_end'] == meta['query_end'] == 55


def test_static_scoring_never_calls_adapt(monkeypatch):
    cfg = ModelConfig(window=16, patch=4, dim=16, heads=4, rank=4, horizon=4)
    model = RiskAlignedModel(cfg, np.zeros(23), np.ones(23))
    case = shot(False)
    monkeypatch.setattr(T, 'adapt', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('adapt')))
    row = T._score_shot(model, case, device='cpu')
    assert len(row['times']) == len(case['times'])
    assert row['scores'].shape == (len(case['times']), 2)


def test_infeasible_threshold_retains_best_tpr_and_is_not_silent():
    rows = [dict(shot=1, times=np.array([0., .1]), scores=np.array([[.1, .1], [.7, .7]]),
                 is_disrupt=True, down=.2),
            dict(shot=2, times=np.array([0., .1]), scores=np.array([[.2, .2], [.8, .8]]),
                 is_disrupt=False, down=None),
            dict(shot=3, times=np.array([.195]), scores=np.array([[.1, .1]]),
                 is_disrupt=True, down=.2)]
    selected = T._select(rows, 0, 10., True)
    assert selected['selection_feasible'] is False
    assert selected['tpr'] == 0.
    assert selected['fpr'] == 0.
    assert selected['target_tpr'] == .95


def test_source_acceptance_is_strict_on_fpr_and_independent_per_split():
    assert T._source_acceptance(dict(tpr=.9, fpr=.199))
    assert not T._source_acceptance(dict(tpr=.9, fpr=.2))
    assert not T._source_acceptance(dict(tpr=.89, fpr=0.))
    assert not T._source_acceptance(dict(tpr=None, fpr=0.))


def test_feasible_epoch_beats_infeasible_even_if_latter_has_lower_fpr():
    feasible = dict(selection_feasible=True, tpr=.9, fpr=.19)
    infeasible = dict(selection_feasible=False, tpr=.5, fpr=0.)
    aux = dict(selection_feasible=True, tpr=.9, fpr=.2)
    assert T._validation_rank(feasible, aux) < T._validation_rank(infeasible, aux)


def test_infeasible_selection_calls_summary_only_for_shot_maxima(monkeypatch):
    rows = [dict(shot=1, times=np.linspace(0., .3, 1001),
                 scores=np.tile([[.4, .4]], (1001, 1)), is_disrupt=True, down=.3),
            dict(shot=2, times=np.linspace(0., .3, 1001),
                 scores=np.tile([[.2, .2]], (1001, 1)), is_disrupt=False, down=None),
            dict(shot=3, times=np.array([.295]), scores=np.array([[.1, .1]]),
                 is_disrupt=True, down=.3)]
    original = T.data.P.summarize
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(T.data.P, 'summarize', counted)
    selected = T._select(rows, 0, 10., True)
    assert not selected['selection_feasible']
    assert len(calls) <= len(rows) + 1


def test_short_record_cannot_supply_future_window_and_later_query():
    case = shot(True)
    for key in ('values', 'available', 'times', 'valid', 'y', 'y125'):
        case[key] = case[key][:3]
    with pytest.raises(ValueError, match='no valid source meta episode'):
        T.sample_episode([case], np.random.default_rng(4), horizon=4,
                         window_length=16, force_index=0)


def test_episode_includes_true_startup_supervision_before_first_fast_update():
    case = shot(True)
    case['y'][:26] = 0
    ep = T.sample_episode([case], np.random.default_rng(9), horizon=25,
                          force_index=0, force_origin=80)
    assert 0 <= ep['prefix_end'] <= 25
    assert ep['prefix_y'] == 0
    assert ep['prefix_y125'] == case['y125'][ep['prefix_end']]
    assert ep['prefix_available'].sum() == (ep['prefix_end'] + 1) * 23


def test_selection_does_not_trade_bad_deployment_head_for_better_source_head():
    good_j = dict(selection_feasible=True, tpr=.9, fpr=.06)
    bad_aux = dict(selection_feasible=True, tpr=.9, fpr=.575)
    balanced_j = dict(selection_feasible=True, tpr=.9, fpr=.125)
    balanced_aux = dict(selection_feasible=True, tpr=.9, fpr=.1875)
    assert T._validation_rank(balanced_j, balanced_aux) < T._validation_rank(good_j, bad_aux)


def test_val_selection_hard_fpr_tradeoff_and_strict_ten():
    rows = [dict(shot=i, times=np.array([0.]), scores=np.array([[score, score]]),
                 is_disrupt=True, down=.02) for i, score in enumerate([.9]*9+[.5])]
    rows += [dict(shot=100+i, times=np.array([0.]), scores=np.array([[.6, .6]]),
                  is_disrupt=False, down=None) for i in range(10)]
    selected = T._select(rows, 0, 10., True, target_tpr=.95)
    assert selected['tpr'] == .9 and selected['fpr'] == 0
    assert not selected['selection_feasible']
    assert T._source_acceptance(selected)
    assert T._select(rows, 0, 10., True, target_tpr=.9)['selection_feasible']
    rows[0]['down'] = .01
    assert T._select(rows, 0, 10., True)['tpr'] == .8


def test_training_sampling_preserves_rare_labels_and_hard_negative_candidates():
    case = shot(True)
    case['y'][:] = 0
    case['y'][-2:] = 1
    case['y125'][:] = 0
    case['y125'][-50:] = 1
    cases = [case, shot(False)]
    pools = T.build_sampling_pools(cases, horizon=4)
    rng = np.random.default_rng(88)
    episodes = [T.sample_episode(cases, rng, horizon=4, window_length=16,
                                sampling_pools=pools) for _ in range(90)]
    assert sum(e['y'] > .5 for e in episodes) >= 15
    assert any(e['y125'] > .5 and e['y'] == 0 for e in episodes)
    assert all(e['negative_values'].shape == (4, 16, 23) for e in episodes)
    assert case['y'].sum() == 2


def test_test_reporting_does_not_select_threshold(monkeypatch):
    rows = [dict(shot=1, times=np.array([0.]), scores=np.array([[.4,.4]]),
                 is_disrupt=True, down=.02),
            dict(shot=2, times=np.array([0.]), scores=np.array([[.1,.1]]),
                 is_disrupt=False, down=None)]
    monkeypatch.setattr(T, '_select', lambda *a, **k: pytest.fail('TEST threshold selection'))
    metrics = T._summarize(rows, 0, .5, 10., True)
    assert not T._source_acceptance(metrics)
    assert metrics['threshold'] == .5


def test_exact_ten_ms_is_not_timely_after_float_roundoff():
    rows = [dict(shot=1, times=np.array([.09]), scores=np.array([[.8, .8]]),
                 is_disrupt=True, down=.1),
            dict(shot=2, times=np.array([0.]), scores=np.array([[.1, .1]]),
                 is_disrupt=False, down=None)]
    assert not T._summarize(rows, 0, .5, 10., True)['predictions'][0]['timely']
    assert T._select(rows, 0, 10., True)['tpr'] == 0
    rows[0]['times'] = np.array([.089999])
    assert T._summarize(rows, 0, .5, 10., True)['predictions'][0]['timely']


def test_source_metrics_with_numpy_clocks_are_json_serializable():
    import json
    rows = [dict(shot=1, times=np.array([.09]), scores=np.array([[.8, .8]]),
                 is_disrupt=True, down=np.float64(.2)),
            dict(shot=2, times=np.array([0.]), scores=np.array([[.1, .1]]),
                 is_disrupt=False, down=None)]
    json.dumps(T._select(rows, 0, 10., True), allow_nan=False)
