import numpy as np
import pytest

from experiments.risk_aligned_online.context_vector_20261030.source_diagnose import (
    balanced_shot_weights, standardize, feedback_controls, grouped_losses, probes,
)


def records():
    return [dict(shot=1, group='safe', label=0.)] * 3 + [
        dict(shot=2, group='safe', label=0.), dict(shot=3, group='danger', label=1.)]


def test_balancing_does_not_give_long_shots_more_weight():
    w = balanced_shot_weights(records())
    assert w[:3].sum() == pytest.approx(.25)
    assert w[3] == pytest.approx(.25)
    assert w[4] == pytest.approx(.5)
    with pytest.raises(ValueError, match='both'):
        balanced_shot_weights(records()[:4])


def test_standardization_uses_training_statistics_only():
    train = np.array([[1., 7.], [3., 7.]])
    held = np.array([[100., 8.]])
    a, b, center, scale = standardize(train, held, np.array([.5, .5]))
    np.testing.assert_array_equal(center, [2., 7.])
    np.testing.assert_array_equal(scale, [1., 1.])
    np.testing.assert_array_equal(a, [[-1., 0.], [1., 0.]])
    np.testing.assert_array_equal(b, [[98., 1.]])


def test_feedback_control_keeps_origin_and_moves_whole_blocks():
    x = np.arange(358., dtype=float)[None]
    mean = np.arange(358., dtype=float)*-1
    reverse, average = feedback_controls(x, mean)
    np.testing.assert_array_equal(reverse[:, :128], x[:, :128])
    np.testing.assert_array_equal(reverse[:, 128:174], x[:, -46:])
    np.testing.assert_array_equal(average[:, :128], x[:, :128])
    np.testing.assert_array_equal(average[:, 128:], mean[None, 128:])


def test_group_summary_is_shot_weighted_and_rejects_nan():
    rec = records()
    pred = np.array([1., 1., 1., -1., 1.])
    result = grouped_losses(rec, np.zeros(5), pred)
    expected = (np.logaddexp(0., 1.)+np.logaddexp(0., -1.))/2-np.log(2.)
    assert result['groups']['safe']['delta_bce'] == pytest.approx(expected)
    assert result['groups']['early']['shots'] == 0
    pred[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        grouped_losses(rec, np.zeros(5), pred)


def test_probe_parameters_do_not_depend_on_held_features_or_labels():
    rng = np.random.default_rng(1)
    rec = [dict(shot=i, group='danger' if i % 2 else 'safe', label=float(i % 2)) for i in range(24)]
    data = dict(context=rng.normal(size=(24, 358)), maturity=rng.normal(size=(24, 128)),
                origin_logit=np.zeros(24), maturity_logit=np.zeros(24), query_logit=np.zeros(24))
    first, _ = probes(data, data, rec, rec)
    changed = {k: v+3. for k, v in data.items()}
    held = [dict(shot=r['shot'], group='safe' if r['label'] else 'danger', label=1-r['label']) for r in rec]
    second, _ = probes(data, changed, rec, held)
    assert first['fit'] == second['fit']
    assert set(first['results']) == {'origin', 'feedback', 'combined', 'maturity',
        'combined_reverse', 'combined_mean', 'origin_logit', 'maturity_logit', 'query_logit'}
