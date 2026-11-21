import numpy as np
import pytest

from experiments.risk_aligned_online.arrival_increment_20261030.run import (
    ARMS, evaluate_shot, gate, neural_scores, require_same_parent,
)


def fixture_arrays():
    data = dict(ids=np.array([7, 7]), ends=np.array([127, 152]),
                origin_time=np.array([.87, .90]), arrival_time=np.array([.895, .925]),
                future_times=np.array([[.926, .970, .971], [.951, .976, .977]]),
                baseline_logits=np.zeros((2, 3)))
    for arm in ARMS:
        data[arm+'_logits'] = np.zeros((2, 3))
    data['real_logits'][:] = .1
    return data


def test_safe_two_dimensional_labels_and_exact_bce():
    rows = evaluate_shot(fixture_arrays(), 7, 0, False, None)
    real = next(r for r in rows if r['arm'] == 'real')
    assert real['group'] == 'safe'
    assert real['timepoints'] == 6
    assert real['delta_bce'] == pytest.approx(np.logaddexp(0, .1)-np.log(2))


def test_strict_thirty_and_original_window_intersection():
    rows = evaluate_shot(fixture_arrays(), 7, 0, True, 1.)
    by_group = {r['group']: r for r in rows if r['arm'] == 'real'}
    assert by_group['danger_timely']['timepoints'] == 2
    assert by_group['danger_origin_intersection']['timepoints'] == 1
    assert by_group['danger_late']['timepoints'] == 4
    assert by_group['danger_timely']['delta_bce'] < 0


def test_noop_cannot_change_neural_score():
    data = fixture_arrays()
    data['identical_logits'][0, 0] = .001
    with pytest.raises(AssertionError, match='uninformative'):
        evaluate_shot(data, 7, 0, False, None)


def test_evidence_potentials_are_not_added_to_output():
    estimates = dict(paired_increment=np.array([2., 3.]),
                     arrival_increment=np.array([.5, .6]),
                     constant_logit=np.array([1., 2.]), q0_logit=np.array([0., 0.]))
    scores = neural_scores(estimates, estimates, estimates)
    assert set(scores) == set(ARMS)
    np.testing.assert_array_equal(scores['real'], [[0., .5], [0., .6]])
    np.testing.assert_array_equal(scores['constant'], [[0., 1.], [0., 2.]])
    assert not scores['identical'].any()


def passing_groups():
    groups = {}
    for arm in ARMS:
        groups[arm] = {g: dict(shots=3, improved=2, mean_delta_bce=-.02 if arm == 'real' else -.01)
                      for g in ('safe', 'danger_timely', 'danger_origin_intersection', 'early_normal')}
    return groups


def test_gate_requires_controls_intersection_and_safe_improvement():
    groups = passing_groups()
    assert gate(groups)[0]
    groups['real']['safe']['mean_delta_bce'] = 0.
    assert not gate(groups)[0]
    groups = passing_groups()
    groups['reverse']['danger_origin_intersection']['mean_delta_bce'] = -.03
    assert not gate(groups)[0]
    groups = passing_groups()
    groups['real']['early_normal']['mean_delta_bce'] = .001
    assert not gate(groups)[0]


def test_absent_group_is_not_zero_or_success():
    groups = passing_groups()
    del groups['real']['safe']
    assert not gate(groups)[0]


def test_parent_guard_compares_matched_neural_batches_exactly():
    old = dict(origin_logit=np.array([.123], dtype=np.float32),
               baseline_logits=np.zeros((1, 25), dtype=np.float32))
    new = {k: v.copy() for k, v in old.items()}
    require_same_parent(new, old)
    new['origin_logit'][0] += 1e-7
    with pytest.raises(AssertionError, match='origin_logit'):
        require_same_parent(new, old)
    new = {k: v.copy() for k, v in old.items()}
    new['baseline_logits'][0, 0] = 1e-8
    with pytest.raises(AssertionError, match='baseline_logits'):
        require_same_parent(new, old)
