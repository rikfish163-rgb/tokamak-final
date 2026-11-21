import numpy as np
import pytest

from experiments.risk_aligned_online.arrival_reference_20261030.run import ARMS, gate, score_shot


def example():
    data = dict(ids=np.array([1, 1]), origin_time=np.array([.87, .90]),
                arrival_time=np.array([.895, .925]),
                future_times=np.array([[.926, .970, .971], [.951, .976, .977]]),
                baseline_logits=np.zeros((2, 3)))
    for arm in ARMS:
        data[arm+'_logits'] = np.zeros((2, 3))
    data['real_logits'][:] = .1
    return data


def test_stage_timing_and_original_window_are_separate():
    rows = score_shot(example(), 1, 0, True, 1.)
    groups = {r['group']: r for r in rows if r['arm'] == 'real'}
    assert groups['danger_timely']['timepoints'] == 2
    assert groups['danger_origin_intersection']['timepoints'] == 1
    assert groups['danger_late']['timepoints'] == 4


def test_safe_exact_bce_and_zero_control():
    data = example()
    rows = score_shot(data, 1, 0, False, None)
    r = next(r for r in rows if r['arm'] == 'real')
    assert r['delta_bce'] == pytest.approx(np.logaddexp(0, .1)-np.log(2))
    data['identical_logits'][0, 0] = 1e-8
    with pytest.raises(AssertionError):
        score_shot(data, 1, 0, False, None)


def passing():
    return {a: {g: dict(shots=3, improved=2, mean_delta_bce=-.02 if a == 'real' else -.01)
                for g in ('safe', 'danger_timely', 'danger_origin_intersection', 'early_normal')}
            for a in ARMS}


def test_conditional_must_outperform_global_mean_and_reference_only():
    groups = passing()
    assert gate(groups)[0]
    for arm in ('mean', 'reference_only', 'reverse', 'misplaced'):
        g = passing()
        g[arm]['safe']['mean_delta_bce'] = -.03
        assert not gate(g)[0]


def test_no_cherry_picking_groups_or_hiding_early_damage():
    g = passing()
    del g['real']['safe']
    assert not gate(g)[0]
    g = passing()
    g['real']['early_normal']['mean_delta_bce'] = .001
    assert not gate(g)[0]
    g = passing()
    g['real']['danger_origin_intersection']['improved'] = 1
    assert not gate(g)[0]
