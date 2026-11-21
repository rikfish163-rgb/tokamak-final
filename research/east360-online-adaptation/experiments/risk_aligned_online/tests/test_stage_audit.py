import copy

import pytest

from experiments.risk_aligned_online.stage_audit import analyze_rows


def rows():
    return {
        'static': [dict(shot=10, disruptive=True, peak=.8),
                   dict(shot=11, disruptive=False, peak=.9),
                   dict(shot=12, disruptive=True, peak=.6),
                   dict(shot=13, disruptive=False, peak=.5)],
        'fixed': [dict(shot=10, disruptive=True, peak=.6),
                  dict(shot=11, disruptive=False, peak=.5),
                  dict(shot=12, disruptive=True, peak=.8),
                  dict(shot=13, disruptive=False, peak=.7)],
    }


def test_stages_cover_all_shots_and_count_harm_as_well_as_benefit():
    out = analyze_rows(rows(), .7, [2, 4])
    assert out['overall']['static']['tp'] == out['overall']['fixed']['tp'] == 1
    assert out['overall']['flips'] == dict(tp_gained=1, tp_lost=1,
                                          fp_removed=1, fp_added=1)
    assert out['stages'][0]['indices'] == [1, 2]
    assert out['stages'][1]['shot_bounds'] == [12, 13]
    assert out['stages'][0]['fixed']['J'] == 0
    assert out['stages'][1]['fixed']['J'] == 0


def test_single_class_stage_has_no_invented_rate():
    out = analyze_rows(rows(), .7, [1, 4])
    assert out['stages'][0]['static']['fpr'] is None
    assert out['stages'][0]['static']['J'] is None


@pytest.mark.parametrize('ends', [[2, 3], [0, 4], [2, 2, 4], [4, 3], [5], []])
def test_invalid_partitions_rejected(ends):
    with pytest.raises(ValueError):
        analyze_rows(rows(), .7, ends)


@pytest.mark.parametrize('bad', ['order', 'label', 'duplicate', 'probability'])
def test_invalid_comparison_rejected(bad):
    r = copy.deepcopy(rows())
    if bad == 'order':
        r['fixed'] = r['fixed'][::-1]
    elif bad == 'label':
        r['fixed'][0]['disruptive'] = False
    elif bad == 'duplicate':
        for arm in r:
            r[arm][1]['shot'] = r[arm][0]['shot']
    else:
        r['fixed'][0]['peak'] = float('nan')
    with pytest.raises(ValueError):
        analyze_rows(r, .7, [4])


def test_missing_timely_opportunity_counts_miss_not_removed_denominator():
    r = rows()
    r['static'][0]['peak'] = None
    out = analyze_rows(r, .7, [4])
    assert out['overall']['static']['tp'] == 0
    assert out['overall']['static']['n_disrupt'] == 2
