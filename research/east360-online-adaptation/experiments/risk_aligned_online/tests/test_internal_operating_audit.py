import pytest

from experiments.risk_aligned_online.internal_operating_audit import summarize


def rows(positive, negative):
    return [dict(shot=i, disruptive=True, peak=p) for i, p in enumerate(positive)] + [
        dict(shot=100+i, disruptive=False, peak=p) for i, p in enumerate(negative)]


def test_uniform_score_shift_is_not_ranking_gain():
    baseline = rows([.9, .6], [.8, .4])
    candidate = rows([.7, .4], [.6, .2])
    report = summarize({'static': baseline, 'intra': candidate}, .65)
    assert report['original']['static']['fp'] == 1
    assert report['original']['intra']['fp'] == 0
    assert report['matched_static']['intra']['tp'] == 1
    assert report['ranking_auc']['intra'] == report['ranking_auc']['static']


def test_same_fp_can_detect_genuine_timely_ranking_difference():
    baseline = rows([.9, .6], [.8, .4])
    candidate = rows([.9, .85], [.8, .4])
    report = summarize({'static': baseline, 'intra': candidate}, .82)
    assert report['original']['intra']['tp'] == 2
    assert report['matched_static']['intra']['tp'] == 1
    assert report['ranking_auc']['intra'] > report['ranking_auc']['static']


def test_mismatched_or_duplicate_identity_is_not_a_paired_comparison():
    baseline = rows([.9], [.4])
    changed = [dict(baseline[0], shot=2), baseline[1]]
    with pytest.raises(ValueError, match='identity'):
        summarize({'static': baseline, 'intra': changed}, .5)
    with pytest.raises(ValueError, match='identity'):
        summarize({'static': baseline + baseline}, .5)

