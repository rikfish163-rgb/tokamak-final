"""Contract tests for report-only aggregation; no model or GPU is loaded."""
import hashlib
import json

import numpy as np
import pytest

from experiments.risk_aligned_online import report_run as R


def _metrics(states, *, threshold=.3):
    predictions = []
    for shot, is_disrupt, timely, alarm in states:
        predictions.append(dict(shot=shot, is_disrupt=is_disrupt, timely=timely,
                                alarm=alarm, lead_ms=50. if timely else None))
    nd = sum(p['is_disrupt'] for p in predictions)
    ns = len(predictions)-nd
    tp = sum(p['is_disrupt'] and p['timely'] for p in predictions)
    fp = sum(not p['is_disrupt'] and p['alarm'] for p in predictions)
    tpr = tp/nd if nd and ns else None
    fpr = fp/ns if nd and ns else None
    return dict(predictions=predictions, tpr=tpr, fpr=fpr,
                J=tpr-fpr if tpr is not None and fpr is not None else None,
                tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, threshold=threshold)


def _run(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    checkpoint = source/'source.pt'
    checkpoint.write_bytes(b'synthetic checkpoint')
    (source/'config.json').write_text(json.dumps(dict(evidence='SMOKE', mode='meta')))
    (source/'test220.json').write_text(json.dumps(dict(evidence='SMOKE', jtext=dict(tpr=.5, fpr=.2))))
    arms = ('static', 'intra', 'inter', 'both')
    (tmp_path/'config.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        arms=arms, domain='EAST', threshold=.3, evidence='DIAGNOSTIC_PREFIX', limit=4)))
    order = [dict(shot=i) for i in (1, 2, 3, 4)]
    (tmp_path/'order.json').write_text(json.dumps(dict(full_order=order, selected_prefix=order)))
    static = [(1, True, True, True), (2, True, False, False),
              (3, False, False, True), (4, False, False, False)]
    intra = [(1, True, True, True), (2, True, True, True),
             (3, False, False, False), (4, False, False, True)]
    both = [(1, True, False, True), (2, True, True, True),
            (3, False, False, False), (4, False, False, False)]
    summary = {arm: dict(parent_hash='same-parent', deployed=_metrics(states),
                         same_parent_no_fast_shadow=_metrics(static))
               for arm, states in [('static', static), ('intra', intra),
                                   ('inter', static), ('both', both)]}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    return summary


def test_report_counts_paired_and_static_transitions_and_provenance(tmp_path):
    _run(tmp_path)
    comparison, markdown = R.build_report(tmp_path)
    pair = comparison['comparisons']['intra']['same_parent_no_fast_shadow']
    assert pair['rescued_disrupt_timely'] == 1
    assert pair['harmed_disrupt_timely'] == 0
    assert pair['net_disrupt_timely'] == 1
    assert pair['avoided_safe_fp'] == 1
    assert pair['new_safe_fp'] == 1
    assert pair['delta_J'] == pytest.approx(.5)
    assert comparison['comparisons']['both']['same_parent_no_fast_shadow']['harmed_disrupt_timely'] == 1
    assert comparison['comparisons']['both']['static_deployed']['harmed_disrupt_timely'] == 1
    assert comparison['arms']['both']['deployed']['fpr_le_0_20'] is True
    assert comparison['arms']['static']['deployed']['fpr_le_0_20'] is False
    assert comparison['provenance']['source_test220_sha256']
    assert comparison['provenance']['online_evidence'] == 'DIAGNOSTIC_PREFIX'
    assert comparison['provenance']['source_evidence'] == 'SMOKE'
    assert '长期状态可能不同' in markdown
    assert '不证明增益' in markdown
    assert '源验收（TPR≥0.9 且 FPR<0.20）未通过。' in markdown


def test_cli_writes_only_report_outputs(tmp_path):
    _run(tmp_path)
    R.main(['--run', str(tmp_path)])
    assert (tmp_path/'RESULTS.md').is_file()
    assert json.loads((tmp_path/'comparison.json').read_text())['roster'] == [1, 2, 3, 4]


@pytest.mark.parametrize('change', ['parent', 'duplicate', 'missing', 'wrong_identity', 'wrong_metric'])
def test_report_rejects_inconsistent_run(tmp_path, change):
    summary = _run(tmp_path)
    if change == 'parent':
        summary['both']['parent_hash'] = 'other-parent'
    elif change == 'duplicate':
        summary['intra']['deployed']['predictions'][1]['shot'] = 1
    elif change == 'missing':
        summary['intra']['deployed']['predictions'].pop()
    elif change == 'wrong_identity':
        summary['intra']['deployed']['predictions'][0]['is_disrupt'] = False
    else:
        summary['intra']['deployed']['fpr'] = 0.
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        R.build_report(tmp_path)


def test_single_class_metrics_stay_null_not_zero(tmp_path):
    summary = _run(tmp_path)
    only_disrupt = [(1, True, True, True), (2, True, False, False)]
    for arm in summary.values():
        arm['deployed'] = _metrics(only_disrupt)
        arm['same_parent_no_fast_shadow'] = _metrics(only_disrupt)
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    order = [dict(shot=i) for i in (1, 2)]
    (tmp_path/'order.json').write_text(json.dumps(dict(full_order=order, selected_prefix=order)))
    config_path = tmp_path/'config.json'
    config = json.loads(config_path.read_text())
    config['limit'] = 2
    config_path.write_text(json.dumps(config))
    comparison, _ = R.build_report(tmp_path)
    assert comparison['arms']['static']['deployed']['fpr'] is None
    assert comparison['arms']['static']['deployed']['J'] is None
    assert comparison['arms']['static']['deployed']['fpr_le_0_20'] is None
    assert comparison['comparisons']['intra']['same_parent_no_fast_shadow']['delta_J'] is None


def test_jtext_report_counts_inter_only_change_without_sealed_parse(tmp_path, monkeypatch):
    summary = _run(tmp_path)
    config_path = tmp_path/'config.json'
    config = json.loads(config_path.read_text())
    config['domain'] = 'J-TEXT'
    config_path.write_text(json.dumps(config))
    summary['inter']['deployed']['predictions'][1].update(timely=True, alarm=True, lead_ms=50.)
    summary['both']['deployed']['predictions'][0].update(timely=True, alarm=True, lead_ms=50.)
    summary['inter']['deployed'].update(tp=2, tpr=1., J=.5)
    summary['both']['deployed'].update(tp=2, tpr=1., J=1.)
    summary['both']['same_parent_no_fast_shadow'] = summary['inter']['deployed']
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(R, '_early_alarm_diagnostic',
                        lambda *args: (_ for _ in ()).throw(AssertionError('sealed parse')))
    result, markdown = R.build_report(tmp_path)
    assert result['comparisons']['inter']['static_deployed']['net_disrupt_timely'] == 1
    assert result['comparisons']['both']['same_parent_no_fast_shadow']['net_disrupt_timely'] == 0
    assert '炮间轨迹相对 static' in markdown


@pytest.mark.parametrize('all_disrupt', [True, False])
def test_runner_single_class_sets_both_rates_null(tmp_path, all_disrupt):
    summary = _run(tmp_path)
    states = ([(1, True, True, True), (2, True, False, False)] if all_disrupt else
              [(1, False, False, True), (2, False, False, False)])
    for arm in summary.values():
        for key in ('deployed', 'same_parent_no_fast_shadow'):
            metrics = _metrics(states)
            metrics.update(tpr=None, fpr=None, J=None)
            arm[key] = metrics
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    order = [dict(shot=i) for i in (1, 2)]
    (tmp_path/'order.json').write_text(json.dumps(dict(full_order=order, selected_prefix=order)))
    config_path = tmp_path/'config.json'
    config = json.loads(config_path.read_text())
    config['limit'] = 2
    config_path.write_text(json.dumps(config))
    result, _ = R.build_report(tmp_path)
    metrics = result['arms']['static']['deployed']
    assert (metrics['tpr'], metrics['fpr'], metrics['J']) == (None, None, None)
    assert metrics['fpr_le_0_20'] is None
    assert metrics['tp'] == (1 if all_disrupt else 0)
    assert metrics['fp'] == (0 if all_disrupt else 1)


def test_limit_zero_rejects_shortened_selected_prefix(tmp_path):
    summary = _run(tmp_path)
    for arm in summary.values():
        for key in ('deployed', 'same_parent_no_fast_shadow'):
            rows = arm[key]['predictions'][:3]
            arm[key] = _metrics([(r['shot'], r['is_disrupt'], r['timely'], r['alarm']) for r in rows])
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    order_path = tmp_path/'order.json'
    order = json.loads(order_path.read_text())
    order['selected_prefix'] = order['selected_prefix'][:3]
    order_path.write_text(json.dumps(order))
    config_path = tmp_path/'config.json'
    config = json.loads(config_path.read_text())
    config['limit'] = 0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='full-order'):
        R.build_report(tmp_path)


def test_postevent_only_threshold_hit_is_not_an_alarm(tmp_path):
    summary = {arm: dict(deployed=dict(predictions=[dict(shot=1, is_disrupt=True,
                           alarm=False, timely=False, lead_ms=None)])) for arm in ('intra', 'both')}
    for arm in ('intra', 'both'):
        directory = tmp_path/arm
        directory.mkdir()
        np.savez_compressed(directory/'1_sealed.npz', times=np.array([0., .001, .002]),
                            scores=np.array([.1, .1, .8]))
        (directory/'1_audit.json').write_text(json.dumps(dict(fast_updates=[dict(effective_from=2)])))
    result = R._early_alarm_diagnostic(tmp_path, summary, [1], .5)
    for arm in ('intra', 'both'):
        assert result[arm]['alarmed'] == 0
        assert result[arm]['unverifiable_postevent_hit_shots'] == [1]
        assert result[arm]['down_time_independently_verified'] is False
