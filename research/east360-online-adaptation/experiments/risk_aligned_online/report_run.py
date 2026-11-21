"""Read-only aggregation of a completed four-arm online replay.

Only RESULTS.md and comparison.json are written; no model is loaded and no
threshold, checkpoint, or prediction is changed. Usage:

    python -m experiments.risk_aligned_online.report_run --run RUN_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ('static', 'intra', 'inter', 'both')
METRICS = ('tpr', 'fpr', 'J', 'tp', 'fp', 'n_disrupt', 'n_safe', 'threshold')
ROOT = Path(__file__).resolve().parents[2]


def _read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(raw, run_dir):
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [ROOT/path, run_dir/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'source checkpoint not found: {raw}')


def _ids(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{name} must be a nonempty ordered roster')
    ids = []
    for item in rows:
        if not isinstance(item, dict) or type(item.get('shot')) is not int:
            raise ValueError(f'{name} contains a noninteger shot ID')
        ids.append(item['shot'])
    if len(ids) != len(set(ids)):
        raise ValueError(f'{name} contains duplicate shots')
    return ids


def _equal_number(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(f'{label} null/non-null mismatch')
    elif not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label} does not agree with per-shot predictions')


def _validate_metrics(metrics, roster, identities, threshold, label):
    if not isinstance(metrics, dict):
        raise ValueError(f'{label} metrics missing')
    predictions = metrics.get('predictions')
    if _ids(predictions, label) != roster:
        raise ValueError(f'{label} roster missing, repeated, reordered, or mismatched')
    for row in predictions:
        sid = row['shot']
        if any(type(row.get(key)) is not bool for key in ('is_disrupt', 'timely', 'alarm')):
            raise ValueError(f'{label} shot {sid} has invalid outcome/alarm flags')
        if row['timely'] and (not row['is_disrupt'] or not row['alarm']):
            raise ValueError(f'{label} shot {sid} has impossible timely flag')
        if sid in identities and identities[sid] != row['is_disrupt']:
            raise ValueError(f'{label} shot {sid} outcome differs across arms/shadows')
        identities[sid] = row['is_disrupt']
    nd = sum(identities[sid] for sid in roster)
    ns = len(roster)-nd
    tp = sum(row['timely'] for row in predictions if row['is_disrupt'])
    fp = sum(row['alarm'] for row in predictions if not row['is_disrupt'])
    # online.summarize_rows deliberately nulls BOTH rates for a single-class
    # prefix. Its tp/fp/counts remain meaningful, but no rate is comparable.
    tpr = tp/nd if nd and ns else None
    fpr = fp/ns if nd and ns else None
    expected = dict(tpr=tpr, fpr=fpr, J=tpr-fpr if nd and ns else None,
                    tp=tp, fp=fp, n_disrupt=nd, n_safe=ns, threshold=threshold)
    for key, value in expected.items():
        if key not in metrics:
            raise ValueError(f'{label} missing {key}')
        _equal_number(metrics[key], value, f'{label}.{key}')
    return {**{key: metrics[key] for key in METRICS},
            'fpr_le_0_20': None if fpr is None else fpr <= .20,
            'predictions': predictions}


def _contrast(current, baseline):
    now = {row['shot']: row for row in current['predictions']}
    before = {row['shot']: row for row in baseline['predictions']}
    rescued = [sid for sid, row in now.items()
               if row['is_disrupt'] and row['timely'] and not before[sid]['timely']]
    harmed = [sid for sid, row in now.items()
              if row['is_disrupt'] and not row['timely'] and before[sid]['timely']]
    avoided = [sid for sid, row in now.items()
               if not row['is_disrupt'] and not row['alarm'] and before[sid]['alarm']]
    added = [sid for sid, row in now.items()
             if not row['is_disrupt'] and row['alarm'] and not before[sid]['alarm']]
    return dict(rescued_disrupt_timely=len(rescued), harmed_disrupt_timely=len(harmed),
                net_disrupt_timely=len(rescued)-len(harmed),
                avoided_safe_fp=len(avoided), new_safe_fp=len(added),
                net_avoided_safe_fp=len(avoided)-len(added),
                delta_J=None if current['J'] is None or baseline['J'] is None
                else current['J']-baseline['J'],
                shot_ids=dict(rescued_disrupt_timely=rescued, harmed_disrupt_timely=harmed,
                              avoided_safe_fp=avoided, new_safe_fp=added))


def _early_alarm_diagnostic(run_dir, summary, roster, threshold):
    """Conditional timing audit; summary lacks DownTime for independent D checks."""
    import numpy as np  # CPU-only; not needed for pure-summary synthetic runs.

    result = {}
    for arm in ('intra', 'both'):
        before, first_three, alarmed = 0, 0, 0
        unverifiable_postevent, disruptive = [], []
        for sid in roster:
            sealed = run_dir/arm/f'{sid}_sealed.npz'
            audit_path = run_dir/arm/f'{sid}_audit.json'
            if not sealed.is_file() or not audit_path.is_file():
                return None
            with np.load(sealed) as payload:
                hits = np.flatnonzero(payload['scores'] >= threshold)
            predicted = next(row for row in summary[arm]['deployed']['predictions'] if row['shot'] == sid)
            if predicted['is_disrupt']:
                disruptive.append(sid)
            if not predicted['alarm']:
                # D is scored only through DownTime. A later threshold hit in
                # the sealed full trace is not an alarm and must not be used.
                if len(hits) and predicted['is_disrupt']:
                    unverifiable_postevent.append(sid)
                elif len(hits):
                    raise ValueError(f'{arm} safe shot {sid} sealed alarm differs from summary')
                continue
            if not len(hits):
                raise ValueError(f'{arm} shot {sid} summary alarm lacks any sealed threshold hit')
            # For D this index is conditional on the runner's summary alarm:
            # without DownTime, the report cannot independently validate the
            # event boundary or a lead time. It never treats D no-alarm hits as
            # evidence of an alarm.
            alarmed += 1
            first_three += int(hits[0] <= 2)
            updates = _read_json(audit_path).get('fast_updates', [])
            if updates:
                before += int(hits[0] < int(updates[0]['effective_from']))
        result[arm] = dict(alarmed=alarmed, first_alarm_before_first_fast_update=before,
                           first_alarm_in_first_three_samples=first_three,
                           down_time_independently_verified=not disruptive,
                           disruptive_shots_without_down_time=disruptive,
                           unverifiable_postevent_hit_shots=unverifiable_postevent)
    return result


def _fmt(value):
    return '不可计算' if value is None else (f'{value:.4f}' if isinstance(value, float) else str(value))


def _markdown(result):
    p = result['provenance']
    lines = ['# 在线回放结果（开发诊断）', '',
             f"运行口径：`{p['online_evidence']}`；源检查点口径：`{p['source_evidence']}`。",
             '这是固定源阈值的离线回放，不是独立验证或实时延迟证明；下表不证明增益。',
             f"装置 `{result['domain']}`，炮数 {len(result['roster'])}；报警阈值 {_fmt(result['threshold'])}。",
             'FPR 上限列仅判断固定阈值下是否 ≤0.20，不作事后阈值重选。', '',
             '| 臂 | 轨迹 | TPR | FPR | J | FPR≤0.20 |',
             '|---|---|---:|---:|---:|---|']
    for arm in ARMS:
        for key, title in (('deployed', '部署'), ('same_parent_no_fast_shadow', '同父模型无炮内更新影子')):
            m = result['arms'][arm][key]
            passed = m['fpr_le_0_20']
            lines.append(f"| {arm} | {title} | {_fmt(m['tpr'])} | {_fmt(m['fpr'])} | "
                         f"{_fmt(m['J'])} | {'不可计算' if passed is None else ('通过' if passed else '未通过')} |")
    lines += ['', '## 逐炮配对变化', '',
              '“救回/伤害”按破裂炮及时报警变化计；安全炮按误报变化计。净救回 = 救回 − 伤害。',
              '同父模型影子是同一臂、同一炮的无炮内更新预测；相对 static 虽同炮序，长期状态可能不同，不能单独归因于炮内更新。', '',
              '| 臂 | 基线 | 救回D及时 | 伤害D及时 | 净救回 | 避免安全FP | 新增安全FP | ΔJ |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('intra', 'both'):
        for key, title in (('same_parent_no_fast_shadow', '同父影子'), ('static_deployed', 'static部署')):
            c = result['comparisons'][arm][key]
            lines.append(f"| {arm} | {title} | {c['rescued_disrupt_timely']} | "
                         f"{c['harmed_disrupt_timely']} | {c['net_disrupt_timely']} | "
                         f"{c['avoided_safe_fp']} | {c['new_safe_fp']} | {_fmt(c['delta_J'])} |")
    inter = result['comparisons']['inter']['static_deployed']
    lines += ['', f"炮间轨迹相对 static：及时破裂炮净变化 {inter['net_disrupt_timely']:+d} "
              f"（救回 {inter['rescued_disrupt_timely']}、伤害 {inter['harmed_disrupt_timely']}）；"
              '这是同炮序比较，长期模型状态已不同。']
    early = result.get('early_alarm_diagnostic')
    if early:
        lines += ['', '## 首报时序辅助核查', '',
                  '根据封存分数和每炮更新审计计算，采样索引从 0 开始；前三采样指索引 0–2。'
                  '破裂炮的 DownTime 不在 summary 中，下述索引以 runner 的报警标志为条件，不能独立复核破裂前有效性；'
                  '未报警破裂炮的事后越阈不计报警。']
        for arm in ('intra', 'both'):
            e = early[arm]
            lines.append(f"- {arm}：{e['alarmed']} 炮报警，{e['first_alarm_before_first_fast_update']} 炮首报早于首次炮内更新生效，"
                         f"{e['first_alarm_in_first_three_samples']} 炮首报在前三采样。")
        lines.append('如果首报均早于更新，当前固定阈值下的报警结果不能作为炮内适应有效的证据。')
    source = result.get('source_test220')
    lines += ['', '## 溯源与限制', '',
              f"- 源检查点：`{p['checkpoint']}`；SHA-256 `{p['checkpoint_sha256']}`。",
              f"- 在线 `summary.json` SHA-256：`{p['summary_sha256']}`；炮序 `order.json` SHA-256：`{p['order_sha256']}`。",
              f"- 四臂共同初始父模型 hash：`{p['parent_hash']}`。",
              '- 没有混用旧实验结果；TEST220 与 EAST 在线回放分别标注，不做跨集直接比较。']
    if source:
        lines.append(f"- 同源 J-TEXT `test220.json`：`{p['source_test220']}`；SHA-256 `{p['source_test220_sha256']}`。"
                     f"其主头 TPR={_fmt(source['tpr'])}、FPR={_fmt(source['fpr'])}、J={_fmt(source['J'])}，"
                     f"源验收（TPR≥0.9 且 FPR<0.20）"
                     f"{'不可计算' if source['tpr'] is None or source['fpr'] is None else ('通过' if source['source_acceptance'] else '未通过')}。")
    else:
        lines.append('- 同源 `test220.json` 不存在，未填造源 TEST 指标。')
    return '\n'.join(lines) + '\n'


def build_report(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    summary_path, config_path, order_path = (run_dir/name for name in
                                             ('summary.json', 'config.json', 'order.json'))
    summary, config, order = map(_read_json, (summary_path, config_path, order_path))
    if set(config.get('arms', [])) != set(ARMS) or set(summary) != set(ARMS):
        raise ValueError('report requires exactly static/intra/inter/both in config and summary')
    if config.get('domain') not in ('EAST', 'J-TEXT'):
        raise ValueError('unknown replay domain')
    threshold = config.get('threshold')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError('invalid fixed source threshold')
    full = _ids(order.get('full_order'), 'full_order')
    roster = _ids(order.get('selected_prefix'), 'selected_prefix')
    limit = config.get('limit', 0)
    if type(limit) is not int or limit < 0:
        raise ValueError('invalid online limit')
    if roster != full[:len(roster)] or (limit == 0 and roster != full) or (limit and len(roster) != limit):
        raise ValueError('selected roster is not the configured full-order prefix')
    parents = {summary[arm].get('parent_hash') for arm in ARMS}
    if len(parents) != 1 or not next(iter(parents)):
        raise ValueError('four arms do not share the same initial parent hash')
    checkpoint = _resolve_checkpoint(config['checkpoint'], run_dir)
    checkpoint_hash = _hash(checkpoint)
    if checkpoint_hash != config.get('checkpoint_sha256'):
        raise ValueError('source checkpoint hash differs from online config')
    source_config_path = checkpoint.parent/'config.json'
    source_config = _read_json(source_config_path) if source_config_path.is_file() else {}
    source_test_path = checkpoint.parent/'test220.json'
    source_test = _read_json(source_test_path) if source_test_path.is_file() else None
    source_head = source_test.get('jtext', {}) if source_test else {}
    source_tpr, source_fpr = source_head.get('tpr'), source_head.get('fpr')
    source_result = None
    if source_test:
        source_result = dict(tpr=source_tpr, fpr=source_fpr, J=source_head.get('J'),
                             source_acceptance=bool(source_tpr is not None and source_fpr is not None
                                                    and source_tpr >= .9 and source_fpr < .20))
    identities = {}
    arms = {}
    checked = {}
    for arm in ARMS:
        checked[arm] = {}
        arms[arm] = {}
        for key in ('deployed', 'same_parent_no_fast_shadow'):
            metrics = _validate_metrics(summary[arm].get(key), roster, identities, threshold, f'{arm}.{key}')
            checked[arm][key] = metrics
            arms[arm][key] = {k: v for k, v in metrics.items() if k != 'predictions'}
    comparisons = {}
    for arm in ('intra', 'both'):
        comparisons[arm] = dict(
            same_parent_no_fast_shadow=_contrast(checked[arm]['deployed'],
                                                 checked[arm]['same_parent_no_fast_shadow']),
            static_deployed=_contrast(checked[arm]['deployed'], checked['static']['deployed']))
    comparisons['inter'] = dict(static_deployed=_contrast(checked['inter']['deployed'],
                                                  checked['static']['deployed']))
    provenance = dict(online_evidence=config.get('evidence'),
                      source_evidence=source_config.get('evidence', config.get('source_evidence')),
                      summary_sha256=_hash(summary_path), online_config_sha256=_hash(config_path),
                      order_sha256=_hash(order_path), checkpoint=str(checkpoint),
                      checkpoint_sha256=checkpoint_hash, parent_hash=next(iter(parents)),
                      source_test220=str(source_test_path) if source_test else None,
                      source_test220_sha256=_hash(source_test_path) if source_test else None)
    result = dict(domain=config['domain'], roster=roster, threshold=threshold,
                  provenance=provenance, arms=arms, comparisons=comparisons,
                  source_test220=source_result,
                  early_alarm_diagnostic=(_early_alarm_diagnostic(run_dir, summary, roster, threshold)
                                          if config['domain'] == 'EAST' else None))
    return result, _markdown(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='completed online replay directory')
    args = parser.parse_args(argv)
    result, markdown = build_report(args.run)
    comparison_path, markdown_path = args.run/'comparison.json', args.run/'RESULTS.md'
    if comparison_path.exists() or markdown_path.exists():
        raise FileExistsError('report already exists; do not overwrite a prior diagnostic')
    with comparison_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with markdown_path.open('x', encoding='utf-8') as handle:
        handle.write(markdown)
    print(f'Wrote {markdown_path} and {comparison_path}')
    return result


if __name__ == '__main__':
    main()
