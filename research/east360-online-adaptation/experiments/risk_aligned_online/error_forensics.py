"""Retrospective EAST error ledger; never trains, reseals, or changes a predictor.

Outcome-anchored deadlines and supervised gradient probes are ORACLE diagnostics,
not online policy observations. Prefix features themselves use observed past only.
The five full-stream arms are existing, explicitly different experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data, simple_train

RUNS = Path(__file__).parent / 'runs'
PARENT = RUNS / 'gradient_aligned512_20260929/source.pt'
ARMS = {
    'static': ('diagnostic_aligned_east360_20260929', 'static'),
    'diagnostic_intra': ('diagnostic_aligned_east360_20260929', 'intra'),
    'inter': ('aligned_maintenance4_east360_20260929', 'inter'),
    'inter_token_intra': ('aligned_maintenance4_east360_20260929', 'both'),
    'mask_density': ('context_input_mask_ne_east360_20260929', 'static'),
}
CLASSES = ('TP', 'TARDY', 'FN', 'FP', 'TN')


def classify(disruptive, alarm, lead_ms):
    if not disruptive:
        return 'FP' if alarm else 'TN'
    if not alarm:
        return 'FN'
    return 'TP' if round(float(lead_ms), 9) > 30. else 'TARDY'


def prefix_features(values, available, end, center, scale, length=128):
    """Observed raw levels and changes: no outcome, future moments, or shot length.

    Source offsets are source-IQR coordinates, not standard deviations or physical
    thresholds. Trend compares first/last thirds in source-asinh coordinates.
    An exact constant channel has zero variation; it is not counted as missing.
    """
    if end < 0 or end >= len(values) or length < 1:
        raise ValueError('observed endpoint and positive lookback required')
    x = np.asarray(values[max(0, end-length+1):end+1], np.float64)
    a = np.asarray(available[max(0, end-length+1):end+1], bool) & np.isfinite(x)
    rows = {}
    for c, name in enumerate(io23.RAW23):
        seen = x[a[:, c], c]
        if not len(seen):
            rows[name] = dict(count=0, available_fraction=0., raw_median=None,
                              offset_iqr=None, ood_fraction=None, log_variation=None,
                              log_trend=None, constant=None)
            continue
        z = (seen-float(center[c]))/float(scale[c])
        q = np.arcsinh(z)
        third = max(1, len(q)//3)
        rows[name] = dict(count=len(seen), available_fraction=float(a[:, c].mean()),
                          raw_median=float(np.median(seen)), offset_iqr=float(np.median(z)),
                          ood_fraction=float(np.mean(np.abs(z) > 5.)),
                          log_variation=float(np.std(q)),
                          log_trend=float(np.median(q[-third:])-np.median(q[:third])),
                          constant=bool(np.max(seen) == np.min(seen)))
    return rows


def record(times, scores, disruptive, down, threshold):
    times, scores = np.asarray(times), np.asarray(scores)
    valid = np.ones(len(times), bool) if not disruptive else (down-times)*1000 >= -1e-6
    indices = np.flatnonzero(valid & (scores >= threshold))
    first = int(indices[0]) if len(indices) else None
    lead = round(float((down-times[first])*1000), 9) if disruptive and first is not None else None
    timely = valid & (np.round((down-times)*1000, 9) > 30.) if disruptive else valid
    timely_indices = np.flatnonzero(timely)
    return dict(category=classify(disruptive, first is not None, lead), first_index=first,
                first_elapsed_ms=None if first is None else round(float((times[first]-times[0])*1000), 9),
                lead_ms=lead, timely_peak=float(scores[timely].max()) if timely.any() else None,
                peak=float(scores[valid].max()) if valid.any() else None,
                deadline_index=int(timely_indices[-1]) if disruptive and len(timely_indices) else None,
                alarm_fraction=float(np.mean(scores[valid] >= threshold)) if valid.any() else None)


def load_arm(label, expected_parent, expected_order, threshold):
    name, arm = ARMS[label]
    root = RUNS/name
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['parent_sha256'] != expected_parent or protocol['threshold'] != threshold:
        raise ValueError('different parent or threshold')
    summary = json.loads((root/'summary.json').read_text())[arm]
    predictions = summary.get('metrics_full', summary)['predictions']
    if [r['shot'] for r in predictions] != expected_order:
        raise ValueError('different order or roster')
    data = {}
    for i, sid in enumerate(expected_order, 1):
        file = root/arm/f'{i:04d}_{sid}_sealed.npz'
        receipt = json.loads(file.with_name(f'{i:04d}_{sid}_seal.json').read_text())
        if simple_train.sha(file) != receipt['sha256']:
            raise ValueError('sealed prediction hash mismatch')
        with np.load(file, allow_pickle=False) as stored:
            times, scores = stored['times'].copy(), stored['scores'].copy()
            if int(stored['shot']) != sid:
                raise ValueError('sealed identity mismatch')
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('invalid probability')
        data[sid] = dict(times=times, scores=scores)
    return data, predictions, summary, protocol


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired(ledger, a, b):
    counts = Counter((r[a+'_category'], r[b+'_category']) for r in ledger)
    flips = [dict(shot=r['shot'], position=r['position'], before=r[a+'_category'],
                  after=r[b+'_category']) for r in ledger if r[a+'_category'] != r[b+'_category']]
    gain = sum((x['before'] in ('FP', 'FN', 'TARDY')) and x['after'] in ('TN', 'TP') for x in flips)
    loss = sum((x['after'] in ('FP', 'FN', 'TARDY')) and x['before'] in ('TN', 'TP') for x in flips)
    return dict(transitions={f'{x}->{y}': n for (x, y), n in counts.items()},
                flipped=flips, corrected=gain, spoiled=loss, net=gain-loss)


def gradient_probe(parent, adapter, shot, query):
    """One hypothetical ZERO-state matured-signal step, ending before query.

    It does NOT replay the original accumulated fast history, train parameters,
    issue a new alarm, or prove whole-shot rescue. True query y is posthoc only.
    """
    from torch.nn import functional as F
    from .diagnostic_targets import trajectory_loss
    from .family_feedback import signal_target23

    horizon = adapter.horizon
    origin = query-horizon-1
    if origin < 8:
        return dict(eligible=False, reason='no_8point_origin_plus_matured_H8_before_endpoint')
    device = parent.center.device
    def tensor_window(end):
        x, a = simple_data.window(shot, end, parent.cfg.window)
        return torch.as_tensor(x[None], device=device), torch.as_tensor(a[None], device=device)
    x, a = tensor_window(origin)
    qx, qa = tensor_window(query)
    truth = torch.as_tensor(shot['values'][origin+1:query][None], device=device)
    available = torch.as_tensor(shot['available'][origin+1:query][None], device=device)
    target, valid = signal_target23(x, a, truth, available)
    if not bool(valid.any()):
        return dict(eligible=False, reason='no_nonconstant_valid_signal_target')
    state = adapter.initial_fast(1).requires_grad_(True)
    issued = adapter(parent, x, a, state)
    loss = trajectory_loss(issued['trajectory'], target, valid).sum()
    gradient, = torch.autograd.grad(loss, state)
    risk_state = adapter.initial_fast(1).requires_grad_(True)
    before = adapter(parent, qx, qa, risk_state)['logit']
    y = torch.as_tensor([float(shot['y'][query])], device=device)
    risk_loss = F.binary_cross_entropy_with_logits(before, y)
    risk_gradient, = torch.autograd.grad(risk_loss, risk_state)
    norm = float(gradient.norm())
    denom = float(gradient.norm()*risk_gradient.norm())
    cosine = float((gradient*risk_gradient).sum())/denom if denom > 0 else None
    updated = adapter.update(state.detach(), gradient.detach(), issued['context'].detach()).detach()
    with torch.no_grad():
        after = adapter(parent, qx, qa, updated)['logit']
        after_loss = F.binary_cross_entropy_with_logits(after, y)
        signal_after = trajectory_loss(adapter(parent, x, a, updated)['trajectory'], target, valid).sum()
    return dict(eligible=True, origin=origin, observed_through=query-1, affects_from=query,
                oracle_risk_y=float(y), probability_before=float(before.detach().sigmoid()),
                probability_after=float(after.sigmoid()), signal_loss_before=float(loss.detach()),
                signal_loss_after=float(signal_after), risk_loss_before=float(risk_loss.detach()),
                risk_loss_after=float(after_loss), signal_gradient_norm=norm,
                signal_risk_gradient_cosine=cosine, state_change=float(updated.norm()),
                signal_loss_decreased=bool(signal_after < loss.detach()),
                risk_loss_decreased=bool(after_loss < risk_loss.detach()))


def configure_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'axes.titleweight': 'bold'})
    skill = io23.ROOT/'.codex/skills/scientific-visualization/scripts/figure_export.py'
    spec = importlib.util.spec_from_file_location('forensic_figure_export', skill)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return plt, module.save_publication_figure


def plots(output, ledger, saved, cases, evidence):
    plt, save = configure_figures()
    colors = {'static': '#333333', 'diagnostic_intra': '#D55E00',
              'inter': '#0072B2', 'inter_token_intra': '#009E73'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ['静态', '原炮内诊断', '炮间诊断', '炮间＋炮内']
    axes[0].bar(labels, [sum(r[a+'_category'] == 'TP' for r in ledger) for a in colors],
                color=list(colors.values()))
    axes[1].bar(labels, [sum(r[a+'_category'] == 'FP' for r in ledger) for a in colors],
                color=list(colors.values()))
    for ax, title in zip(axes, ['及时预警炮数 / 180破裂炮', '误报炮数 / 180安全炮']):
        ax.set_ylim(0, 180)
        ax.set_title(title)
        for bar in ax.patches:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, str(int(bar.get_height())), ha='center')
    fig.suptitle('同一父模型、阈值和炮序；不同适应实验不得混称一条主线')
    save(fig, output/'figures/01_full360_counts', formats=['pdf', 'png'], dpi=600)
    plt.close(fig)
    for sid in cases:
        row = next(r for r in ledger if r['shot'] == sid)
        shot = simple_data.read_shot('EAST', sid)
        disruptive, down = io23.read_outcome('EAST', sid)
        duration = (shot['times']-shot['times'][0])*1000
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for arm, color in colors.items():
            axes[0].plot(duration, saved[arm][sid]['scores'], color=color,
                         linestyle='--' if arm == 'diagnostic_intra' else '-', lw=1.2,
                         label=f'{arm}: {row[arm+"_category"]}')
        axes[0].axhline(evidence['threshold'], color='#555555', ls=':', label='固定报警门槛')
        axes[0].set_ylim(0, 1.02)
        axes[0].set_ylabel('风险分数')
        axes[0].legend(ncol=2, fontsize=8, loc='upper left')
        for ax, names in zip(axes[1:], [('ip', 'Bt', 'ne_nG'),
                                      ('Mir_avg_amp', 'n=1 amplitude', 'dx_a', 'dy_a'),
                                      ('CIII', 'sxr_mean', 'xuv_var')]):
            for n, ls in zip(names, ['-', '--', ':', '-.']):
                c = io23.RAW23.index(n)
                q = np.arcsinh((shot['values'][:, c].astype(np.float64)-evidence['center'][c])/evidence['scale'][c])
                q[~shot['available'][:, c]] = np.nan
                ax.plot(duration, q, lw=1., linestyle=ls, label=n)
            ax.set_ylabel('源域尺度下的\nasinh信号值')
            ax.legend(ncol=len(names), fontsize=8, loc='upper left')
        if disruptive:
            deadline = (down-shot['times'][0])*1000-30.
            for ax in axes:
                ax.axvline(deadline, color='#CC79A7', ls='--', lw=1.)
                ax.axvline(deadline+30, color='#777777', ls=':', lw=.8)
        axes[-1].set_xlabel('从已保存信号起点开始的观测时间（ms；不是等离子体启动时刻）')
        timing_note = '；竖虚线为事后30 ms截止线' if disruptive else '；安全炮无破裂截止线'
        fig.suptitle(f'EAST {sid} · 第{row["position"]}炮 · {"破裂" if disruptive else "安全"}' + timing_note)
        save(fig, output/f'figures/shot_{sid}', formats=['pdf', 'png'], dpi=600)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true', help='read-only GPU oracle direction probes')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh analysis directory')
    torch.set_num_threads(4)
    ckpt = torch.load(PARENT, map_location='cpu', weights_only=False)
    parent_hash, threshold = simple_train.sha(PARENT), float(ckpt['threshold'])
    center, scale = np.asarray(ckpt['source_center']), np.asarray(ckpt['source_scale'])
    protocol = json.loads((RUNS/ARMS['static'][0]/'protocol.json').read_text())
    order = protocol['order']
    ids = [int(x['shot']) for x in order]
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('complete unique 360-shot manifest required')
    saved, originals, input_summaries = {}, {}, {}
    for arm in ARMS:
        saved[arm], originals[arm], input_summaries[arm], _ = load_arm(arm, parent_hash, ids, threshold)
    ledger, features, event_details = [], [], []
    records_by_shot, probe_queries = {}, {}
    for position, sid in enumerate(ids, 1):
        shot = simple_data.attach_outcome(simple_data.read_shot('EAST', sid), 'EAST', {})
        row = dict(shot=sid, position=position, is_disrupt=bool(shot['is_disrupt']), down=shot['down'],
                   initial_ip_A=order[position-1]['initial_ip_A'],
                   initial_abs_Bt_raw=order[position-1]['initial_abs_Bt_raw'], proxy=order[position-1]['proxy'])
        records = {}
        for arm in ARMS:
            scores = saved[arm][sid]
            if not np.array_equal(scores['times'], shot['times']):
                raise ValueError('different saved times')
            rec = record(scores['times'], scores['scores'], shot['is_disrupt'], shot['down'], threshold)
            original = originals[arm][position-1]
            if rec['category'] != classify(original['is_disrupt'], original['alarm'], original['lead_ms']):
                raise ValueError('rescored sealed category does not match original ledger')
            records[arm] = rec
            row.update({arm+'_'+k: v for k, v in rec.items()})
        records_by_shot[sid] = records
        valid = np.flatnonzero(shot['valid'])
        if not len(valid):
            raise ValueError('no preterminal observations')
        # Fixed 100th observed sample; short records are explicitly counted.
        early_end = min(99, int(valid[-1]))
        anchors = {'fixed100': early_end}
        static = records['static']
        if static['category'] == 'FP':
            anchors['oracle_error'] = static['first_index']
            probe_queries[sid] = static['first_index']
            audit = json.loads((RUNS/ARMS['diagnostic_intra'][0]/'intra'/f'{position:04d}_{sid}_audit.json').read_text())
            effects = [e['affects_from'] for e in audit['events']]
            row['actual_first_diagnostic_effect'] = min(effects) if effects else None
            row['fp_precedes_diagnostic_feedback'] = bool(not effects or static['first_index'] < min(effects))
            row['fp_precedes_H8_theoretical_feedback'] = static['first_index'] < 18
        if shot['is_disrupt'] and static['deadline_index'] is not None:
            anchors['oracle_deadline'] = static['deadline_index']
            if static['category'] in ('FN', 'TARDY') or records['inter_token_intra']['category'] in ('FN', 'TARDY'):
                probe_queries[sid] = static['deadline_index']
        row['fixed100_actual_end'] = early_end
        for anchor, end in anchors.items():
            feat = prefix_features(shot['values'], shot['available'], end, center, scale)
            event_details.append(dict(shot=sid, position=position, anchor=anchor, observed_end=end,
                                      elapsed_ms=float((shot['times'][end]-shot['times'][0])*1000), channels=feat))
            for channel, measures in feat.items():
                features.append(dict(shot=sid, position=position, static_category=static['category'],
                                     inter_category=records['inter']['category'], anchor=anchor,
                                     end=end, channel=channel, **measures))
        ledger.append(row)
        if position % 60 == 0:
            print(f'forensic raw-signal and sealed-score audit {position}/360', flush=True)
    summary = dict(scope='RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_POLICY_PERFORMANCE',
                   parent=str(PARENT.resolve()), parent_sha256=parent_hash, threshold=threshold,
                   order=ids, source_code_sha256=simple_train.sha(__file__), sealed_files_verified=360*len(ARMS),
                   counts={arm: dict(Counter(r[arm+'_category'] for r in ledger)) for arm in ARMS},
                   pairs={f'{a}_vs_{b}': paired(ledger, a, b) for a, b in
                          [('static', 'diagnostic_intra'), ('static', 'inter'),
                           ('inter', 'inter_token_intra'), ('static', 'mask_density')]},
                   early_FP_before_actual_diagnostic_feedback=sum(r.get('fp_precedes_diagnostic_feedback', False) for r in ledger),
                   early_FP_before_earliest_H8_feedback=sum(r.get('fp_precedes_H8_theoretical_feedback', False) for r in ledger),
                   static_errors={c: [r['shot'] for r in ledger if r['static_category'] == c] for c in ('FP', 'FN', 'TARDY')},
                   both_errors={c: [r['shot'] for r in ledger if r['inter_token_intra_category'] == c] for c in ('FP', 'FN', 'TARDY')})
    # Descriptive, shot-level fixed-elapsed comparisons; no significance search.
    comparisons = {}
    for channel in io23.RAW23:
        channel_rows = [r for r in features if r['anchor'] == 'fixed100' and r['channel'] == channel]
        comparisons[channel] = {}
        for category in CLASSES:
            subset = [r for r in channel_rows if r['static_category'] == category]
            measures = {}
            for metric in ('raw_median', 'offset_iqr', 'ood_fraction', 'log_variation', 'log_trend'):
                vals = [r[metric] for r in subset if r[metric] is not None]
                measures[metric] = dict(n=len(vals), q25_median_q75=np.quantile(vals, [.25, .5, .75]).tolist()) if vals else None
            comparisons[channel][category] = measures
    summary['fixed100_group_descriptives'] = comparisons
    # Verify that the paired inter/both intervention did not change persistent training.
    inter_rows = input_summaries['inter']['rows']
    both_rows = input_summaries['inter_token_intra']['rows']
    for pos, (a, b) in enumerate(zip(inter_rows, both_rows), 1):
        for key in ('sample_sha256', 'forecast_sample_sha256', 'classifier_sha256'):
            if a['persistent'][key] != b['persistent'][key]:
                raise ValueError(f'paired persistent training differs at {pos}: {key}')
        expected_history = ids[max(0, pos-200):pos]
        if a['persistent']['history_ids'] != expected_history:
            raise ValueError('persistent history includes wrong/future shots')
    summary['inter_both_persistent_rounds_verified'] = len(inter_rows)
    args.output.mkdir(parents=True)
    (args.output/'figures').mkdir()
    write_csv(args.output/'all360_error_ledger.csv', ledger)
    write_csv(args.output/'past_window_channel_features.csv', features)
    write_json(args.output/'event_channels.json', event_details)
    if args.probe:
        from .diagnostic_training import load_parent
        from .family_feedback import FamilyFeedback
        parent, _ = load_parent(PARENT)
        file = RUNS/'family_aligned512_20260929/adapter.pt'
        stored = torch.load(file, map_location='cpu', weights_only=False)
        if stored['protocol']['parent_sha256'] != parent_hash:
            raise ValueError('mismatched oracle probe adapter')
        adapter = FamilyFeedback(parent.cfg.dim, horizon=8).cuda().eval().requires_grad_(False)
        adapter.load_state_dict(stored['adapter'])
        immutable = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
        probes = []
        for n, (sid, query) in enumerate(probe_queries.items(), 1):
            shot = simple_data.attach_outcome(simple_data.read_shot('EAST', sid), 'EAST', {})
            probes.append(dict(shot=sid, query=query, static_category=records_by_shot[sid]['static']['category'],
                               inter_category=records_by_shot[sid]['inter']['category'],
                               **gradient_probe(parent, adapter, shot, query)))
            if n % 30 == 0:
                print(f'read-only oracle gradient probe {n}/{len(probe_queries)}', flush=True)
        if any(not torch.equal(v.detach().cpu(), immutable[k]) for k, v in parent.state_dict().items()):
            raise AssertionError('forensic probe changed the parent')
        write_csv(args.output/'oracle_single_step_probes.csv', probes)
        summary['oracle_probes'] = {}
        for group in ('FP', 'FN', 'TARDY', 'TP'):
            subset = [r for r in probes if r['static_category'] == group and r['eligible']]
            summary['oracle_probes'][group] = dict(n=len(subset),
                signal_improved=sum(r['signal_loss_decreased'] for r in subset),
                risk_improved=sum(r['risk_loss_decreased'] for r in subset),
                signal_improved_risk_worsened=sum(r['signal_loss_decreased'] and not r['risk_loss_decreased'] for r in subset),
                median_probability_change=float(np.median([r['probability_after']-r['probability_before'] for r in subset])) if subset else None,
                median_gradient_cosine=float(np.median([r['signal_risk_gradient_cosine'] for r in subset if r['signal_risk_gradient_cosine'] is not None])) if subset else None)
        summary['oracle_probe_caveat'] = 'hypothetical zero-state one-step H8; true risk y is evaluator-only; not a full-shot rescue or accumulated-state replay'
        summary['probe_parent_unchanged'] = True
    write_json(args.output/'summary.json', summary)
    cases = [69855, 67102, 70327, 90874, 74087, 86756]
    plots(args.output, ledger, saved, cases, dict(threshold=threshold, center=center, scale=scale))
    if simple_train.sha(PARENT) != parent_hash:
        raise AssertionError('parent file changed')
    print(json.dumps({k: v for k, v in summary.items() if k in
                      ('counts', 'early_FP_before_actual_diagnostic_feedback',
                       'early_FP_before_earliest_H8_feedback', 'oracle_probes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
