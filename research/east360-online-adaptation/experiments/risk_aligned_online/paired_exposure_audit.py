"""Retrospective full2953 exposure/input audit; never changes deployed metrics.

Unlike the archived exposure_audit (VAL160/balanced360), this pairs the current
complete sealed runs, retains short records and reports conditional prefix
statistics separately. Channel summaries are numerical, not physical causes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23, simple_data
from .causal_input import apply_input_contract
from .score_shift_audit import read_seals, sha


def summarize_trace(times, scores, threshold, caps):
    times, scores = np.asarray(times, float), np.asarray(scores, float)
    if (times.ndim != 1 or not len(times) or scores.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or not np.isfinite(threshold) or not 0 < threshold <= 1
            or any(type(k) is not int or k < 1 for k in caps)):
        raise ValueError('finite increasing times, probabilities and positive caps required')
    exceeds = scores >= threshold
    alarm = np.flatnonzero(exceeds)
    return dict(samples=len(times), start_s=float(times[0]), span_s=float(times[-1]-times[0]),
        alarm_index=int(alarm[0]) if len(alarm) else None,
        exceedance_fraction=float(exceeds.mean()),
        prefix={str(k): dict(complete=len(times) >= k, points=min(k, len(times)),
                            alarm=bool(exceeds[:k].any())) for k in caps})


def prefix_features(values, available, center, scale, count):
    x, m = np.asarray(values)[:count], np.asarray(available)[:count]
    center, scale = np.asarray(center, float), np.asarray(scale, float)
    if (type(count) is not int or count < 1 or x.ndim != 2 or not len(x)
            or x.shape[1] != 23 or m.shape != x.shape or m.dtype != bool
            or center.shape != (23,) or scale.shape != (23,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(x[m]).all()):
        raise ValueError('observed finite flat23 values and source coordinate required')
    out = []
    for channel in range(23):
        a = m[:, channel]
        u = (x[a, channel].astype(float)-center[channel])/scale[channel]
        out.append(dict(available_fraction=float(a.mean()),
            median_source_coordinate=float(np.median(np.arcsinh(u))) if len(u) else None,
            beyond_10_source_scales=float(np.mean(np.abs(u) > 10.)) if len(u) else None))
    return out


def quantiles(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), q10_q50_q90=np.quantile(values, [.1, .5, .9]).tolist()
                if values else None)


def summarize_safe(rows, caps):
    safe = [r for r in rows if not r['disruptive']]
    n = len(safe)
    out = dict(n_safe=n, arms={}, prefix={})
    for arm in ('static', 'fixed'):
        fp = sum(r[arm]['alarm_index'] is not None for r in safe)
        out['arms'][arm] = dict(fp=fp, fpr=fp/n if n else None,
            span_s=quantiles([r[arm]['span_s'] for r in safe]),
            start_s=quantiles([r[arm]['start_s'] for r in safe]),
            exceedance_fraction=quantiles([r[arm]['exceedance_fraction'] for r in safe]))
    for cap in map(str, caps):
        out['prefix'][cap] = {}
        for name, subset in [('all_available', safe), ('complete_only', [
                r for r in safe if r['static']['prefix'][cap]['complete']])]:
            part = dict(n_safe=len(subset))
            for arm in ('static', 'fixed'):
                fp = sum(r[arm]['prefix'][cap]['alarm'] for r in subset)
                part.update({f'{arm}_fp': fp, f'{arm}_fpr': fp/len(subset) if subset else None})
            out['prefix'][cap][name] = part
    return out


def channel_profile(rows):
    return {name: {key: quantiles([r['features'][j][key] for r in rows]) for key in (
        'available_fraction', 'median_source_coordinate', 'beyond_10_source_scales')}
        for j, name in enumerate(io23.RAW23)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior audit')
    paired = json.loads(args.paired.read_text())
    if not paired['verification']['full2953_verified'] or sha(args.checkpoint) != paired['source_sha256']:
        raise ValueError('verified full2953 and unchanged source required')
    import torch
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    center, scale = np.asarray(state['source_center']), np.asarray(state['source_scale'])
    threshold = paired['threshold']
    traces = {arm: read_seals(path) for arm, path in paired['runs'].items()}
    if (set(traces) != {'static', 'fixed'} or len(traces['static']) != 2953
            or set(traces['static']) != set(traces['fixed'])):
        raise ValueError('complete identical paired roster required')
    caps = [26, 128, 1000]  # Fixed before inspecting this audit, not tuned for gains.
    rows = []
    for index, sid in enumerate(sorted(traces['static']), 1):
        s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        a, b = traces['static'][sid], traces['fixed'][sid]
        if (any(t['threshold'] != threshold for t in (a, b))
                or not np.array_equal(a['times'], b['times'])
                or not np.array_equal(a['times'], s['times'])):
            raise ValueError('sealed comparison/input time axis mismatch')
        disruptive, down = io23.read_outcome('EAST', sid)
        rows.append(dict(index=index, shot=sid, disruptive=bool(disruptive),
            static=summarize_trace(a['times'], a['scores'], threshold, caps),
            fixed=summarize_trace(b['times'], b['scores'], threshold, caps),
            features=prefix_features(s['values'], s['available'], center, scale, 128),
            data_sha256=sha(io23.path_for('EAST', sid)),
            seals={arm: traces[arm][sid]['sha256'] for arm in traces}))
        if index % 250 == 0:
            print(json.dumps(dict(phase='target', completed=index)), flush=True)
    overall = summarize_safe(rows, caps)
    for arm in traces:
        if (overall['n_safe'] != paired['segments']['overall'][arm]['n_safe']
                or overall['arms'][arm]['fp'] != paired['segments']['overall'][arm]['fp']):
            raise ValueError('safe full-record metrics changed')
    # Reference is current processed source TRAIN input, NOT proof of the exact
    # prefix sampling distribution seen in historical source optimization.
    source = []
    for index, sid in enumerate(simple_data.source_roles()['train'], 1):
        s = simple_data.read_shot('J-TEXT', sid)
        disruptive, _ = io23.read_outcome('J-TEXT', sid)
        source.append(dict(shot=sid, disruptive=bool(disruptive),
            features=prefix_features(s['values'], s['available'], center, scale, 128),
            samples=len(s['times']), start_s=float(s['times'][0]),
            span_s=float(s['times'][-1]-s['times'][0]),
            data_sha256=sha(io23.path_for('J-TEXT', sid))))
        if index % 250 == 0:
            print(json.dumps(dict(phase='source', completed=index)), flush=True)
    stages = []
    for start, end in zip([0, 30, 120, 600, 1500], [30, 120, 600, 1500, 2953]):
        selected = rows[start:end]
        safe = [r for r in selected if not r['disruptive']]
        stages.append(dict(indices=[start+1, end], **summarize_safe(selected, caps),
            channels={group: channel_profile(subset) for group, subset in (
                ('safe_all', safe),
                ('static_FP', [r for r in safe if r['static']['alarm_index'] is not None]),
                ('static_TN', [r for r in safe if r['static']['alarm_index'] is None]))}))
    report = dict(scope='RETROSPECTIVE_DEVELOPMENT_EXPOSURE_INPUT_DIAGNOSIS',
        code_sha256=sha(__file__), paired_sha256=sha(args.paired), source_sha256=sha(args.checkpoint),
        feature_names=list(io23.RAW23), input_prefix_points=128, threshold=threshold,
        source_input_contract='simple_data JTEXT: ip*1000; no new CIII masking',
        target_input_contract='simple_data EAST nominal + MNM mask + pointwise CIII sentinel mask',
        coordinate='asinh((x-source_center)/source_scale); scale is robust source statistic, not SD',
        caps=caps, overall=overall, stages=stages,
        source_profiles={key: channel_profile(subset) for key, subset in (
            ('all', source), ('safe', [r for r in source if not r['disruptive']]),
            ('disruptive', [r for r in source if r['disruptive']]))},
        source_rows=source, rows=rows, model_updates=False, threshold_selection=False,
        full_npz_hashes_verified=True,
        interpretation='Full-record FPR retained. Complete-prefix subsets condition on record length; '
            'not deployed metrics or a causal adjustment. Input associations do not identify physics '
            'or prove miscalibration. First recorded samples need not be the start of a discharge.')
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(overall=overall, stages=[{k: v for k, v in s.items()
        if k != 'channels'} for s in stages])), flush=True)


if __name__ == '__main__':
    main()
