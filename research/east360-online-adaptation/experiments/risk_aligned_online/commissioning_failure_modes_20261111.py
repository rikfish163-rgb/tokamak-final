"""Trace which fixed-model failures are shared, late, or untouched by updates."""

import argparse
from pathlib import Path

import numpy as np

from .history_gradient_metric_20261102.run import read, sha, write
from .mask_repair_20261104.east_evaluation import BASELINE
from .source_anticipation_20261104.static_stream import timing_decomposition


def opportunity(times, fixed, adapted, disruptive, down, cutoff):
    times, fixed, adapted = map(np.asarray, (times, fixed, adapted))
    lead = (down - times) * 1000. if disruptive else None
    legal = lead >= -1e-6 if disruptive else np.ones(len(times), bool)
    timely = np.round(lead, 9) > 30. if disruptive else legal
    hits = np.flatnonzero(legal & (fixed >= cutoff))
    changed = np.flatnonzero(legal & (fixed != adapted))
    first_alarm = int(hits[0]) if len(hits) else None
    first_change = int(changed[0]) if len(changed) else None
    info = dict(first_fixed_alarm_index=first_alarm, first_change_index=first_change,
                eligible_points=int(timely.sum()), changed_eligible_points=int((timely & (fixed != adapted)).sum()),
                alarm_before_any_change=first_alarm is not None and (
                    first_change is None or first_alarm < first_change))
    if not timely.any():
        return info

    def logit(x):
        x = np.clip(np.asarray(x, np.float64), 1e-7, 1. - 1e-7)
        return np.log(x) - np.log1p(-x)

    z, a = logit(fixed[timely]), logit(adapted[timely])
    c = logit(cutoff)
    j = int(z.argmax())
    best_point = int(np.flatnonzero(timely)[j])
    info.update(gap=float(c - z[j]), updated_gap=float(c - a.max()),
                largest_lift=float((a - z).max()), lift_at_best_static_point=float(a[j] - z[j]),
                best_static_index=best_point,
                best_static_lead_ms=round(float(lead[best_point]), 9) if disruptive else None)
    return info


def consensus(arms):
    anchor = next(iter(arms.values()))
    if any(len(r) != len(anchor) for r in arms.values()):
        raise ValueError('same ordered labeled shots required')
    counts = {k: [] for k in ('always_untimely', 'always_no_alarm', 'always_false_alarm',
                             'seed_or_condition_sensitive_disruptions', 'seed_or_condition_sensitive_safe')}
    for rows in zip(*arms.values(), strict=True):
        sid, d = rows[0]['shot'], rows[0]['disruptive']
        if any((r['shot'], r['disruptive']) != (sid, d) for r in rows):
            raise ValueError('same ordered labeled shots required')
        hits = sum(r['alarm'] for r in rows)
        timely = sum(r['timely'] for r in rows)
        if d:
            if not timely:
                counts['always_untimely'].append(sid)
            elif timely < len(rows):
                counts['seed_or_condition_sensitive_disruptions'].append(sid)
            if not hits:
                counts['always_no_alarm'].append(sid)
        elif hits == len(rows):
            counts['always_false_alarm'].append(sid)
        elif hits:
            counts['seed_or_condition_sensitive_safe'].append(sid)
    return counts


def transitions(before, after):
    consensus(dict(before=before, after=after))
    result = {k: [] for k in ('timely_lost', 'timely_rescued', 'false_added', 'false_removed')}
    for a, b in zip(before, after, strict=True):
        if a['disruptive'] and a['timely'] != b['timely']:
            result['timely_rescued' if b['timely'] else 'timely_lost'].append(a['shot'])
        elif not a['disruptive'] and a['alarm'] != b['alarm']:
            result['false_added' if b['alarm'] else 'false_removed'].append(a['shot'])
    return result


def run(ordinal, history):
    p, seal, per, audit = (read(ordinal / k) for k in (
        'protocol.json', 'seal.json', 'per_shot.json', 'independent_audit.json'))
    if (p['scope'] != 'EAST360_FIXED_HEAD_EARLY_ORDINAL_TEACHING_DEVELOPMENT'
            or len(seal['curve_sha256']) != len(p['order']) or len(p['order']) != 360
            or audit['status'] != 'INDEPENDENT_HDF5_RECOUNT'):
        raise ValueError('verified ordinal reference required')
    static_path = BASELINE / 'per_shot.json'
    if sha(static_path) != read(BASELINE / 'completed.json')['artifact_sha256']['per_shot.json']:
        raise ValueError('sealed source-only comparisons changed')
    original = read(static_path)
    current = per['0.2']['fixed']
    methods = dict(unmapped=original['unmapped'], map_after20=original['map_after20'], current=current)
    stages = consensus(methods)
    errors = []
    for i, (sid, row) in enumerate(zip(p['order'], current, strict=True), 1):
        if row['shot'] != sid:
            raise ValueError('different ordinal shot order')
        if (row['disruptive'] and row['timely']) or (not row['disruptive'] and not row['alarm']):
            continue
        curve = ordinal / 'shots' / f'{i:04d}_{sid}.npz'
        if sha(curve) != seal['curve_sha256'][curve.name]:
            raise ValueError(f'sealed prediction changed: {sid}')
        cutoff = p['original'] if i <= p['startup'] else p['calibration']['0.2']
        with np.load(curve, allow_pickle=False) as f:
            o = opportunity(f['times'], f['fixed'], f['ordered'], row['disruptive'], row['down_s'], cutoff)
            if o['first_fixed_alarm_index'] is not None:
                o['first_alarm_ms_since_record_start'] = float(
                    (f['times'][o['first_fixed_alarm_index']] - f['times'][0]) * 1000.)
        typ = ('tardy' if row['alarm'] else 'no_alarm') if row['disruptive'] else 'false_alarm'
        errors.append(dict(shot=sid, index=i, cold=i <= p['startup'],
                           error=typ, lead_ms=row['lead_ms'], **o))
    missed = [o for o in errors if not o['cold'] and o['error'] != 'false_alarm']
    fp = [o for o in errors if not o['cold'] and o['error'] == 'false_alarm']
    prior = read(history / 'recurring_errors.json')
    hard = set(prior['hard_untimely_all_static'])
    result = dict(
        scope='POST_SEAL_FIXED_MODEL_FAILURES_NOT_A_SEED_STUDY',
        current_stage_metrics={k: timing_decomposition(v) for k, v in methods.items()},
        stage_consensus=stages, direct_to_aligned=transitions(methods['unmapped'], methods['map_after20']),
        aligned_to_current=transitions(methods['map_after20'], current),
        earlier_hard_still_untimely=sorted(hard & {r['shot'] for r in current if r['disruptive'] and not r['timely']}),
        earlier_hard_now_timely=sorted(hard & {r['shot'] for r in current if r['timely']}),
        suffix_missed=dict(n=len(missed), tardy=sum(o['error'] == 'tardy' for o in missed),
                           no_alarm=sum(o['error'] == 'no_alarm' for o in missed),
                           with_changed_timely_points=sum(o['changed_eligible_points'] > 0 for o in missed),
                           median_gap=float(np.median([o['gap'] for o in missed])),
                           median_largest_lift=float(np.median([o['largest_lift'] for o in missed])),
                           median_lift_at_best_static_point=float(np.median(
                               [o['lift_at_best_static_point'] for o in missed])),
                           nominally_large_lift_shots=[o['shot'] for o in missed if o['largest_lift'] >= o['gap']]),
        suffix_false_alarms=dict(n=len(fp),
                                 before_score_change=[o['shot'] for o in fp if o['alarm_before_any_change']]),
        bindings={str(f): sha(f) for f in (static_path, history / 'recurring_errors.json',
                                          ordinal / 'seal.json', ordinal / 'per_shot.json',
                                          ordinal / 'independent_audit.json', Path(__file__))},
        errors=errors)
    write(ordinal / 'failure_modes_diagnostic.json', result)
    print(result['suffix_missed'], result['suffix_false_alarms'], flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sealed', type=Path, required=True)
    cli.add_argument('--recurring', type=Path, required=True)
    args = cli.parse_args()
    run(args.sealed.resolve(), args.recurring.resolve())
