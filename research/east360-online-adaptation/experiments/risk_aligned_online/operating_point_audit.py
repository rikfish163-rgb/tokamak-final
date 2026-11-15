"""Post-hoc matched-FP diagnosis, never training or deploying a threshold.

Use exact score breakpoints, treating tied safe maxima as indivisible. The
positive score is the maximum BEFORE the strict 30 ms deadline, not the
maximum including a late alarm. All outcomes here are retrospective labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23
from .score_shift_audit import read_seals, sha


def peak_for_shot(times, scores, disruptive, down):
    times, scores = np.asarray(times, np.float64), np.asarray(scores, np.float64)
    if (times.ndim != 1 or not len(times) or scores.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError('finite aligned increasing times and probabilities required')
    if disruptive:
        if down is None or not np.isfinite(down):
            raise ValueError('finite disruption time required')
        eligible = np.round((down-times)*1000., 9) > 30.
        return float(scores[eligible].max()) if eligible.any() else None
    return float(scores.max())


def _arrays(rows):
    pos, neg = [], []
    for row in rows:
        if type(row['disruptive']) is not bool:
            raise ValueError('boolean disruptive indicator required')
        peak = row['peak']
        if peak is None:
            if not row['disruptive']:
                raise ValueError('safe shot requires an observed peak')
            peak = -np.inf  # No timely opportunity: never counted as a TP.
        elif not np.isfinite(peak) or not 0 <= peak <= 1:
            raise ValueError('finite probability peak required')
        (pos if row['disruptive'] else neg).append(peak)
    if not pos or not neg:
        raise ValueError('both outcome classes required')
    return np.sort(pos), np.sort(neg)


def at_threshold(rows, threshold):
    pos, neg = _arrays(rows)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('nonnegative finite threshold required')
    tp, fp = int(np.sum(pos >= threshold)), int(np.sum(neg >= threshold))
    return dict(threshold=float(threshold), tp=tp, fp=fp,
                n_disrupt=len(pos), n_safe=len(neg), tpr=tp/len(pos), fpr=fp/len(neg),
                J=tp/len(pos)-fp/len(neg))


def best_at_count(rows, max_fp):
    pos, neg = _arrays(rows)
    if type(max_fp) is not int or not 0 <= max_fp <= len(neg):
        raise ValueError('integer FP budget within safe count required')
    thresholds = np.unique(np.r_[pos[np.isfinite(pos)], neg, np.nextafter(1., 2.)])
    tp = len(pos)-np.searchsorted(pos, thresholds, side='left')
    fp = len(neg)-np.searchsorted(neg, thresholds, side='left')
    feasible = np.flatnonzero(fp <= max_fp)
    best = max(feasible, key=lambda k: (tp[k], -fp[k], thresholds[k]))
    return dict(at_threshold(rows, float(thresholds[best])), max_fp=max_fp)


def timely_auc(rows):
    pos, neg = _arrays(rows)
    less = np.searchsorted(neg, pos, side='left')
    equal = np.searchsorted(neg, pos, side='right')-less
    return float(np.sum(less+.5*equal)/(len(pos)*len(neg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous diagnostic')
    paired = json.loads(args.paired.read_text())
    if not paired['verification']['full2953_verified']:
        raise ValueError('completed verified full2953 comparison required')
    traces = {arm: read_seals(Path(path)) for arm, path in paired['runs'].items()}
    if set(traces) != {'static', 'fixed'} or set(traces['static']) != set(traces['fixed']):
        raise ValueError('identical static/fixed paired shots required')
    if len(traces['static']) != 2953:
        raise ValueError('full2953 only, no selective subset')
    threshold = float(paired['threshold'])
    rows = {'static': [], 'fixed': []}
    for sid in sorted(traces['static']):
        a, b = traces['static'][sid], traces['fixed'][sid]
        if (not np.array_equal(a['times'], b['times']) or
                any(t['threshold'] != threshold for t in (a, b))):
            raise ValueError('same times and original threshold required')
        disruptive, down = io23.read_outcome('EAST', sid)
        for arm, trace in (('static', a), ('fixed', b)):
            rows[arm].append(dict(shot=sid, disruptive=bool(disruptive),
                peak=peak_for_shot(trace['times'], trace['scores'], disruptive, down),
                sealed_sha256=trace['sha256']))
    original = {arm: at_threshold(r, threshold) for arm, r in rows.items()}
    for arm, metrics in original.items():
        expected = paired['segments']['overall'][arm]
        for key in ('tp', 'fp', 'n_safe', 'n_disrupt', 'tpr', 'fpr', 'J'):
            if metrics[key] != expected[key]:
                raise ValueError(f'original metrics changed: {arm}/{key}')
    # Same-count comparison is selected after outcomes, never a deployed arm.
    matched = best_at_count(rows['static'], original['fixed']['fp'])
    tp_flips = {'gained': [], 'lost': []}; fp_flips = {'added': [], 'removed': []}
    for a, b in zip(rows['static'], rows['fixed']):
        before = a['peak'] is not None and a['peak'] >= matched['threshold']
        after = b['peak'] is not None and b['peak'] >= threshold
        if before != after:
            if a['disruptive']:
                tp_flips['gained' if after else 'lost'].append(a['shot'])
            else:
                fp_flips['added' if after else 'removed'].append(a['shot'])
    budgets = [175, 263, 351, original['fixed']['fp']]
    report = dict(scope='ALL_TARGET_LABEL_POSTHOC_DIAGNOSTIC_NOT_DEPLOYABLE',
        model_updates=False, threshold_deployed=False, source_sha256=paired['source_sha256'],
        paired_sha256=sha(args.paired), code_sha256=sha(__file__),
        lead_rule='round((down-time)*1000,9)>30ms; no maximum lead cutoff',
        original=original, static_at_candidate_fp_budget=matched,
        matched_tp_flips=tp_flips, matched_fp_flips=fp_flips,
        timely_peak_ranking_auc={arm: timely_auc(r) for arm, r in rows.items()},
        caps={str(budget): {arm: best_at_count(r, budget) for arm, r in rows.items()}
              for budget in budgets}, rows=rows,
        interpretation='Numerical operating-point comparison, NOT a causal attribution of mechanism. '
            'Both thresholds and comparisons use developed target outcomes. No physical-law claim.')
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('original', 'static_at_candidate_fp_budget',
          'timely_peak_ranking_auc', 'caps')}, indent=2))
    print(json.dumps(dict(matched_tp_flips={k: len(v) for k, v in tp_flips.items()},
                          matched_fp_flips={k: len(v) for k, v in fp_flips.items()})))


if __name__ == '__main__':
    main()
