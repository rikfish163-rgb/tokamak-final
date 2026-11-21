"""Post-run attribution only; never used for prediction or parameter selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..io23 import RAW23
from ..simple_online import _durable_json
from ..simple_train import sha


def read(path):
    return json.loads(path.read_text())


def summary(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return dict(n=0)
    if not np.isfinite(a).all():
        raise ValueError('diagnostics require finite measured values')
    return dict(n=int(a.size), minimum=float(a.min()), median=float(np.median(a)),
                maximum=float(a.max()), mean=float(a.mean()))


def correct(row):
    return bool(row['timely'] if row['is_disrupt'] else not row['alarm'])


def acceptance(metrics):
    own = metrics['online/calibrated']
    def j(arm):
        m = metrics[arm+'/calibrated']
        # A safety-only smoke test has no TPR denominator. Preserve unknown,
        # never substitute zero and pretend the pilot demonstrates benefit.
        return None if m['tpr'] is None or m['fpr'] is None else m['tpr']-m['fpr']
    conditions = dict(fpr_at_most_point2=None if own['fpr'] is None else own['fpr'] <= .20,
                      no_tp_loss_vs_static=own['tp'] >= metrics['static/calibrated']['tp'])
    for arm in ('uniform', 'fixed', 'hold', 'mean'):
        ours, other = j('online'), j(arm)
        conditions[f'better_J_than_{arm}'] = None if ours is None or other is None else ours > other
    return conditions


def main(root):
    completed = read(root/'completed.json')
    verification = read(root/'verification.json')
    if verification['completed_sha256'] != sha(root/'completed.json'):
        raise ValueError('matching completed-run verification required')
    per_shot = read(root/'per_shot.json')
    protocol = read(root/'protocol.json')
    ids = protocol['shot_ids']
    outcomes = {r['shot']:r for r in read(root/'outcomes.json')}
    static_rows = {r['shot']:r for r in per_shot['static/calibrated']['predictions']}
    arms = ('static', 'uniform', 'fixed', 'online', 'hold', 'mean')
    weights = {arm: [] for arm in ('fixed', 'online', 'hold', 'mean')}
    differences = {arm: [] for arm in ('static', 'uniform', 'fixed', 'hold', 'mean')}
    gradients, movements, episode_effects = [], [], []
    early_fp, missed = [], []
    for sid in ids:
        intra = read(root/'shots'/f'{sid}_intra.json')
        for arm in weights:
            weights[arm].extend(r['weights'] for r in intra[arm])
        with np.load(root/'shots'/f'{sid}.npz') as z:
            row, threshold = static_rows[sid], protocol['threshold']
            valid = (np.ones(len(z['times']), dtype=bool) if not row['is_disrupt'] else
                     (outcomes[sid]['down']-z['times'])*1000 >= -1e-6)
            for arm in differences:
                differences[arm].append(float(np.max(np.abs(z['online'][valid]-z[arm][valid]))))
            if not row['is_disrupt'] and row['alarm']:
                first = int(np.flatnonzero(z['static'] >= threshold)[0])
                if first < 256:
                    early_fp.append(dict(shot=sid, first_alarm_index=first))
            if row['is_disrupt'] and not row['timely']:
                eligible = np.round((outcomes[sid]['down']-z['times'])*1000, 9) > 30.
                peak = float(z['static'][eligible].max()) if eligible.any() else None
                missed.append(dict(shot=sid, timely_region_peak=peak,
                                   peak_threshold_ratio=None if peak is None else peak/threshold))
        for step in read(root/'shots'/f'{sid}_inter.json')['steps']:
            if 'episodes' not in step:
                continue
            gradients.append(step['gradient_norm'])
            movements.append(step['parameter_delta_norm'])
            episode_effects.extend(e['after_bce']-e['before_bce'] for e in step['episodes'])
    weight_stats = {}
    for arm, rows in weights.items():
        w = np.asarray(rows, dtype=float)
        weight_stats[arm] = dict(
            updates=len(w), mean_by_channel=dict(zip(RAW23, w.mean(0).tolist())),
            std_by_channel=dict(zip(RAW23, w.std(0).tolist())),
            near_lower_bound_fraction=float((w < np.exp(-2)*1.01).mean()),
            near_upper_bound_fraction=float((w > np.exp(2)*.99).mean()))
    blocks = []
    for lo in range(0, len(ids), 60):
        rs = per_shot['static/calibrated']['predictions'][lo:lo+60]
        block = dict(start=lo+1, end=lo+len(rs),
                     disrupt=sum(r['is_disrupt'] for r in rs),
                     safe=sum(not r['is_disrupt'] for r in rs), arms={})
        for arm in arms:
            rs = per_shot[arm+'/calibrated']['predictions'][lo:lo+60]
            block['arms'][arm] = dict(tp=sum(r['timely'] for r in rs),
                fp=sum(r['alarm'] and not r['is_disrupt'] for r in rs))
        blocks.append(block)
    paired = {}
    online_rows = per_shot['online/calibrated']['predictions']
    for arm in differences:
        prior = per_shot[arm+'/calibrated']['predictions']
        changed = [dict(shot=b['shot'], beneficial=correct(b), before=a, after=b)
                   for a, b in zip(prior, online_rows) if correct(a) != correct(b)]
        assert changed == verification['flips'][arm]
        paired[arm] = dict(improved=sum(r['beneficial'] for r in changed),
                           harmed=sum(not r['beneficial'] for r in changed), changes=changed)
    conditions = acceptance(completed['metrics'])
    result = dict(scope='POST_HOC_DEVELOPMENT_DIAGNOSIS; not new performance run',
        completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(root/'verification.json'),
        diagnostic_code_sha256=sha(Path(__file__)),
        acceptance_conditions=conditions, passed_all=all(v is True for v in conditions.values()),
        paired=paired, blocks60=blocks, weights=weight_stats,
        per_shot_max_probability_difference_valid_times_only={k:summary(v) for k,v in differences.items()},
        policy_gradient_norm=summary(gradients), policy_parameter_movement=summary(movements),
        historical_simulated_after_minus_before_bce=summary(episode_effects),
        historical_improved=sum(x < 0 for x in episode_effects),
        historical_harmed=sum(x > 0 for x in episode_effects),
        errors_before_current_feedback=dict(
            static_false_alarms_in_immutable_first256=early_fp,
            static_missed_timely_shots=missed,
            missed_with_peak_below_half_threshold=sum(r['peak_threshold_ratio'] is not None
                and r['peak_threshold_ratio'] < .5 for r in missed),
            missed_with_peak_above_90percent_threshold=sum(r['peak_threshold_ratio'] is not None
                and r['peak_threshold_ratio'] >= .9 for r in missed),
            limitation='This bound applies only to the tested frozen-parent/first256 scheme, '
                       'not to all possible earlier or persistent updates.'),
        warning='Historical episode risk loss is not a held-out future effect. '
                'Online/hold/mean share policy history; not independent seeds. '
                'Compute budgets differ from cached static/uniform baselines.')
    _durable_json(root/'diagnostics.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired', 'weights', 'errors_before_current_feedback')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
