"""Post-hoc full-stream loss/metric semantics; NEVER imported by an online learner.

This measures a conflict in scalar logit targets, not a causal attribution of
parameter-gradient effects. A positive bag derivative is a diagnostic comparator,
not a proposed ground-truth label for every early point of a disruptive shot.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from .audit import classify, digest
from .protocol import sampled_endpoints


def describe(times, scores, disruptive, down, threshold, selected):
    times, scores = np.asarray(times), np.asarray(scores)
    selected = np.asarray(selected, dtype=int)
    if times.ndim != 1 or scores.shape != times.shape or not np.isfinite(scores).all():
        raise ValueError('finite matched time/score arrays required')
    labels, valid = io23.labels_at(times, disruptive, down, 125.)
    lead = np.round((down - times) * 1000, 9) if disruptive else None
    eligible = valid & (lead > 30) if disruptive else valid
    indices = np.flatnonzero(eligible)
    peak = int(indices[np.argmax(scores[indices])]) if len(indices) else None
    high_negative = selected[(labels[selected] == 0) & (scores[selected] >= threshold)]
    decision = classify(times, scores, disruptive, down, threshold)
    return dict(**decision, timely_peak_index=peak,
                timely_peak_score=float(scores[peak]) if peak is not None else None,
                timely_peak_lead_ms=float(lead[peak]) if disruptive and peak is not None else None,
                timely_peak_point_label=int(labels[peak]) if peak is not None else None,
                point_gradient_at_timely_peak=float(scores[peak] - labels[peak]) if peak is not None else None,
                positive_bag_gradient_at_timely_peak=float(scores[peak] - 1) if disruptive and peak is not None else None,
                selected_high_negative_count=len(high_negative),
                selected_high_negative_indices=high_negative.tolist(),
                selected_count=len(selected))


def main(root, output):
    completed = json.loads((root / 'completed.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    order = json.loads((root / 'order.json').read_text())
    inputs = json.loads((root / 'input_hashes.json').read_text())
    if completed['status'] != 'COMPLETED' or len(order) != 360:
        raise ValueError('complete canonical 360 stream required')
    ids = [r['shot'] for r in order]
    if len(set(ids)) != 360 or set(ids) != set(protocol['contract']['shot_ids']):
        raise ValueError('not the canonical full stream')
    sampler = Path(__file__).with_name('protocol.py')
    if digest(sampler) != protocol['code_sha256'][str(sampler.relative_to(io23.ROOT))]:
        raise ValueError('endpoint sampler differs from archived experiment')
    arms = ('source', 'aligned', 'cold', 'intra', 'inter', 'both')
    rows = []
    for index, sid in enumerate(ids, 1):
        path = root / 'shots' / f'{index:04d}_{sid}.npz'
        seal = json.loads(path.with_name(path.stem + '_seal.json').read_text())
        if digest(path) != seal['sha256'] or digest(io23.path_for('EAST', sid)) != inputs[str(sid)]:
            raise ValueError('sealed curves or raw input changed')
        flag, down = io23.read_outcome('EAST', sid)
        with np.load(path) as f:
            t = f['times']
            y, valid = io23.labels_at(t, flag, down, 125.)
            arm_rows = {}
            for arm in arms:
                selected = sampled_endpoints(t, y, valid, f[arm], down=down if flag else None)
                arm_rows[arm] = describe(t, f[arm], flag, down, protocol['threshold'], selected)
            peak = arm_rows['cold']['timely_peak_index']
            drift = float(f['inter'][peak] - f['cold'][peak]) if peak is not None else None
        rows.append(dict(index=index, shot=sid, disruptive=bool(flag), arms=arm_rows,
                         inter_minus_cold_at_cold_peak=drift))
    totals = {}
    for arm in arms:
        d = [r for r in rows if r['disruptive']]
        tp = [r for r in d if r['arms'][arm]['timely']]
        post = [r for r in d if r['index'] > 20]
        totals[arm] = dict(
            timely_count=len(tp),
            timely_first_alarm_earlier_than125=sum(r['arms'][arm]['lead_ms'] > 125.000001 for r in tp),
            timely_peak_label_zero_ids=[r['shot'] for r in tp if r['arms'][arm]['timely_peak_point_label'] == 0],
            poststartup_D_with_selected_high_negative=[r['shot'] for r in post if r['arms'][arm]['selected_high_negative_count']],
            poststartup_D_selected_high_negative_points=sum(r['arms'][arm]['selected_high_negative_count'] for r in post))
        if len(tp) != completed['metrics'][arm]['full360']['tp']:
            raise AssertionError('full-stream timely count does not reproduce')
    lost = [r for r in rows if r['disruptive'] and r['arms']['cold']['timely'] and not r['arms']['inter']['timely']]
    summary = dict(
        scope='POSTHOC_LOSS_SEMANTICS_DIAGNOSTIC_NOT_DEPLOYED',
        completed_run=str(root), n_shots=len(rows), n_disrupt=sum(r['disruptive'] for r in rows),
        completed_sha256=digest(root / 'completed.json'), threshold=protocol['threshold'],
        classification_label='fixed125ms', timely_metric='strictly_more_than30ms_no_upper_bound',
        totals=totals,
        lost_by_inter=[r['shot'] for r in lost],
        lost_by_inter_with_negative_cold_peak=[r['shot'] for r in lost if r['arms']['cold']['timely_peak_point_label'] == 0],
        lost_by_inter_first_cold_alarm_earlier_than125=[r['shot'] for r in lost if r['arms']['cold']['lead_ms'] > 125.000001],
        poststartup_sampling_reconstructs_candidate_training_points_not_accepted_gradients=True,
        parameter_gradient_causality_proven=False,
        original_predictions_or_model_modified=False, per_shot=rows)
    with output.open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_shot'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
