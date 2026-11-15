"""Post-run threshold capacity diagnostic; no predictor or neural adaptation."""

import argparse
from pathlib import Path
import time

import numpy as np

from . import io23
from .history_gradient_metric_20261102.run import read, sha, write

REFERENCE = io23.ROOT / 'probe_outputs/qualified_source_static360_20261104_v1'
CAPS = (0.10, 0.15, 0.20)


def shot_maximum(times, scores, disruptive, down):
    times, scores = np.asarray(times), np.asarray(scores)
    if (times.ndim != 1 or scores.shape != times.shape or not len(times)
            or not np.isfinite(times).all() or not np.isfinite(scores).all()
            or (np.diff(times) <= 0).any() or (scores < 0).any() or (scores > 1).any()
            or type(disruptive) is not bool):
        raise ValueError('finite increasing clock, probabilities and HDF outcome required')
    if disruptive:
        if down is None or not np.isfinite(down):
            raise ValueError('finite HDF disruption time required')
        # Match saved first-alarm evaluation: nine-decimal ms rounding and >30.
        eligible = np.round((down - times) * 1000, 9) > 30
    else:
        if down is not None:
            raise ValueError('a safe shot has no disruption time')
        eligible = np.ones(len(times), bool)
    return float(np.max(scores[eligible])) if eligible.any() else None


def cutoff_from_safe(maxima, cap):
    scores = np.asarray(maxima, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()
            or not np.isfinite(cap) or not 0 <= cap < 1):
        raise ValueError('nonempty safe preparation probabilities and cap in [0,1) required')
    boundary = np.sort(scores)[::-1][int(np.floor(cap * len(scores)))]
    # Saved probabilities are float32. A float32 successor also remains strict
    # when a future implementation compares a Python cutoff with float32 data.
    return float(np.nextafter(np.float32(boundary), np.float32(np.inf)))


def operating_point(rows, cutoff):
    nd = sum(r['disruptive'] for r in rows)
    ns = len(rows) - nd
    if not nd or not ns or not np.isfinite(cutoff):
        raise ValueError('both HDF outcome classes and finite cutoff required')
    tp = sum(r['disruptive'] and r['maximum'] is not None
             and r['maximum'] >= cutoff for r in rows)
    fp = sum(not r['disruptive'] and r['maximum'] is not None
             and r['maximum'] >= cutoff for r in rows)
    return dict(tp=tp, fp=fp, n_disruptive=nd, n_safe=ns,
                tpr=tp / nd, fpr=fp / ns, J=tp / nd - fp / ns)


def oracle_point(rows, cap):
    # This searches all360 outcomes and is deliberately NOT a deployment arm.
    candidates = {0.0}
    for r in rows:
        if r['maximum'] is not None:
            candidates.add(r['maximum'])
            candidates.add(float(np.nextafter(np.float32(r['maximum']), np.float32(np.inf))))
    options = [(cut, operating_point(rows, cut)) for cut in sorted(candidates)]
    feasible = [(cut, point) for cut, point in options if point['fpr'] <= cap]
    cutoff, point = max(feasible, key=lambda v: (v[1]['tp'], -v[1]['fp'], -v[0]))
    return dict(cutoff=cutoff, **point, uses_all360_outcomes=True, deployable=False)


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    protocol, completed = read(REFERENCE / 'protocol.json'), read(REFERENCE / 'completed.json')
    for name, expected in completed['artifact_sha256'].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError('sealed baseline artifact changed')
    ids = [r['shot'] for r in protocol['order']]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids('EAST')):
        raise ValueError('locked EAST360 membership required')
    rows = {arm: [] for arm in protocol['arms']}
    inputs = {str(REFERENCE / 'completed.json'): sha(REFERENCE / 'completed.json')}
    for i, sid in enumerate(ids, 1):
        path = REFERENCE / 'shots' / f'{i:04d}_{sid}.npz'
        seal = read(path.with_name(path.stem + '_seal.json'))
        if sha(path) != seal['curve_sha256'] or seal['current_outcome_used_in_prediction'] is not False:
            raise ValueError('baseline prediction seal changed')
        disruptive, down = io23.read_outcome('EAST', sid)
        inputs[str(path)] = seal['curve_sha256']
        with np.load(path) as curve:
            for arm in rows:
                rows[arm].append(dict(shot=sid, disruptive=disruptive,
                                     maximum=shot_maximum(curve['times'], curve[arm], disruptive, down)))
    fixed = {arm: operating_point(values, protocol['threshold']) for arm, values in rows.items()}
    reference = read(REFERENCE / 'summary.json')
    for arm, point in fixed.items():
        if any(point[k] != reference[arm]['full360'][k] for k in point):
            raise AssertionError('shot-maximum diagnostic changed primary evaluation')
    capacity = {arm: {str(cap): oracle_point(values, cap) for cap in CAPS}
                for arm, values in rows.items()}
    preparation = {}
    for arm in ('unmapped', 'retrospective_map_all'):
        values = rows[arm]
        preparation[arm] = {}
        for cap in CAPS:
            cutoff = cutoff_from_safe([r['maximum'] for r in values[:20] if not r['disruptive']], cap)
            mixed = []
            for i, r in enumerate(values):
                current, threshold = (rows['unmapped'][i], protocol['threshold']) if i < 20 else (r, cutoff)
                alarm = current['maximum'] is not None and current['maximum'] >= threshold
                mixed.append(dict(current, maximum=float(alarm)))
            preparation[arm][str(cap)] = dict(
                cutoff=cutoff, preparation_safe_shots=12, preparation_labeled_shots=20,
                preparation_fpr_on_ended_data=operating_point(values[:20], cutoff)['fpr'],
                full360=operating_point(mixed, 0.5), suffix340=operating_point(mixed[20:], 0.5),
                first20_predictions_unchanged=True, empirical_suffix_fpr_not_guaranteed=True)
    output.mkdir(parents=True)
    write(output / 'rows.json', rows)
    write(output / 'summary.json', dict(
        fixed_threshold=fixed, retrospective_all360_threshold_capacity=capacity,
        first20_label_threshold_controls=preparation))
    result = dict(
        status='COMPLETED_POSTRUN_THRESHOLD_DIAGNOSTIC_NOT_NEURAL_GAIN',
        seconds=time.monotonic() - started, raw_signal_reads=0, source_model_loaded=False,
        neural_updates=0, alarm_method_changed_in_mainline=False,
        first20_threshold_selection_uses_only_ended_safe_shots=True,
        independent_future_validation=False, input_sha256=inputs,
        code_sha256=sha(Path(__file__)), artifact_sha256={p.name: sha(p) for p in output.iterdir()})
    write(output / 'completed.json', result)
    print(dict(status=result['status'], seconds=result['seconds'], fixed_threshold=fixed,
               oracle20={a: v['0.2'] for a, v in capacity.items()},
               preparation20={a: v['0.2'] for a, v in preparation.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
