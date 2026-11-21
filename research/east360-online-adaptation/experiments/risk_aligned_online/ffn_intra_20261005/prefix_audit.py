"""Post-hoc causal-prefix feasibility bounds on already sealed EAST risks.

This never fits a model or chooses a threshold. A fixed prefix cannot be
rescored: an alarm before the first effective update is irrevocable. The
remainder is allowed arbitrary oracle scores only to derive an optimistic
bound, NOT to assert that any neural update can attain it.
"""
import argparse
from pathlib import Path

import numpy as np

from .. import representation_analysis as analysis
from . import compare_external as paired


def prefix_bound(row, threshold, effect_index):
    """Indices refer to the original emitted grid, before outcome filtering."""
    times, scores = analysis._row_arrays(row)
    n = len(times)
    valid = np.asarray(row['valid'])
    if (not n or np.any(np.diff(times) <= 0) or valid.shape != (n,)
            or valid.dtype != bool):
        raise ValueError('nonempty increasing times and aligned boolean validity required')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('fixed probability threshold required')
    if effect_index is not None and (type(effect_index) is not int or effect_index < 0):
        raise ValueError('effect index must be a nonnegative integer or None')
    effect = min(n, effect_index) if effect_index is not None else n
    disruptive = row['is_disrupt']
    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError('boolean outcome required')
    valid = valid.copy()
    timely = np.zeros(n, bool)
    if disruptive:
        down = row['down']
        if down is None or not np.isfinite(down):
            raise ValueError('disruptive outcome requires finite DownTime')
        valid &= times <= down
        timely = valid & (np.round((down - times) * 1000., 9) > 30.)
    alarms = np.flatnonzero(valid & (scores >= threshold))
    first = int(alarms[0]) if len(alarms) else None
    alarm = first is not None
    lead = analysis.rounded_lead_ms(row['down'], times[first]) if disruptive and alarm else None
    category = analysis.classify_prediction(disruptive, alarm, lead)
    prefix = np.arange(n) < effect
    locked_timely = bool(np.any(prefix & timely & (scores >= threshold)))
    opportunity = bool(np.any(~prefix & timely))
    return dict(shot=int(row['shot']), is_disrupt=bool(disruptive), static_class=category,
        effect_index=effect_index, fixed_prefix_points=effect,
        effect_elapsed_ms=float((times[effect] - times[0]) * 1000.) if effect < n else None,
        first_alarm_index=first,
        first_alarm_elapsed_ms=float((times[first] - times[0]) * 1000.) if alarm else None,
        unavoidable_fp=bool(not disruptive and np.any(prefix & valid & (scores >= threshold))),
        locked_timely=locked_timely,
        mutable_timely_opportunity=opportunity,
        unavoidable_miss=bool(disruptive and not locked_timely and not opportunity))


def summarize_bounds(rows):
    negative = [r for r in rows if not r['is_disrupt']]
    positive = [r for r in rows if r['is_disrupt']]
    locked = [r['shot'] for r in negative if r['unavoidable_fp']]
    missed = [r['shot'] for r in positive if r['unavoidable_miss']]
    return dict(n_safe=len(negative), n_disrupt=len(positive), shot_count=len(rows),
        static_fp=sum(r['static_class'] == 'FP' for r in negative),
        static_tp=sum(r['static_class'] == 'TP' for r in positive),
        unavoidable_fp_count=len(locked), unavoidable_fp_shots=locked,
        unavoidable_miss_count=len(missed), unavoidable_miss_shots=missed,
        locked_timely_count=sum(r['locked_timely'] for r in positive),
        fpr_lower_bound=len(locked)/len(negative) if negative else None,
        tpr_upper_bound=1-len(missed)/len(positive) if positive else None)


def audit(paired_path, output):
    paired_path, output = Path(paired_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    prior = analysis._read_json(paired_path)
    if not prior['verification']['full2953_verified']:
        raise ValueError('full verified paired report required')
    runs = [paired._read_run(Path(prior['runs'][arm]).resolve(), arm, 2953)
            for arm in ('static', 'fixed')]
    if any(output.is_relative_to(run['root']) for run in runs):
        raise ValueError('new analysis must remain outside both sealed input runs')
    contract = paired._contracts(*runs)
    for module in (analysis.simple_data, analysis.simple_data.io23):
        path = Path(module.__file__).resolve()
        paired._check_sha(path, contract['helper_sha256'][path.name])
    receipts = [paired._sealed_receipts(run) for run in runs]
    loader = paired._strict_outcomes(analysis.simple_data.attach_outcome)
    threshold = runs[0]['protocol']['threshold']
    methods = {'partial_earliest6': [], 'full_earliest26': [], 'fixed_recorded_effect': []}
    predictions = [[], []]
    prefix_max_abs = 0.
    alarm_disagreements = []
    all_rows = []
    audit_hashes = []
    for index, (left, right) in enumerate(zip(*receipts), 1):
        traces = [paired._trace(run['path'], receipt)
                  for run, receipt in zip(runs, (left, right))]
        if not np.array_equal(traces[0]['times'], traces[1]['times']):
            raise ValueError('paired grids changed')
        trace = traces[0]
        outcome = loader(dict(shot=trace['shot'], device='EAST', times=trace['times']), 'EAST', {})
        row = dict(trace, **{k: outcome[k] for k in ('valid', 'down', 'is_disrupt')})
        stem = Path(right['npz']).name.removesuffix('_sealed.npz')
        audit_path = runs[1]['path'] / f'{stem}_audit.json'
        saved = analysis._read_json(audit_path)
        if saved['seal_sha256'] != right['sha256'] or saved['shot'] != trace['shot']:
            raise ValueError('fixed audit/seal mismatch')
        audit_hashes.append(dict(path=str(audit_path), sha256=analysis._sha256(audit_path)))
        effects = [e['effective_index'] for e in saved['events']
                   if e['accepted'] and e['effective_index'] is not None]
        if any(type(i) is not int or i < 26 or i >= len(trace['times']) for i in effects):
            raise ValueError('invalid recorded effect index')
        effect = min(effects) if effects else None
        end = effect if effect is not None else len(trace['times'])
        difference = np.abs(traces[0]['scores'][:end] - traces[1]['scores'][:end])
        prefix_max_abs = max(prefix_max_abs, float(difference.max(initial=0.)))
        if np.any((traces[0]['scores'][:end] >= threshold) !=
                  (traces[1]['scores'][:end] >= threshold)):
            alarm_disagreements.append(trace['shot'])
        cuts = dict(partial_earliest6=6, full_earliest26=26, fixed_recorded_effect=effect)
        records = {key: prefix_bound(row, threshold, cut) for key, cut in cuts.items()}
        for key, value in records.items():
            methods[key].append(value)
        all_rows.append(dict(stream_index=index, shot=trace['shot'], bounds=records))
        valid = outcome['valid'].copy()
        if outcome['is_disrupt']:
            valid &= trace['times'] <= outcome['down']
        for destination, candidate in zip(predictions, traces):
            metric_row = dict(shot=trace['shot'], is_disrupt=outcome['is_disrupt'], down=outcome['down'],
                              times=candidate['times'][valid], scores=candidate['scores'][valid])
            destination.append(analysis.metrics_for_rows([metric_row], threshold)['predictions'][0])
        if index % 500 == 0:
            print(f'verified {index}/2953 sealed pairs', flush=True)
    for run, rows in zip(runs, predictions):
        analysis._verify_summary(run['summary'], run['summary']['metrics'], paired._metrics(rows, threshold), 2953)
    result = dict(kind='POST_HOC_IMMUTABLE_PREFIX_BOUND_NOT_ACHIEVABLE_MODEL',
        paired_sha256=analysis._sha256(paired_path), source_sha256=prior['source_sha256'],
        analysis_code_sha256={str(Path(module.__file__).resolve()): analysis._sha256(Path(module.__file__))
                              for module in (analysis, paired)},
        code_sha256=analysis._sha256(Path(__file__)),
        threshold=threshold, complete2953=True, model_updated=False, gpu_used=False,
        meaning='arbitrary oracle risk is permitted after effect solely for optimistic bounds; no rescues claimed',
        partial_note='earliest6 is a hypothetical timing bound; partial model was NOT replayed on EAST here',
        index_note='zero-based original grid, not filtered indices or guaranteed milliseconds',
        prefix_fixed_vs_static=dict(max_abs_probability_difference=prefix_max_abs,
                                    alarm_disagreement_shots=alarm_disagreements),
        segments={name: {method: summarize_bounds(rows[lo:hi]) for method, rows in methods.items()}
                  for name, lo, hi in (('overall', 0, 2953), ('first30', 0, 30),
                                      ('31to120', 30, 120), ('121toend', 120, 2953))},
        audit_sha256=audit_hashes, per_shot=all_rows,
        limitations='development exposed; no raw upstream causality proof, physical mechanism or independent validation')
    analysis._write_json_once(output, result)
    print({method: {key: value for key, value in values.items() if not key.endswith('_shots')}
           for method, values in result['segments']['overall'].items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.paired, args.output)


if __name__ == '__main__':
    main()
