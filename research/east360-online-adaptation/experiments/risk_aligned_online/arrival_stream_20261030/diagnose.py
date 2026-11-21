"""Post-seal descriptive error diagnosis; never used for fitting or selection."""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..conditional_likelihood_20261029.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def describe_shot(times, curves, disruptive, down, threshold, first_update):
    times = np.asarray(times)
    valid = (down-times)*1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    relevant = valid & (np.round((down-times)*1000, 9) > 30) if disruptive else valid
    arms = {}
    for name, probability in curves.items():
        hit = np.flatnonzero(valid & (probability >= threshold))
        first = int(hit[0]) if len(hit) else None
        lead = float(round((down-times[first])*1000, 9)) if disruptive and first is not None else None
        arms[name] = dict(first_alarm_index=first, lead_ms=lead,
                          margin=float(np.max(probability[relevant])-threshold) if relevant.any() else None,
                          mean_abs_probability_change=float(np.abs(probability-curves['cold'])[valid].mean()))
    base = arms['cold']
    error = (base['lead_ms'] is None or base['lead_ms'] <= 30) if disruptive else base['first_alarm_index'] is not None
    return dict(disruptive=bool(disruptive), cold_error=bool(error), arms=arms,
                first_update_index=first_update,
                timely_update_possible=bool(first_update is not None and relevant[first_update:].any()),
                early_sealed_false_alarm=bool(not disruptive and error and
                    (first_update is None or base['first_alarm_index'] < first_update)))


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if len(values) else None


def run(output):
    done, protocol = read(output/'complete.json'), read(output/'protocol.json')
    audit = read(output/'independent_audit.json')
    assert audit['saved_array_audit'] and audit['complete_sha256'] == sha(output/'complete.json')
    arms = protocol['arms']
    rows, feedback_rows = [], []
    for index, sid in enumerate(protocol['order'][20:], 21):
        name = f'{index:04d}_{sid}'
        seal = read(output/'shots'/(name+'_seal.json'))
        assert sha(output/'shots'/(name+'.npz')) == seal['sha256']
        applied = [r for r in seal['updates'] if r['apply_count'] > 0]
        first = applied[0]['first_affected'] if applied else None
        flag, down = io23.read_outcome('EAST', sid)
        with np.load(output/'shots'/(name+'.npz'), allow_pickle=False) as saved:
            row = describe_shot(saved['times'], {a: saved[a] for a in arms},
                                flag, down, protocol['threshold'], first)
        rows.append(dict(shot=sid, index=index, **row))
        feedback_rows.extend(dict(shot=sid, **r) for r in applied)
    groups = {}
    for name, disruptive in (('miss_or_tardy', True), ('false_alarm', False)):
        selected = [r for r in rows if r['cold_error'] and r['disruptive'] == disruptive]
        base = np.array([r['arms']['cold']['margin'] for r in selected], float)
        real = np.array([r['arms']['real']['margin'] for r in selected], float)
        groups[name] = dict(shots=len(selected),
            cold_margin=quantiles(base), real_minus_cold_margin=quantiles(real-base),
            margin_increases=int(np.sum(real > base)), margin_decreases=int(np.sum(real < base)),
            timely_update_possible=sum(r['timely_update_possible'] for r in selected),
            early_sealed_false_alarm=sum(r['early_sealed_false_alarm'] for r in selected))
    evidence = {}
    grouped = {}
    for receipt in feedback_rows:
        grouped.setdefault(receipt['shot'], []).append(receipt)
    real = np.array([r['e']['real'] for r in feedback_rows])
    for arm in arms:
        if arm == 'cold':
            continue
        other = np.array([r['e'][arm] for r in feedback_rows])
        shot_mae = [np.mean([abs(r['e'][arm]-r['e']['real']) for r in receipts])
                    for receipts in grouped.values()]
        evidence[arm] = dict(pooled_pearson=float(np.corrcoef(real, other)[0, 1]) if np.std(other) else None,
                            pooled_sign_agreement=float(np.mean(np.sign(real) == np.sign(other))),
                            equal_shot_mean_abs_difference=float(np.mean(shot_mae)))
    changed = sorted({s for c in done['comparisons'].values() for field in ('rescued', 'harmed') for s in c[field]})
    result = dict(posthoc_diagnostic_only=True, tuning_authorization=False,
                  complete_sha256=sha(output/'complete.json'), suffix_shots=len(rows),
                  groups=groups, evidence_comparison=evidence, changed_shots=[r for r in rows if r['shot'] in changed],
                  all_shots=rows, applied_intervals=len(feedback_rows),
                  no_architecture_upper_bound_claim=True)
    _durable_json(output/'posthoc_diagnosis.json', result)
    print({k: v for k, v in result.items() if k not in ('all_shots', 'changed_shots')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output.absolute())
