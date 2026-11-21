"""Post-seal error/direction audit, never used for fitting this experiment."""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..arrival_stream_20261030.diagnose import describe_shot, quantiles
from ..conditional_likelihood_20261029.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def run(output):
    done, protocol = read(output/'complete.json'), read(output/'protocol.json')
    audit = read(output/'independent_audit.json')
    assert audit['saved_array_audit'] and audit['complete_sha256'] == sha(output/'complete.json')
    arms = protocol['arms']
    rows, feedback = [], []
    curve_agreement = {a: [] for a in arms}
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
            for arm in arms:
                curve_agreement[arm].append(float(np.mean(np.abs(saved[arm]-saved['operational']))))
        rows.append(dict(shot=sid, index=index, **row))
        feedback.extend(dict(shot=sid, **r) for r in applied)
    groups = {}
    for name, disruptive in (('miss_or_tardy', True), ('false_alarm', False)):
        selected = [r for r in rows if r['cold_error'] and r['disruptive'] == disruptive]
        base = np.array([r['arms']['cold']['margin'] for r in selected], float)
        arm_differences = {}
        for arm in arms:
            current = np.array([r['arms'][arm]['margin'] for r in selected], float)
            arm_differences[arm] = dict(delta_margin=quantiles(current-base),
                increases=int(np.sum(current > base)), decreases=int(np.sum(current < base)))
        groups[name] = dict(shots=len(selected), cold_margin=quantiles(base),
            differences=arm_differences,
            timely_update_possible=sum(r['timely_update_possible'] for r in selected),
            early_sealed_false_alarm=sum(r['early_sealed_false_alarm'] for r in selected))
    geometry = {arm: dict(norm=quantiles([r['state_norm'][arm] for r in feedback]),
        cosine_signal=quantiles([r['cosine_signal'][arm] for r in feedback]),
        cosine_risk=quantiles([r['cosine_risk'][arm] for r in feedback]),
        cosine_operational=quantiles([r['cosine_operational'][arm] for r in feedback]),
        first_order_origin_logit_change=quantiles([r['state_norm'][arm] *
            r['risk_gradient_norm'] * r['cosine_risk'][arm] for r in feedback]),
        equal_shot_mean_absolute_difference_from_operational=float(np.mean(curve_agreement[arm])))
        for arm in arms}
    changed = sorted({s for c in done['comparisons'].values() for field in ('rescued', 'harmed') for s in c[field]})
    result = dict(posthoc_diagnostic_only=True, complete_sha256=sha(output/'complete.json'),
        suffix_shots=len(rows), groups=groups, geometry=geometry,
        changed_shots=[r for r in rows if r['shot'] in changed], all_shots=rows,
        applied_intervals=len(feedback), no_architecture_upper_bound_claim=True)
    _durable_json(output/'posthoc_diagnosis.json', result)
    print({k: v for k, v in result.items() if k not in ('all_shots', 'changed_shots')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output.absolute())
