"""Post-run comparison: only feedback arrival timing changes intentionally."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.compare_label_policy import read
from ..source_task_memory_20261026.compare import verify_source_receipt


def validate_pair(delayed, progressive):
    declared = {'feedback', 'feedback_plan_sha256', 'source_memory_training_feedback',
                'source_deployment_feedback_match', 'code_sha256'}
    for key in delayed.keys() | progressive.keys():
        if key not in declared and delayed.get(key) != progressive.get(key):
            raise ValueError('changed paired protocol: '+key)
    if (delayed.get('feedback', 'delayed') != 'delayed'
            or progressive.get('feedback') != 'progressive'
            or progressive.get('source_memory_training_feedback') != 'delayed'
            or progressive.get('source_deployment_feedback_match') is not False
            or len(progressive.get('feedback_plan_sha256', '')) != 64):
        raise ValueError('explicit delayed-to-progressive feedback contract required')
    approved = {'lifecycle_stream_20261025/model.py', 'lifecycle_stream_20261025/run.py',
                'delayed_recurrent_memory_20261025/model.py',
                'progressive_feedback_20261026/__init__.py', 'progressive_feedback_20261026/model.py'}
    a, b = delayed['code_sha256'], progressive['code_sha256']
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if not changed <= {'experiments/risk_aligned_online/'+p for p in approved}:
        raise ValueError('unreviewed code changes outside feedback implementation')


def compare(delayed, progressive):
    roots = [delayed, progressive]
    protocols = [read(r, 'protocol.json') for r in roots]
    audits = [read(r, 'independent_audit.json') for r in roots]
    if not all(a['complete360'] for a in audits):
        raise ValueError('two complete independently reconstructed streams required')
    validate_pair(*protocols)
    provenance = verify_source_receipt(protocols[0]['source_memory'], protocols[0]['source_sha256'])
    files = ('order.json', 'input_map.json', 'source_replay.json', 'input_hashes.json')
    for filename in files:
        if read(delayed, filename) != read(progressive, filename):
            raise ValueError('changed paired data: '+filename)
    arms, threshold = protocols[0]['arms'], protocols[0]['threshold']
    rows = [{arm: [] for arm in arms} for _ in roots]
    cases = []
    probability = dict(changed=0, total=0, absolute_sum=0., maximum=0.)
    for index, item in enumerate(read(delayed, 'order.json'), 1):
        sid = item['shot']; name = f'{index:04d}_{sid}.npz'
        disruptive, down = io23.read_outcome('EAST', sid)
        with np.load(delayed/'shots'/name) as a, np.load(progressive/'shots'/name) as b:
            for key in ('times', 'source', 'aligned'):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f'unchanged curve differs: {sid}/{key}')
            case = dict(shot=sid, index=index, disruptive=disruptive)
            for run, values in enumerate((a, b)):
                for arm in arms:
                    result = dict(shot=sid, disruptive=disruptive,
                                  **classify(values['times'], values[arm], disruptive, down, threshold))
                    rows[run][arm].append(result)
                case['delayed' if run == 0 else 'progressive'] = {
                    arm: rows[run][arm][-1] for arm in arms}
            delta = np.abs(b['intra'].astype(float)-b['cold'])
            probability['changed'] += int(np.count_nonzero(delta))
            probability['total'] += delta.size
            probability['absolute_sum'] += float(delta.sum())
            probability['maximum'] = max(probability['maximum'], float(delta.max()))
            cases.append(case)
    for i in range(2):
        if {k: counts(v) for k, v in rows[i].items()} != audits[i]['metrics']:
            raise ValueError('reconstructed scores differ from independent audit')
    receipts = read(progressive, 'receipts.json')
    fast = [r['fast'] for r in receipts[20:]]
    if not all(f['feedback'] == 'progressive' and f['first_affected_index'] == 16
               and f['reset_each_shot'] for f in fast):
        raise ValueError('wrong per-shot feedback implementation')
    probability['mean_absolute'] = probability.pop('absolute_sum')/probability['total']
    paired = {arm: flips(rows[0][arm], rows[1][arm]) for arm in arms}
    within = audits[1]['flips']
    changed_shots = {sid for d in [*paired.values(), *within.values()] for v in d.values() for sid in v}
    result = dict(delayed_run=str(delayed.resolve()), progressive_run=str(progressive.resolve()),
        source_provenance=provenance, identical_source_aligned_shots=360,
        unchanged_inputs=list(files), threshold=threshold,
        metrics={name: audit['metrics'] for name, audit in zip(('delayed', 'progressive'), audits, strict=True)},
        progressive_minus_delayed=paired, progressive_within_arm_flips=within,
        intra_minus_cold_probability_change=probability,
        writes=sum(f['updates'] for f in fast),
        horizon_targets=np.array([f['horizon_targets'] for f in fast]).sum(0).tolist(),
        target_scalars=sum(f['target_scalars'] for f in fast),
        accepted_updates=audits[1]['accepted_updates'],
        cases=[c for c in cases if c['shot'] in changed_shots],
        scope='ONE_SEED_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION',
        attribution='Cold-start parameters may differ; use within-run intra versus cold for isolated fast gain.',
        shadow='both_shadow is fast-off in the both trajectory, not an independent inter run')
    with (progressive/'paired_feedback_audit.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('delayed', type=Path); p.add_argument('progressive', type=Path)
    args = p.parse_args(); compare(args.delayed, args.progressive)
