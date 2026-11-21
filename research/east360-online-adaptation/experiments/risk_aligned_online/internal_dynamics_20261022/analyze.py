"""Recount full sealed TEST360 and attribute beneficial/harmful shot changes."""
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..simple_train import sha
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256


def wilson(x, n):
    z = 1.959963984540054; p = x/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-radius, center+radius]


def main():
    root = Path(__file__).parents[1]/'runs'
    arms = ['static', 'F', 'M', 'E', 'recent3']
    contract = load_test_contract('EAST')
    outcomes = {sid:io23.read_outcome('EAST', sid) for sid in contract['shot_ids']}
    all_results = {}; predictions = {}; traces = {}
    protocols = []
    for name in arms:
        directory = root/f'internal_EAST_{name}_20261022'
        completed = json.loads((directory/'completed.json').read_text())
        protocol = json.loads((directory/'protocol.json').read_text()); protocols.append(protocol)
        if completed['status'] != 'COMPLETED': raise AssertionError('incomplete test arm')
        if sha(Path(protocol['source'])) != protocol['source_sha256']: raise AssertionError('changed prepared artifact')
        z = np.load(directory/'test360_traces.npz'); rows = []; audits = []
        for index, sid in enumerate(contract['shot_ids'], 1):
            receipt = json.loads((directory/f'{index:04d}_{sid}_seal.json').read_text())
            if sha(directory/receipt['npz']) != receipt['sha256']: raise AssertionError('changed prediction seal')
            sealed = np.load(directory/receipt['npz']); times=z[f'{sid}_times']; scores=z[f'{sid}_scores']
            disruptive, down = outcomes[sid]
            valid = io23.labels_at(sealed['times'], disruptive, down, 125.)[1]
            np.testing.assert_array_equal(sealed['scores'][valid], scores)
            np.testing.assert_array_equal(sealed['times'][valid], times)
            rows.append(dict(shot=sid, is_disrupt=disruptive, down=down, times=times, scores=scores))
            audit = json.loads((directory/f'{index:04d}_{sid}_audit.json').read_text())
            for u in audit['updates']:
                if not u['last_target_index'] <= u['now_index'] < u['first_affected_prediction']:
                    raise AssertionError('noncausal update receipt')
            audits.append(audit)
        metrics = score_fixed_test('EAST', rows, protocol['threshold'])
        for key in ('tp', 'fp', 'tpr', 'fpr', 'J'):
            if metrics[key] != completed['metrics'][key]: raise AssertionError(f'recount mismatch {name} {key}')
        predictions[name] = {r['shot']:r for r in metrics['predictions']}
        traces[name] = rows
        all_results[name] = dict(metrics={k:v for k,v in metrics.items() if k != 'predictions'},
            TPR_Wilson95_reference=wilson(metrics['tp'], 180), FPR_Wilson95_reference=wilson(metrics['fp'], 180),
            updates=sum(a['accepted_updates'] for a in audits),
            gradient_steps=(sum(a['gradient_steps'] for a in audits) if name == 'recent3'
                            else sum(a['accepted_updates'] for a in audits)),
            final_fast_norm_quantiles=np.quantile([a['final_fast_norm'] for a in audits], [0,.25,.5,.75,1]).tolist(),
            audit_recount='PASS', elapsed_seconds=completed['elapsed_seconds'])
    if len({p['threshold'] for p in protocols}) != 1 or {p['parent_sha256'] for p in protocols} != {PARENT_SHA256}:
        raise AssertionError('comparison threshold/parent mismatch')
    paired = {}
    baseline = predictions['static']
    for name in arms[1:]:
        changes = dict(rescued_timely=[], lost_timely=[], removed_FP=[], new_FP=[])
        for sid, before in baseline.items():
            after=predictions[name][sid]
            if before['is_disrupt']:
                if not before['timely'] and after['timely']: changes['rescued_timely'].append(sid)
                if before['timely'] and not after['timely']: changes['lost_timely'].append(sid)
            else:
                if before['alarm'] and not after['alarm']: changes['removed_FP'].append(sid)
                if not before['alarm'] and after['alarm']: changes['new_FP'].append(sid)
        changes['net_beneficial_flips'] = len(changes['rescued_timely'])+len(changes['removed_FP'])-len(changes['lost_timely'])-len(changes['new_FP'])
        changes['delta_J'] = changes['net_beneficial_flips']/180.
        paired[name] = changes
    early=[]
    for row in traces['static']:
        if not row['is_disrupt']:
            alarms=np.flatnonzero(row['scores']>=protocols[0]['threshold'])
            if len(alarms): early.append(dict(shot=row['shot'], first_index=int(alarms[0]), start_time_s=float(row['times'][0])))
    prefix = dict(FP_first_index_counts={str(n):sum(r['first_index']<n for r in early) for n in [5,8,10,16,25,50,100,128,500,1000]},
                  earliest_feedback_affects_index=50, records=early,
                  interpretation='FP issued before index50 cannot be undone by this unchanged-alarm intra schedule')
    if sha(DEFAULT_PARENT) != PARENT_SHA256: raise AssertionError('original parent checkpoint changed')
    result = dict(status='VERIFIED_RECOUNT', fixed_test=contract, arms=all_results, paired_vs_static=paired,
        startup_feedback_bound=prefix, parent_sha256=PARENT_SHA256,
        statistical_scope='single preparation seed; Wilson intervals are binomial references, not seed/development-selection uncertainty; no significance claim',
        independent_confirmation=False)
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/internal_dynamics_summary.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output':str(output),'arms':all_results, 'paired':paired}, ensure_ascii=False))


if __name__ == '__main__': main()
