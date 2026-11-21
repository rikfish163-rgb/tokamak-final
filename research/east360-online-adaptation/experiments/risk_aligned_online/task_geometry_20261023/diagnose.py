"""Read sealed results only: failure types, first-update limits and ROC diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..operating_curve_audit import operating_curve
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import NAMES, READ


def main(root):
    p, full, verified = READ(root/'protocol.json'), READ(root/'per_shot.json'), READ(root/'verification.json')
    assert verified['status'] == 'VERIFIED' and verified['completed_sha256'] == sha(root/'completed.json')
    records = {a:[] for a in NAMES}; endpoints = []
    early_fp = {a:[] for a in NAMES}; errors = {}
    trajectory_change = {a:dict(points=0,changed_over_1e6=0,max_abs=0.,sum_abs=0.) for a in NAMES[2:]}
    for a in NAMES:
        errors[a] = dict(never_alarm=[],late_alarm=[],false_alarm=[])
        for r in full[a]['predictions']:
            if r['is_disrupt'] and not r['timely']:
                errors[a]['late_alarm' if r['alarm'] else 'never_alarm'].append(r['shot'])
            elif not r['is_disrupt'] and r['alarm']:
                errors[a]['false_alarm'].append(r['shot'])
    for sid in p['contract']['shot_ids']:
        disruptive, down = io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            times = z['times']
            valid = (down-times)*1000 >= -1e-6 if disruptive else np.ones(len(times),bool)
            eligible = valid & (np.round((down-times)*1000,9)>30) if disruptive else valid
            detail = dict(shot=sid,is_disrupt=bool(disruptive),arms={})
            for a in NAMES:
                score = z[a]
                if a in trajectory_change:
                    difference = np.abs(score.astype('float64')-z['ordinary'])
                    stat = trajectory_change[a]; stat['points'] += len(difference)
                    stat['changed_over_1e6'] += int((difference>1e-6).sum())
                    stat['sum_abs'] += float(difference.sum())
                    stat['max_abs'] = max(stat['max_abs'],float(difference.max()))
                records[a].append(dict(shot=sid,is_disrupt=disruptive,down=down,valid=valid,
                                       times=times.copy(),scores=score.copy()))
                if not disruptive and (score[:256] >= p['threshold']).any():
                    early_fp[a].append(sid)
                indices = np.flatnonzero(eligible)
                end = int(indices[np.argmax(score[indices])]) if len(indices) else None
                detail['arms'][a] = dict(eligible_peak=float(score[end]) if end is not None else None,
                    peak_index=end,peak_after_first_update=bool(end is not None and end>=256),
                    peak_lead_ms=float((down-times[end])*1000) if disruptive and end is not None else None)
            endpoints.append(detail)
    operating = {}
    for a in NAMES:
        curve = operating_curve(records[a],p['threshold'])
        actual = READ(root/'completed.json')['metrics'][a]
        for k in ('tp','fp','tpr','fpr','J'):
            assert curve['fixed_source_threshold'][k] == actual[k]
        operating[a] = {k:v for k,v in curve.items() if k != 'curve'}
    assert all(early_fp[a] == early_fp['static'] for a in NAMES)
    for stat in trajectory_change.values():
        stat['mean_abs'] = stat['sum_abs']/stat['points']
    result = dict(scope='POSTHOC DEVELOPMENT DIAGNOSIS; no learner or threshold changes',
        code_sha256=sha(Path(__file__)),verification_sha256=sha(root/'verification.json'),
        first256_false_alarm_ids=early_fp,errors=errors,peak_details=endpoints,operating=operating,
        trajectory_change_vs_ordinary=trajectory_change,
        first_update_limit='A past sealed false alarm cannot be undone by any subsequent intra update.',
        selected_thresholds_are_future_oracles=True,independent_confirmation=False)
    _durable_json(root/'posthoc_diagnostic.json',result)
    print(json.dumps(dict(early_fp=len(early_fp['static']),trajectory_change=trajectory_change,
        errors={a:{k:len(v) for k,v in e.items()} for a,e in errors.items()},
        oracle_caps={a:o['posthoc_best_at_cap'] for a,o in operating.items()})),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
