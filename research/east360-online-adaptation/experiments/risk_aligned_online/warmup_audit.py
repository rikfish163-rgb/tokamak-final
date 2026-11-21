"""Irrevocable first-alarm floor for delayed pure-intra proposals.

Retrospective sealed static evidence only. No model updates or thresholds.
Conditional bound: unchanged source/threshold/first-alarm and no inter state.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .score_shift_audit import sha


def first_effect(window):
    if type(window) is not int or window<=0: raise ValueError('positive integer window')
    origin=((window-1)//26)*26
    return next(end+1 for end in (origin+5,origin+25,origin+31) if end>=window-1)


def prefix_bound(first_alarms,effect):
    if not first_alarms or type(effect) is not int or effect<0:
        raise ValueError('nonempty safe roster and nonnegative effect index required')
    count=sum(index is not None and index<effect for index in first_alarms)
    return dict(first_effect_index=effect,irrevocable_fp=count,n_safe=len(first_alarms),
                fpr_lower_bound=count/len(first_alarms))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve previous bound')
    audit=json.loads(args.score_audit.read_text())
    labels={r['shot']:r['is_disrupt'] for r in audit['rows']}
    if len(labels)!=2953: raise ValueError('fixed complete2953 evidence required')
    root=Path(audit['baseline']); rows=[]; threshold=None; seen=set()
    for path in sorted(root.glob('*_seal.json')):
        receipt=json.loads(path.read_text()); sid=int(receipt['shot'])
        if sid in seen or sid not in labels: raise ValueError('duplicate/unknown shot')
        seen.add(sid)
        if labels[sid]: continue
        if threshold is None: threshold=receipt['threshold']
        if threshold!=receipt['threshold']: raise ValueError('static threshold changed')
        file=root/receipt['npz']; seal=sha(file)
        if seal!=receipt['sha256'] or seal!=audit['seals'][str(sid)]['baseline']:
            raise ValueError('static score seal changed')
        with np.load(file,allow_pickle=False) as trace:
            if int(trace['shot'])!=sid: raise ValueError('shot identity mismatch')
            hits=np.flatnonzero(trace['scores']>=threshold)
            index=int(hits[0]) if len(hits) else None
            rows.append(dict(shot=sid,first_alarm_index=index,seal_sha256=seal))
    if seen!=set(labels) or len(rows)!=1755: raise ValueError('same complete safe cohort required')
    indices=[r['first_alarm_index'] for r in rows]
    if sum(i is not None for i in indices)!=1044: raise ValueError('static FP total differs')
    effect=first_effect(768)
    result=dict(scope='CONDITIONAL_DIAGNOSTIC_NOT_A_DEPLOYED_METHOD',
        assumptions='same source, threshold, first-alarm rule, no inter-shot change; no update before given index',
        source_audit_sha256=sha(args.score_audit),code_sha256=sha(__file__),threshold=threshold,
        window768_first_effect=effect,bounds={str(e):prefix_bound(indices,e) for e in (6,26,128,768,effect)},
        rows=rows,model_loaded=False,training=False)
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['bounds'],indent=2))


if __name__=='__main__': main()
