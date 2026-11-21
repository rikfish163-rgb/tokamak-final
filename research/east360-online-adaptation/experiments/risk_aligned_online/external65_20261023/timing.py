"""Posthoc immutable-first-alarm lower bound, not an alarm suppression policy."""
import argparse
from pathlib import Path

import numpy as np

from ..credit_horizon_20261023.run import READ
from ..simple_online import _durable_json
from ..simple_train import sha


def first_alarm(scores,threshold,first_affected):
    indices=np.flatnonzero(np.asarray(scores)>=threshold)
    index=int(indices[0]) if len(indices) else None
    return dict(index=index,before_update=bool(index is not None and index<first_affected))


def main(root):
    out=root/'posthoc_timing.json'
    if out.exists():raise FileExistsError(out)
    p=READ(root/'protocol.json');done=READ(root/'completed.json');summary={};details=[]
    safe=[r['shot'] for r in p['roster'] if not r['is_disrupt']]
    for f in range(4):
        summary[str(f)]={}
        for model in ('warm','prepared'):
            early=[];before128=[]
            for sid in safe:
                path=root/f'fold{f}'/'shots'/f'{sid}.npz'
                assert sha(path)==done['artifact_sha256'][str(path.relative_to(root))]
                updates=READ(path.with_name(f'{sid}_updates.json'))[model]
                with np.load(path) as z:
                    affected=updates[0]['first_affected_prediction'] if updates else len(z['times'])
                    assert not updates or affected==256
                    a=first_alarm(z[f'{model}_static'],p['threshold'],affected)
                    b=first_alarm(z[f'{model}_intra'],p['threshold'],affected)
                    np.testing.assert_array_equal(z[f'{model}_static'][:affected],z[f'{model}_intra'][:affected])
                    assert a['before_update']==b['before_update']
                    if a['before_update']:
                        assert a['index']==b['index'];early.append(sid)
                    if a['index'] is not None and a['index']<128:before128.append(sid)
                    details.append(dict(fold=f,model=model,shot=sid,first_affected=affected,
                        record_start_seconds=float(z['times'][0]),
                        first_alarm_seconds=None if a['index'] is None else float(z['times'][a['index']]),**a))
            summary[str(f)][model]=dict(early_false_alarms=early,before128=before128,
                safe_shots=len(safe),fpr_floor=len(early)/len(safe),
                cap20_feasible_from_later_updates=len(early)/len(safe)<=.2)
    result=dict(status='POSTHOC_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        code_sha256=sha(Path(__file__)),predictions_changed=False,summary=summary,details=details,
        limitation='Record start is not physical discharge start; bound assumes immutable first alarm and unchanged pre-update predictions.')
    _durable_json(out,result);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
