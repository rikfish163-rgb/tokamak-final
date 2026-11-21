"""Independent counts from sealed curves, source identity and complete order."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23,simple_data
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha
from ..simple_online import _durable_json
from .stream import ARMS


def run(root):
    p=read(root,'protocol.json');done=read(root,'complete.json')
    if not done['complete'] or len(p['order'])!=360 or len(set(p['order']))!=360:
        raise ValueError('completed unique360 experiment required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('source changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('experiment code changed')
    training=Path(p['source_root']);tp=read(training,'protocol.json');td=read(training,'complete.json')
    if sha(training/'complete.json')!=p['source_complete_sha256']:raise ValueError('source teaching result changed')
    roles=simple_data.source_roles()
    for key in ('train','val'):
        if tp[key+'_ids']!=roles[key]:raise ValueError('source roles changed')
    validation=read(training,'val_windows.json')
    if [r['shot'] for r in validation]!=roles['val'] or any(len(r['ends'])!=4 for r in validation):
        raise ValueError('VAL160 window coverage changed')
    if read(training,'val_epoch2.json')!=td['val']:raise ValueError('source validation summary changed')
    for epoch in (1,2):
        windows=read(training,f'windows_epoch{epoch}.json')
        if sorted(r['shot'] for r in windows)!=sorted(roles['train']) or any(len(r['ends'])!=4 for r in windows):
            raise ValueError('incomplete source epoch')
    for name,h in td['checkpoints'].items():
        if sha(training/f'{name}.pt')!=h:raise ValueError('imputer changed')
    reference=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('reference changed')
    declared=read(root,'per_shot.json');actual={k:[] for k in ARMS};norms=[];updates={k:0 for k in ARMS[1:]}
    whole_differences={k:[] for k in ARMS[1:]}
    for i,sid in enumerate(p['order']):
        path=root/'shots'/f'{i+1:04d}_{sid}.npz';seal=read(root/'shots',f'{i+1:04d}_{sid}_seal.json')
        if seal['shot']!=sid or seal['index']!=i+1 or seal['outcome_used'] or sha(path)!=seal['sha256']:
            raise ValueError('invalid sealed prediction')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z:
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool)
            times=z['times'][valid]
            for name in ARMS:
                score=z[name][valid];cross=np.flatnonzero(score>=p['threshold']);alarm=bool(len(cross))
                lead=float(round((down-times[cross[0]])*1000,9)) if flag and alarm else None
                record=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))
                if record!=declared[name][i]:raise ValueError('per-shot decision mismatch')
                actual[name].append(record)
                if name!='cold':
                    delta=score-z['cold'][valid]
                    diagnostic=dict(shot=sid,flag=flag,mean_delta=float(delta.mean()),max_abs_delta=float(np.abs(delta).max()))
                    if flag:
                        lead_grid=np.round((down-times)*1000,9)
                        for label,keep in (('timely',lead_grid>30),('near',(lead_grid>30)&(lead_grid<=125)),('early',lead_grid>125)):
                            diagnostic[label+'_points']=int(keep.sum())
                            diagnostic[label+'_mean_delta']=float(delta[keep].mean()) if keep.any() else None
                            diagnostic[label+'_peak_delta']=float(score[keep].max()-z['cold'][valid][keep].max()) if keep.any() else None
                    whole_differences[name].append(diagnostic)
                    if i<20 and not np.array_equal(z[name],z['cold']):raise ValueError('cold20 changed')
                    if i>=20 and not np.array_equal(z[name][:128],z['cold'][:128]):raise ValueError('update affected prefeedback prefix')
            if i>=20:
                for name in ARMS[1:]:
                    info=seal['updates'][name]
                    if info['updates']!=(len(z['times'])-1)//128:raise ValueError('update count does not match causal schedule')
                    if not 0<=info['max_state_norm']<=1.000001:raise ValueError('state radius violated')
                    updates[name]+=info['updates'];norms.append(info['max_state_norm'])
    metrics={}
    for name,r in actual.items():
        nd=sum(q['is_disrupt'] for q in r);ns=360-nd
        tp_count=sum(q['timely'] for q in r);fp=sum(q['alarm'] and not q['is_disrupt'] for q in r)
        if (nd,ns)!=(180,180):raise ValueError('full stream class balance changed')
        recomputed=dict(tp=tp_count,fp=fp,tpr=tp_count/nd,fpr=fp/ns,J=(tp_count-fp)/180)
        reported=done['results'][name]['full360']
        for k,v in recomputed.items():
            if abs(v-reported[k])>1e-12:raise ValueError('aggregate table mismatch')
        metrics[name]=recomputed
    if updates!=done['update_counts']:raise ValueError('update total mismatch')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),source_unchanged=True,
        sealed360_verified=True,source_train1353_two_epochs_verified=True,source_val160_verified=True,
        first128_predictions_before_update_verified=True,counts=metrics,update_counts=updates,
        maximum_temporary_state_norm=max(norms),whole_shot_risk_differences=whole_differences,
        note='One developmental seed. Counts do not establish physical causality or independent generalization.')
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    _durable_json(root/'audit.json',result);print({k:v for k,v in result.items() if k!='whole_shot_risk_differences'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
