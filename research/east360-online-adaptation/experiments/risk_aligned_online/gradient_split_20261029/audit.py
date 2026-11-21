"""Recompute all decisions from sealed curves, independently of stream metrics."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha


def count(records):
    nd=sum(r['is_disrupt'] for r in records);ns=len(records)-nd
    tp=sum(r['timely'] for r in records);fp=sum(not r['is_disrupt'] and r['alarm'] for r in records)
    leads=[r['lead_ms'] for r in records if r['timely']]
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,tpr=tp/nd,fpr=fp/ns,J=tp/nd-fp/ns,
        median_lead_ms=float(np.median(leads)) if leads else None,early_over3000=sum(v>3000 for v in leads))


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');declared=read(root,'per_shot.json')
    if not done['complete'] or len(p['order'])!=360 or len(set(p['order']))!=360:raise ValueError('completed distinct360 required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256:raise ValueError('source changed')
    if sha(Path(p['source_root'])/'task.pt')!=p['task_sha256']:raise ValueError('imputer changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed')
    if sha(io23.ROOT/'GRADIENT_SPLIT_STREAM_PLAN.md')!=p['plan_sha256']:raise ValueError('predeclared plan changed')
    previous=Path(p['previous']);reference=Path(p['reference'])
    if sha(previous/'complete.json')!=p['previous_complete_sha256']:raise ValueError('previous result changed')
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold provenance changed')
    arms=p['arms'];adapt=arms[2:];records={k:[] for k in arms};differences={k:[] for k in adapt};cosines=[]
    updates={k:0 for k in adapt};maximum_norm=0.;hashes=read(reference,'input_hashes.json')
    for i,sid in enumerate(p['order']):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw input changed')
        name=f'{i+1:04d}_{sid}';path=root/'shots'/f'{name}.npz';seal=read(root/'shots',f'{name}_seal.json')
        old_path=previous/'shots'/f'{name}.npz';old_seal=read(previous/'shots',f'{name}_seal.json')
        if seal['outcome_used'] or seal['shot']!=sid or seal['index']!=i+1 or sha(path)!=seal['sha256']:raise ValueError('invalid curve seal')
        if sha(old_path)!=old_seal['sha256']:raise ValueError('previous curve changed')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z,np.load(old_path) as old:
            if not np.array_equal(z['times'],old['times']) or not np.array_equal(z['both_paths'],old['task']):raise ValueError('archived comparator changed')
            if np.max(np.abs(z['cold']-old['cold']))>2e-6 or not np.array_equal(z['cold']>=p['threshold'],old['cold']>=p['threshold']):
                raise ValueError('cold decisions changed')
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool)
            times=z['times'][valid]
            for k in arms:
                score=z[k][valid]
                if not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError('invalid probabilities')
                cross=np.flatnonzero(score>=p['threshold']);alarm=bool(len(cross))
                lead=float(round((down-times[cross[0]])*1000,9)) if alarm and flag else None
                r=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))
                if r!=declared[k][i]:raise ValueError('per-shot decision discrepancy')
                records[k].append(r)
                if i<20 and not np.array_equal(z[k],z['cold']):raise ValueError('cold20 altered')
                if k in adapt:
                    if not np.array_equal(z[k][:128],z['cold'][:128]):raise ValueError('earlier sealed predictions changed')
                    delta=score-z['cold'][valid]
                    item=dict(shot=sid,flag=flag,mean_delta=float(delta.mean()),max_abs_delta=float(np.abs(delta).max()))
                    if flag:
                        timely=np.round((down-times)*1000,9)>30
                        item['timely_peak_delta']=float(score[timely].max()-z['cold'][valid][timely].max()) if timely.any() else None
                    differences[k].append(item)
            if i>=20:
                for k in adapt:
                    q=seal['updates'][k]
                    if q['updates']!=(len(z['times'])-1)//128 or q['max_state_norm']>1.000001:raise ValueError('update count/radius mismatch')
                    updates[k]+=q['updates'];maximum_norm=max(maximum_norm,q['max_state_norm'])
                a=z['stop_target'][valid].astype('float64')-z['cold'][valid]
                b=z['stop_target_reverse'][valid].astype('float64')-z['cold'][valid]
                den=np.linalg.norm(a)*np.linalg.norm(b)
                cosines.append(dict(shot=sid,flag=flag,cosine=float(a@b/den) if den>0 else None))
    comparisons={};metrics={}
    for k in arms:
        metrics[k]={label:count(sub) for label,sub in (('full360',records[k]),('cold20',records[k][:20]),('suffix340',records[k][20:]))}
        for section,values in metrics[k].items():
            for key,val in values.items():
                actual=done['results'][k][section][key]
                if actual is None and val is None:continue
                if actual is None or val is None or abs(actual-val)>1e-12:raise ValueError('aggregate result mismatch')
        if k!='cold':
            rescued=[];harmed=[]
            for old,new in zip(records['cold'],records[k]):
                a=old['timely'] if old['is_disrupt'] else not old['alarm']
                b=new['timely'] if new['is_disrupt'] else not new['alarm']
                if b and not a:rescued.append(old['shot'])
                if a and not b:harmed.append(old['shot'])
            comparisons[k]=dict(rescued=rescued,harmed=harmed)
    if comparisons!=done['comparisons'] or updates!=done['update_counts']:raise ValueError('paired comparison or update mismatch')
    values=[r['cosine'] for r in cosines if r['cosine'] is not None]
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),sealed360_verified=True,source_unchanged=True,
        first128_preserved=True,cold20_included=True,matched_parent_verified=True,metrics=metrics,comparisons=comparisons,
        updates=updates,maximum_temporary_norm=maximum_norm,
        median_candidate_reverse_risk_delta_cosine=float(np.median(values)) if values else None,
        per_shot_delta_cosines=cosines,risk_differences=differences,
        scope='one developed stream/seed; not independent confirmation or physical causal proof')
    _durable_json(root/'audit.json',result)
    print(dict(metrics={k:v['full360'] for k,v in metrics.items()},comparisons=comparisons,updates=updates,
               cosine=result['median_candidate_reverse_risk_delta_cosine']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
