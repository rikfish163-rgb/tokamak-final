"""Independent full360 decisions and matured-feedback accounting."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..gradient_split_20261029.audit import count
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');declared=read(root,'per_shot.json')
    if not done['complete'] or len(set(p['order']))!=360 or len(p['order'])!=360:raise ValueError('completed360 required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256:raise ValueError('source changed')
    if sha(io23.ROOT/'ATTENTION_LOWRANK_PLAN.md')!=p['plan_sha256']:raise ValueError('plan changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed')
    previous=Path(p['previous']);reference=Path(p['reference'])
    if sha(previous/'complete.json')!=p['previous_complete_sha256']:raise ValueError('prior result changed')
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold reference changed')
    hashes=read(reference,'input_hashes.json');mapping=read(reference,'input_map.json');arms=p['arms']
    records={k:[] for k in arms};differences={k:[] for k in arms[1:]}
    totals={k:dict(opportunities=0,updates=0,observed_targets=0,loss_decreased=0,loss_before_sum=0.,loss_after_sum=0.) for k in arms[1:]}
    maxnorm=0.;matrixnorm=0.;first_error=0.
    for i,sid in enumerate(p['order']):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        name=f'{i+1:04d}_{sid}';path=root/'shots'/f'{name}.npz';seal=read(root/'shots',f'{name}_seal.json')
        if seal['outcome_used'] or seal['shot']!=sid or seal['index']!=i+1 or sha(path)!=seal['sha256']:raise ValueError('invalid seal')
        oldpath=previous/'shots'/f'{name}.npz';oldseal=read(previous/'shots',f'{name}_seal.json')
        if sha(oldpath)!=oldseal['sha256']:raise ValueError('previous curve changed')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z,np.load(oldpath) as old:
            if not np.array_equal(z['times'],old['times']) or np.max(np.abs(z['cold']-old['cold']))>2e-6:raise ValueError('cold mismatch')
            if not np.array_equal(z['cold']>=p['threshold'],old['cold']>=p['threshold']):raise ValueError('cold alarm changed')
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool)
            times=z['times'][valid];n=len(z['times'])
            for k in arms:
                score=z[k][valid]
                if not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError('invalid probability')
                cross=np.flatnonzero(score>=p['threshold']);alarm=bool(len(cross))
                lead=float(round((down-times[cross[0]])*1000,9)) if alarm and flag else None
                r=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))
                if r!=declared[k][i]:raise ValueError('decision mismatch')
                records[k].append(r)
                if i<20 and not np.array_equal(z[k],z['cold']):raise ValueError('cold20 altered')
                if k!='cold':
                    err=float(np.max(np.abs(z[k][:256]-z['cold'][:256])));first_error=max(first_error,err)
                    if err>2e-6 or not np.array_equal(z[k][:256]>=p['threshold'],z['cold'][:256]>=p['threshold']):raise ValueError('first256 decision changed')
                    delta=score-z['cold'][valid]
                    differences[k].append(dict(shot=sid,flag=flag,mean_delta=float(delta.mean()),max_abs_delta=float(np.abs(delta).max())))
        if i>=20:
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            clocks=list(range(256,n,128));observed=0;eligible=0
            for hi in clocks:
                count_targets=0
                for origin in (hi-26,hi-51,hi-76,hi-101):
                    if origin<127:continue
                    ok=sig['available'][origin+1:origin+26]&np.isfinite(sig['values'][origin+1:origin+26])
                    ok=ok&sig['available'][origin:origin+1]&np.isfinite(sig['values'][origin:origin+1])
                    ok[:,10]=False;count_targets+=int(ok.sum())
                observed+=count_targets;eligible+=int(count_targets>0)
            for k in arms[1:]:
                q=seal['updates'][k]
                if (q['last_observations']!=[c-1 for c in clocks] or q['opportunities']!=len(clocks)
                        or q['observed_targets']!=observed or not 0<=q['updates']<=eligible or q['max_state_norm']>1.000001):
                    raise ValueError('matured feedback count/radius mismatch')
                for key in totals[k]:totals[k][key]+=q[key]
                maxnorm=max(maxnorm,q['max_state_norm']);matrixnorm=max(matrixnorm,q['max_matrix_norm'])
    metrics={};comparisons={}
    for k in arms:
        metrics[k]={section:count(sub) for section,sub in (('full360',records[k]),('cold20',records[k][:20]),('suffix340',records[k][20:]))}
        for section,values in metrics[k].items():
            for key,val in values.items():
                actual=done['results'][k][section][key]
                if actual is None and val is None:continue
                if actual is None or val is None or abs(actual-val)>1e-12:raise ValueError('aggregate mismatch')
        if k!='cold':
            rescued=[];harmed=[]
            for old,new in zip(records['cold'],records[k]):
                a=old['timely'] if old['is_disrupt'] else not old['alarm'];b=new['timely'] if new['is_disrupt'] else not new['alarm']
                if b and not a:rescued.append(old['shot'])
                if a and not b:harmed.append(old['shot'])
            comparisons[k]=dict(rescued=rescued,harmed=harmed)
    if comparisons!=done['comparisons'] or {k:v['updates'] for k,v in totals.items()}!=done['update_counts']:raise ValueError('totals mismatch')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),sealed360_verified=True,
        first256_max_abs_difference=first_error,cold20_included=True,metrics=metrics,comparisons=comparisons,
        update_accounting=totals,max_temporary_norm=maxnorm,max_matrix_delta=matrixnorm,risk_differences=differences,
        audit_code_sha256=sha(Path(__file__)),independent_confirmation=False)
    _durable_json(root/'audit.json',result);print(dict(metrics={k:v['full360'] for k,v in metrics.items()},comparisons=comparisons,updates=totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
