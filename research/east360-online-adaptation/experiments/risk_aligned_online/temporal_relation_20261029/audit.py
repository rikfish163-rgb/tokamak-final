"""Independent decision recount and source-pair accounting for relation task."""
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


def source_check(root):
    p=read(root,'protocol.json');done=read(root,'complete.json');v=read(root,'val_epoch6.json')
    if not done['complete'] or not done['source_only_gate'] or sha(root/'judge.pt')!=done['checkpoint_sha256']:
        raise ValueError('source prerequisite failed')
    if p['epochs']!=6 or p['selected_epoch']!='FIXED_FINAL_6':raise ValueError('epoch selection changed')
    if sha(io23.ROOT/'TEMPORAL_RELATION_PLAN.md')!=p['plan_sha256']:raise ValueError('source plan changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('source code changed')
    coverage={}
    for role in ('train','val'):
        windows=read(root,f'{role}_windows.json')
        if [r['shot'] for r in windows]!=p[f'{role}_ids'] or any(len(r['ends'])!=4 for r in windows):
            raise ValueError('source window identities changed')
    for mode in ('random','interleave'):
        records=v[mode]['records'];by={sid:[] for sid in p['val_ids']}
        if len(records)!=640:raise ValueError('VAL160 four-window accounting failed')
        for r in records:by[r['shot']].append(r)
        if any(len(rows)!=4 for rows in by.values()):raise ValueError('missing source validation shots')
        eligible={sid:[r for r in rows if r['eligible']] for sid,rows in by.items()}
        for group in ('all','D','S'):
            rows=[rs for rs in eligible.values() if rs and (group=='all' or rs[0]['flag']==(group=='D'))]
            actual=dict(shots=len(rows),windows=sum(map(len,rows)),
                accuracy=float(np.mean([np.mean([1. if r['margin']>0 else .5 if r['margin']==0 else 0. for r in rs]) for rs in rows])),
                roughness_accuracy=float(np.mean([np.mean([1. if r['roughness_margin']>0 else .5 if r['roughness_margin']==0 else 0. for r in rs]) for rs in rows])))
            for k,val in actual.items():
                if abs(val-done['val'][mode][group][k])>1e-12:raise ValueError('source auxiliary metric mismatch')
        coverage[mode]=dict(eligible_shots=sum(bool(rs) for rs in eligible.values()),
            eligible_windows=sum(map(len,eligible.values())),no_pair_shots=[sid for sid,rs in eligible.items() if not rs])
    return dict(coverage=coverage,complete_sha256=sha(root/'complete.json'),maximum_mean_only_margin=v['maximum_mean_only_margin'])


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');declared=read(root,'per_shot.json')
    if not done['complete'] or len(set(p['order']))!=360 or len(p['order'])!=360:raise ValueError('completed360 required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256:raise ValueError('source changed')
    source_root=Path(p['source_root']);source=source_check(source_root)
    if source['complete_sha256']!=p['source_complete_sha256'] or sha(source_root/'judge.pt')!=p['judge_sha256']:
        raise ValueError('auxiliary changed')
    if sha(io23.ROOT/'TEMPORAL_RELATION_STREAM_PLAN.md')!=p['plan_sha256']:raise ValueError('stream plan changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('stream implementation changed')
    previous=Path(p['previous']);reference=Path(p['reference'])
    if sha(previous/'complete.json')!=p['previous_complete_sha256']:raise ValueError('prior result changed')
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold provenance changed')
    hashes=read(reference,'input_hashes.json');mapping=read(reference,'input_map.json')
    arms=p['arms'];records={k:[] for k in arms};differences={k:[] for k in arms[1:]}
    totals={k:dict(opportunities=0,eligible=0,updates=0) for k in arms[1:]};maximum_norm=0.
    for i,sid in enumerate(p['order']):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw input changed')
        name=f'{i+1:04d}_{sid}';path=root/'shots'/f'{name}.npz';seal=read(root/'shots',f'{name}_seal.json')
        if seal['outcome_used'] or seal['shot']!=sid or seal['index']!=i+1 or sha(path)!=seal['sha256']:
            raise ValueError('invalid curve seal')
        oldpath=previous/'shots'/f'{name}.npz';oldseal=read(previous/'shots',f'{name}_seal.json')
        if sha(oldpath)!=oldseal['sha256']:raise ValueError('previous curve changed')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z,np.load(oldpath) as old:
            if not np.array_equal(z['times'],old['times']) or np.max(np.abs(z['cold']-old['cold']))>2e-6:
                raise ValueError('cold numerical mismatch')
            if not np.array_equal(z['cold']>=p['threshold'],old['cold']>=p['threshold']):raise ValueError('cold alarm changed')
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool)
            times=z['times'][valid]
            for k in arms:
                score=z[k][valid]
                if not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError('invalid probability')
                cross=np.flatnonzero(score>=p['threshold']);alarm=bool(len(cross))
                lead=float(round((down-times[cross[0]])*1000,9)) if alarm and flag else None
                r=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))
                if r!=declared[k][i]:raise ValueError('per-shot decision mismatch')
                records[k].append(r)
                if i<20 and not np.array_equal(z[k],z['cold']):raise ValueError('cold20 altered')
                if k in differences:
                    if not np.array_equal(z[k][:128],z['cold'][:128]):raise ValueError('sealed first128 altered')
                    delta=score-z['cold'][valid]
                    differences[k].append(dict(shot=sid,flag=flag,mean_delta=float(delta.mean()),
                        max_abs_delta=float(np.abs(delta).max()),peak=float(score.max()),cold_peak=float(z['cold'][valid].max())))
            if i>=20:
                raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                sig=transform({k:raw[k] for k in ('values','available')},mapping)
                rng=np.random.default_rng(131);eligible=0;opportunities=(len(z['times'])-1)//128
                for hi in range(128,len(z['times']),128):
                    x=sig['values'][hi-128:hi];a=sig['available'][hi-128:hi]
                    inner=rng.permutation(np.arange(1,15))
                    if np.array_equal(inner,np.arange(1,15)):inner=np.roll(inner,1)
                    xx=x.reshape(16,8,23)[np.r_[0,inner,15]].reshape(128,23)
                    eligible+=int(np.all(a==a[:1]) and a.any() and np.any((xx!=x)&a))
                for k in arms[1:]:
                    q=seal['updates'][k]
                    if q['opportunities']!=opportunities or q['eligible']!=eligible or not 0<=q['updates']<=eligible or q['max_state_norm']>1.000001:
                        raise ValueError('update eligibility/count/radius mismatch')
                    for key in totals[k]:totals[k][key]+=q[key]
                    maximum_norm=max(maximum_norm,q['max_state_norm'])
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
    if comparisons!=done['comparisons'] or {k:v['updates'] for k,v in totals.items()}!=done['update_counts']:
        raise ValueError('comparison or update totals mismatch')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),source_auxiliary=source,
        sealed360_verified=True,first128_preserved=True,cold20_included=True,matched_parent_verified=True,
        metrics=metrics,comparisons=comparisons,update_accounting=totals,maximum_temporary_norm=maximum_norm,
        risk_differences=differences,scope='single developed stream and seed, not independent confirmation')
    _durable_json(root/'audit.json',result)
    print(dict(metrics={k:v['full360'] for k,v in metrics.items()},comparisons=comparisons,updates=totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
