"""Recount sealed full360 curves and verify historical-basis provenance."""
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
    p=read(root,'protocol.json');d=read(root,'complete.json');records=read(root,'per_shot.json')
    if not d['complete'] or len(p['order'])!=360 or len(set(p['order']))!=360:raise ValueError('complete unique360 required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256:raise ValueError('source changed')
    if sha(io23.ROOT/'ATTENTION_BASIS_PLAN.md')!=p['plan_sha256']:raise ValueError('base plan changed')
    policy=Path(p['policy_dir']);fit=read(policy,'complete.json');tp=read(policy,'protocol.json');w=read(policy,'policy.json')
    if (sha(policy/'complete.json')!=p['policy_complete_sha256'] or sha(policy/'policy.json')!=p['policy_sha256']
            or p['policy_sha256']!=fit['policy_sha256'] or tp['fit_ids']!=p['order'][:20] or tp['order']!=p['order']
            or tp['reference_hashes']!=p['reference_hashes'] or fit['protocol_sha256']!=sha(policy/'protocol.json')
            or fit['crossfit_sha256']!=sha(policy/'crossfit.json')):raise ValueError('fit provenance changed')
    if tp['scope']=='ENDED_COLD20_ONLY_QK_BASIS_STAGED_FIT':
        if (tp['fixed_step_updates']!=20 or tp['amendment']!='ATTENTION_BASIS_STAGED_PLAN.md'
                or sha(io23.ROOT/tp['amendment'])!=tp['amendment_sha256']):raise ValueError('staged amendment changed')
    elif tp['scope']!='ENDED_COLD20_ONLY_QK_BASIS_FIT':raise ValueError('unknown fitting protocol')
    for manifest in (p,tp):
        for path,h in manifest['code_sha256'].items():
            if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed')
    a=np.asarray(w['basis']);eta=w['step'];g=np.asarray(w['mean_gradient'])
    if (a.shape!=(2,2,64) or g.shape!=(2,64,64) or not np.isfinite(g).all() or not np.isfinite(a).all()
            or not np.allclose(a@a.transpose(0,2,1),np.eye(2),atol=1e-5,rtol=0) or not 0<=eta<=1
            or eta!=p['step'] or eta!=fit['step']):raise ValueError('invalid basis or step')
    sample=read(policy,'sampling.json');fit_ids=p['order'][:20]
    if [r['shot'] for r in sample['shots']]!=fit_ids or any(r['shot'] not in fit_ids for r in sample['receipts']):raise ValueError('suffix used for fitting')
    held=[]
    for fold in range(5):
        f=read(policy,f'fold{fold}.json');expected=[sid for i,sid in enumerate(fit_ids) if i%5!=fold]
        if [r['shot'] for r in f['per_shot'] if r['fitted']]!=expected or len(f['history'])!=41:raise ValueError('fold membership or epochs differ')
        if tp['scope'].endswith('STAGED_FIT') and any(abs(r['step']-.05)>1e-8 for r in f['history'][:21]):raise ValueError('stage1 step changed')
        held.extend(r for r in f['per_shot'] if not r['fitted'])
    if sorted(held,key=lambda r:fit_ids.index(r['shot']))!=read(policy,'crossfit.json'):raise ValueError('held ledger differs')
    if [r['shot'] for r in w['per_shot']]!=fit_ids or not all(r['fitted'] for r in w['per_shot']):raise ValueError('final fit IDs differ')
    ref=Path(p['reference']);previous=Path(p['previous'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    if sha(previous/'complete.json')!=p['previous_complete_sha256']:raise ValueError('previous stream changed')
    hashes=read(ref,'input_hashes.json');mapping=read(ref,'input_map.json');arms=p['arms'];actual={k:[] for k in arms}
    updates={k:0 for k in arms[1:]};opportunities=0;maxnorm=0.;early_error=0.;reverse_error=0.;constant_error=0.
    for i,sid in enumerate(p['order']):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw data changed')
        name=f'{i+1:04d}_{sid}';path=root/'shots'/f'{name}.npz';seal=read(root/'shots',f'{name}_seal.json')
        if (sha(path)!=seal['sha256'] or seal['outcome_used'] or seal['shot']!=sid or seal['index']!=i+1
                or seal['policy_active']!=(i>=20)):raise ValueError('seal or activation wrong')
        oldpath=previous/'shots'/f'{name}.npz';oldseal=read(previous/'shots',f'{name}_seal.json')
        if sha(oldpath)!=oldseal['sha256']:raise ValueError('previous curve changed')
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(path) as z,np.load(oldpath) as old:
            if (not np.array_equal(z['times'],old['times']) or np.max(np.abs(z['cold']-old['cold']))>2e-6
                    or not np.array_equal(z['cold']>=p['threshold'],old['cold']>=p['threshold'])):raise ValueError('cold changed')
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool);times=z['times'][valid];n=len(z['times'])
            for k in arms:
                score=z[k][valid]
                if not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError('invalid probability')
                cross=np.flatnonzero(score>=p['threshold']);alarm=bool(len(cross))
                lead=float(round((down-times[cross[0]])*1000,9)) if flag and alarm else None
                row=dict(shot=sid,is_disrupt=flag,alarm=alarm,lead_ms=lead,timely=bool(lead is not None and lead>30))
                if row!=records[k][i]:raise ValueError('per-shot record differs')
                actual[k].append(row)
                if i<20 and not np.array_equal(z[k],z['cold']):raise ValueError('first20 changed')
                error=float(np.max(np.abs(z[k][:256]-z['cold'][:256])));early_error=max(early_error,error)
                if error>2e-6 or not np.array_equal(z[k][:256]>=p['threshold'],z['cold'][:256]>=p['threshold']):raise ValueError('pre-feedback prediction changed')
                if eta==0 and not np.array_equal(z[k],z['cold']):raise ValueError('zero-step curve must be identical')
            reverse_error=max(reverse_error,float(np.abs(z['learned_basis'][valid]-z['reverse_target'][valid]).max()))
            constant_error=max(constant_error,float(np.abs(z['learned_basis'][valid]-z['constant_feedback'][valid]).max()))
        if i>=20:
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));signal=transform({k:raw[k] for k in ('values','available')},mapping)
            clocks=list(range(256,n,128));opportunities+=len(clocks);observed=0;eligible=0
            for hi in clocks:
                here=0
                for origin in (hi-26,hi-51,hi-76,hi-101):
                    if origin<127:continue
                    mask=signal['available'][origin+1:origin+26]&np.isfinite(signal['values'][origin+1:origin+26])
                    mask=mask&signal['available'][origin:origin+1]&np.isfinite(signal['values'][origin:origin+1]);mask[:,10]=False;here+=int(mask.sum())
                observed+=here;eligible+=int(here>0)
            for k in arms[1:]:
                r=seal['updates'][k]
                if (r['last_observations']!=[c-1 for c in clocks] or r['opportunities']!=len(clocks) or r['observed_targets']!=observed
                        or not 0<=r['updates']<=eligible or r['max_matrix_norm']>eta+1e-6 or eta==0 and r['updates']!=0):raise ValueError('feedback clock/mask/norm changed')
                updates[k]+=r['updates'];maxnorm=max(maxnorm,r['max_matrix_norm'])
    metrics={};comparisons={}
    for k in arms:
        metrics[k]={s:count(v) for s,v in (('full360',actual[k]),('cold20',actual[k][:20]),('suffix340',actual[k][20:]))}
        for section,values in metrics[k].items():
            for key,value in values.items():
                other=d['results'][k][section][key]
                if value is None and other is None:continue
                if value is None or other is None or abs(value-other)>1e-12:raise ValueError('aggregate differs')
        if k!='cold':
            rescued=[];harmed=[]
            for old,new in zip(actual['cold'],actual[k]):
                before=old['timely'] if old['is_disrupt'] else not old['alarm'];after=new['timely'] if new['is_disrupt'] else not new['alarm']
                if after and not before:rescued.append(old['shot'])
                if before and not after:harmed.append(old['shot'])
            comparisons[k]=dict(rescued=rescued,harmed=harmed)
    if comparisons!=d['comparisons'] or updates!=d['update_counts']:raise ValueError('pairing/updates differ')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),metrics=metrics,comparisons=comparisons,updates=updates,
        opportunities=opportunities,max_matrix_norm=maxnorm,first256_max_abs_error=early_error,max_reverse_difference=reverse_error,
        max_constant_difference=constant_error,code_sha256=sha(Path(__file__)),independent_confirmation=False)
    _durable_json(root/'audit.json',result);print(dict(metrics={k:v['full360'] for k,v in metrics.items()},comparisons=comparisons,updates=updates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
