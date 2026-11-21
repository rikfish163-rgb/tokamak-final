"""Independent LP/matrix reconstruction and sealed full360 recount."""
import argparse
import math
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


def verify_fit(root):
    from scipy.optimize import linprog
    p=read(root,'protocol.json');done=read(root,'complete.json');final=read(root,'policy.json')
    if (p['scope']!='ENDED_COLD20_PROTECTED_QK_CROSSFIT' or not done['complete']
            or done['protocol_sha256']!=sha(root/'protocol.json') or done['policy_sha256']!=sha(root/'policy.json')
            or done['crossfit_sha256']!=sha(root/'crossfit.json') or done['pairs_sha256']!=sha(root/'historical_pairs.npz')
            or p['fit_ids']!=p['order'][:20]):raise ValueError('fit provenance invalid')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('fit code changed')
    diagnostic=Path(p['diagnostic'])
    if sha(diagnostic/'complete.json')!=p['diagnostic_complete_sha256']:raise ValueError('prior diagnostic changed')
    prior=read(diagnostic,'complete.json');dp=read(diagnostic,'protocol.json')
    if (sha(diagnostic/'protocol.json')!=prior['protocol_sha256']
            or sha(io23.ROOT/'ATTENTION_SPECTRAL_PROTECTION_DIAGNOSTIC.md')!=dp['plan_sha256']):
        raise ValueError('diagnostic contract changed')
    if final['step']!=prior['step'] or not np.allclose(final['matrix'],prior['projector'],atol=1e-12,rtol=0):
        raise ValueError('final policy differs from previous historical diagnostic')
    sample=read(root,'sampling.json')
    if [r['shot'] for r in sample['shots']]!=p['fit_ids'] or any(r['shot'] not in p['fit_ids'] for r in sample['receipts']):
        raise ValueError('sampling outside cold20')
    with np.load(root/'historical_pairs.npz') as z:
        g=z['g'];h=z['h'];baseline=z['baseline']
    ng=g/np.maximum(np.linalg.norm(g.reshape(20,-1),axis=1),1e-12)[:,None,None,None]
    flags=np.array([r['flag'] for r in sample['shots']],bool);held=[];fits=[];max_error=0.
    for fold in list(range(5))+[None]:
        fit=final if fold is None else read(root,f'fold{fold}.json')
        take=np.array([i for i in range(20) if fold is None or i%5!=fold]);f=flags[take]
        raw=h[take].transpose(0,1,3,2)@ng[take];s=(raw+raw.transpose(0,1,3,2))/2
        ev,u=np.linalg.eigh(s[f].mean(0));atoms=[];groups=[];gain=[]
        for group in [0,1]:
            for j in np.flatnonzero(ev[group]>np.abs(ev[group]).max()*1e-12):
                atom=np.outer(u[group,:,j],u[group,:,j]);atoms.append(atom);groups.append(group)
                gain.append(np.einsum('ij,nji->n',atom,s[:,group]))
        gains=np.stack(gain,axis=1) if gain else np.empty((len(take),0));groups=np.array(groups)
        linear=fit['linear'];w=np.array(linear['weights']);actual=np.array(fit['matrix']);expected=np.zeros_like(actual)
        if groups.tolist()!=linear['atom_groups'] or not np.allclose(gains,linear['gains'],atol=1e-12,rtol=1e-8):
            raise ValueError('reconstructed eigenatom gains differ')
        for weight,atom,group in zip(w,atoms,groups):expected[group]+=weight*atom
        error=float(np.abs(actual-expected).max());max_error=max(max_error,error)
        if error>1e-10:raise ValueError('contraction reconstruction differs')
        eigenvalues=np.linalg.eigvalsh(actual)
        if (eigenvalues.min() < -1e-8 or eigenvalues.max()>1+1e-8 or (eigenvalues.sum(1)>2+1e-8).any()
                or (eigenvalues>1e-8).sum(1).tolist()!=linear['ranks']):raise ValueError('PSD/rank/budget differs')
        if len(w):
            objective=gains[f].mean(0);norm=np.maximum(np.abs(gains).max(1),1e-12)
            budget=np.array([groups==i for i in [0,1]],float);a=np.r_[-gains/norm[:,None],budget];b=np.r_[np.zeros(len(take)),[2.,2.]]
            if (w.min() < -2e-8 or w.max()>1+2e-8 or np.max(a@w-b)>2e-8):raise ValueError('LP constraints violated')
            scale=max(float(np.abs(objective).max()),1e-12)
            optimum=linprog(-objective/scale,A_ub=a,b_ub=b,bounds=(0.,1.),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            if not optimum.success or abs(objective@w+optimum.fun*scale)>1e-10:raise ValueError('LP objective not reproduced')
        history=fit['history'];eta=fit['step']
        if [r['step'] for r in history]!=p['backtracking_steps'][:len(history)]:raise ValueError('step order differs')
        accepted=[]
        for row in history:
            d=np.array(row['loss_changes'])[take]
            ok=bool(np.all(d<=1e-6) and d[f].mean() < -1e-6);accepted.append(ok)
            if ok!=row['accepted'] or abs(d.max()-row['max_harm'])>1e-10:raise ValueError('fitted-only selection differs')
        if any(accepted[:-1]) or eta!=(history[-1]['step'] if accepted[-1] else 0.) or not accepted[-1] and len(history)!=11:
            raise ValueError('first accepted step not selected')
        if [r['shot'] for r in fit['per_shot']]!=p['fit_ids']:raise ValueError('fit records differ')
        for i,row in enumerate(fit['per_shot']):
            if row['fitted']!=(i in take) or abs(row['baseline_peak']-baseline[i])>1e-7:raise ValueError('membership/baseline differs')
        sign=np.where(flags,-1.,1.);boundary=math.log(p['threshold']/(1-p['threshold']))
        current=np.array([r['updated_peak'] for r in fit['per_shot']])
        delta=np.logaddexp(0,sign*(current-boundary))-np.logaddexp(0,sign*(baseline-boundary))
        if not np.allclose(delta,[r['loss_change'] for r in fit['per_shot']],atol=2e-6,rtol=1e-5):raise ValueError('finite losses differ')
        if eta and not np.allclose(delta,history[-1]['loss_changes'],atol=2e-6,rtol=1e-5):raise ValueError('accepted losses differ')
        fits.append(dict(fold=fold,step=eta,ranks=linear['ranks'],matrix_error=error))
        if fold is not None:held.extend(r for r in fit['per_shot'] if not r['fitted'])
    if sorted(held,key=lambda r:p['fit_ids'].index(r['shot']))!=read(root,'crossfit.json'):raise ValueError('crossfit records differ')
    for name,value in [('improved',sum(r['loss_change'] < -1e-6 for r in held)),('worsened',sum(r['loss_change']>1e-6 for r in held)),('tied',sum(abs(r['loss_change'])<=1e-6 for r in held))]:
        if done['crossfit_'+name]!=value:raise ValueError('crossfit count differs')
    rand=np.array(final['random_matrix']);matrix=np.array(final['matrix'])
    if not np.allclose(rand,rand.transpose(0,2,1),atol=1e-12,rtol=0) or not np.allclose(np.linalg.eigvalsh(rand),np.linalg.eigvalsh(matrix),atol=1e-12,rtol=1e-8):
        raise ValueError('random contraction capacity not matched')
    return dict(max_matrix_error=max_error,fits=fits,random_spectrum_matched=True)


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');d=read(root,'complete.json');records=read(root,'per_shot.json')
    if not d['complete'] or len(p['order'])!=360 or len(set(p['order']))!=360:raise ValueError('complete unique360 required')
    if sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256:raise ValueError('source changed')
    if sha(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md')!=p['plan_sha256']:raise ValueError('plan changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('stream code changed')
    policy=Path(p['policy_dir']);tp=read(policy,'protocol.json');w=read(policy,'policy.json')
    if (sha(policy/'complete.json')!=p['policy_complete_sha256'] or sha(policy/'policy.json')!=p['policy_sha256']
            or tp['fit_ids']!=p['order'][:20] or tp['order']!=p['order'] or tp['reference_hashes']!=p['reference_hashes']):
        raise ValueError('historical policy changed')
    fitting=verify_fit(policy);eta=w['step']
    if eta!=p['step'] or not 0<eta<=1:raise ValueError('nonzero matched step required')
    ref=Path(p['reference']);previous=Path(p['previous'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    if sha(previous/'complete.json')!=p['previous_complete_sha256']:raise ValueError('previous stream changed')
    hashes=read(ref,'input_hashes.json');mapping=read(ref,'input_map.json');arms=p['arms'];actual={k:[] for k in arms}
    updates={k:0 for k in arms[1:]};opportunities=0;maxnorm=0.;early_error=0.;reverse_error=0.;constant_error=0.;changes=[]
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
            valid=np.round((down-z['times'])*1000,9)>=-1e-6 if flag else np.ones(len(z['times']),bool)
            times=z['times'][valid];n=len(z['times']);item=dict(shot=sid,flag=flag,max_probability_changes={})
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
                item['max_probability_changes'][k]=float(np.abs(score-z['cold'][valid]).max())
            changes.append(item)
            reverse_error=max(reverse_error,float(np.abs(z['protected'][valid]-z['reverse_target'][valid]).max()))
            constant_error=max(constant_error,float(np.abs(z['protected'][valid]-z['constant_feedback'][valid]).max()))
        if i>=20:
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:raw[k] for k in ('values','available')},mapping);clocks=list(range(256,n,128))
            opportunities+=len(clocks);observed=0;eligible=0
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
                        or not 0<=r['updates']<=eligible or r['max_matrix_norm']>eta+1e-6):raise ValueError('feedback clock/mask/norm changed')
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
        max_constant_difference=constant_error,changes=changes,historical_fit=fitting,code_sha256=sha(Path(__file__)),independent_confirmation=False)
    _durable_json(root/'audit.json',result);print(dict(metrics={k:v['full360'] for k,v in metrics.items()},comparisons=comparisons,updates=updates,
        historical_fit=fitting,max_reverse_difference=reverse_error,max_constant_difference=constant_error),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
