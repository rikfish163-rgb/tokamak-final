"""Fit23 loss weights using only ended cold20; no suffix selection."""
import argparse
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..attention_lowrank_20261029.model import AttentionAdapter
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import feedback_parts,weighted_state,operational_objective,validate_fit_ids


def make_cache(net,slow,p,ref):
    ids=p['order'][:20];validate_fit_ids(ids,p['order']);mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json')
    parts=[];xs=[];masks=[];case_indices=[];shot_indices=[];shots=[];receipts=[];count_cases=0;count_queries=0
    for j,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('cold input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        flag,down=io23.read_outcome('EAST',sid)
        sig=transform({k:raw[k] for k in ('values','available')},mapping)
        eligible=np.round((down-raw['times'])*1000,9)>30 if flag else np.ones(len(raw['times']),bool)
        early=np.flatnonzero(eligible & (np.arange(len(eligible))<256))
        early_logits=[]
        with torch.no_grad():
            for lo in range(0,len(early),32):
                x,a=batch(sig,early[lo:lo+32],'cuda')
                early_logits.extend(net(x,a,x.new_zeros(len(x),512),slow,'qk',forecast=False)['logit'].cpu().tolist())
        early_peak=max(early_logits) if early_logits else -math.inf
        endpoints=np.arange(256,len(eligible),8);endpoints=endpoints[eligible[endpoints]]
        if flag:
            ends=np.flatnonzero(eligible & (np.arange(len(eligible))>=256))
            if len(ends):endpoints=np.unique(np.r_[endpoints,ends[-1]])
        for count in sorted(set((endpoints//128*128).tolist())):
            ends=endpoints[(endpoints//128*128)==count]
            g,info=feedback_parts(net,{k:v[:count] for k,v in sig.items()},slow)
            x,a=batch(sig,ends,'cpu');xs.append(x);masks.append(a);parts.append(g.cpu())
            case_indices.extend([count_cases]*len(x));shot_indices.extend([j]*len(x))
            receipts.append(dict(shot=sid,count=count,query_ends=ends.tolist(),**{k:v for k,v in info.items() if k!='count'}))
            count_cases+=1;count_queries+=len(x)
        shots.append(dict(shot=sid,flag=flag,early_peak=early_peak if math.isfinite(early_peak) else None,
                          query_count=len(endpoints),fixed_prefix_count=len(early)))
        print(dict(cache_shot=j+1,shot=sid,cases=count_cases,queries=count_queries),flush=True)
    return dict(parts=torch.stack(parts).cuda(),x=torch.cat(xs),mask=torch.cat(masks),
        cases=torch.tensor(case_indices,device='cuda'),shot_index=torch.tensor(shot_indices),shots=shots,receipts=receipts)


def peaks(net,slow,cache,weights):
    states=weighted_state(cache['parts'],weights)
    values=[]
    with torch.no_grad():
        for lo in range(0,len(cache['x']),256):
            x=cache['x'][lo:lo+256].cuda();a=cache['mask'][lo:lo+256].cuda()
            s=states[cache['cases'][lo:lo+256]]
            values.append(net(x,a,s,slow,'qk',forecast=False)['logit'].cpu())
    z=torch.cat(values);out=[];selected=[]
    for j,r in enumerate(cache['shots']):
        ids=torch.nonzero(cache['shot_index']==j).flatten()
        fixed=-math.inf if r['early_peak'] is None else r['early_peak']
        index=int(ids[z[ids].argmax()]) if len(ids) else -1
        if index>=0 and float(z[index])>fixed:out.append(float(z[index]));selected.append(index)
        else:out.append(fixed);selected.append(-1)
    if not np.isfinite(out).all():raise ValueError('historical shot has no eligible operational endpoint')
    return torch.tensor(out,device='cuda'),selected


def baseline_peaks(net,slow,cache):
    result=[]
    with torch.no_grad():
        for lo in range(0,len(cache['x']),256):
            x=cache['x'][lo:lo+256].cuda();a=cache['mask'][lo:lo+256].cuda()
            result.append(net(x,a,x.new_zeros(len(x),512),slow,'qk',forecast=False)['logit'].cpu())
    z=torch.cat(result);values=[]
    for j,r in enumerate(cache['shots']):
        local=z[cache['shot_index']==j]
        values.append(max(-math.inf if r['early_peak'] is None else r['early_peak'],float(local.max()) if len(local) else -math.inf))
    if not np.isfinite(values).all():raise ValueError('missing static operational peak')
    return torch.tensor(values,device='cuda')


def fit(net,slow,cache,baseline,fit_indices,threshold):
    weights=torch.zeros(23,device='cuda',requires_grad=True);opt=torch.optim.Adam([weights],lr=.05)
    flags=torch.tensor([r['flag'] for r in cache['shots']],device='cuda');take=torch.tensor(fit_indices,device='cuda')
    history=[]
    for epoch in range(21):
        current,selected=peaks(net,slow,cache,weights)
        active=[i for i in fit_indices if selected[i]>=0]
        values=current.detach().clone()
        if active:
            queries=torch.tensor([selected[i] for i in active],dtype=torch.long)
            case=cache['cases'][queries.cuda()]
            states=weighted_state(cache['parts'][case],weights)
            v=net(cache['x'][queries].cuda(),cache['mask'][queries].cuda(),states,slow,'qk',forecast=False)['logit']
            values=values.index_copy(0,torch.tensor(active,device='cuda'),v)
        # Static pre-feedback winners still participate in the objective. Their
        # derivative is exactly zero; do not fabricate a different endpoint.
        objective,delta=operational_objective(values[take]+weights.sum()*0,baseline[take],flags[take],threshold)
        grad,=torch.autograd.grad(objective,weights)
        if not torch.isfinite(objective) or not torch.isfinite(grad).all():raise FloatingPointError('invalid policy fitting')
        history.append(dict(epoch=epoch,objective=float(objective.detach()),gradient_norm=float(grad.norm()),
            active_shots=len(active),log_weights=weights.detach().cpu().tolist(),deltas=delta.detach().cpu().tolist()))
        if epoch==20:break
        weights.grad=grad;opt.step();opt.zero_grad(set_to_none=True)
    final,_=peaks(net,slow,cache,weights)
    sign=torch.where(flags,-1.,1.);boundary=math.log(threshold/(1-threshold))
    delta=torch.nn.functional.softplus(sign*(final-boundary))-torch.nn.functional.softplus(sign*(baseline-boundary))
    return dict(log_weights=weights.detach().cpu().tolist(),history=history,
        per_shot=[dict(shot=r['shot'],flag=r['flag'],fitted=i in fit_indices,baseline_peak=float(baseline[i]),
                      updated_peak=float(final[i]),loss_change=float(delta[i])) for i,r in enumerate(cache['shots'])])


def run(output,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree-local output required')
    p=read(previous,'protocol.json');a=read(previous,'audit.json')
    if not a['complete'] or a['complete_sha256']!=sha(previous/'complete.json'):raise ValueError('audited previous stream required')
    ref=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=AttentionAdapter(base)
    frozen={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='ENDED_COLD20_ONLY_POLICY_FIT',source_sha256=PARENT_SHA256,
        previous=str(previous),previous_audit_sha256=sha(previous/'audit.json'),reference=str(ref),reference_hashes=p['reference_hashes'],
        order=p['order'],fit_ids=p['order'][:20],threshold=parent['threshold'],epochs=20,learning_rate=.05,
        plan_sha256=sha(io23.ROOT/'ATTENTION_WEIGHT_PLAN.md'),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        policy_only_crossfit=True,cold_parent_uses_all20=True,seed=131,independent_confirmation=False)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_WEIGHT_PLAN.md','ATTENTION_WEIGHT_PLAN.md')
    cache=make_cache(net,slow,p,ref);baseline=baseline_peaks(net,slow,cache)
    _durable_json(output/'sampling.json',dict(shots=cache['shots'],receipts=cache['receipts']))
    crossfit=[]
    for fold in range(5):
        indices=[i for i in range(20) if i%5!=fold]
        validate_fit_ids([p['order'][i] for i in indices],p['order'])
        result=fit(net,slow,cache,baseline,indices,parent['threshold'])
        _durable_json(output/f'fold{fold}.json',result)
        held=[r for r in result['per_shot'] if not r['fitted']];crossfit.extend(held)
        print(dict(fold=fold,fit_objective=result['history'][-1]['objective'],held=held,seconds=time.monotonic()-started),flush=True)
    final=fit(net,slow,cache,baseline,list(range(20)),parent['threshold']);_durable_json(output/'policy.json',final)
    _durable_json(output/'crossfit.json',sorted(crossfit,key=lambda r:p['order'].index(r['shot'])))
    if not all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('persistent model changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during fit')
    result=dict(complete=True,scope=protocol['scope'],seconds=time.monotonic()-started,
        policy_sha256=sha(output/'policy.json'),crossfit_sha256=sha(output/'crossfit.json'),protocol_sha256=sha(output/'protocol.json'),
        training_objective=final['history'][-1]['objective'],crossfit_improved=sum(r['loss_change']<0 for r in crossfit),
        crossfit_worsened=sum(r['loss_change']>0 for r in crossfit),crossfit_tied=sum(r['loss_change']==0 for r in crossfit),
        source_unchanged=True,suffix_accessed_for_fit=False,scientific_goal_achieved=False)
    _durable_json(output/'complete.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous)
