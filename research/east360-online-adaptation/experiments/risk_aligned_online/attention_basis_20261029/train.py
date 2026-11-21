"""Ended-cold20 meta-fit of an orthogonal Q/K update basis and step."""
import argparse
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..attention_weights_20261029.learning import operational_objective,validate_fit_ids
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import MatrixAttention,project_rows,proposal,feedback_gradient


def make_cache(net,slow,p,ref):
    ids=p['order'][:20];validate_fit_ids(ids,p['order']);mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json')
    gradients=[];xs=[];masks=[];cases=[];indices=[];shots=[];receipts=[]
    for j,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('cold input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));flag,down=io23.read_outcome('EAST',sid)
        sig=transform({k:raw[k] for k in ('values','available')},mapping)
        eligible=np.round((down-raw['times'])*1000,9)>30 if flag else np.ones(len(raw['times']),bool)
        early=np.flatnonzero(eligible&(np.arange(len(eligible))<256));early_logits=[]
        with torch.no_grad():
            for lo in range(0,len(early),32):
                x,a=batch(sig,early[lo:lo+32],'cuda')
                early_logits.extend(net(x,a,x.new_zeros(len(x),8192),slow,'qk',forecast=False)['logit'].cpu().tolist())
        endpoints=np.arange(256,len(eligible),8);endpoints=endpoints[eligible[endpoints]]
        if flag:
            ends=np.flatnonzero(eligible&(np.arange(len(eligible))>=256))
            if len(ends):endpoints=np.unique(np.r_[endpoints,ends[-1]])
        for count in sorted(set((endpoints//128*128).tolist())):
            ends=endpoints[endpoints//128*128==count];g,info=feedback_gradient(net,{k:v[:count] for k,v in sig.items()},slow)
            x,a=batch(sig,ends,'cpu');xs.append(x);masks.append(a);cases.extend([len(gradients)]*len(x));indices.extend([j]*len(x))
            gradients.append(g.cpu());receipts.append(dict(shot=sid,query_ends=ends.tolist(),**info))
        shots.append(dict(shot=sid,flag=flag,early_peak=max(early_logits) if early_logits else None,
            query_count=len(endpoints),fixed_prefix_count=len(early)))
        print(dict(cache_shot=j+1,cases=len(gradients)),flush=True)
    return dict(g=torch.stack(gradients).cuda(),x=torch.cat(xs),mask=torch.cat(masks),cases=torch.tensor(cases,device='cuda'),
        shot_index=torch.tensor(indices),shots=shots,receipts=receipts)


def peaks(net,slow,cache,states):
    values=[]
    with torch.no_grad():
        for lo in range(0,len(cache['x']),256):
            x=cache['x'][lo:lo+256].cuda();a=cache['mask'][lo:lo+256].cuda();s=states[cache['cases'][lo:lo+256]].flatten(1)
            values.append(net(x,a,s,slow,'qk',forecast=False)['logit'].cpu())
    z=torch.cat(values);out=[];selected=[]
    for j,r in enumerate(cache['shots']):
        ids=torch.nonzero(cache['shot_index']==j).flatten();fixed=-math.inf if r['early_peak'] is None else r['early_peak']
        index=int(ids[z[ids].argmax()]) if len(ids) else -1
        if index>=0 and float(z[index])>fixed:out.append(float(z[index]));selected.append(index)
        else:out.append(fixed);selected.append(-1)
    if not np.isfinite(out).all():raise ValueError('no eligible operational endpoint')
    return torch.tensor(out,device='cuda'),selected


def fit(net,slow,cache,baseline,fit_indices,threshold):
    a=torch.stack((net.q_basis,net.k_basis)).detach().clone().requires_grad_();eta=a.new_tensor(.05,requires_grad=True)
    opt=torch.optim.Adam([a,eta],lr=.03);flags=torch.tensor([r['flag'] for r in cache['shots']],device='cuda')
    take=torch.tensor(fit_indices,device='cuda');history=[]
    for epoch in range(41):
        states=proposal(cache['g'],a,eta);current,selected=peaks(net,slow,cache,states)
        active=[i for i in fit_indices if selected[i]>=0];values=current.detach().clone()
        if active:
            queries=torch.tensor([selected[i] for i in active],dtype=torch.long);case=cache['cases'][queries.cuda()]
            s=proposal(cache['g'][case],a,eta).flatten(1)
            v=net(cache['x'][queries].cuda(),cache['mask'][queries].cuda(),s,slow,'qk',forecast=False)['logit']
            values=values.index_copy(0,torch.tensor(active,device='cuda'),v)
        objective,delta=operational_objective(values[take]+a.sum()*0+eta*0,baseline[take],flags[take],threshold)
        ga,ge=torch.autograd.grad(objective,(a,eta))
        if not all(torch.isfinite(t).all() for t in (objective,ga,ge)):raise FloatingPointError('invalid meta gradient')
        history.append(dict(epoch=epoch,objective=float(objective.detach()),basis_gradient_norm=float(ga.norm()),
            step_gradient=float(ge),step=float(eta.detach()),active_shots=len(active),deltas=delta.detach().cpu().tolist()))
        if epoch==40:break
        a.grad=ga;eta.grad=ge;opt.step();opt.zero_grad(set_to_none=True)
        with torch.no_grad():a.copy_(project_rows(a));eta.clamp_(0.,1.)
    final,_=peaks(net,slow,cache,proposal(cache['g'],a,eta));sign=torch.where(flags,-1.,1.);boundary=math.log(threshold/(1-threshold))
    change=torch.nn.functional.softplus(sign*(final-boundary))-torch.nn.functional.softplus(sign*(baseline-boundary))
    return dict(basis=a.detach().cpu().tolist(),step=float(eta.detach()),history=history,
        per_shot=[dict(shot=r['shot'],flag=r['flag'],fitted=i in fit_indices,baseline_peak=float(baseline[i]),
                      updated_peak=float(final[i]),loss_change=float(change[i])) for i,r in enumerate(cache['shots'])])


def run(output,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    p=read(previous,'protocol.json');audit=read(previous,'audit.json');ref=Path(p['reference'])
    if not audit['complete'] or audit['complete_sha256']!=sha(previous/'complete.json'):raise ValueError('audited previous stream required')
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base);frozen={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='ENDED_COLD20_ONLY_QK_BASIS_FIT',source_sha256=PARENT_SHA256,previous=str(previous),
        previous_audit_sha256=sha(previous/'audit.json'),reference=str(ref),reference_hashes=p['reference_hashes'],order=p['order'],fit_ids=p['order'][:20],
        threshold=parent['threshold'],epochs=40,learning_rate=.03,plan_sha256=sha(io23.ROOT/'ATTENTION_BASIS_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},policy_only_crossfit=True,cold_parent_uses_all20=True,seed=131,independent_confirmation=False)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_BASIS_PLAN.md','ATTENTION_BASIS_PLAN.md')
    cache=make_cache(net,slow,p,ref);baseline,_=peaks(net,slow,cache,torch.zeros_like(cache['g']))
    _durable_json(output/'sampling.json',dict(shots=cache['shots'],receipts=cache['receipts']))
    held=[]
    for fold in range(5):
        indices=[i for i in range(20) if i%5!=fold];validate_fit_ids([p['order'][i] for i in indices],p['order'])
        result=fit(net,slow,cache,baseline,indices,parent['threshold']);_durable_json(output/f'fold{fold}.json',result)
        held.extend(r for r in result['per_shot'] if not r['fitted']);print(dict(fold=fold,step=result['step'],objective=result['history'][-1]['objective']),flush=True)
    final=fit(net,slow,cache,baseline,list(range(20)),parent['threshold']);final['mean_gradient']=cache['g'].mean(0).cpu().tolist()
    _durable_json(output/'policy.json',final);_durable_json(output/'crossfit.json',sorted(held,key=lambda r:p['order'].index(r['shot'])))
    if not all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('frozen parent changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('fit implementation changed')
    done=dict(complete=True,scope=protocol['scope'],seconds=time.monotonic()-started,policy_sha256=sha(output/'policy.json'),
        crossfit_sha256=sha(output/'crossfit.json'),protocol_sha256=sha(output/'protocol.json'),training_objective=final['history'][-1]['objective'],
        step=final['step'],crossfit_improved=sum(r['loss_change']<0 for r in held),crossfit_worsened=sum(r['loss_change']>0 for r in held),
        crossfit_tied=sum(r['loss_change']==0 for r in held),source_unchanged=True,suffix_accessed_for_fit=False,scientific_goal_achieved=False)
    _durable_json(output/'complete.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous)
