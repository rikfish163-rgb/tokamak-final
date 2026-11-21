"""Ended-cold20 exact spectral diagnostic with finite-step backtracking."""
import argparse
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..attention_basis_20261029.train import make_cache,peaks
from ..attention_weights_20261029.learning import operational_objective,validate_fit_ids
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import cross_moment,positive_projector,proposal


def historical_pairs(net,slow,cache,selected,threshold):
    active=[i for i,index in enumerate(selected) if index>=0]
    g=cache['g'].new_zeros(20,2,64,64);h=torch.zeros_like(g)
    if active:
        queries=torch.tensor([selected[i] for i in active]);case=cache['cases'][queries.cuda()]
        state=g.new_zeros(len(active),8192,requires_grad=True)
        z=net(cache['x'][queries].cuda(),cache['mask'][queries].cuda(),state,slow,'qk',forecast=False)['logit']
        flags=torch.tensor([cache['shots'][i]['flag'] for i in active],device=z.device)
        sign=torch.where(flags,-1.,1.);boundary=math.log(threshold/(1-threshold))
        loss=torch.nn.functional.softplus(sign*(z-boundary)).sum();grad,=torch.autograd.grad(loss,state)
        g[active]=cache['g'][case];h[active]=grad.detach().reshape(-1,2,64,64)
    return g,h


def fit(net,slow,cache,baseline,g,h,fit_indices,threshold):
    flags=torch.tensor([r['flag'] for r in cache['shots']],device=g.device)
    take=torch.tensor(fit_indices,device=g.device);f=flags[take]
    if not f.any() or f.all():raise ValueError('both ended-shot classes required')
    weights=torch.where(f,.5/f.sum(),.5/(~f).sum()).to(g)
    s=cross_moment(g[take],h[take],weights);p,eigenvalues=positive_projector(s)
    selected_step=0.;selected_peaks=baseline;history=[]
    for exponent in range(11):
        eta=2.**(-exponent)
        current,_=peaks(net,slow,cache,proposal(cache['g'],p,eta))
        objective,delta=operational_objective(current[take],baseline[take],f,threshold)
        value=float(objective)
        history.append(dict(step=eta,objective=value,deltas=delta.cpu().tolist()))
        if value < -1e-6:
            selected_step=eta;selected_peaks=current;break
    sign=torch.where(flags,-1.,1.);boundary=math.log(threshold/(1-threshold))
    delta=torch.nn.functional.softplus(sign*(selected_peaks-boundary))-torch.nn.functional.softplus(sign*(baseline-boundary))
    normalized=g/g.flatten(1).norm(dim=1).clamp_min(1e-12)[:,None,None,None]
    raw_gain=(h*normalized).flatten(1).sum(1)
    projected_gain=(h*(normalized@p.to(g))).flatten(1).sum(1)
    return dict(projector=p.cpu().tolist(),step=selected_step,history=history,
        eigenvalues=eigenvalues.cpu().tolist(),linear_macro_gain=float((projected_gain[take]*weights).sum()),
        raw_linear_macro_gain=float((raw_gain[take]*weights).sum()),
        per_shot=[dict(shot=r['shot'],flag=r['flag'],fitted=i in fit_indices,
            active=bool(g[i].norm()>0),baseline_peak=float(baseline[i]),updated_peak=float(selected_peaks[i]),
            loss_change=float(delta[i]),raw_linear_gain=float(raw_gain[i]),projected_linear_gain=float(projected_gain[i]))
            for i,r in enumerate(cache['shots'])])


def run(output,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    p=read(previous,'protocol.json');audit=read(previous,'audit.json');ref=Path(p['reference'])
    if not audit['complete'] or audit['complete_sha256']!=sha(previous/'complete.json'):raise ValueError('audited previous stream required')
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    validate_fit_ids(p['order'][:20],p['order'])
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base);frozen={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='ENDED_COLD20_ONLY_SPECTRAL_QK_DIAGNOSTIC',source_sha256=PARENT_SHA256,previous=str(previous),
        previous_audit_sha256=sha(previous/'audit.json'),reference=str(ref),reference_hashes=p['reference_hashes'],order=p['order'],fit_ids=p['order'][:20],
        threshold=parent['threshold'],plan_sha256=sha(io23.ROOT/'ATTENTION_SPECTRAL_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},policy_only_crossfit=True,cold_parent_uses_all20=True,
        rank_per_matrix=2,backtracking_steps=[2.**(-i) for i in range(11)],accept_objective_below=-1e-6,seed=131,independent_confirmation=False)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_SPECTRAL_PLAN.md','ATTENTION_SPECTRAL_PLAN.md')
    cache=make_cache(net,slow,p,ref);baseline,selected=peaks(net,slow,cache,torch.zeros_like(cache['g']))
    g,h=historical_pairs(net,slow,cache,selected,parent['threshold'])
    with (output/'historical_pairs.npz').open('xb') as file:
        np.savez_compressed(file,g=g.cpu().numpy(),h=h.cpu().numpy(),baseline=baseline.cpu().numpy(),selected=selected)
    _durable_json(output/'sampling.json',dict(shots=cache['shots'],receipts=cache['receipts']))
    held=[]
    for fold in range(5):
        indices=[i for i in range(20) if i%5!=fold]
        result=fit(net,slow,cache,baseline,g,h,indices,parent['threshold']);_durable_json(output/f'fold{fold}.json',result)
        held.extend(r for r in result['per_shot'] if not r['fitted'])
        print(dict(fold=fold,step=result['step'],linear_gain=result['linear_macro_gain'],last_objective=result['history'][-1]['objective']),flush=True)
    final=fit(net,slow,cache,baseline,g,h,list(range(20)),parent['threshold']);final['mean_gradient']=cache['g'].mean(0).cpu().tolist()
    _durable_json(output/'policy.json',final);_durable_json(output/'crossfit.json',sorted(held,key=lambda r:p['order'].index(r['shot'])))
    if not all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('frozen parent changed')
    for path,digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=digest:raise ValueError('fit implementation changed')
    done=dict(complete=True,scope=protocol['scope'],seconds=time.monotonic()-started,policy_sha256=sha(output/'policy.json'),
        crossfit_sha256=sha(output/'crossfit.json'),protocol_sha256=sha(output/'protocol.json'),pairs_sha256=sha(output/'historical_pairs.npz'),
        training_objective=final['history'][-1]['objective'] if final['step'] else 0.,last_tried_objective=final['history'][-1]['objective'],
        step=final['step'],linear_macro_gain=final['linear_macro_gain'],raw_linear_macro_gain=final['raw_linear_macro_gain'],
        crossfit_improved=sum(r['loss_change'] < -1e-6 for r in held),crossfit_worsened=sum(r['loss_change']>1e-6 for r in held),
        crossfit_tied=sum(abs(r['loss_change'])<=1e-6 for r in held),source_unchanged=True,suffix_accessed_for_fit=False,scientific_goal_achieved=False)
    _durable_json(output/'complete.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous)
