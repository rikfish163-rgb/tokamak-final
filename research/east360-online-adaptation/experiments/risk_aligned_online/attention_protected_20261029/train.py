"""Policy-only crossfit of the frozen historical no-compensation rule."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..attention_basis_20261029.train import make_cache,peaks
from ..attention_spectral_20261029.protection import contraction
from ..attention_spectral_20261029.train import historical_pairs
from ..attention_spectral_20261029.learning import proposal
from ..attention_weights_20261029.learning import operational_objective,validate_fit_ids
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import accept_history,matched_random


def fit(net,slow,cache,baseline,g,h,indices,threshold):
    flags=np.array([r['flag'] for r in cache['shots']],bool);take=np.asarray(indices)
    p,linear=contraction(g[take],h[take],flags[take]);matrix=torch.as_tensor(p,device='cuda',dtype=cache['g'].dtype)
    history=[];chosen=0.;chosen_delta=np.zeros(20);chosen_peaks=baseline.cpu().numpy()
    for exponent in range(11):
        eta=2.**(-exponent);current,_=peaks(net,slow,cache,proposal(cache['g'],matrix,eta))
        _,delta=operational_objective(current,baseline,torch.as_tensor(flags,device='cuda'),threshold)
        d=delta.cpu().numpy();accepted=accept_history(d,flags,take)
        history.append(dict(step=eta,accepted=accepted,mean_disruptive=float(d[take][flags[take]].mean()),
            mean_safe=float(d[take][~flags[take]].mean()),max_harm=float(d[take].max()),loss_changes=d.tolist()))
        if accepted:chosen=eta;chosen_delta=d;chosen_peaks=current.cpu().numpy();break
    return dict(matrix=p.tolist(),step=chosen,linear=linear,history=history,
        per_shot=[dict(shot=r['shot'],flag=r['flag'],fitted=i in indices,baseline_peak=float(baseline[i]),
            updated_peak=float(chosen_peaks[i]),loss_change=float(chosen_delta[i])) for i,r in enumerate(cache['shots'])])


def run(output,previous,diagnostic):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    p=read(previous,'protocol.json');a=read(previous,'audit.json');dp=read(diagnostic,'protocol.json');dd=read(diagnostic,'complete.json')
    ref=Path(p['reference']);oldfit=Path(dp['fit_root'])
    if not a['complete'] or a['complete_sha256']!=sha(previous/'complete.json') or not dd['complete']:raise ValueError('completed verified references required')
    if dd['protocol_sha256']!=sha(diagnostic/'protocol.json') or dp['fit_ids']!=p['order'][:20] or dp['reference_hashes']!=p['reference_hashes']:
        raise ValueError('diagnostic provenance changed')
    if sha(oldfit/'complete.json')!=dp['fit_complete_sha256']:raise ValueError('older spectral fit changed')
    for path,h in dp['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('historical diagnostic implementation changed')
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    validate_fit_ids(p['order'][:20],p['order'])
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base);before={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='ENDED_COLD20_PROTECTED_QK_CROSSFIT',source_sha256=PARENT_SHA256,previous=str(previous),previous_audit_sha256=sha(previous/'audit.json'),
        reference=str(ref),reference_hashes=p['reference_hashes'],order=p['order'],fit_ids=p['order'][:20],threshold=parent['threshold'],
        diagnostic=str(diagnostic),diagnostic_complete_sha256=sha(diagnostic/'complete.json'),plan_sha256=sha(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},policy_only_crossfit=True,cold_parent_uses_all20=True,
        backtracking_steps=[2.**(-i) for i in range(11)],max_fit_harm=1e-6,disruptive_improvement_below=-1e-6,seed=131,independent_confirmation=False)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md','ATTENTION_PROTECTED_STREAM_PLAN.md')
    cache=make_cache(net,slow,p,ref);baseline,selected=peaks(net,slow,cache,torch.zeros_like(cache['g']))
    gt,ht=historical_pairs(net,slow,cache,selected,parent['threshold']);g=gt.cpu().double().numpy();h=ht.cpu().double().numpy()
    with np.load(oldfit/'historical_pairs.npz') as z:
        if not np.array_equal(g,z['g']) or not np.array_equal(h,z['h']):raise ValueError('historical gradients changed')
    with (output/'historical_pairs.npz').open('xb') as f:np.savez_compressed(f,g=g,h=h,baseline=baseline.cpu().numpy(),selected=selected)
    _durable_json(output/'sampling.json',dict(shots=cache['shots'],receipts=cache['receipts']))
    held=[]
    for fold in range(5):
        indices=[i for i in range(20) if i%5!=fold];result=fit(net,slow,cache,baseline,g,h,indices,parent['threshold'])
        _durable_json(output/f'fold{fold}.json',result);held.extend(r for r in result['per_shot'] if not r['fitted'])
        print(dict(fold=fold,step=result['step'],ranks=result['linear']['ranks'],last_check=result['history'][-1]),flush=True)
    final=fit(net,slow,cache,baseline,g,h,list(range(20)),parent['threshold'])
    if final['step']!=dd['step'] or not np.allclose(final['matrix'],dd['projector'],atol=1e-12,rtol=0):raise ValueError('full policy failed to reproduce prior diagnostic')
    final['mean_gradient']=cache['g'].mean(0).cpu().tolist();final['random_matrix']=matched_random(torch.tensor(final['matrix'],dtype=torch.float64)).tolist()
    _durable_json(output/'policy.json',final);_durable_json(output/'crossfit.json',sorted(held,key=lambda r:p['order'].index(r['shot'])))
    if not all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('parent changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during fit')
    done=dict(complete=True,scope=protocol['scope'],seconds=time.monotonic()-started,step=final['step'],ranks=final['linear']['ranks'],
        policy_sha256=sha(output/'policy.json'),crossfit_sha256=sha(output/'crossfit.json'),protocol_sha256=sha(output/'protocol.json'),pairs_sha256=sha(output/'historical_pairs.npz'),
        crossfit_improved=sum(r['loss_change'] < -1e-6 for r in held),crossfit_worsened=sum(r['loss_change']>1e-6 for r in held),
        crossfit_tied=sum(abs(r['loss_change'])<=1e-6 for r in held),source_unchanged=True,suffix_accessed_for_fit=False,scientific_goal_achieved=False)
    _durable_json(output/'complete.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous,a.diagnostic)
