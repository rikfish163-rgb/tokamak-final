"""Historical feasibility diagnostic: do not compensate one shot's harm by another."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import linprog
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..attention_basis_20261029.train import make_cache,peaks
from ..attention_weights_20261029.learning import operational_objective
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .audit import verify_fit
from .learning import proposal
from .train import historical_pairs


def safe_weights(gains,flags,groups):
    if (gains.ndim!=2 or flags.shape!=(len(gains),) or flags.dtype!=np.bool_ or not flags.any() or flags.all()
            or groups.shape!=(gains.shape[1],) or not np.isin(groups,[0,1]).all() or not np.isfinite(gains).all()):
        raise ValueError('finite shot/atom gains, both outcome classes and Q/K groups required')
    if not gains.shape[1]:
        return np.zeros(0),dict(objective=0.,min_linear_gain=0.,solver_status='empty_positive_eigenspace')
    objective=gains[flags].mean(0);scale=max(float(np.abs(objective).max()),1e-12)
    row_norm=np.maximum(np.abs(gains).max(1),1e-12)
    budget=np.array([groups==i for i in [0,1]],dtype='float64')
    a=np.concatenate((-gains/row_norm[:,None],budget));b=np.r_[np.zeros(len(gains)),[2.,2.]]
    fit=linprog(-objective/scale,A_ub=a,b_ub=b,bounds=(0.,1.),method='highs',
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not fit.success:raise RuntimeError(f'feasible zero-point LP failed: {fit.message}')
    w=fit.x
    if np.max(a@w-b)>2e-8 or w.min() < -2e-8 or w.max()>1+2e-8:raise ValueError('LP returned infeasible weights')
    return w,dict(objective=float(objective@w),min_linear_gain=float((gains@w).min()),solver_status=str(fit.message))


def contraction(g,h,flags):
    ng=g/np.maximum(np.linalg.norm(g.reshape(len(g),-1),axis=1),1e-12)[:,None,None,None]
    raw=h.transpose(0,1,3,2)@ng;s=(raw+raw.transpose(0,1,3,2))/2
    means=s[flags].mean(0);ev,u=np.linalg.eigh(means);atoms=[];groups=[];gains=[]
    for group in [0,1]:
        for j in np.flatnonzero(ev[group]>np.abs(ev[group]).max()*1e-12):
            atom=np.outer(u[group,:,j],u[group,:,j]);atoms.append(atom);groups.append(group)
            gains.append(np.einsum('ij,nji->n',atom,s[:,group]))
    a=np.stack(gains,axis=1) if gains else np.empty((len(g),0));groups=np.asarray(groups)
    weights,info=safe_weights(a,flags,groups);p=np.zeros((2,64,64))
    for w,atom,group in zip(weights,atoms,groups):p[group]+=w*atom
    spectra=np.linalg.eigvalsh(p)
    if spectra.min() < -1e-8 or spectra.max()>1+1e-8 or (spectra.sum(1)>2+1e-8).any():raise ValueError('not a trace-bounded PSD contraction')
    return p,dict(**info,atom_groups=groups.tolist(),weights=weights.tolist(),gains=a.tolist(),
        per_shot_linear_gain=(a@weights).tolist(),spectra=spectra.tolist(),ranks=(spectra>1e-8).sum(1).tolist())


def run(output,fit_root):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    verified=verify_fit(fit_root);p=read(fit_root,'protocol.json');ref=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('reference changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base);before={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='HISTORICAL_ONLY_NO_COMPENSATION_DIAGNOSTIC',fit_root=str(fit_root),fit_complete_sha256=sha(fit_root/'complete.json'),
        fit_ids=p['fit_ids'],reference=str(ref),reference_hashes=p['reference_hashes'],source_sha256=p['source_sha256'],
        plan_sha256=sha(io23.ROOT/'ATTENTION_SPECTRAL_PROTECTION_DIAGNOSTIC.md'),threshold=parent['threshold'],
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},independent_confirmation=False)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_SPECTRAL_PROTECTION_DIAGNOSTIC.md','ATTENTION_SPECTRAL_PROTECTION_DIAGNOSTIC.md')
    cache=make_cache(net,slow,p,ref);baseline,selected=peaks(net,slow,cache,torch.zeros_like(cache['g']))
    g,h=historical_pairs(net,slow,cache,selected,parent['threshold'])
    with np.load(fit_root/'historical_pairs.npz') as z:
        if not np.array_equal(g.cpu().numpy(),z['g']) or not np.array_equal(h.cpu().numpy(),z['h']):raise ValueError('recomputed historical gradients changed')
    flags=np.array([r['flag'] for r in cache['shots']],bool)
    contraction_matrix,linear=contraction(g.cpu().double().numpy(),h.cpu().double().numpy(),flags)
    matrix=torch.tensor(contraction_matrix,device='cuda',dtype=g.dtype);history=[];chosen=0.;chosen_delta=np.zeros(20)
    for exponent in range(11):
        eta=2.**(-exponent);current,_=peaks(net,slow,cache,proposal(cache['g'],matrix,eta))
        objective,delta=operational_objective(current,baseline,torch.tensor(flags,device='cuda'),parent['threshold'])
        d=delta.cpu().numpy();accepted=bool(np.all(d<=1e-6) and d[flags].mean() < -1e-6)
        history.append(dict(step=eta,accepted=accepted,macro_harm_doubled=float(objective),
            mean_disruptive=float(d[flags].mean()),mean_safe=float(d[~flags].mean()),max_harm=float(d.max()),loss_changes=d.tolist()))
        if accepted:chosen=eta;chosen_delta=d;break
    if not all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('parent changed')
    for path,digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=digest:raise ValueError('code changed during run')
    result=dict(complete=True,scope=protocol['scope'],seconds=time.monotonic()-started,step=chosen,
        linear=linear,projector=contraction_matrix.tolist(),history=history,
        per_shot=[dict(shot=r['shot'],flag=r['flag'],loss_change=float(chosen_delta[i])) for i,r in enumerate(cache['shots'])],
        historical_recomputed_exactly=True,source_unchanged=True,suffix_accessed=False,protocol_sha256=sha(output/'protocol.json'),
        prior_verification=verified,scientific_goal_achieved=False,not_online_metrics=True)
    _durable_json(output/'complete.json',result)
    print(dict(step=chosen,linear_objective=linear['objective'],min_linear_gain=linear['min_linear_gain'],ranks=linear['ranks'],
               history=history,seconds=result['seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fit',type=Path,required=True)
    a=parser.parse_args();run(a.output,a.fit)
