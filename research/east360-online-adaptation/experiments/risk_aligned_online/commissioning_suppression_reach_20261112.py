"""Postseal comparison of per-point and shared risk suppression at the original write dose."""

import argparse
from pathlib import Path
from shutil import copyfile
import time

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_feedback_span_20261112 as span
from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .commissioning_group_carry_20261112 import HISTORY
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


lag, layer = group.lag, group.layer
REFERENCE = io23.ROOT/'probe_outputs/commissioning_write_delay_east360_20261112_v2'
BAND, WAIT = '0.15', 5
MODELS = ('archived','group_point','free_point','group_shared')
STEPS, LR, TEMP = 80, .03, .15
BATCH = 96


def pointwise(base, x, mask, phase, correction, hidden, z, beta, dose):
    """Frozen risk derivative and two finite interventions, with the actual archived token norm."""
    edges = []
    patcher = layer.patcher(phase,correction)

    def transform(raw):
        value = patcher(raw)
        edges.append(group.group_edges(value,base.embedding.weight))
        return value

    sites, risk = layer.hidden_pass(base,x,mask,patcher=transform,differentiable=True,beta=beta)
    grad, = torch.autograd.grad(risk.sum(),sites[0])
    with torch.no_grad():
        op = span.edge_matrix(edges[0])
        grad = grad[:,:-1]
        gram = op.transpose(1,2)@op
        rhs = (op.transpose(1,2)@grad.flatten(1)[...,None]).squeeze(-1)
        eye = torch.eye(gram.shape[-1],device=x.device,dtype=x.dtype)
        coef = torch.linalg.solve(gram+1e-5*eye,rhs[...,None])
        down = -(op@coef).reshape_as(grad)
        valid = (mask&torch.isfinite(x)).reshape(len(x),grad.shape[1],-1).any(-1)
        mean = grad.masked_fill(~valid[...,None],0.).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        free = (mean[:,None]-grad).masked_fill(~valid[...,None],0.)
        moves = torch.stack([layer.match_dose(g,dose) for g in (down,free)])
        tokens = sites[0][:,:-1]
        actual = ((tokens+moves)-tokens).flatten(2).norm(dim=2)
        err = float((actual-dose).abs().max())
        _, candidate = layer.hidden_pass(base,x.repeat(2,1,1),mask.repeat(2,1,1),site=0,
            change=moves.flatten(0,1),patcher=layer.patcher(phase.repeat(2,*([1]*(phase.ndim-1))),
                correction.repeat(2,*([1]*(correction.ndim-1)))),beta=beta)
        pred = torch.cat((risk[None],candidate.reshape(2,-1)),0)+z-hidden@beta
    if err>1e-5 or not torch.isfinite(pred).all():
        raise ValueError('pointwise projected witness cannot realize the archived dose')
    return pred,gram,rhs,err


def linear_change(state,slot,fraction,gram,rhs,dose):
    """First-order risk change after normalizing the old and new token moves and their blend."""
    def metric_norm(c):
        return ((gram@c[...,None]).squeeze(-1)*c).sum(1).clamp_min(1e-14).sqrt()

    old,new = state[slot-1],state[slot]
    f = fraction[:,None]
    direction = (1.-f)*old/metric_norm(old)[:,None]+f*new/metric_norm(new)[:,None]
    return dose*(rhs*direction).sum(1)/metric_norm(direction)


def fit_shared(original,slot,fraction,gram,rhs,dose,risk):
    """One optimized coefficient per release; the exact output is verified separately."""
    init = torch.zeros_like(original)
    for i,a in ((slot-1,1.-fraction),(slot,fraction)):
        init.index_add_(0,i,-rhs*(a*dose)[:,None])
    touched = (original.norm(dim=1)>group.EPS)&(init.norm(dim=1)>group.EPS)
    initial = torch.where(touched[:,None],F.normalize(init,dim=1),original)
    free = initial.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([free],lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        candidate = torch.where(touched[:,None],F.normalize(free,dim=1),original)
        z = risk+linear_change(candidate,slot,fraction,gram,rhs,dose)
        loss = TEMP*torch.logsumexp(z/TEMP,dim=0)
        (free.grad,) = torch.autograd.grad(loss,free)
        opt.step()
    if not torch.isfinite(free).all():
        raise ValueError('nonfinite shared witness')
    return torch.where(touched[:,None],F.normalize(free.detach(),dim=1),original)


@torch.no_grad()
def shared_forward(base, signal, track, state, active, dose, old):
    q,h,z,beta,phase,correction = track
    r = q+lag.H+1
    original = np.searchsorted(r,active,side='right')
    delayed = np.searchsorted(r+WAIT,active,side='right')
    frac = group.handoff_fraction(r[delayed-1]+WAIT,active,base.cfg.patch)
    out = old.copy()
    errors = []
    for start in range(0,len(active),BATCH):
        sl = slice(start,start+BATCH)
        ends,s,t = active[sl],original[sl],delayed[sl]
        x,m = lag._windows(base,signal,ends)
        view,_,err = group.forward_hidden(base,x,m,layer.patcher(phase[s],correction[s]),
            writes=state[None,t],handoff=(state[None,t-1],z.new_tensor(frac[sl])),
            target_norm=z.new_tensor(dose[ends]))
        prob = (z[ends]+((view[1]-h[ends])*beta).sum(-1)).sigmoid().cpu().numpy()
        out[ends] = prob
        errors.append(float(err))
    error = max(errors,default=0.)
    if error>1e-5 or not np.isfinite(out).all():
        raise ValueError(f'shared witness violates finite dose: {error:g}')
    return out,error


def diagnose(signal,args,decoder,path,weights,release,eligible,dose,old,parent,interval,tau):
    base = args[0]
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    avail,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
    state,common = group._estimate_innovation(base,signal,q,avail,phase,correction,prior,fields,wave,decoder,weights)
    np.testing.assert_array_equal(q+lag.H+1,release)
    np.testing.assert_array_equal(common,eligible)
    if not np.any(interval & (dose==0.) & (old>=tau)):
        active = np.flatnonzero(dose>0.)
    else:
        raise ValueError('probe cohort includes an immutable crossing')
    state = state[-2]
    track = q,h,z,beta,phase,correction
    rebuilt, ref_error = shared_forward(base,signal,track,state,active,dose,old)
    replay_error = float(np.abs(rebuilt-old).max())
    np.testing.assert_array_equal(rebuilt>=tau,old>=tau)
    # Include every original crossing, plus parent and adapted near-threshold
    # points in the scoring interval to discourage moving the peak elsewhere.
    near = float(1/(1+np.exp(-(logit(tau)-TEMP))))
    ends = np.flatnonzero(interval & (dose>0.) & ((old>=near)|(parent>=near)))
    src = np.searchsorted(release,ends,side='right')
    pieces = []
    for start in range(0,len(ends),BATCH):
        sl = slice(start,start+BATCH)
        i,s = ends[sl],src[sl]
        x,m = lag._windows(base,signal,i)
        pieces.append(pointwise(base,x,m,phase[s],correction[s],h[i],z[i],beta,z.new_tensor(dose[i])))
    pp = torch.cat([r[0] for r in pieces],dim=1)
    gram,rhs = (torch.cat([r[k] for r in pieces]) for k in (1,2))
    np.testing.assert_allclose(pp[0].sigmoid().cpu().numpy(),parent[ends],atol=3e-6,rtol=0.)
    dose_error = max(ref_error,*(r[3] for r in pieces))
    scores = np.broadcast_to(old,(len(MODELS),len(old))).copy()
    for j,v in enumerate(pp[1:].sigmoid().cpu().numpy(),1):
        scores[j,ends] = np.minimum(old[ends],v)
    slot = np.searchsorted(release+WAIT,ends,side='right')
    frac = group.handoff_fraction(release[slot-1]+WAIT,ends,base.cfg.patch)
    fit = fit_shared(state.flatten(1),torch.as_tensor(slot,device=z.device),z.new_tensor(frac),
                     gram,rhs,z.new_tensor(dose[ends]),pp[0]).reshape_as(state)
    scores[3], err = shared_forward(base,signal,track,fit,active,dose,old)
    if replay_error>3e-6:
        raise ValueError(f'original direction failed to reproduce the archived curve: {replay_error:g}')
    return scores,len(ends),len(active),replay_error,max(err,dose_error)


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,ref,args,decoder,path = group.load_reference(lock)
    s,protocol = (read(REFERENCE/k) for k in ('seal.json','protocol.json'))
    key = ('order','startup','original','calibration','source_sha256')
    hist = read(HISTORY/'protocol.json')
    asset = HISTORY/'teaching.json'
    if (any(p[k]!=protocol[k] for k in key) or protocol['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or protocol['history_seal']!=sha(HISTORY/'seal.json') or hist['teaching_sha256']!=sha(asset)
            or len(s['curve_sha256'])!=360 or protocol['additional_wait_samples'][-1]!=WAIT):
        raise ValueError('the sealed same-dose 5-point replay is required')
    out.mkdir()
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_POSTSEAL_SUPPRESSION_WITNESS_NOT_A_POLICY',
        frozen_source=p['source_sha256'],reference_seal=sha(REFERENCE/'seal.json'),
        executed_sha256=sha(out/'executed_probe.py'),band=BAND,wait=WAIT,models=MODELS,
        scope_of_labels='sealed HDF5 results choose entire safe curves and timely spans of true positive curves',
        doses='same archived realized L2 at each writable endpoint; pointwise min of two finite actions '
              'relaxes shared coefficient; shared replays same 8-point handoff over full writable support',
        solver=dict(steps=STEPS,lr=LR,temp=TEMP,near_logit=TEMP,objective='smooth maximum of first-order risk')))
    base = args[0]
    weights = torch.as_tensor(read(asset)['weights'],device=base.center.device,dtype=torch.float32)
    frozen = [state_digest(a) for a in args[:2]]
    scores = read(REFERENCE/'per_shot.json')[BAND][f'current_{WAIT}']
    rows,maxima = [],np.zeros(2)
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        f = REFERENCE/'shots'/name
        if sha(f)!=s['curve_sha256'][name]:
            raise ValueError(f'changed delay curve: {sid}')
        y,down,t = group.hdf._raw_truth(sid)
        tau = p['original'] if rank<=p['startup'] else p['calibration'][BAND]
        with np.load(f,allow_pickle=False) as u:
            old,parent = (u[k] for k in (f'current_{WAIT}',group.bias.BIAS))
            release,eligible,dose = u['release'],u['eligible'],u['dose'][-1]
        truth = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,old,y,down,tau))
        if truth!=scores[rank-1]:
            raise ValueError(f'raw HDF5 scoring changed: {sid}')
        interval = np.round((down-t)*1000.,9)>30. if y else np.ones(len(t),bool)
        relevant = interval & (old>=tau)
        if not relevant.any():
            continue
        role = 'timely_disruption' if y else 'false_alarm'
        immutable = relevant & (dose==0.)
        old_wrote = int((relevant & (dose>0.)).sum())
        entry = dict(rank=rank,shot=sid,role=role,original_crossings=int(relevant.sum()),
                     immutable_crossings=int(immutable.sum()),original_writable_crossings=old_wrote,
                     inherited_alarm=bool((interval & (parent>=tau)).any()))
        if not immutable.any():
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError(f'raw signal changed: {sid}')
            np.testing.assert_array_equal(t,raw['times'])
            signal = lag.prepare_actual(raw,rank,lock['mapping'])
            v,n,m,e,d = diagnose(signal,args,decoder,path,weights,release,eligible,dose,old,parent,interval,tau)
            maxima = np.maximum(maxima,(e,d))
            entry.update(fit_endpoints=n,writable_endpoints=m,methods={
                k:dict(crossings=int((interval & (a>=tau)).sum()),
                       peak=float(a[interval].max()),
                       newly_crossing_points=int((interval & (a>=tau)&(old<tau)).sum()),
                       **score_curve(t,a,y,down,tau))
                for k,a in zip(MODELS,v,strict=True)})
        rows.append(entry)
        if rank%35==0:
            print(dict(stage='probe_full_warning_intervals',rank=rank,seconds=round(time.monotonic()-start,2)),
                  flush=True)
    if [state_digest(a) for a in args[:2]]!=frozen:
        raise ValueError('frozen network changed during probe')
    summary = {}
    for role in ('false_alarm','timely_disruption'):
        cases = [r for r in rows if r['role']==role]
        reachable = [r for r in cases if 'methods' in r]
        summary[role] = dict(all=len(cases),immutable=len(cases)-len(reachable),evaluated=len(reachable),
            total_writable_crossings=sum(r['original_writable_crossings'] for r in reachable),
            removed={k:[r['shot'] for r in reachable if r['methods'][k]['crossings']==0] for k in MODELS})
    write(out/'result.json',dict(status='POSTSEAL_SUPPRESSION_WITNESS_COMPLETE',raw_verified=360,summary=summary,
        cases=rows,max_archived_error=maxima[0],max_dose_error=maxima[1],seconds=time.monotonic()-start))
    print(summary,flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    a = ap.parse_args()
    run(a.out.resolve())
