"""Fit a small shot-local change to observed diagnostic motion, then verify on an unseen block."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_cross_diagnostic_20261113 as cross
from . import commissioning_inverse_core_20261114 as core
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer, precision = cross.group, cross.lag, cross.layer, cross.precision
ARCHIVE, WAIT, LOCK = precision.ARCHIVE, precision.WAIT, cross.LOCK_KEYS
HISTORY = 4
DAMPING = .10
CONTROL = (*group.OLD, 'current_5', 'earlier_5')
NEW = ('linear_current', 'inverse_current', 'inverse_shifted', 'inverse_without_latest')
FACTORIAL = ('linear_history', 'inverse_single')
CASES = (70320, 73405, 79649, 71076)


def jacobian(pred, state):
    """Batched forecast Jacobians via reverse-mode on the actual Transformer."""
    outputs = pred.flatten(1).sum(0)
    # The sum over batch preserves each row's derivative: examples have no
    # cross-batch attention or normalization. Each row has its own state.
    return torch.stack([torch.autograd.grad(v, state, retain_graph=i+1<len(outputs))[0].flatten(1)
                        for i,v in enumerate(outputs.unbind())],dim=1)


def follow_indices(q, eligible, length):
    """Disjoint validation for a single completed action; unrelated intervening writes excluded."""
    source = np.flatnonzero(eligible)
    if not len(source):
        return source, source
    desired = q[source]+2*lag.STRIDE
    i = np.searchsorted(q,desired).clip(max=len(q)-1)
    good = (q[i]==desired)&(q[i]+lag.H+1<length)&eligible[i]
    return source[good], i[good]


def aggregate(gram, rhs, index):
    keep = torch.as_tensor(index>=0,device=gram.device)
    i = index.clip(min=0)
    return ((gram[i]*keep[...,None,None]).sum(1),(rhs[i]*keep[...,None]).sum(1))


def candidate_steps(gram,rhs,sham,selected,windows,*,factorial=False):
    old=windows.copy()
    old[:,-1]=-1
    g,real=aggregate(gram,rhs,windows)
    _,wrong=aggregate(gram,sham,windows)
    gh,rh=aggregate(gram,rhs,old)
    linear=rhs[selected]
    inverse,shifted,historical=(core.solve_step(a,b,damping=DAMPING)
        for a,b in ((g,real),(g,wrong),(gh,rh)))
    state=torch.stack((linear,inverse,shifted,historical))
    if factorial:
        state=torch.cat((state,torch.stack((real,core.solve_step(gram[selected],linear,damping=DAMPING)))))
    return state


def observation(base, signal, q, phase, correction, prior, fields, decoder, i):
    pos=q[i]
    s=lag.delayed_slots(q,pos)[0]
    x,m=lag._windows(base,signal,pos)
    u=x.new_zeros((len(i),base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)),requires_grad=True)
    f=(prior[i],*(a[i] for a in fields[:3]),fields[3])
    view=group.forward_hidden(base,x,m,layer.patcher(phase[s],correction[s]),state=u)
    return cross.decode(view,f,decoder),u


def track(signal, args, decoder, path, teacher, *, factorial=False):
    """The same moving finite coefficient is evaluated in later, already observed windows."""
    q,h,z,beta,prior,phase,correction=group.parent_track(signal,args,decoder,path)
    base=args[0]
    ready,fields,waveform=group.feedback_waveform(signal,q,args[2],z.device)
    obs=group.signed_levels(waveform,fields[1],fields[2])
    state_shape=(base.cfg.window//base.cfg.patch-1,len(group.TRUSTED))
    width=int(np.prod(state_shape))
    force=z.new_zeros(obs.shape)
    gram=z.new_zeros((len(q),width,width))
    rhs=z.new_zeros((len(q),width))
    sham=torch.zeros_like(rhs)
    modes=(*NEW,*FACTORIAL) if factorial else NEW
    states=z.new_zeros((2+len(modes),len(q)+1,*state_shape))
    _,_,_,common=group.past_force_parts(q,force,ready)
    using=np.flatnonzero(ready)
    # Compute normals at issue, then solve only on rows with the matched
    # original eligibility. Not a replacement for the archived gradient.
    for j in range(0,len(using),group.BATCH):
        i=using[j:j+group.BATCH]
        pred,u=observation(base,signal,q,phase,correction,prior,fields,decoder,i)
        delta=obs[i]-pred.detach()
        force[i]=(delta/group.BETA).clamp(-1.,1.)
        J=jacobian(pred,u).unflatten(1,(lag.H//group.BIN,len(group.USE)))
        gram[i],rhs[i],sham[i]=core.block_system(J,delta.clamp(-1.,1.))
    fitted=core.mature_history(q,ready,horizon=lag.H,stride=lag.STRIDE,count=HISTORY)
    optional=np.flatnonzero(common)
    info=np.zeros((len(q),5),np.float32)
    if len(optional):
        parts=candidate_steps(gram,rhs,sham,optional,fitted[optional],factorial=factorial)
        linear,inverse,shifted,historical=parts[:4]
        states[2:,optional+1]=parts.unflatten(-1,state_shape)
        info[optional]=torch.stack((torch.as_tensor((fitted[optional,:-1]>=0).sum(1),device=z.device),
            F.cosine_similarity(inverse,shifted,dim=-1,eps=group.EPS),
            F.cosine_similarity(inverse,historical,dim=-1,eps=group.EPS),
            F.cosine_similarity(inverse,linear,dim=-1,eps=group.EPS),
            (gram[optional].diagonal(dim1=-2,dim2=-1).sum(-1)>0.).to(z.dtype)),dim=1).cpu().numpy()

    _,now,previous,_=group.past_force_parts(q,force,ready)
    for j in range(0,len(optional),group.BATCH):
        i=optional[j:j+group.BATCH]
        issued,current,state,wave,center,spread=group.signed_window(
            base,signal,q,i,phase,correction,prior,fields,decoder)
        # Verify our issue graph is truly the frozen existing diagnostic graph
        # by comparing its original backpropagated current/earlier alarm arms.
        pred=group.signed_levels(wave,center,spread)
        states[:2,i+1]=group.signed_directions(issued,current,state,(
            (pred,group.shape_force(now[i],teacher)),(pred,group.shape_force(previous[i],teacher))))
    if not torch.isfinite(states).all():
        raise ValueError('nonfinite shot-local inverse or matched controls')
    return (q,h,z,beta,phase,correction,states,common), (prior,fields,obs,info)


@torch.no_grad()
def held_forecasts(signal, base, tr, future, decoder, dose):
    q,h,z,beta,phase,correction,states,common=tr
    prior,fields,obs,_=future
    action=q+lag.H+1+WAIT+base.cfg.patch
    legal=common & (action<len(z))
    # The later forecast is q+64: same issue and input for every arm while
    # only moving the representation with this one old action. A newer write
    # is deliberately not substituted into this attribution comparison.
    origin,held=follow_indices(q,common,len(z))
    i=origin[legal[origin]]
    j=held[legal[origin]]
    same=dose[action[i]]>0.
    i,j=i[same],j[same]
    rows=[]
    for c in range(0,len(i),group.BATCH):
        origin,held=i[c:c+group.BATCH],j[c:c+group.BATCH]
        x,m=lag._windows(base,signal,q[held])
        slot=lag.delayed_slots(q,q[held])[0]
        t=layer.patcher(phase[slot],correction[slot])
        frozen,local_dose,check=group.forward_hidden(base,x,m,t,writes=states[:,origin+1])
        if float(check)>1e-5:
            raise ValueError('future diagnostic forward failed the same-dose match')
        prediction=cross.decode(frozen,(prior[held],*(a[held] for a in fields[:3]),fields[3]),decoder)
        residual=prediction-obs[held]
        dynamic=(prediction-prediction.mean(dim=2,keepdim=True)
                -(obs[held]-obs[held].mean(dim=1,keepdim=True)))
        losses=[F.smooth_l1_loss(v,torch.zeros_like(v),reduction='none',beta=group.BETA).mean((2,3))
                for v in (residual,dynamic)]
        action_scores,dose_,err=cross.risk_trial(signal,q,action[origin],(base,),h,z,beta,phase,
            correction,states[:,origin+1],torch.as_tensor(dose[action[origin]],device=z.device))
        if err>1e-5:
            raise ValueError('same-dose warning witness changed')
        supported=(local_dose>0.) & (dose_>0.)
        values=torch.stack((*losses,action_scores),dim=-1).transpose(0,1)
        rows.append((np.c_[q[origin],action[origin],q[held],q[held]+lag.H+1][supported.cpu().numpy()],
                     values[supported].cpu().numpy()))
    return tuple(np.concatenate(v) for v in zip(*rows,strict=True)) if rows else (
        np.empty((0,4),np.int64),np.empty((0,len(states)+1,3),np.float32))


def seal_run(out, *, factorial=False):
    lag.configure()
    start=time.monotonic()
    lock=lag.load_inputs()
    ref,_,args,decoder,path=group.load_reference(lock)
    previous,seal=(read(ARCHIVE/f'{k}.json') for k in ('protocol','seal'))
    donor=precision.delay.HISTORY/'teaching.json'
    if (any(ref[k]!=previous[k] for k in LOCK) or len(seal['curve_sha256'])!=360
            or previous['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or read(donor.parent/'protocol.json')['teaching_sha256']!=sha(donor)):
        raise ValueError('same frozen source, thresholds, and parent required')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    copyfile(core.__file__,out/'executed_core.py')
    base=args[0]
    frozen=[state_digest(m) for m in args[:2]]
    teacher=torch.as_tensor(read(donor)['weights'],device=base.center.device,dtype=torch.float32)
    tau=tuple(previous['calibration'].values())
    modes=(*NEW,*FACTORIAL) if factorial else NEW
    hist=('no_added_write',*CONTROL[2:],*modes)
    write(out/'protocol.json',dict(scope='EAST360_OBSERVED_MOTION_INVERSE_DEVELOPMENT',
        **{k:previous[k] for k in LOCK}, arms=(*CONTROL,*modes), witness_arms=hist, witness_columns=(
            'level_huber','centered_motion_huber','subsequent_logit'), reference=str(ARCHIVE),
        reference_seal=sha(ARCHIVE/'seal.json'), regularization=DAMPING, history_windows=HISTORY,
        coefficient='same 60 first-block temporal exchange coefficients; '
                    'actual issue-time forecast Jacobian of independently centered 5-bin diagnostics; '
                    'ridge on the actual Jacobian curvature. No target shot labels or risk gradients.',
        controls='last centered gradient, damped recent windows, identical system with 2-bin error roll, '
                 'same system without the latest finished target. The factorial also crosses summed '
                 'first-order gradients with a single-block damped inverse. All use the same dose and clock.',
        onset='original q+26 maturity, +5 samples wait, +8 handoff; first20 and all zero dose unchanged',
        confirmation='same follow-up origin and unsupervised target q+65..q+89, omits subsequent writes; '
                     'no information from follow-up target influences the action or alarm',
        source_fit='no new fits and no new target labels; score only after seal'))
    files,witnesses={},{}
    maximum=np.zeros(4)
    for rank,sid in enumerate(previous['order'],1):
        name=f'{rank:04d}_{sid}.npz'
        rfile=ARCHIVE/'shots'/name
        if sha(rfile)!=seal['curve_sha256'][name]:
            raise ValueError('locked raw curve has changed')
        with np.load(rfile,allow_pickle=False) as f:
            locked={k:f[k] for k in CONTROL}
            dose=f['dose'][precision.delay.WAITS.index(WAIT)]
            release,eligible=f['release'],f['eligible']
        parent=locked['bias_only']
        if rank<=previous['startup']:
            scores=np.broadcast_to(parent,(len(modes),len(parent)))
            frames=np.empty((0,4),np.int64)
            follow=np.empty((0,len(hist),3),np.float32)
            geometry=np.empty((0,5),np.float32)
        else:
            with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before seal')):
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('changed EAST recorded features')
                raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal=lag.prepare_actual(raw,rank,lock['mapping'])
                traced,future=track(signal,args,decoder,path,teacher,factorial=factorial)
                np.testing.assert_array_equal(traced[0]+lag.H+1,release)
                np.testing.assert_array_equal(traced[-1],eligible)
                rendered,pe,de=precision.render(base,signal,traced,parent,dose)
                baseline=np.stack([locked[k] for k in CONTROL[2:]])
                np.testing.assert_allclose(rendered[:2],baseline,rtol=0.,atol=3e-6)
                for t in tau:
                    np.testing.assert_array_equal(rendered[:2]>=t,baseline>=t)
                frames,follow=held_forecasts(signal,base,traced,future,decoder,dose)
                scores=rendered[2:]
                geometry=future[-1][eligible]
                maximum[:3]=np.maximum(maximum[:3],(pe,de,float(np.abs(rendered[:2]-baseline).max())))
                if rank in (21,120,360) or sid in CASES:
                    n=min(len(parent),590)
                    prefix={k:signal[k][:n] for k in ('values','available')}
                    pt,pf=track(prefix,args,decoder,path,teacher,factorial=factorial)
                    ps,*_=precision.render(base,prefix,pt,parent[:n],dose[:n])
                    np.testing.assert_allclose(ps,rendered[:,:n],rtol=0.,atol=3e-6)
                    for t in tau:
                        np.testing.assert_array_equal(ps>=t,rendered[:,:n]>=t)
                    f,w=held_forecasts(prefix,base,pt,pf,decoder,dose[:n])
                    subset=frames[:,-1]<n
                    np.testing.assert_array_equal(f,frames[subset])
                    np.testing.assert_allclose(w,follow[subset],rtol=0.,atol=7e-6)
                    maximum[3]=max(maximum[3],float(np.abs(ps-rendered[:,:n]).max()))
                    witnesses[str(sid)]=dict(prefix=n,unseen_blocks=len(f))
        dest=out/'shots'/name
        with dest.open('xb') as fp:
            np.savez_compressed(fp,**locked,**dict(zip(modes,scores,strict=True)),dose=dose,
                                frames=frames,follow=follow,geometry=geometry)
        files[name]=sha(dest)
        if rank==21 or rank%60==0:
            print(dict(stage='seal_motion_inverse',rank=rank,elapsed_s=round(time.monotonic()-start,2)),flush=True)
    if frozen!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen source or inherited network changed')
    write(out/'seal.json',dict(curves=files,witnesses=witnesses,max_error=maximum.tolist(),
        elapsed_s=time.monotonic()-start))


def observation_summary(traces, hist):
    names=('level_huber','motion_huber','action_logit')
    out={}
    for k,rows in traces.items():
        if not rows:
            out[k]=dict(shots=0,blocks=0)
            continue
        value=np.stack([s.mean(0) for s in rows])
        out[k]=dict(shots=len(rows),blocks=sum(map(len,rows)),
            means=dict(zip(hist,[dict(zip(names,a,strict=True)) for a in value.mean(0).tolist()],strict=True)),
            inverse_better=dict(
                motion_vs_parent=int((value[:,4,1]<value[:,0,1]).sum()),
                motion_vs_shift=int((value[:,4,1]<value[:,5,1]).sum()),
                action_up_vs_parent=int((value[:,4,2]>value[:,0,2]).sum())))
    return out


def score(out):
    p,s=(read(out/f'{k}.json') for k in ('protocol','seal'))
    if len(s['curves'])!=360 or p['reference_seal']!=sha(ARCHIVE/'seal.json'):
        raise ValueError('whole stream must be sealed before viewing HDF5 outcomes')
    arms=tuple(p['arms'])
    modes=arms[len(CONTROL):]
    archived=read(ARCHIVE/'per_shot.json')
    per={b:{k:[] for k in arms} for b in p['calibration']}
    groups={k:[] for k in ('all','late_before_30','late_validated_before_30','false_alarm_before_first')}
    geom,history,late=[],[],[]
    n=0
    for rank,sid in enumerate(p['order'],1):
        name=f'{rank:04d}_{sid}.npz'
        if sha(out/'shots'/name)!=s['curves'][name]:
            raise ValueError('changed sealed prediction')
        y,down,t=group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            active=f['dose']>0.
            n+=int(active.sum())
            for band,tau in p['calibration'].items():
                tau=p['original'] if rank<=p['startup'] else tau
                for k in arms:
                    if k in modes:
                        np.testing.assert_array_equal(f[k][~active],f['bias_only'][~active])
                    row=dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,f[k],y,down,tau))
                    if k in CONTROL and row!=archived[band][k][rank-1]:
                        raise ValueError('archive first alarm does not reproduce')
                    per[band][k].append(row)
            tau=p['calibration']['0.2']
            frame,value=f['frames'],f['follow']
            geom.append(f['geometry'])
            history.append(len(f['geometry']))
            ref=per['0.2']['current_5'][-1]
            if len(frame):
                groups['all'].append(value)
            if y and rank>p['startup'] and not ref['timely']:
                dt=np.round((down-t)*1000.,9)
                take=active&(dt>30.)&(dt<=125.)
                if take.any():
                    original=logit(np.clip(f['current_5'][take],1e-6,1.-1e-6))
                    uplift={k:float((logit(np.clip(f[k][take],1e-6,1.-1e-6))-original).max()) for k in modes}
                    late.append(dict(shot=sid,points=int(take.sum()),gap=float(logit(tau)-original.max()),
                        max_lift=uplift,near_cross={k:bool((f[k][take]>=tau).any()) for k in modes}))
                valid=(dt[frame[:,1]]>30.)&(dt[frame[:,1]]<=125.)
                mature=valid&(dt[frame[:,-1]]>30.)
                if valid.any():
                    groups['late_before_30'].append(value[valid])
                if mature.any():
                    groups['late_validated_before_30'].append(value[mature])
            elif not y and rank>p['startup'] and ref['alarm']:
                first=int(np.flatnonzero(f['current_5']>=tau)[0])
                valid=frame[:,1]<=first
                if valid.any():
                    groups['false_alarm_before_first'].append(value[valid])
    metrics={b:{k:lead.summarize(v) for k,v in rows.items()} for b,rows in per.items()}
    comparisons=(('bias_only','inverse_current'),('current_5','inverse_current'),
       ('linear_current','inverse_current'),('inverse_shifted','inverse_current'),
       ('inverse_without_latest','inverse_current'))
    if 'linear_history' in modes:
        comparisons+=tuple((a,b) for a,b in (('linear_current','linear_history'),
           ('linear_current','inverse_single'),('inverse_single','inverse_current'),
           ('linear_history','inverse_current'),('bias_only','linear_current'),
           ('current_5','linear_current')))
    paired={b:{f'{a}__to__{c}':dict(flips=flips(rows[a],rows[c]),lead=lead.paired(rows[a],rows[c]))
                for a,c in comparisons} for b,rows in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(metrics=metrics,paired=paired,raw_verified=360*len(arms)*len(per),
        write_endpoints=n,steps=sum(history),inverse_geometry=dict(
            columns=('prior_windows','true_vs_shift','true_vs_without_latest','true_vs_linear','nonzero_system'),
            median=np.median(np.concatenate(geom),axis=0).tolist()),
        held_observation=observation_summary(groups,p['witness_arms']),original_late=late))
    print({b:{k:(r['by_deadline']['30']['tp'],r['fp']) for k,r in rows.items()} for b,rows in metrics.items()},
          flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--factorial',action='store_true')
    parser.add_argument('--score-only',action='store_true')
    a=parser.parse_args()
    if not a.score_only:
        seal_run(a.output.resolve(),factorial=a.factorial)
    score(a.output.resolve())
