"""Matched-dose neural trial: separate SXR brightness and profile-shape feedback."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_sxr_moments_20261114 as moments
from . import commissioning_sxr_shape_20261114 as probe
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_history_span_attribution_20261114 import confirmation
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


history, group, lag, layer = probe.history, probe.group, probe.lag, probe.layer
WAIT, PATCH, OLD = probe.WAIT, probe.PATCH, probe.OLD
BETA, RIDGE = .2, .10
ARMS = ('sxr_joint', 'sxr_gain', 'sxr_shape', 'sxr_shape_shifted', 'sxr_shape_earlier')
CASES = (70320, 73405, 69813, 73790, 71076)


def fit_head(packets):
    """Weighted direct H25 linear decoder from frozen risk representations."""
    n = len(packets)
    mu = np.mean([x.mean(0, dtype=np.float64) for x,y in packets],axis=0)
    scale = np.sqrt(np.mean([np.square(x-mu).mean(0) for x,y in packets],axis=0)).clip(1e-6)
    d = len(mu)+1
    g,t = np.zeros((d,d)),np.zeros((d,20))
    for x,y in packets:
        z = np.c_[(x-mu)/scale, np.ones(len(x))]
        g += (z.T@z)/len(z)
        t += (z.T@y.reshape(-1,20))/len(z)
    reg = np.diag([RIDGE]*(d-1)+[0.])
    return mu,scale,np.linalg.solve(g/n+reg,t/n)


def decode(hidden, head):
    center, scale, affine = head
    return (((hidden-center)/scale)@affine[:-1]+affine[-1]).unflatten(-1, (5,4))


def closed_head(hidden, head):
    mu,scale,w = head
    return (((hidden-mu)/scale)@w[:-1]+w[-1]).reshape(-1,5,4)


def force(target, forecast):
    """Negative derivative of scaled SmoothL1, apart from the fixed reduction."""
    return ((target-forecast.detach())/BETA).clamp(-1.,1.)


def separate(f):
    g = f.clone()
    g[...,1:] = 0.
    return g, f-g


def compose(base, aux, supported):
    """Equal unit mixture in coefficient space; unchanged legacy if not supported."""
    b, a = (F.normalize(v.flatten(-2),dim=-1,eps=group.EPS) for v in (base,aux))
    mixed = (b[None]+a).reshape_as(aux)
    remainder = torch.where(supported[None,:,None,None],mixed,base[None])
    return torch.cat((base[None],remainder))


def cosine_rows(states):
    reference, candidate = states[:1].flatten(2), states[1:].flatten(2)
    return F.cosine_similarity(reference,candidate,dim=-1).T.contiguous()


def sampled_features(base, signal, q):
    packets=[]
    with torch.no_grad():
        for j in range(0,len(q),group.BATCH):
            x,m = lag._windows(base,signal,q[j:j+group.BATCH])
            y=[]
            handle=base.risk_head.register_forward_pre_hook(lambda _m,args:y.append(args[0]))
            try:
                base(x,m)
            finally:
                handle.remove()
            packets.append(y[0].cpu().numpy())
    return np.concatenate(packets) if packets else np.empty((0,base.cfg.dim),np.float32)


def fit_source(out, observed):
    lag.configure()
    base, _ = load_parent(DEFAULT_PARENT)
    base.cuda()
    roles = simple_data.source_roles()
    _, target_scale = probe.load_forecast(observed)
    d, counts = {}, {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('decoder training uses signals only')):
        for role in ('train','val'):
            packets, rows = [], []
            for sid in roles[role]:
                s = simple_data.read_shot('J-TEXT',sid)
                q = np.arange(lag.FIRST,len(s['values']),lag.STRIDE)
                _, y, _, ok = probe.make_packet(s,q)
                rows.append(dict(shot=sid,blocks=int(ok.sum())))
                if ok.any():
                    packets.append((sampled_features(base,s,q[ok]),y[ok]/target_scale))
            d[role],counts[role] = packets,rows
    head=fit_head(d['train'])
    e=[]
    for v,y in d['val']:
        prediction=closed_head(v,head)
        e.append(np.stack([probe.shape_loss(p,y,1.).mean((0,1)) for p in (np.zeros_like(y),prediction)]))
    stats = dict(val_shots=len(e),val_blocks=sum(len(y) for _,y in d['val']),
                 columns=('gain','relative_spread','skew','gap'),
                 hold_actual=np.mean(e,axis=0).tolist())
    out.mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    copyfile(probe.__file__,out/'executed_probe.py')
    copyfile(moments.__file__,out/'executed_moments.py')
    write(out/'decoder.json',dict(counts=counts,ridge=RIDGE,checkpoint=PARENT_SHA256,
        head=[x.tolist() for x in head],target_scale=target_scale.tolist(),stats=stats))
    print(dict(stage='fit',**stats),flush=True)


def estimate(signal,args,decoder,path,teacher,head,scale,archive):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    np.testing.assert_array_equal(q+lag.H+1,archive['release'])
    base = args[0]
    eligible,fields,waveform=group.feedback_waveform(signal,q,args[2],z.device)
    s = prior.new_zeros((len(ARMS)+1,len(q)+1,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    old_force = prior.new_zeros((len(q),lag.H//group.BIN,len(group.USE)))
    model_force = prior.new_zeros((2,len(q),5,4))
    _,_,_,common = group.past_force_parts(q,old_force,eligible)
    np.testing.assert_array_equal(common,archive['eligible'])
    _, target, issued_valid, complete=probe.make_packet(signal,q)
    target=torch.as_tensor(target/scale,device=z.device,dtype=z.dtype)
    pred_trace=target.new_zeros(target.shape)
    previous=probe.previous_rows(q,complete & eligible)
    qualified=common & complete & (previous>=0)
    predicted=np.zeros(len(q),bool)
    support, spatial = np.flatnonzero(eligible),prior.new_zeros((len(q),len(ARMS)))
    # The forecast is computed in the issue-time graph. Validity only decides
    # whether the arrived target is usable, not the value of the forecast.
    for a in range(0,len(support),group.BATCH):
        i=support[a:a+group.BATCH]
        issue,current,state,wave,c,m=group.signed_window(base,signal,q,i,phase,correction,prior,fields,decoder)
        pred,old_force[i]=history.cross.precision.signed_residual(wave,waveform[i],c,m)
        _,now,_,_=group.past_force_parts(q,old_force,eligible)
        ready=torch.as_tensor(common[i],device=z.device)[:,None,None]
        sxr=decode(issue,head)
        pred_trace[i]=sxr.detach()
        predicted[i]=issued_valid[i]
        can=torch.as_tensor(complete[i],device=z.device)[:,None,None]
        model_force[0,i]=force(target[i],sxr)*can
        model_force[1,i]=force(target[i].roll(2,1),sxr)*can
        # Earlier residual comes from an actually issued forecast at its own target.
        older=model_force[0,np.maximum(previous[i],0)]
        r=model_force[0,i]
        gain,shape=separate(r)
        specs=((pred,group.shape_force(now[i],teacher)*ready),
               *((sxr,f) for f in (r,gain,shape,separate(model_force[1,i])[1],separate(older)[1])))
        g = group.signed_directions(issue,current,state,specs)
        s[:,i+1] = compose(g[0],g[1:],torch.as_tensor(qualified[i],device=z.device))
        spatial[i] = cosine_rows(s[:,i+1])
    valid = predicted & complete
    errors=pred_trace.new_zeros((len(q),2,4))
    errors[:,0] = F.smooth_l1_loss(target,torch.zeros_like(target),reduction='none').mean(1)
    errors[:,1] = F.smooth_l1_loss(target,pred_trace,reduction='none').mean(1)
    errors[~valid] = 0.
    if not torch.isfinite(s).all():
        raise ValueError('nonfinite SXR write')
    return (q,h,z,beta,phase,correction,s,qualified),dict(q=q,qualified=qualified,
           valid=valid,errors=errors.cpu().numpy(),cosine=spatial.cpu().numpy())


@torch.no_grad()
def render(base, signal, track, archive, thresholds):
    q,h,z,beta,phase,correction,states,qualified=track
    rel=q+lag.H+1+WAIT
    parent, legacy, dose=(archive[k] for k in ('bias_only','current_5','dose'))
    pos = np.flatnonzero(dose>0.)
    use, exp, frac=probe.clock(rel,qualified,pos)
    old_cur=pos[use]
    curves = np.broadcast_to(legacy, (len(ARMS),len(legacy))).copy()
    radius, seen = np.zeros_like(dose), np.zeros_like(dose,dtype=bool)
    maxima=np.zeros(3)
    norm=states.flatten(-2).norm(dim=-1)
    # If a specialized gradient is exactly zero, it adds no signal. If the
    # legacy gradient needed for a dose-matched comparison isn't present, use
    # the sealed common legacy in every arm at that endpoint instead.
    for a in range(0,len(old_cur),group.BATCH):
        sl=slice(a,a+group.BATCH)
        t=old_cur[sl]
        k,f=exp[use][sl],frac[use][sl]
        slot=lag.delayed_slots(q,t)[0]
        x,m=lag._windows(base,signal,t)
        transform=layer.patcher(phase[slot],correction[slot])
        cap=torch.as_tensor(dose[t],device=z.device)
        k0=np.maximum(k-1,0)
        # The archived dose proves an inherited action can write at this same
        # endpoint. The current and previous vectors are blended as before.
        good=((norm[:,k]>group.EPS).all(0) & (norm[:,k0]>group.EPS).all(0)).cpu().numpy()
        f=torch.as_tensor(f,device=z.device,dtype=z.dtype)
        v,actually,error=group.forward_hidden(base,x,m,transform,writes=states[:,k],
                                      handoff=(states[:,k0],f),target_norm=cap)
        score=(z[t]+((v-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        active=good & (actually.cpu().numpy()>0.)
        difference=np.abs(score[1,active]-legacy[t[active]])
        if difference.size:
            maxima[1]=max(maxima[1],float(difference.max()))
            for th in thresholds:
                np.testing.assert_array_equal(score[1,active]>=th,legacy[t[active]]>=th)
        maxima[0]=max(maxima[0],float(np.abs(score[0]-parent[t]).max()))
        maxima[2]=max(maxima[2],float(error))
        k=t[active]
        curves[:,k]=score[2:,active]
        radius[k]=actually.cpu().numpy()[active]
        seen[k]=True
    if (maxima[:2]>3e-6).any() or maxima[2]>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'SXR intervention violates same-parent or dose parity: {maxima}')
    return dict(zip(ARMS,curves,strict=True)),dict(applied=seen,dose=radius),maxima


def replay(out, *, rank_only=None):
    start=time.monotonic()
    lock,p,s,args,decoder,path=history.inputs()
    source=read(out/'decoder.json')
    if source['checkpoint']!=p['source_sha256'] or len(source['counts']['train'])!=1353:
        raise ValueError('the fitted source forecast and the archived source must match')
    head=tuple(torch.as_tensor(v,device=args[0].center.device,dtype=torch.float32) for v in source['head'])
    scale=np.asarray(source['target_scale'])
    teaching=history.cross.precision.delay.HISTORY/'teaching.json'
    committed=read(history.cross.precision.delay.HISTORY/'protocol.json')
    if committed['teaching_sha256']!=sha(teaching):
        raise ValueError('existing legacy teacher changed')
    teacher=torch.as_tensor(read(teaching)['weights'],device=args[0].center.device,dtype=torch.float32)
    thresholds=tuple(p['calibration'].values())
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(scope='EAST360_SXR_NEURAL_REPRESENTATION_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK},arms=(*OLD,*ARMS),rank_only=rank_only,
        reference=str(history.ARCHIVE),reference_seal=sha(history.ARCHIVE/'seal.json'),
        auxiliary='J-TEXT TRAIN-only linear H25 frozen CLS head; no endpoint outcomes',
        gradient='real magnitude or shape residual at issue graph through the release Jacobian; '
                 'unit-normalize auxiliary, add to unit legacy, equalize actual first-layer token dose',
        feedback_clock=dict(horizon=25,release=26,wait=WAIT,handoff=PATCH,
                            previous_control='immediately preceding complete 32-point slot'),
        reset='each shot; cold20 unchanged; no new inter-shot update or alarm policy'))
    snapshots=[state_digest(m) for m in args[:2]]
    files,witnesses,maxima={},{},np.zeros(3)
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed target outcome')):
        for rank,sid in enumerate(p['order'],1):
            if rank_only is not None and rank!=rank_only:
                continue
            arc=history.archived_shot(rank,sid,s)
            empty=np.zeros_like(arc['dose'])
            if rank<=p['startup']:
                curves=dict.fromkeys(ARMS,arc['current_5'])
                trace=dict(applied=empty.astype(bool),dose=empty,q=np.empty(0,np.int64),
                    qualified=np.empty(0,bool),valid=np.empty(0,bool),errors=np.empty((0,2,4),np.float32),
                    cosine=np.empty((0,len(ARMS)),np.float32))
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('EAST signals changed relative to the lock')
                raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal=lag.prepare_actual(raw,rank,lock['mapping'])
                tr, trace=estimate(signal,args,decoder,path,teacher,head,scale,arc)
                curves, w, err=render(args[0],signal,tr,arc,thresholds)
                trace.update(w)
                maxima=np.maximum(maxima,err)
                if rank in (21,120,360) or sid in CASES or rank_only==rank:
                    n=min(len(raw['times']),620)
                    sig={k:signal[k][:n] for k in ('values','available')}
                    cut={k:arc[k][:n] for k in (*OLD,'dose')}
                    maturity=arc['release']<n
                    cut.update(release=arc['release'][maturity],eligible=arc['eligible'][maturity])
                    tr, _=estimate(sig,args,decoder,path,teacher,head,scale,cut)
                    u,w,_=render(args[0],sig,tr,cut,thresholds)
                    for k in ARMS:
                        np.testing.assert_allclose(u[k],curves[k][:n],atol=3e-6,rtol=0.)
                        for th in thresholds:
                            np.testing.assert_array_equal(u[k]>=th,curves[k][:n]>=th)
                    np.testing.assert_array_equal(w['applied'],trace['applied'][:n])
                    witnesses[str(sid)]=dict(points=n,active=int(w['applied'].sum()),
                        probability_error=max(float(np.abs(u[k]-curves[k][:n]).max()) for k in ARMS))
            name=f'{rank:04d}_{sid}.npz'
            dest=out/'shots'/name
            with dest.open('xb') as h:
                np.savez_compressed(h,**{k:arc[k] for k in OLD},**curves,**trace)
            files[name]=sha(dest)
            if rank in (21,60,120,180,240,300,360) or rank_only:
                print(dict(stage='sealed',rank=rank,applied=int(trace['applied'].sum()),
                           elapsed=round(time.monotonic()-start,2)),flush=True)
    if snapshots!=[state_digest(m) for m in args[:2]]:
        raise ValueError('a persistent source or parent model changed')
    write(out/'seal.json',dict(curves=files,prefix=witnesses,maximum=maxima.tolist(),
                              elapsed_s=time.monotonic()-start))


def score(out):
    p,s=(read(out/f'{n}.json') for n in ('protocol','seal'))
    if p['rank_only'] is not None or len(s['curves'])!=360 or p['reference_seal']!=sha(history.ARCHIVE/'seal.json'):
        raise ValueError('all 360 forecasts must have been sealed')
    old=read(history.ARCHIVE/'per_shot.json')
    per={b:{a:[] for a in (*OLD,*ARMS)} for b in p['calibration']}
    counts, spans, e = [], [], defaultdict(list)
    for rank,sid in enumerate(p['order'],1):
        key=f'{rank:04d}_{sid}.npz'
        dest=out/'shots'/key
        if sha(dest)!=s['curves'][key]:
            raise ValueError('sealed SXR prediction changed')
        y,down,t=group.hdf._raw_truth(sid)
        with np.load(dest,allow_pickle=False) as f:
            active, q, qualified=f['applied'],f['q'],f['qualified']
            ready,_,_=probe.clock(q+lag.H+1+WAIT,qualified,np.flatnonzero(active))
            if not ready.all():
                raise ValueError('SXR auxiliary change before legitimate release')
            counts.append(dict(shot=sid,applied=int(active.sum()),mature=int(f['valid'].sum()),
                               releases=int(qualified.sum())))
            for k in ARMS:
                np.testing.assert_array_equal(f[k][~active],f['current_5'][~active])
            for b,tau in p['calibration'].items():
                th=p['original'] if rank<=p['startup'] else tau
                for a in per[b]:
                    row=dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,f[a],y,down,th))
                    if a in OLD and row!=old[b][a][rank-1]:
                        raise ValueError('archived baseline changed')
                    per[b][a].append(row)
            valid, es=f['valid'],f['errors']
            if valid.any():
                e['all_EAST'].append(es[valid])
            ref=per['0.2']['current_5'][-1]
            tau=p['calibration']['0.2']
            if rank>p['startup'] and y and not ref['timely']:
                ld=np.round((down-t)*1000.,9)
                near=(ld>30.)&(ld<=125.)&active
                sample=np.flatnonzero(valid & (q+lag.H+1+WAIT+PATCH<len(t)))
                j=q[sample]+lag.H+1+WAIT+PATCH
                sample=sample[(ld[j]>30.)&(ld[j]<=125.)]
                if len(sample):
                    e['near_miss'].append(es[sample])
                if near.any():
                    def logits(a):
                        return logit(f[a][near].clip(1e-6,1.-1e-6))
                    vz=logits('current_5')
                    spans.append(dict(shot=sid,window=int(near.sum()),gap=float(logit(tau)-vz.max()),
                        max_lift={a:float((logits(a)-vz).max()) for a in ARMS},
                        crosses={a:bool((f[a][near]>=tau).any()) for a in ARMS}))
            if not y and rank>p['startup'] and ref['alarm']:
                end=int(np.flatnonzero(f['current_5']>=tau)[0])
                v=valid & (q+lag.H+1+WAIT+PATCH<=end)
                if v.any():
                    e['before_actual_false_alarm'].append(es[v])
    pairs=[('current_5',k) for k in ARMS]+[(k,'sxr_shape') for k in ARMS if k!='sxr_shape']
    metrics={b:{a:lead.summarize(x) for a,x in table.items()} for b,table in per.items()}
    paired={b:{f'{a}__to__{c}':dict(flips=flips(table[a],table[c]),lead=lead.paired(table[a],table[c]))
               for a,c in pairs} for b,table in per.items()}
    fcast={role:dict(shots=len(rows),windows=sum(map(len,rows)),
                     hold_and_decoder=np.mean([v.mean(0,dtype=np.float64) for v in rows],axis=0).tolist())
           for role,rows in e.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(metrics=metrics,paired=paired,support=counts,late=spans,forecast=fcast,
                                 hdf5_scored=360*len(per)*len((*OLD,*ARMS))))
    write(out/'postseal_confirmation.json',confirmation(out,names=('bias_only','current_5',*ARMS)))
    print({b:{a:(row['by_deadline']['30']['tp'],row['fp']) for a,row in t.items()}
           for b,t in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--observed',type=Path)
    parser.add_argument('--rank-only',type=int)
    parser.add_argument('--stage',choices=('source','replay','score'),required=True)
    a=parser.parse_args()
    if a.stage=='source':
        if a.observed is None:
            parser.error('--observed is required for source fit')
        fit_source(a.output.resolve(),a.observed.resolve())
    elif a.stage=='replay':
        replay(a.output.resolve(),rank_only=a.rank_only)
    else:
        score(a.output.resolve())
