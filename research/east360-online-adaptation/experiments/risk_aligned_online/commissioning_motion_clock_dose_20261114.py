"""Test shared motion at the legacy dose and condition the forecast on prior arrivals."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_shared_motion_20261114 as motion
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


history,lag,layer = motion.history,motion.lag,motion.layer
RAW,USES = 'fixed_source', 'past_gain'
DOSE = ('large_current','large_earlier','gain_large_current','gain_large_earlier')
ARMS = ('fixed','bias_only','current_5','source_representation',*DOSE)
CASES = (70320,73405,69813,73790,74970)
MIN_NORM = 1e-4


def issued_gain(q, pred, actual, known):
    """Constrained per-channel gain, fit only to the last three arrived H25 pairs."""
    if not len(q):
        return pred.new_empty((0,pred.shape[-1])),np.empty(0,dtype=int)
    right = np.searchsorted(q+motion.OFFSET,q,side='right')
    prior = np.maximum.accumulate(np.r_[-1,np.where(known,np.arange(len(q)),-1)])
    i = prior[right]
    use = []
    for _ in range(3):
        use.append(i)
        i = prior[np.maximum(i,0)]
    ids = np.column_stack(use)
    obs = ids>=0
    keep = torch.as_tensor(obs,device=pred.device)[:,:,None,None]
    accepted = ids.clip(min=0)
    x,y = (v[accepted].masked_fill(~keep,0.) for v in (pred,actual))
    energy = x.square().mean(2).sum(1)
    covariance = (x*y).mean(2).sum(1)
    # One source prior packet, in the same asinh displacement units as the forecast.
    gain = ((covariance+1.)/(energy+1.)).clamp(0.,1.)
    return gain,obs.sum(1)


def fixed_radius(offsets, cap):
    norm = offsets.flatten(2).norm(dim=-1)
    r = cap.masked_fill(~(norm>MIN_NORM).all(0),0.)
    return offsets*(r/norm.clamp_min(MIN_NORM))[...,None,None],r


def paired_directions(q, grad, observed):
    old,legal = motion.past_eligible(q,observed)
    z = torch.zeros_like(grad[:1]) if len(grad) else grad.new_zeros((1,2,motion.C))
    enabled = torch.as_tensor(np.r_[False,legal],device=grad.device)[:,None,None]
    a = torch.cat((z,grad))*enabled
    b = torch.cat((z,torch.cat((z,grad))[old+1]))*enabled
    return torch.stack((a,b)),legal


def paid_feedback(signal,base,model,scale,q,support):
    knots, targets, masks = [], [], []
    for j in range(0,len(q),motion.BATCH):
        t = q[j:j+motion.BATCH]
        x,m = lag._windows(base,signal,t)
        with torch.no_grad():
            g = motion.input_view(base,x,m,scale)
            target,mask = motion.target_view(base,*motion.future(base,signal,t),g,scale)
            knots.append(model(g))
            targets.append(target)
            masks.append(mask)
    k = torch.cat(knots) if knots else base.center.new_empty((0,motion.K,motion.C))
    y = torch.cat(targets) if targets else base.center.new_empty((0,motion.H,motion.C))
    m = torch.cat(masks) if masks else torch.empty_like(y,dtype=torch.bool)
    observed = support & m.all(dim=(1,2)).cpu().numpy()
    gain,n = issued_gain(q,motion.increment(motion.waveform(k)),motion.increment(y),observed)
    variants = torch.stack((k,k*gain[:,None]))
    error = torch.stack([motion.absolute_errors(motion.waveform(v),y,m) for v in variants])
    if len(q):
        step = k.new_zeros((2,len(k),2,motion.C),requires_grad=True)
        ls = torch.stack([motion.sample_loss(motion.shared_state(v,s),y,m) for v,s in zip(variants,step,strict=True)])
        g, = torch.autograd.grad(ls.sum(),step)
        step = (-g).clamp(-1.,1.)
    else:
        step = k.new_empty((2,0,2,motion.C))
    direct,legal = paired_directions(q,step[0],observed)
    calibrated,_ = paired_directions(q,step[1],observed)
    return direct,calibrated,observed,legal,error.detach().cpu().numpy(),n


@torch.no_grad()
def evaluate(signal,base,model,carrier,scale,track,feedback,capped,known):
    q,h,z,beta,phase,correction = track
    n = len(z)
    curves = np.empty((5,n),np.float32)
    dose = np.zeros((2,n),np.float32)
    err = 0.
    for j in range(0,n,motion.BATCH):
        t = np.arange(j,min(j+motion.BATCH,n))
        x,m = lag._windows(base,signal,t)
        view = motion.input_view(base,x,m,scale)
        knots = model(view)
        keep = motion.masking(m)[...,None]
        root = carrier.static(knots,view.known)*keep
        first,last,fraction = motion.available_states(q,t,known)
        f = z.new_tensor(fraction)[None,:,None,None]
        joint = []
        for k,states in enumerate(feedback):
            trial = states[:,last].lerp(states[:,first],f)
            add = torch.stack([carrier.static(motion.shared_state(knots,a)-knots,view.known) for a in trial])
            delta,want = fixed_radius(add*keep[None],z.new_tensor(capped[t]))
            joint.append(root[None]+delta)
            dose[k,t] = want.cpu().numpy()
        edits = torch.cat((root[None],*joint),dim=0)
        slot = lag.delayed_slots(q,t)[0]
        patcher = layer.patcher(phase[slot].repeat(5,1,1),correction[slot].repeat(5,1))
        check = []

        def check_delta(tokens):
            a = tokens[:,:-1].reshape(5,len(t),-1)
            used = (a[1:]-a[:1]).norm(dim=-1).reshape(2,2,len(t))
            check.append(float((used-z.new_tensor(dose[:,t])[:,None]).abs().max()))

        out = motion.frozen_hidden(base,x.repeat(5,1,1),m.repeat(5,1,1),edits.flatten(0,1),patcher,
                                   observe_tokens=check_delta)
        curves[:,t] = (z[t]+((out.reshape(5,len(t),-1)-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        for i in range(2):
            sl = slice(2*i+1,2*i+3)
            curves[sl,t] = np.where(dose[i,t]>0.,curves[sl,t],curves[0,t])
        err = max(err,check[0])
    return curves,dose,err


def track_and_run(signal,args,decoder,path,model,carrier,scale,arc):
    q,h,z,beta,_,phase,correction = motion.group.parent_track(signal,args,decoder,path)
    np.testing.assert_array_equal(q+motion.H+1,arc['release'])
    a,b,observed,usable,e,n = paid_feedback(signal,args[0],model,scale,q,arc['eligible'])
    curves,dose,err = evaluate(signal,args[0],model,carrier,scale,(q,h,z,beta,phase,correction),
                              (a,b),arc['dose'],usable)
    return curves,dose,err,observed,usable,e,n


def run(source,out):
    begun = time.monotonic()
    lock,p,seal,args,decoder,path = history.inputs()
    base = args[0]
    model,carrier,scale = motion.inputs_from_fit(source,base)
    v1, sv1 = read(source/'protocol.json'),read(source/'seal.json')
    if len(sv1['curves'])!=360 or sha(source/'shared.pt')!=v1['source_fit']:
        raise ValueError('fixed full-stream source carrier required')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(scope='EAST360_SHARED_MOTION_FIXED_LEGACY_RADIUS_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK},arms=ARMS,archive_seal=sha(history.ARCHIVE/'seal.json'),
        source=str(source),source_seal=sha(source/'seal.json'),
        fit='last three complete old pairs from same discharge; fit a nonnegative shrinkage of slope forecast; '
            'no outcome and no later packet; shrink forecasts used only for feedback',
        risk='same source carrier state in all arms; no threshold or direct score adjustment',
        dose='fixed old token L2 when both algebraic directions are nonzero; diagnostic stress not a natural step',
        outcome='read after all curves sealed; same cold20 and no inter-shot update'))
    weights = [state_digest(m) for m in (*args[:2],model,carrier)]
    files, measured = {},{}
    norms, source_diff = 0.,0.
    cutoffs = tuple(p['calibration'].values())
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('no EAST target outcome in prediction')):
        for rank,sid in enumerate(p['order'],1):
            key = f'{rank:04d}_{sid}.npz'
            arc = history.archived_shot(rank,sid,seal)
            n = len(arc['bias_only'])
            cuts = len(arc['release'])
            curves = np.broadcast_to(arc['bias_only'],(5,n))
            dose = np.zeros((2,n),np.float32)
            forecast = np.zeros((2,cuts,4),np.float32)
            observed = legal = np.zeros(cuts,bool)
            count = np.zeros(cuts,int)
            if rank>p['startup']:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('the referenced input changed')
                x = lag.prepare_actual(simple_data.read_shot('EAST',sid,rate_contract='nominal'),rank,lock['mapping'])
                curves,dose,e,observed,legal,forecast,count = track_and_run(x,args,decoder,path,model,carrier,scale,arc)
                norms = max(norms,e)
                if sha(source/'shots'/key)!=sv1['curves'][key]:
                    raise ValueError('previous source-only curves were not sealed')
                with np.load(source/'shots'/key,allow_pickle=False) as f:
                    old = f['source_representation']
                    motion.probe.check_thresholds(curves[0],old,cutoffs)
                    source_diff = max(source_diff,float(abs(curves[0]-old).max()))
                    np.testing.assert_array_equal(f['eligible'],observed)
                    np.testing.assert_allclose(f['forecast'],forecast[0],atol=2e-6,rtol=0.)
                if rank in (21,120,360) or sid in CASES:
                    hit = np.flatnonzero((curves[1]>=p['calibration']['0.2']) & (dose[0]>0.))
                    cut = min(n,max(768,int(hit[0])+2) if len(hit) else 768)
                    short = {a:v[:cut] if a in ('values','available') else v for a,v in x.items()}
                    i = arc['release']<cut
                    little = dict(dose=arc['dose'][:cut],release=arc['release'][i],eligible=arc['eligible'][i])
                    zz,dd,*_ = track_and_run(short,args,decoder,path,model,carrier,scale,little)
                    motion.probe.check_thresholds(zz,curves[:,:cut],cutoffs)
                    np.testing.assert_allclose(dd,dose[:,:cut],atol=1e-5,rtol=0.)
                    measured[str(sid)] = dict(points=cut,delta=float(abs(zz-curves[:,:cut]).max()))
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f,**dict(zip(ARMS[3:],curves,strict=True)),
                                   dose=dose,forecast=forecast,observed=observed,legal=legal,prior=count)
            files[key] = sha(out/'shots'/key)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='legacy_dose_seal',rank=rank,seconds=time.monotonic()-begun),flush=True)
    if norms>1e-5 or weights!=[state_digest(m) for m in (*args[:2],model,carrier)]:
        raise ValueError('actual dose mismatch or frozen parameter mutation')
    write(out/'seal.json',dict(curves=files,prefix=measured,max_norm_error=norms,
                             max_static_error=source_diff,seconds=time.monotonic()-begun))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    ref = read(history.ARCHIVE/'seal.json')
    old = read(history.ARCHIVE/'per_shot.json')
    source = Path(p['source'])
    if (len(s['curves'])!=360 or p['source_seal']!=sha(source/'seal.json')
            or p['archive_seal']!=sha(history.ARCHIVE/'seal.json')):
        raise ValueError('complete fixed stream required')
    first = read(source/'per_shot.json')
    rows = {c:{k:[] for k in ARMS} for c in p['calibration']}
    fit = {n:{k:[] for k in ('all','critical')} for n in (RAW,USES)}
    covered = np.zeros(3,np.int64)
    dose_support = np.zeros(2,np.int64)
    dose_values = []
    measured = []
    tau = p['calibration']['0.2']
    for rank,sid in enumerate(p['order'],1):
        f = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=s['curves'][f.name]:
            raise ValueError('changed candidate curve')
        d,down,t = motion.group.hdf._raw_truth(sid)
        arc = history.archived_shot(rank,sid,ref)
        with np.load(f,allow_pickle=False) as a:
            view = dict(a)
        affected = view['dose']>0.
        dose_support += affected.sum(1)
        if (np.abs(view['dose']-arc['dose'])*affected>1e-6).any():
            raise ValueError('new pair outside fixed old budget')
        for j,k in enumerate(DOSE):
            np.testing.assert_array_equal(view[k][~affected[j//2]],view['source_representation'][~affected[j//2]])
        for c,cut in p['calibration'].items():
            if rank<=p['startup']:
                cut = p['original']
            for k in ARMS:
                v = dict(shot=sid,disruptive=d,down_s=down,
                         **score_curve(t,(arc if k in ARMS[:3] else view)[k],d,down,cut))
                if k in ARMS[:3] and v!=old[c][k][rank-1]:
                    raise ValueError('archived parent outcome differs on original HDF5')
                if k=='source_representation' and v!=first[c][k][rank-1]:
                    raise ValueError('source carrier outcome changed')
                rows[c][k].append(v)
        if rank<=p['startup']:
            continue
        m = view['observed']
        b = view['legal']
        observed = arc['release']+history.WAIT
        k = np.searchsorted(observed,np.arange(len(t)),side='right')
        if (affected & ~np.r_[False,b][k]).any():
            raise ValueError('update outside allowed feedback support')
        covered += (int(m.sum()),int(b.sum()),int(np.count_nonzero(m & (view['prior']>0))))
        dose_values.append(view['dose'][:,affected.all(0)])
        checked = np.flatnonzero(b & (observed+motion.PATCH<len(t)))
        remaining = (np.round((down-t[observed[checked]+motion.PATCH])*1e3,9)
                     if d else np.full(len(checked),np.nan))
        late = d and not old['0.2']['current_5'][rank-1]['timely']
        critical = late & (remaining>30.) & (remaining<=125.)
        for j,k0 in enumerate((RAW,USES)):
            fit[k0]['all'].append(view['forecast'][j,m])
            fit[k0]['critical'].append(view['forecast'][j,checked[critical]])
        if late:
            ld = np.round((down-t)*1e3,9)
            at = np.flatnonzero((ld>30.) & (ld<=125.) & affected.all(0))
            if len(at):
                parent = logit(view['source_representation'][at].clip(1e-6,1-1e-6))
                child = {k:logit(view[k][at].clip(1e-6,1-1e-6)) for k in DOSE}
                measured.append(dict(shot=sid,points=len(at),gap=float(logit(tau)-parent.max()),
                    lifts={k:float((v-parent).max()) for k,v in child.items()},
                    alarm={k:bool((v>=logit(tau)).any()) for k,v in child.items()}))
    metrics = {c:{k:lead.summarize(v) for k,v in r.items()} for c,r in rows.items()}
    pairs = {c:{f'{a}__to__{b}':dict(flips=flips(r[a],r[b]),lead=lead.paired(r[a],r[b]))
        for a in ('bias_only','source_representation','current_5',DOSE[1],DOSE[3]) for b in DOSE if a!=b}
        for c,r in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,paired=pairs, raw_verified=360*len(ARMS)*len(p['calibration']),
        fit={k:{n:motion.forecast_summary(a) for n,a in v.items()} for k,v in fit.items()},
        feedback_counts=covered.tolist(), active_endpoints=dose_support.tolist(),
        dose_quantiles=np.quantile(np.concatenate(dose_values,axis=1),[.1,.5,.9],axis=1).tolist(),
        critical_endpoints=measured))
    for b,m in metrics.items():
        print(b,{k:(a['by_deadline']['30']['tp'],a['fp']) for k,a in m.items()},flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--score',action='store_true')
    a = p.parse_args()
    if a.score:
        score(a.output.resolve())
    else:
        run(a.source.resolve(),a.output.resolve())
