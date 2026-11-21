"""Probe warning information in causal trajectories and old intra-shot states."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
import torch

from . import commissioning_history_span_20261114 as history
from . import io23, simple_data
from .commissioning_excursion_order_20261111 import disjoint_pairs
from .commissioning_roughness_20261110 import coordinates
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = history.group, history.lag, history.layer
precision = history.cross.precision
WIDTH, BIN, STEP, LATER = 128, 16, 4, 32
BATCH, RIDGE, CLIP = 128, .10, 5.
COLUMNS = tuple(lag.CHANNELS[c] for c in group.USE)
CONTEXT = tuple(io23.RAW23.index(k) for k in ('ip', 'Bt'))
NAMES = ('operating_only', 'raw_trajectory', 'patch_projection', 'risk_state')
CASES = (70320, 73405, 74970, 79649, 71076, 85422)
CALIPERS = np.array([.50,.35,.35,.50,.35,.25])


def trajectory(z, available, endpoint):
    """Eight measured means per channel: window level plus seven signed differences."""
    index = endpoint[:,None]+np.arange(1-WIDTH,1)
    ok = available[index].all((1,2))
    mean = z[index].reshape(len(index),WIDTH//BIN,BIN,z.shape[1]).mean(2)
    design = np.concatenate((mean.mean(1,keepdims=True),np.diff(mean,axis=1)),axis=1)
    return design.transpose(0,2,1).reshape(len(index),-1).astype(np.float32),ok


def operating(signal, ends):
    z,m = signal['values'][:,CONTEXT], signal['available'][:,CONTEXT]
    now,c0 = _window_mean(z,m,ends,-15,16)
    old,c1 = _window_mean(z,m,ends,-31,16)
    x = np.arcsinh(np.c_[abs(now[:,0])/1e5,abs(now[:,1]),(abs(now[:,0])-abs(old[:,0]))/1e4])
    return x,((c0>=8) & (c1>=8)).all(1)


def later_rows(ends, chosen):
    """Retrospective +32-sample control; callers never treat it as live evidence."""
    seek = ends[chosen]+LATER
    place = np.searchsorted(ends,seek)
    idx = np.flatnonzero(place<len(ends))
    idx = idx[ends[place[idx]]==seek[idx]]
    return place[idx],idx


def training_weights(shot, target):
    w = np.zeros(len(shot))
    for label in (False,True):
        j = target==label
        _,owner,count = np.unique(shot[j],return_inverse=True,return_counts=True)
        w[j] = .5/(len(count)*count[owner])
    return w


def fit_probe(context, features, shot, target):
    """Balance the two endpoint classes, then give each contributing shot equal mass."""
    design = np.c_[context,features].astype(np.float64)
    mass = training_weights(shot,target)
    center = mass@design
    spread = np.sqrt(np.maximum(mass@np.square(design-center), .05**2))
    x = np.c_[np.clip((design-center)/spread,-CLIP,CLIP),np.ones(len(design))]
    r = np.r_[np.full(x.shape[1]-1,RIDGE),0.]

    def loss(w):
        z = x@w
        return (mass@(np.logaddexp(0.,z)-target*z)+.5*(r*w)@w,
                x.T@(mass*(expit(z)-target))+r*w)

    f = minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(ftol=1e-12,gtol=1e-8))
    if not f.success or not np.isfinite(f.x).all():
        raise ValueError('ridge logistic probe did not converge')
    return dict(center=center.tolist(),spread=spread.tolist(),coef=f.x.tolist())


def apply_probe(context, features, model):
    x = np.c_[context,features]
    z = np.clip((x-np.asarray(model['center']))/np.asarray(model['spread']),-CLIP,CLIP)
    w = np.asarray(model['coef'])
    return z@w[:-1]+w[-1]


@torch.no_grad()
def capture(base, x, m, *, transform=None, writes=None, handoff=None, budget=None):
    """Get all four CLS views from the same encoder call, and the untouched patch projection."""
    seen=[]
    count=len(x)
    hook=base.embedding.register_forward_hook(lambda _m,_a,y:seen.append(y[:count].detach()))
    try:
        if writes is None:
            blank=x.new_zeros((count,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
            hidden=group.forward_hidden(base,x,m,transform,state=blank)[None]
            dose=x.new_zeros(count)
            error=0.
        else:
            hidden,dose,error=group.forward_hidden(base,x,m,transform,writes=writes,handoff=handoff,
                                                  target_norm=budget)
    finally:
        hook.remove()
    # Each 16-sample bin contains two original patch embeddings. The linear
    # projection is summarized without inspecting attention or a label.
    patch=seen[0].reshape(count,WIDTH//BIN,-1,base.cfg.dim).mean(2)
    levels=torch.cat((patch.mean(1,keepdim=True),patch[:,1:]-patch[:,:-1]),1)
    return levels.flatten(1).cpu().numpy(),hidden,dose,float(error)


def views(signal, base, endpoints, track=None, archive=None):
    """Same-prefix diagnostic views. The additional traces never enter prediction."""
    center,scale=(v.cpu().numpy() for v in (base.center,base.scale))
    raw,m=coordinates(signal['values'][:,COLUMNS],signal['available'][:,COLUMNS],
                     center[list(COLUMNS)],scale[list(COLUMNS)])
    v,ok=trajectory(raw,m,endpoints)
    work,valid=operating(signal,endpoints)
    patch,states,base_prob=[],[],[]
    err=np.zeros(3)
    for start in range(0,len(endpoints),BATCH):
        sl=slice(start,start+BATCH)
        t=endpoints[sl]
        x,mask=lag._windows(base,signal,t)
        if track is None:
            p,h,_,_=capture(base,x,mask,transform=lambda v:v)
            z=base.risk_head(h[0]).flatten().sigmoid().detach().cpu().numpy()[None]
        else:
            q,origin_logits,base_hidden,beta,phase,correction,writes=track
            issued=lag.delayed_slots(q,t)[0]
            matured=q+lag.H+1+history.WAIT
            using=np.searchsorted(matured,t,side='right')
            a=group.handoff_fraction(np.r_[-1,matured][using],t,base.cfg.patch)
            cap=x.new_tensor(archive['dose'][t])
            p,h,d,e=capture(base,x,mask,transform=layer.patcher(phase[issued],correction[issued]),
                writes=writes[:,using],handoff=(writes[:,np.maximum(using-1,0)],x.new_tensor(a)),budget=cap)
            z=(origin_logits[t]+((h-base_hidden[t])*beta).sum(-1)).sigmoid().cpu().numpy()
            err[0]=max(err[0],e)
            err[1]=max(err[1],float((d-cap).abs().max()))
            arc=np.stack([archive[k][t] for k in ('bias_only','current_5','earlier_5')])
            err[2]=max(err[2],float(abs(z-arc).max()))
        patch.append(p)
        states.append(h.transpose(0,1).cpu().numpy())
        base_prob.append(z.T)
    width=base.cfg.dim
    packet=dict(ends=endpoints, operating=work.astype(np.float32),raw=v,valid=ok & valid,
        excitation=np.tanh(np.sqrt(np.square(v.reshape(-1,8,8)[...,1:]).mean((1,2)))),
        patch=np.concatenate(patch) if patch else np.empty((0,8*width),np.float32),
        states=np.concatenate(states) if states else np.empty((0,1 if track is None else 3,width),np.float32),
        base=np.concatenate(base_prob) if base_prob else np.empty((0,1 if track is None else 3),np.float32))
    return packet,err


def inherited_track(signal,args,decoder,path,teacher,archive):
    q,h,z,beta,phase_prior,phase,correction=group.parent_track(signal,args,decoder,path)
    ok,fields,waveform=group.feedback_waveform(signal,q,args[2],z.device)
    writes,common=group._estimate_innovation(args[0],signal,q,ok,phase,correction,phase_prior,
                                            fields,waveform,decoder,teacher)
    np.testing.assert_array_equal(q+lag.H+1,archive['release'])
    np.testing.assert_array_equal(common,archive['eligible'])
    return q,z,h,beta,phase,correction,writes[-2:]


def end_grid(n):
    return np.arange(WIDTH-1,n,STEP,dtype=np.int64)


def empty_context(row):
    return np.empty((len(row),0),np.float32)


def source_fit(base):
    ids=simple_data.source_roles()['train']
    if len(ids)!=1353:
        raise ValueError('fitting must use fixed source TRAIN1353')
    rows=[]
    advanced=io23.advanced_times()
    for sid in ids:
        raw=simple_data.read_shot('J-TEXT',sid)
        truth=simple_data.attach_outcome(raw,'J-TEXT',advanced)
        t=end_grid(len(raw['times']))
        # The training task compares only source-legal timely positive
        # endpoints and endpoints from genuinely safe source discharges.
        if truth['is_disrupt']:
            positive=truth['y'][t].astype(bool) & (np.round((truth['down']-raw['times'][t])*1000.,9)>10.)
            t=t[positive]
        else:
            # Fixed source-only subsampling limits repeated within-shot points.
            t=t[::32//STEP]
        if len(t):
            state,_=views(raw,base,t)
            k=state['valid']
            if k.any():
                rows.append(dict(**{key:a[k] for key,a in state.items()},
                    shot=np.full(k.sum(),sid),positive=np.full(k.sum(),truth['is_disrupt'],bool)))
    data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    x,shot,y=data['operating'],data['shot'],data['positive']
    features=(empty_context(x),data['raw'],data['patch'],data['states'][:,0])
    result={k:fit_probe(x,a,shot,y) for k,a in zip(NAMES,features,strict=True)}
    summary=dict(ids=ids,usable=train_counts(shot,y),weights='equal within each class by shot',
                 positive_selection='source positive window and lead strictly >10 ms')
    return result,summary


def train_counts(shot,y):
    return {name:dict(shots=len(np.unique(shot[y==label])),points=int((y==label).sum()))
            for label,name in ((False,'safe'),(True,'precursor'))}


def projected(row, probes):
    x=row['operating']
    # The same source-trained probe reads frozen, currently adapted, and
    # equally dosed earlier-feedback states; no separate fits to gain an edge.
    result=[apply_probe(x,v,probes[k]) for k,v in zip(NAMES[:3],
        (empty_context(x),row['raw'],row['patch']),strict=True)]
    result.extend(apply_probe(x,h,probes['risk_state']) for h in row['states'].transpose(1,0,2))
    return np.column_stack(result).astype(np.float32)


VIEWS=(*NAMES,'risk_state_current','risk_state_earlier')


def check_thresholds(actual, archived, thresholds):
    np.testing.assert_allclose(actual,archived,atol=3e-6,rtol=0.)
    for tau in thresholds:
        np.testing.assert_array_equal(actual>=tau,archived>=tau)


def observe(out):
    begun=time.monotonic()
    lock,p,seal,args,decoder,path=history.inputs()
    base=args[0]
    models=[state_digest(k) for k in args[:2]]
    probes,fit=source_fit(base)
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'fit.json',dict(probes=probes,**fit))
    write(out/'protocol.json',dict(scope='EAST360_CAUSAL_WEAK_SIGNAL_INFORMATION_PROBE',
        **{k:p[k] for k in history.LOCK}, reference=str(history.ARCHIVE),
        reference_seal=sha(history.ARCHIVE/'seal.json'), views=VIEWS,folds='J-TEXT TRAIN fits; VAL held',
        endpoint=dict(window=WIDTH,bin=BIN,step=STEP,late_witness=LATER),
        selection='same eight measured channels for raw changes with complete support; '
                  'base projection and final CLS from original full-23 model',
        limits='internal probes have no influence on risk, threshold, or inter-shot decisions'))
    teaching=precision.delay.HISTORY/'teaching.json'
    if sha(teaching)!=read(teaching.parent/'protocol.json')['teaching_sha256']:
        raise ValueError('frozen teacher changed')
    teacher=torch.as_tensor(read(teaching)['weights'],device=base.center.device,dtype=torch.float32)
    traces,maxima,witnesses={},np.zeros(3),{}
    thresholds=tuple(p['calibration'].values())
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('probe observations cannot access outcomes')):
        values=[]
        for sid in simple_data.source_roles()['val']:
            raw=simple_data.read_shot('J-TEXT',sid)
            row,_=views(raw,base,end_grid(len(raw['times'])))
            values.append(dict(shot=np.full(len(row['ends']),sid),ends=row['ends'],
                               valid=row['valid'],predict=projected(row,probes),base=row['base'][:,0]))
        with (out/'val.npz').open('xb') as f:
            np.savez_compressed(f,**{k:np.concatenate([r[k] for r in values]) for k in values[0]})

        for rank,sid in enumerate(p['order'],1):
            arc=history.archived_shot(rank,sid,seal)
            raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
            if len(raw['times'])!=len(arc['bias_only']):
                raise ValueError('frozen signal clock no longer matches')
            t=end_grid(len(raw['times']))
            if rank<=p['startup']:
                # Preparation alarms are fixed, with no new probe/intra state.
                compact=dict(ends=np.empty(0,np.int64),valid=np.empty(0,bool),
                    score=np.empty((0,len(VIEWS)),np.float32),dose=np.empty(0,np.float32),
                    context=np.empty((0,len(CALIPERS)),np.float32))
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('frozen input file changed')
                signal=lag.prepare_actual(raw,rank,lock['mapping'])
                track=inherited_track(signal,args,decoder,path,teacher,arc)
                row,error=views(signal,base,t,track,arc)
                maxima=np.maximum(maxima,error)
                check_thresholds(row['base'].T,np.stack([arc[k][t]
                    for k in ('bias_only','current_5','earlier_5')]),thresholds)
                if error[0]>1e-5 or error[1]>1e-5:
                    raise ValueError(f'actual token budget differs: {sid}')
                live=projected(row,probes)
                if rank in (21,120,360) or sid in CASES:
                    n=min(len(raw['times']),620)
                    slice_sig={k:signal[k][:n] for k in ('values','available')}
                    mask=arc['release']<n
                    cut={k:arc[k][:n] for k in (*history.OLD,'dose')}
                    cut.update(release=arc['release'][mask],eligible=arc['eligible'][mask])
                    cut_track=inherited_track(slice_sig,args,decoder,path,teacher,cut)
                    short,_=views(slice_sig,base,t[t<n],cut_track,cut)
                    k=len(short['ends'])
                    checks=('raw','patch','states','operating')
                    delta={}
                    for field in checks:
                        np.testing.assert_allclose(short[field],row[field][:k],atol=2e-5,rtol=0.)
                        delta[field]=float(abs(short[field]-row[field][:k]).max())
                    new=projected(short,probes)
                    np.testing.assert_allclose(new,live[:k],atol=3e-5,rtol=0.)
                    np.testing.assert_array_equal(short['valid'],row['valid'][:k])
                    check_thresholds(short['base'].T,row['base'][:k].T,thresholds)
                    witnesses[str(sid)]=dict(points=n,errors=delta,max_probe_error=float(abs(new-live[:k]).max()))
                ctx=np.c_[logit(arc['bias_only'][t].clip(1e-6,1.-1e-6))-logit(p['calibration']['0.2']),
                          row['operating'],np.log1p((raw['times'][t]-raw['times'][0])*1000.),row['excitation']]
                compact=dict(ends=t,valid=row['valid'],score=live,dose=arc['dose'][t],context=ctx)
            key=f'{rank:04d}_{sid}.npz'
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f,**compact)
            traces[key]=sha(out/'shots'/key)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='sealed',rank=rank,seconds=round(time.monotonic()-begun,2)),flush=True)
    if models!=[state_digest(m) for m in args[:2]]:
        raise ValueError('persistent parent altered by observational probe')
    write(out/'seal.json',dict(curves=traces,source_val=sha(out/'val.npz'),fit=sha(out/'fit.json'),
        prefixes=witnesses,max_errors=maxima.tolist(),elapsed_s=time.monotonic()-begun))


def describe_difference(delta, names=VIEWS):
    return {name:dict(correct=int((x>0.).sum()),ties=int((x==0.).sum()),
                      median=float(np.median(x)) if len(x) else None,
                      pair_log_loss=float(np.logaddexp(0.,-x).mean()) if len(x) else None)
            for name,x in zip(names,delta.T,strict=True)}


def pair_report(rows, a, b):
    p,q=np.flatnonzero(a),np.flatnonzero(b)
    context,shot=rows['context'],rows['shot']
    matches=disjoint_pairs(context[p],context[q],shot[p],shot[q],CALIPERS)
    i,j=p[matches[:,0]],q[matches[:,1]]
    difference=rows['score'][i]-rows['score'][j]
    pairs=[]
    for m,n in zip(i,j,strict=True):
        pairs.append(dict(precursor_shot=int(shot[m]),safe_shot=int(shot[n]),precursor_index=int(rows['end'][m]),
                          safe_index=int(rows['end'][n]),lead_ms=float(rows['lead'][m]),
                          distance=abs(context[m]-context[n]).tolist()))
    result=dict(pairs=len(i),eligible_precursor_shots=len(np.unique(shot[p])),
        eligible_safe_shots=len(np.unique(shot[q])),
        scores=describe_difference(difference),
        same_fixed_risk=describe_difference((context[i,0]-context[j,0])[:,None],('strong_parent_risk',)),
        pair_rows=pairs)
    both=np.isfinite(rows['later'][i]).all(1)&np.isfinite(rows['later'][j]).all(1)
    # Exact same endpoints and pairs, but the next 32 samples are exposed.
    # This uses look-ahead intentionally for a diagnostic control only.
    result['paired_future_diagnostic']=dict(pairs=int(both.sum()),
        later=describe_difference(rows['later'][i[both]]-rows['later'][j[both]]),
        before=describe_difference(difference[both]),
        later_still_in_time_for_both=int((rows['still_timely'][i[both]] & rows['still_timely'][j[both]]).sum()))
    return result


def source_auc(out):
    with np.load(out/'val.npz',allow_pickle=False) as f:
        data=dict(f)
    advanced=io23.advanced_times()
    rows=[]
    for sid in simple_data.source_roles()['val']:
        y,down=io23.read_outcome('J-TEXT',sid)
        j=np.flatnonzero((data['shot']==sid)&data['valid'])
        if y:
            raw=simple_data.read_shot('J-TEXT',sid)
            ld=np.round((down-raw['times'][data['ends'][j]])*1000.,9)
            j=j[(ld>10.) & (ld<=io23.effective_horizon_ms(sid,advanced)[0])]
        if len(j):
            rows.append((y,np.c_[data['base'][j],data['predict'][j]].max(0)))
    label=np.asarray([r[0] for r in rows])
    ranks=rankdata(np.stack([r[1] for r in rows]),axis=0)
    k=int(label.sum())
    auc=(ranks[label].sum(0)-k*(k+1)/2)/(k*(len(label)-k))
    return dict(positive_shots=k,safe_shots=len(label)-k,
                descriptive_peak_auc=dict(zip(('source_risk',*NAMES),auc.tolist(),strict=True)))


def score(out):
    p,s,fit=[read(out/f'{k}.json') for k in ('protocol','seal','fit')]
    if (len(s['curves'])!=360 or p['reference_seal']!=sha(history.ARCHIVE/'seal.json')
            or sha(out/'val.npz')!=s['source_val'] or sha(out/'fit.json')!=s['fit']):
        raise ValueError('complete sealed diagnostic views required')
    old,archive=(read(history.ARCHIVE/f'{k}.json') for k in ('per_shot','seal'))
    rows=[]
    all_result={k:{'timely':0,'fp':0} for k in history.OLD}
    fails=dict(old_late=[],old_false=[],unseen_false=[])
    p0=p['calibration']['0.2']
    for rank,sid in enumerate(p['order'],1):
        name=f'{rank:04d}_{sid}.npz'
        if sha(out/'shots'/name)!=s['curves'][name]:
            raise ValueError('sealed diagnostic for this discharge changed')
        y,down,t=group.hdf._raw_truth(sid)
        arc=history.archived_shot(rank,sid,archive)
        tau=p['original'] if rank<=p['startup'] else p0
        for k,v in all_result.items():
            fixed=dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,arc[k],y,down,tau))
            if fixed!=old['0.2'][k][rank-1]:
                raise ValueError('HDF5 first alarm differs from original reference')
            v['timely']+=int(fixed['timely'])
            v['fp']+=int(fixed['alarm'] and not y)
        if rank<=p['startup']:
            continue
        reference=old['0.2']['current_5'][rank-1]
        late=y and not reference['timely']
        fp=not y and reference['alarm']
        if late:
            fails['old_late'].append(sid)
        if fp:
            fails['old_false'].append(sid)
        first=int(np.flatnonzero(arc['current_5']>=tau)[0]) if fp else len(t)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            end=f['ends']
            if not np.array_equal(end,end_grid(len(t))):
                raise ValueError('packed endpoints differ from recorded physical clock')
            score_logit,good=f['score'],f['valid'] & (f['dose']>0.)
            lead_ms=np.round((down-t[end])*1000.,9) if y else np.full(len(end),np.nan)
            if y:
                use=good & (lead_ms>30.) & (lead_ms<=125.)
            elif fp:
                use=good & (t[end]<=t[first]) & (t[end]>=t[first]-.125)
                if not use.any():
                    fails['unseen_false'].append(sid)
            else:
                use=good
            selected=np.flatnonzero(use)
            n=len(selected)
            after,receiver=later_rows(end,selected)
            visit=good[after] & ((lead_ms[after]>=-1e-6) if y else True)
            after,receiver=after[visit],receiver[visit]
            later=np.full((n,len(VIEWS)),np.nan,np.float32)
            from_delay=np.zeros(n,bool)
            later[receiver]=score_logit[after]
            from_delay[receiver]=(lead_ms[after]>30.) if y else end[after]<=first
            rows.append(dict(shot=np.full(n,sid),end=end[selected],context=f['context'][selected],
                lead=lead_ms[selected],score=score_logit[selected],later=later,still_timely=from_delay,
                y=np.full(n,y,bool),late=np.full(n,late,bool),fp=np.full(n,fp,bool)))
    data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    comparisons=dict(all=pair_report(data,data['y'],~data['y']),
                     difficult=pair_report(data,data['late'],data['fp']))
    result=dict(calipers=CALIPERS.tolist(),source_fit=fit['usable'],source_held=source_auc(out),
                full_reference=all_result,comparisons=comparisons,**fails)
    write(out/'result.json',result)
    for k,v in comparisons.items():
        print(k,{a:b for a,b in v.items() if a!='pair_rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--score',action='store_true')
    a=p.parse_args()
    (score if a.score else observe)(a.output.resolve())
