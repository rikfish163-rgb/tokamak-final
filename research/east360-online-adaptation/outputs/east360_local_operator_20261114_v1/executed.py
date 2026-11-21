"""Can a measured local transition help forecast an unused EAST diagnostic?"""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, write
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


A = tuple(io23.RAW23.index(k) for k in ('n=1 amplitude','dy_a'))
B = tuple(io23.RAW23.index(k) for k in ('v_loop','sxr_var'))
NAMES = ('hold','own_trend','context','cold_operator','observed_operator','earlier_operator','diagonal_operator')
WIDTH, STEP, STRIDE, H = 128,8,32,25
RELEASE, HANDOFF = 31,39
RIDGE, MIN_RANK_FRACTION = .1,.02


def past_terms(a,mask,q):
    windows = q[:,None]+np.arange(1-WIDTH,1)
    x = a[windows].reshape(-1,WIDTH//STEP,STEP,len(A)).mean(2)
    x -= x.mean(1,keepdims=True)
    return x[:,:-1],x[:,1:],mask[windows].all((1,2))


def fit_operator(x,y,*,ridge=RIDGE,diagonal=False):
    g,h = x.transpose(0,2,1)@x,x.transpose(0,2,1)@y
    trace = np.trace(g,axis1=1,axis2=2)
    # Scale the regularizer with the actual measured excitation; zero
    # excitation is separately marked unavailable by rank_fraction.
    regularizer = ridge*np.maximum(trace,np.finfo(x.dtype).eps)/g.shape[-1]
    system = g+regularizer[:,None,None]*np.eye(g.shape[-1])
    if diagonal:
        return np.eye(g.shape[-1])*(h.diagonal(axis1=1,axis2=2)/
                                    system.diagonal(axis1=1,axis2=2))[:,None]
    return np.linalg.solve(system,h)


def rank_fraction(x):
    g = x.transpose(0,2,1)@x
    v = np.linalg.eigvalsh(g)
    return np.maximum(v[:,0],0.)/v.sum(1).clip(min=1e-10)


def earlier(q,usable):
    p = np.maximum.accumulate(np.r_[-1,np.where(usable,np.arange(len(q)),-1)])
    use = p[np.searchsorted(q+RELEASE,q,side='right')]
    return use,usable & (use>=0)


def modes(op):
    return abs(np.linalg.eigvals(op)).max(1)


def design(a,x):
    # Three powers are applied to the *observed* history state at the issue
    # time. The readout emits an entire H25 target in one call, not a rollout
    # driven by B predictions or extra realized A measurements.
    x1 = np.einsum('ni,nij->nj',x,a)
    x2 = np.einsum('ni,nij->nj',x,a@a)
    x3 = np.einsum('ni,nij->nj',x,a@a@a)
    rel = np.stack((x1-x,x2-x,x3-x),axis=1)
    return np.tanh(rel).reshape(len(x),-1)


def parts(sig,center,scale):
    z = np.arcsinh((sig['values']-center)/scale)
    q = np.arange(WIDTH-1,len(z)-H,STRIDE)
    a,b = z[:,list(A)],z[:,list(B)]
    am,bm = (sig['available'][:,list(k)] for k in (A,B))
    x,y,raw = past_terms(a,am,q)
    fraction = rank_fraction(x)
    allowed = raw & (fraction>=MIN_RANK_FRACTION)
    idx,okay = earlier(q,allowed)
    xa,xb = (z[q[:,None]+np.arange(-15,1)][...,list(k)].reshape(len(q),2,STEP,2)
             .mean(2) for k in (A,B))
    ctx,cv = probe.operating(sig,q)
    recent_b = bm[q[:,None]+np.arange(-15,1)].all((1,2))
    known = okay & recent_b & cv
    # Disjoint 64-point halves fitted in their own local coordinates.
    near0,near1 = (v-v.mean(1,keepdims=True) for v in (x[:,:8],y[:,7:]))
    first = fit_operator(near0[:,:-1],near0[:,1:])
    last = fit_operator(near1[:,:-1],near1[:,1:])
    split_ok = (rank_fraction(near0[:,:-1])>=MIN_RANK_FRACTION)&(
        rank_fraction(near1[:,:-1])>=MIN_RANK_FRACTION)
    context = np.concatenate((ctx,np.tanh(xa[:,1]),np.tanh(xa[:,1]-xa[:,0]),
                              np.tanh(xb[:,1]),np.tanh(xb[:,1]-xb[:,0])),axis=1)
    future = q[:,None]+np.arange(1,H+1)
    d = b[future]-b[q,None]
    target_ok = bm[future].all((1,2))
    op = fit_operator(x,y)
    diag = fit_operator(x,y,diagonal=True)
    return dict(q=q,signal_support=allowed,target_valid=target_ok,
                observed=known & target_ok,op=op,diag=diag,older=idx,x=x,y=y,
                current=y[:,-1],base=context,target=d,trend=xb[:,1]-xb[:,0],
                split_ok=split_ok,split_diff=abs(modes(first)-modes(last)),
                split_matrix=np.linalg.norm(first-last,axis=(1,2))/
                    ((np.linalg.norm(first,axis=(1,2))+np.linalg.norm(last,axis=(1,2)))*.5+1e-8))


def frozen(rows):
    g,h = [],[]
    for r in rows:
        x,y = (r[k][r['signal_support']] for k in ('x','y'))
        if len(x):
            g.append((x.transpose(0,2,1)@x).mean(0))
            h.append((x.transpose(0,2,1)@y).mean(0))
    v = np.mean(g,axis=0)
    return np.linalg.solve(v+(RIDGE*np.trace(v)/len(A))*np.eye(len(A)),np.mean(h,axis=0))


def measurement(r,fixed):
    m = r['observed']
    op = r['op']
    bank = (np.broadcast_to(fixed,op.shape),op,op[r['older'].clip(min=0)],r['diag'])
    return [np.c_[r['base'][m],design(x[m],r['current'][m])] for x in bank]


def ridge_fit(data,target,w):
    axis = w@data
    spread = np.sqrt((w@np.square(data-axis)).clip(min=.01**2))
    g = (data-axis)/spread
    x = np.c_[g,np.ones(len(g))]
    gram = x.T@(w[:,None]*x)
    # Do not shrink the intercept of the same readout.
    penalty = .1*np.eye(x.shape[1])
    penalty[-1,-1] = 0.
    coef = np.linalg.solve(gram+penalty,x.T@(w[:,None]*target))
    return axis,spread,coef


def fit_cold(rows,fixed):
    # The OP readout learns on all three causal orientation controls and the
    # diagonal comparator with equal weight. No held-out arm is favored.
    t,d,p,w,targets,mass = [],[],[],[],[],[]
    for r in rows:
        m = r['observed']
        if m.any():
            x = measurement(r,fixed)
            n = int(m.sum())
            y = r['target'][m].reshape(n,-1)
            p.append(np.concatenate(x))
            d.append(r['base'][m])
            t.append(y)
            w.append(np.full(n,1./n))
            targets.append(np.tile(y,(len(x),1)))
            mass.append(np.full(n*len(x),1./(n*len(x))))
    return (ridge_fit(np.concatenate(d),np.concatenate(t),np.concatenate(w)/len(w)),
            ridge_fit(np.concatenate(p),np.concatenate(targets),np.concatenate(mass)/len(mass)))


def output(x,model):
    a,s,w = model
    return (np.c_[(x-a)/s,np.ones(len(x))]@w).reshape(len(x),H,len(B))


def audit(out):
    protocol = read(probe.history.ARCHIVE/'protocol.json')
    frozen_base,_ = load_parent(DEFAULT_PARENT)
    center,scale = frozen_base.center.numpy().astype(float),frozen_base.scale.numpy().astype(float)
    lock = probe.lag.load_inputs()
    if protocol['order']!=lock['ids'] or len(lock['ids'])!=360:
        raise ValueError('same fixed causal EAST360 required')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    cold,completed,prefix = [],[],{}
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes unavailable')),
          patch.object(probe.group.hdf,'_raw_truth',side_effect=RuntimeError('outcomes unavailable'))):
        for rank,sid in enumerate(protocol['order'],1):
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            x = probe.lag.prepare_actual(raw,protocol['startup']+1,lock['mapping'])
            r = parts(x,center,scale)
            if rank<=protocol['startup']:
                cold.append(r)
                if rank==protocol['startup']:
                    f = frozen(cold)
                    models = fit_cold(cold,f)
                    write(out/'cold_fit.json',dict(matrix=f.tolist(),windows=[int(v['observed'].sum()) for v in cold],
                        weights=[[u.tolist() for u in v] for v in models], outcomes_used=False,
                        coordinates='completed first 20 waveform recordings in locked postcold map'))
                continue
            m,q = r['observed'],r['q']
            t = r['target'][m]
            trend = r['trend'][m,None,:]*(np.arange(1,H+1)[:,None]/STEP)
            forecasts = np.stack((np.zeros_like(t),trend,output(r['base'][m],models[0]),
                                  *(output(z,models[1]) for z in measurement(r,f))))
            error = np.abs(forecasts-t).mean(2).transpose(1,0,2)
            if rank in (21,120,360):
                h = min(512,len(x['values']))
                shortened = {k:x[k][:h] for k in ('values','available')}
                rr = parts(shortened,center,scale)
                sl = slice(0,len(rr['q']))
                np.testing.assert_array_equal(rr['q'],q[sl])
                for k in ('op','observed','older','base','current'):
                    np.testing.assert_allclose(rr[k],r[k][sl],atol=1e-11,rtol=0.)
                prefix[str(sid)] = dict(samples=h,observations=int(rr['observed'].sum()))
            filename = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/filename).open('xb') as stream:
                np.savez_compressed(stream,q=q,observed=m,error=error,operator=modes(r['op'])[m],
                                   enough=r['signal_support'],split_ok=r['split_ok'][m],
                                   split_matrix=r['split_matrix'][m],split_diff=r['split_diff'][m])
            completed.append(filename)
            if rank%60==0:
                print('sealed',rank,flush=True)
    write(out/'seal.json',dict(order=protocol['order'],preparation=protocol['startup'],completed=completed,
        readout='only the 20 earlier ended waveform recordings, no disruption/DownTime label',
        choices=NAMES,rank_fraction=MIN_RANK_FRACTION,ridge=RIDGE,
        a=[io23.RAW23[k] for k in A],b=[io23.RAW23[k] for k in B],prefix=prefix))


def statistics(rows):
    a = [r for r in rows if len(r)]
    if not a:
        return dict(shots=0,windows=0,mae=None)
    errors = np.stack([v.mean(0) for v in a])
    return dict(shots=len(a), windows=sum(map(len,a)),
                mae=dict(zip(NAMES,errors.mean(0).tolist(),strict=True)),
                better_than_hold=dict(zip(NAMES,(errors<errors[:,:1]).sum(0).tolist(),strict=True)))


def diagnose(out):
    p = read(out/'seal.json')
    n = p['preparation']
    old = read(probe.history.ARCHIVE/'per_shot.json')['0.2']['current_5']
    seal = read(probe.history.ARCHIVE/'seal.json')
    cut = read(probe.history.ARCHIVE/'protocol.json')['calibration']['0.2']
    order = p['order']
    files = [f'{i:04d}_{s}.npz' for i,s in enumerate(order,1)]
    if [r['shot'] for r in old]!=order or files[n:]!=p['completed'] or len(files)!=360:
        raise ValueError('scoring requires complete sealed reference information')
    by = {k:[] for k in ('all','near_never','near_late','near_timely','before_false')}
    cases,rates,motion = {},[],[]
    for rank in range(n,len(order)):
        s = old[rank]
        sid = order[rank]
        d,down,t = probe.group.hdf._raw_truth(sid)
        if (d,down)!=(s['disruptive'],s['down_s']):
            raise ValueError('wrong reference outcome')
        with np.load(out/'shots'/files[rank],allow_pickle=False) as f:
            row = dict(f)
        q,m = row['q'],row['observed']
        np.testing.assert_array_equal(q,np.arange(WIDTH-1,len(t)-H,STRIDE))
        a = row['error']
        if len(a):
            r = row['split_ok']
            rates.append(row['split_diff'][r])
            motion.append(row['split_matrix'][r])
        by['all'].append(a)
        time = q+HANDOFF
        usable = time<len(t)
        if d:
            lead = np.round((down-t[time.clip(max=len(t)-1)])*1e3,9)
            at = usable & (lead>30.) & (lead<=125.)
            group = 'near_timely' if s['timely'] else 'near_late' if s['alarm'] else 'near_never'
        elif s['alarm']:
            arc = probe.history.archived_shot(rank+1,sid,seal)
            alarm = np.flatnonzero(arc['current_5']>=cut)[0]
            at = usable & (time<=alarm)
            group = 'before_false'
        else:
            continue
        by[group].append(a[at[m]])
        cases[sid] = dict(group=group,possible=int(at.sum()),with_rank=int((at&row['enough']).sum()),
                          common=int((at&m).sum()))
    metrics = {k:statistics(r) for k,r in by.items()}
    all_diff = np.concatenate(rates)
    write(out/'result.json',dict(complete_stream=360,evaluated_after_preparation=360-n,
        meaning='H25 masked common-support direct signal prediction, no alarms generated',
        b=p['b'], metrics=metrics, cases=cases,split_valid_windows=len(all_diff),
        spectral_radius_abs_difference=np.quantile(all_diff,[.1,.5,.9]).tolist(),
        relative_matrix_difference=np.quantile(np.concatenate(motion),[.1,.5,.9]).tolist()))
    for k,v in metrics.items():
        print(k,v,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    audit(a.output.resolve())
    diagnose(a.output.resolve())
