"""Prequential, outcome-free test of device- and shot-specific diagnostic dynamics."""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np

from . import commissioning_shared_motion_20261114 as motion
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, write


NAMES = ('hold', 'source', 'frozen_cold', 'diagonal', 'cross', 'recent_cross')
RIDGE, RECENT = 1., 8
PHASE = np.arange(1,motion.BIN+1)/motion.BIN
HORIZON = np.arange(1,motion.H+1)


def waveform(knots):
    precede = np.concatenate((np.zeros_like(knots[:,:1]),knots[:,:-1]),axis=1)
    return (precede[:,:,None]+(knots-precede)[:,:,None]*PHASE[None,None,:,None]).reshape(
        len(knots),motion.H,knots.shape[-1])


BASIS = waveform(np.eye(motion.K)[:,:,None]).squeeze(-1).T
PROJECT = np.linalg.pinv(BASIS)


def fit_knots(y):
    return PROJECT @ y


def direct(q, x, target, observed):
    """Estimate parameters from prior, fully landed issue/target pairs only."""
    n,c = x.shape
    s = np.zeros((3,n,motion.K,c))
    cumulative = np.eye(c)*RIDGE
    rhs = np.zeros((c,motion.K*c))
    hit = 0
    accepted = []
    flat = target.reshape(n,motion.K*c)
    for i in range(n):
        while hit < i and q[hit]+motion.OFFSET <= q[i]:
            if observed[hit]:
                a = x[hit]
                cumulative += np.outer(a,a)
                rhs += a[:,None]*flat[hit]
                accepted.append(hit)
            hit += 1
        if accepted:
            diagonal = rhs.reshape(c,motion.K,c)[np.arange(c),:,np.arange(c)]
            s[0,i] = ((x[i]/cumulative.diagonal())[:,None]*diagonal).T
            s[1,i] = (x[i]@np.linalg.solve(cumulative,rhs)).reshape(motion.K,c)
            j = accepted[-RECENT:]
            x0,y0 = x[j],flat[j]
            s[2,i] = (x[i]@np.linalg.solve(x0.T@x0+np.eye(c)*RIDGE,x0.T@y0)).reshape(motion.K,c)
    return s


def packets(signal,center,spread,scale,q):
    channels = list(motion.CHANNELS)
    m = signal['available'][:,channels]
    z = np.arcsinh((signal['values'][:,channels]-center)/spread)
    recent = q[:,None]+np.arange(-7,1)
    former = recent-8
    future = q[:,None]+HORIZON
    delta = ((z[recent].mean(1)-z[former].mean(1))/scale).clip(-8.,8.)
    y = (z[future]-z[q,None])/scale
    complete = m[recent].all((1,2)) & m[former].all((1,2)) & m[future].all((1,2))
    return delta,y,complete


def prep_fit(rows):
    """Cold-shot balanced linear dynamics fit; no outcomes are inputs."""
    x,y = (np.concatenate([v[k]/np.sqrt(len(v[k])) for v in rows],axis=0) for k in (0,1))
    return np.linalg.solve(x.T@x+np.eye(x.shape[1])*RIDGE,x.T@y.reshape(len(y),-1))


def metrics(last, y):
    """Keep target support and all normalizations identical in each arm."""
    point = np.abs(last-y[None]).mean(axis=(2,3))
    delta = np.diff(np.pad(last.reshape(len(last),len(y),motion.K,motion.BIN,-1).mean(3),
                           ((0,0),(0,0),(1,0),(0,0))),axis=2)
    target = np.diff(np.pad(y.reshape(len(y),motion.K,motion.BIN,-1).mean(2),
                            ((0,0),(1,0),(0,0))),axis=1)
    return np.stack((point,np.abs(delta-target[None]).mean(axis=(2,3))),axis=-1)


def audit(source,reference,out):
    from .redesign_20261004.prepare_prior import DEFAULT_PARENT,load_parent
    from . import causal_input
    import torch

    motion.lag.configure()
    base,_ = load_parent(DEFAULT_PARENT)
    base.cuda()
    model,_,scale = motion.inputs_from_fit(source,base)
    p = read(reference/'protocol.json')
    anchors = read(motion.history.ARCHIVE/'protocol.json')
    if any(p[k]!=anchors[k] for k in motion.history.LOCK) or len(p['order'])!=360:
        raise ValueError('exact fixed parent stream required')
    out.mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    xscale = [a[list(motion.CHANNELS)].cpu().numpy() for a in (base.center,base.scale)]
    params = (*xscale, scale.cpu().numpy())
    prepared, result, witness = [],{},{}
    cold_fit = None
    cohort = p['order']
    for rank,sid in enumerate(cohort,1):
        with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcomes in diagnostic fit')),
              patch.object(motion.group.hdf,'_raw_truth',side_effect=RuntimeError('no outcomes in diagnostic fit'))):
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        signal = causal_input.apply_input_contract(raw)
        n = len(signal['values'])
        q = np.arange(motion.WIDTH-1,n-motion.H,motion.lag.STRIDE)
        x,y,ok = packets(signal,*params,q)
        knots = fit_knots(y)
        if rank<=p['startup']:
            prepared.append((x[ok],knots[ok]))
            if rank==p['startup']:
                cold_fit = prep_fit(prepared)
            continue
        if cold_fit is None:
            raise ValueError('cold prep must precede any evaluation')
        d = direct(q,x,knots,ok)
        fixed = (x@cold_fit).reshape(len(q),motion.K,motion.C)
        f = []
        with torch.no_grad():
            for i in range(0,len(q),motion.BATCH):
                a,m = motion.lag._windows(base,signal,q[i:i+motion.BATCH])
                f.append(motion.waveform(model(motion.input_view(base,a,m,scale))).cpu().numpy())
        source_wave = np.concatenate(f)
        pred = np.concatenate((np.zeros_like(y)[None],source_wave[None],waveform(fixed)[None],
                               np.stack([waveform(v) for v in d])))
        error = metrics(pred,y)
        support = np.r_[0,ok.astype(int).cumsum()][np.searchsorted(q+motion.OFFSET,q,side='right')]
        past = np.r_[-1,np.maximum.accumulate(np.where(ok,np.arange(len(q)),-1))][
            np.searchsorted(q+motion.OFFSET,q,side='right')]
        # Selection for metrics only. At issuance not knowing coverage of the
        # target window must never filter which previous pair can be learned.
        if rank in (21,121,360):
            cutoff = min(n,motion.WIDTH+8*motion.lag.STRIDE)
            prefix_signal = {k:signal[k][:cutoff] for k in ('values','available')}
            early = q+motion.H<cutoff
            x0,y0,a0 = packets(prefix_signal,*params,q[early])
            d0 = direct(q[early],x0,fit_knots(y0),a0)
            np.testing.assert_allclose(d[:,early],d0,rtol=0,atol=1e-12)
            witness[sid] = dict(length=cutoff, comparisons=int(early.sum()))
        result[sid] = dict(rank=rank, q=q[ok].tolist(), support=support[ok].tolist(),
                           last_issue=np.where(past>=0,q[past],-1)[ok].tolist(), error=error[:,ok].tolist())
        if rank in (60,120,180,240,300,360):
            print(dict(stage='signal_predictions_finished',rank=rank),flush=True)
    write(out/'forecast_sealed.json',dict(order=cohort,startup=p['startup'],source=str(source),
        reference=str(reference),names=NAMES, q_offset=motion.OFFSET, basis=BASIS.tolist(),
        cold_windows=[len(z[0]) for z in prepared], cold_matrix=cold_fit.tolist(),
        witness=witness, shots=result))


def summarize(rows):
    z = [v for v in rows if len(v)]
    if not z:
        return dict(shots=0, windows=0, average=None)
    a = np.stack([v.mean(axis=0) for v in z])
    return dict(shots=len(z),windows=sum(map(len,z)),
                average={k:a[:,i].mean(0).tolist() for i,k in enumerate(NAMES)},
                better_than_hold={k:(a[:,i,1]<a[:,0,1]).sum().item() for i,k in enumerate(NAMES[1:],1)})


def analyze(out):
    p = read(out/'forecast_sealed.json')
    base = read(Path(p['reference'])/'per_shot.json')['0.2']['current_5']
    theta = read(motion.history.ARCHIVE/'protocol.json')['calibration']['0.2']
    groups = {k:[] for k in ('all_complete','own_1_to_7','own_8_or_more','timely_risk_window',
                              'near_late','near_unwarned','false_alarm_before_alarm')}
    if len(p['shots'])!=340 or p['order']!=[v['shot'] for v in base]:
        raise ValueError('diagnostic predictions must be sealed for same full stream')
    for s in base[p['startup']:]:
        sid = s['shot']
        v = p['shots'][str(sid)]
        d,down,clock = motion.group.hdf._raw_truth(sid)
        if (d,down)!=(s['disruptive'],s['down_s']):
            raise ValueError('scoring must use original HDF5 only')
        q = np.asarray(v['q'],dtype=int)
        n = np.asarray(v['support'])
        pred = np.array(v['error']).transpose(1,0,2)
        i = q+motion.READY
        after = (i<len(clock)) & (n>0)
        t = clock[i.clip(max=len(clock)-1)]
        groups['all_complete'].append(pred)
        groups['own_1_to_7'].append(pred[(n>0)&(n<RECENT)])
        groups['own_8_or_more'].append(pred[n>=RECENT])
        if d:
            ahead = np.round((down-t)*1e3,9)
            use = after & (ahead>30.) & (ahead<=125.)
            if s['timely']:
                groups['timely_risk_window'].append(pred[use])
            else:
                groups['near_late' if s['alarm'] else 'near_unwarned'].append(pred[use])
        elif s['alarm']:
            # First false-alarm time is recovered from the already recorded
            # curve because the operational safe-shot record has no lead time.
            key = f"{v['rank']:04d}_{sid}.npz"
            with np.load(motion.history.ARCHIVE/'shots'/key,allow_pickle=False) as file:
                alarm = np.flatnonzero(file['current_5']>=theta)[0]
            groups['false_alarm_before_alarm'].append(pred[after & (i<=alarm)])
    result = {k:summarize(rows) for k,rows in groups.items()}
    write(out/'result.json',dict(metrics=result,columns=('waveform_mae','bin_increment_mae'),
        mode='No EAST outcomes accessed until signal predictions were sealed; near alarm uses q+39 handoff',
        whole_stream_shots=len(base),scored_diagnostic_shots=len(p['shots'])))
    for k,row in result.items():
        print(k,row,flush=True)


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('source',type=Path)
    a.add_argument('reference',type=Path)
    a.add_argument('output',type=Path)
    ns = a.parse_args()
    audit(ns.source.resolve(),ns.reference.resolve(),ns.output.resolve())
    analyze(ns.output.resolve())
