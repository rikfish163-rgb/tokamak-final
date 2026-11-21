"""Check independent diagnostic forecasting value without changing EAST alarms."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import io23, simple_data
from .causal_input import apply_input_contract
from .commissioning_embedding_fast_20261114 import OFFSET, PATCH, VALUES
from .commissioning_mnm_witness_20261114 import model as fit_ridge
from .commissioning_mnm_witness_20261114 import predict as linear
from .commissioning_split_learning_20261111 import FIRST, HORIZON as H, STRIDE
from .history_gradient_metric_20261102.run import read, write


REF = io23.ROOT/'outputs/east360_latent_teacher_20261114_v3_value_scope'
EXTRA = tuple(io23.RAW23.index(k) for k in ('Mir_VV','CIII'))
OP = tuple(io23.RAW23.index(k) for k in ('ip','Bt'))
# Held-out outputs are outside both tested extra fields and include an
# electrical, a positional and a radiative record; no mechanism labels.
OUT = tuple(io23.RAW23.index(k) for k in ('v_loop','dy_a','sxr_var'))
CORE = (*VALUES,*OP)
RELEASE = OFFSET+PATCH
NAMES = ('persistence','core','current','earlier','reversed','level')
LOCAL = ('shot_latest','shot_earlier','hold_with_shot_latest','hold_with_shot_earlier')
SHRINK = 6.


def bins(y):
    return y.reshape(*y.shape[:-2],5,5,y.shape[-1]).mean(axis=-2)


def waveform(x,valid,q,cols):
    idx = q[:,None]+np.arange(1,1+H)
    v = x[idx][:,:,cols]
    good = valid[idx][:,:,cols].all(1) & valid[q][:,cols]
    return bins(v-x[q,None][:,:,cols]),good


def make_issues(x,valid,risk,q,eligible,dose):
    """At issue use current and previous wholly observed diagnostic packets."""
    index = np.flatnonzero(eligible & (q+RELEASE<len(x)) & (np.arange(len(q))>0))
    at = q[index]+RELEASE
    index = index[(q[index]-q[index-1]==STRIDE) & (dose[at]>0.)]
    q,prev,r = q[index],q[index-1],q[index]+RELEASE
    new,ok = waveform(x,valid,q,EXTRA)
    old,old_ok = waveform(x,valid,prev,EXTRA)
    ix = r[:,None]+np.arange(-47,1)
    bank = x[ix][:,:,CORE].reshape(-1,3,16,len(CORE)).mean(2)
    valid_core = valid[ix][:,:,CORE].all((1,2)) & valid[r][:,OUT].all(1)
    base = np.c_[bank[:,-1], np.diff(bank,axis=1).reshape(len(bank),2*len(CORE)),
                 logit(risk[r].astype(float).clip(1e-6,1.-1e-6)),np.log1p(r)]
    return dict(q=q,release=r,core=base,extra=new.transpose(0,2,1),earlier=old.transpose(0,2,1),
                available=ok & old_ok & valid_core[:,None])


def targets(x,valid,r):
    seen = np.flatnonzero(r+H+1<len(x))
    y = np.zeros((len(r),5*len(OUT)))
    complete = np.zeros(len(r),dtype=bool)
    future, observed = waveform(x,valid,r[seen],OUT)
    y[seen] = future.reshape(len(seen),5*len(OUT))
    complete[seen] = observed.all(1)
    return y,complete


def fit_pair(x,extra,y,shots):
    reduced = fit_ridge(x,y,shots)
    conditional = fit_ridge(x,extra,shots)
    residual = extra-linear(x,conditional)
    larger = fit_ridge(np.c_[x,residual],y,shots)
    return reduced,conditional,larger


def eval_models(x,e,earlier,rev,reduced,conditional,expanded):
    expected = linear(x,conditional)
    level = np.broadcast_to(e.mean(1,keepdims=True),e.shape)
    full = [linear(np.c_[x,v-expected],expanded) for v in (e,earlier,rev,level)]
    return (linear(x,reduced),*full)


def local_update(current,earlier,estimate,estimate_earlier,y,r,usable,received,fit):
    """Ridge response of an independent output to one group, reset every shot."""
    gram = np.diag(fit['feature_energy'])*fit['local_prior']
    rhs = np.zeros((current.shape[1],y.shape[1]))
    correction = np.zeros_like(y)
    stale = np.zeros_like(y)
    pointer = 0
    for i,when in enumerate(r):
        while pointer<i and r[pointer]+H+1<=when:
            if usable[pointer] and received[pointer]:
                a = current[pointer]
                gram += a[:,None]*a
                rhs += a[:,None]*(y[pointer]-estimate[pointer])
            pointer += 1
        if pointer and usable[i]:
            step = np.linalg.solve(gram,rhs)
            correction[i] = current[i]@step
            stale[i] = earlier[i]@step
    return estimate+correction,estimate_earlier+stale


def observe(sid,rank,seal,p):
    raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    with np.load(REF/'shots'/seal['shots'][rank-1],allow_pickle=False) as f:
        risk = f['bias_only']
        if rank<=p['startup']:
            # Cold curves did not enable this write. The auxiliary fit
            # reconstructs the matching hardware sample grid from signals.
            q = np.arange(FIRST,len(raw['times'])-H-1,STRIDE)
            eligible = np.ones(len(q),bool)
            dose = np.ones(len(raw['times']))
        else:
            q,eligible,dose = (f[k] for k in ('q','eligible','dose'))
    # Single monotone transform per numerical field; no silent completion of
    # missing values and no statistics from a future part of this discharge.
    x = np.arcsinh(raw['values'].astype(np.float64))
    return x,raw['available'],risk,q,eligible,dose


def build_fit(data,adapt_response):
    fits = []
    for i in range(len(EXTRA)):
        arrays = []
        for sid,u,y,good in data:
            mask = u['available'][:,i]&good
            arrays.append((u['core'][mask],u['extra'][mask,i],y[mask],np.repeat(sid,mask.sum())))
        x,e,y,shots = map(np.concatenate,zip(*arrays,strict=True))
        if len(set(shots))<2:
            raise ValueError('extra target has insufficient preparation-shot support')
        models = fit_pair(x,e,y,shots)
        scale = np.sqrt(np.maximum(np.square(y.reshape(-1,5,len(OUT))).mean((0,1)),1e-10))
        record = dict(models=models,scale=scale,shots=np.unique(shots),packets=len(shots))
        if adapt_response:
            # Use the cold-shot residual scale, not any statistic of the
            # future portion of the active discharge.
            residual = e-linear(x,models[1])
            record.update(feature_energy=np.maximum(np.mean(residual**2,axis=0),1e-8),
                          local_prior=SHRINK)
        fits.append(record)
    return fits


def issue_and_verify(x,valid,risk,q,elig,dose,fits,adapt_response=False):
    u = make_issues(x,valid,risk,q,elig,dose)
    y,received = targets(x,valid,u['release'])
    out = []
    for j,fit in enumerate(fits):
        ok = u['available'][:,j]&received
        observe = u['available'][:,j]
        v = u['extra'][:,j]
        estimate = eval_models(u['core'],v,u['earlier'][:,j],v[:,::-1],*fit['models'])
        zero = np.zeros_like(y)
        suffix = ()
        if adapt_response:
            expected = linear(u['core'],fit['models'][1])
            a,b = v-expected,u['earlier'][:,j]-expected
            suffix = (*local_update(a,b,estimate[1],estimate[2],y,u['release'],observe,received,fit),
                      *local_update(a,b,zero,zero,y,u['release'],observe,received,fit))
        forecast = np.stack((zero,*estimate,*suffix),axis=1)[ok]
        err = abs(forecast-y[ok,None]).reshape(int(ok.sum()),forecast.shape[1],5,len(OUT))
        out.append(dict(q=u['q'][ok],release=u['release'][ok],
            ready=u['release'][ok]+H+1,mae=err.mean(2)/fit['scale']))
    return out


def collect(out,adapt_response=False):
    start = monotonic()
    p,s = (read(REF/f'{k}.json') for k in ('protocol','seal'))
    order = p['order']
    if p['startup']!=20 or len(order)!=360 or s['shots']!=[f'{i:04d}_{v}.npz' for i,v in enumerate(order,1)]:
        raise ValueError('only the original 360-shot sequence is supported')
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    pre,cut,sizes = [],{},[]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('truth after outcomes-free score seal only')):
        for r,sid in enumerate(order[:p['startup']],1):
            x,m,risk,q,e,d = observe(sid,r,s,p)
            issue = make_issues(x,m,risk,q,e,d)
            y,ok = targets(x,m,issue['release'])
            pre.append((sid,issue,y,ok))
        fits = build_fit(pre,adapt_response)
        model_def = [{k:(v.tolist() if k in ('scale','shots','feature_energy') else [[a.tolist() for a in m] for m in v]
                         if k=='models' else v) for k,v in g.items()} for g in fits]
        write(out/'protocol.json',dict(order=order,startup=p['startup'],models=model_def,
            extras=[io23.RAW23[i] for i in EXTRA],target=[io23.RAW23[i] for i in OUT],
            cohort_note='no prediction is changed; post-seal HDF5 only determines analysis strata',
            controls=(*NAMES,*LOCAL) if adapt_response else NAMES,
            response='shot-local ridge fit using already matured pairs only' if adapt_response else None,
            clock=dict(use=RELEASE,own_nonoverlapping_target_ready=H+1)))
        for rank,sid in enumerate(order[p['startup']:],p['startup']+1):
            z = observe(sid,rank,s,p)
            records = issue_and_verify(*z,fits,adapt_response)
            name = s['shots'][rank-1]
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,**{f'{k}_{i}':v for i,d in enumerate(records) for k,v in d.items()})
            sizes.append(name)
            if rank in (21,120,360) or sid in (90874,74970,91689):
                x,m,risk,q,e,d = z
                n = min(len(x),650)
                use = q+H+1<n
                small = issue_and_verify(x[:n],m[:n],risk[:n],q[use],e[use],d[:n],fits,adapt_response)
                for f,g in zip(small,records,strict=True):
                    ready = g['ready']<n
                    np.testing.assert_allclose(f['mae'],g['mae'][ready],atol=1e-10,rtol=0.)
                    np.testing.assert_array_equal(f['q'],g['q'][ready])
                cut[str(sid)] = n
    write(out/'seal.json',dict(files=sizes,real_prefix=cut,wall_s=monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    reference = read(REF/'protocol.json')
    names = p['controls']
    rows = read(REF/'per_shot.json')['0.2']
    slices = {}
    assert len(s['files'])==340 and len(p['order'])==360
    for rank,fn in enumerate(s['files'],p['startup']+1):
        sid = p['order'][rank-1]
        if fn!=f'{rank:04d}_{sid}.npz':
            raise ValueError('forecast seal does not match cohort')
        y,down = io23.read_outcome('EAST',sid)
        t = io23.read_signals('EAST',sid,rate_contract='nominal')['times']
        old,new = (rows[k][rank-1] for k in ('current_5','bias_only'))
        if (sid,y,down)!=(old['shot'],old['disruptive'],old['down_s']):
            raise ValueError('original HDF5 truth does not match the frozen cohort')
        if y and not old['timely']:
            which = 'late' if old['alarm'] else 'never'
            cutoff = down-.030
        elif not y and new['alarm']:
            which = 'false'
            tau = reference['calibration']['0.2']
            with np.load(REF/'shots'/fn,allow_pickle=False) as f:
                cutoff = t[np.flatnonzero(f['bias_only']>=tau)[0]]
        else:
            which = 'timely' if y else 'safe'
            cutoff = np.inf
        with np.load(out/'shots'/fn,allow_pickle=False) as v:
            for j,k in enumerate(p['extras']):
                r,ready,error = (v[f'{key}_{j}'] for key in ('release','ready','mae'))
                rr = np.round((cutoff-t[r])*1000.,9)
                near = (rr>0.) & (rr<=95.)
                arrived = np.round((cutoff-t[ready])*1000.,9)>0.
                for label,mask in (('suffix',np.ones(len(r),bool)),(which,np.ones(len(r),bool)),
                                   (which+'_near',near),(which+'_near_mature',near & arrived)):
                    # “suffix” is not an outcome partition.
                    if mask.any():
                        slices.setdefault(k,{}).setdefault(label,[]).append((sid,error[mask]))
    result = {}
    for extra,groups in slices.items():
        result[extra] = {}
        for label,rs in groups.items():
            a = np.stack([v.mean(0) for _,v in rs])
            avg = a.mean(0)
            result[extra][label] = dict(shots=len(a),packets=sum(len(v) for _,v in rs),
                mae=dict(zip(names,avg.mean(1).tolist(),strict=True)),
                by_channel={k:dict(zip(p['target'],v.tolist(),strict=True)) for k,v in zip(names,avg,strict=True)},
                current_better_than_core_shots=int((a[:,2].mean(1)<a[:,1].mean(1)).sum()))
    write(out/'result.json',result)
    print({k:{g:(v['shots'],v['packets'],v['mae']) for g,v in z.items()
              if g in ('suffix','never_near_mature','late_near_mature','false_near_mature')}
           for k,z in result.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--adapt-response', action='store_true')
    a = cli.parse_args()
    collect(a.output.resolve(),a.adapt_response)
    score(a.output.resolve())
