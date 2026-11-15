"""Known-error positive control for causal same-shot measurement adaptation."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_diagnostic_feedback_20261114 as feedback
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips


history,group,lag,layer = feedback.history,feedback.group,feedback.lag,feedback.layer
BT,IP,QA = tuple(io23.RAW23.index(k) for k in ('Bt','ip','qa_proxy'))
RAW, FIXED = 'faulted', 'first_gain_only'
MODES = (RAW,FIXED,'shot_gain','static_qa','oracle_gain','shared_fault','shared_shot_gain','clean_sham')
ARMS = (*feedback.embed.OLD,*MODES)
W,MIN = 32,16
CHANGE = 128
GAINS = (1.5,.67)
DEADBAND = .01  # Reject roundoff and small numerical-convention variation.
CASES = (90874,70320,73405,81406,74970,85422,73743)


def ratios(s):
    x,m = (s[k][:,[BT,IP,QA]].astype(np.float64) for k in ('values','available'))
    b,i,q = x.T
    with np.errstate(divide='ignore',invalid='ignore'):
        g = q*i/b
    return np.where((b!=0.) & (abs(i)>1.) & m.all(1) & (g>0.) & np.isfinite(g),g,np.nan)


def inject(s,g,*,common_mode=False):
    x = s['values'].copy()
    x[:,BT] *= g
    if common_mode:
        x[:,QA] *= g
    return dict(s,values=x)


def gain_clock(n,rank):
    # Synthetic injection only: the onset/phase script is never supplied to
    # the adaptive estimator. Only two changes, with order balanced by rank.
    stage = np.minimum(np.arange(n)//CHANGE,2)
    return np.where(stage==0,1.,np.asarray(GAINS)[(rank+stage)%2])


def estimate(s,k,*,fixed=False):
    """Each block estimates a bounded multiplicative gain from earlier pairs."""
    a = ratios(s)
    g = np.ones(len(a))
    current = 1.
    captured = False
    for r in range(W,len(a),MIN):
        packet = a[r-W:r]
        if not captured and np.isfinite(packet).all():
            proposal = k/np.median(packet)
            current = float(np.clip(proposal,.5,2.)) if abs(proposal-1.)>DEADBAND else 1.
            captured = fixed and current!=1.
        g[r:r+MIN] = current
    return g


def repair(s,g):
    x = s['values'].copy()
    x[:,BT] /= g
    return dict(s,values=x)


def reconstruct(s,k):
    x,m = s['values'].copy(),s['available']
    use = m[:,[BT,IP,QA]].all(1)&(abs(x[:,IP])>1.)
    x[use,BT] = (x[use,QA].astype(np.float64)*x[use,IP].astype(np.float64))/k
    return dict(s,values=x)


def fit_k(p,lock):
    m = lock['mapping']
    if any(m['gain'][j]!=1. or m['offset'][j]!=0. for j in (BT,IP,QA)):
        raise ValueError('measurement convention requires the original Bt, Ip and QA coordinates')
    v = []
    for sid in p['order'][:p['startup']]:
        x = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('locked cold-start signals changed')
        a = ratios(x)
        v.append(float(np.nanmedian(a)))
    if not np.isfinite(v).all() or min(v)<=0.:
        raise ValueError('cold shots must identify a stable numerical convention')
    return float(np.median(v)),v


@torch.no_grad()
def encode_views(base,views,tr):
    # Keep parent side paths and inherited correction parameters fixed.
    # The frozen non-linear encoder re-evaluates each changed input.
    q,h,z,beta,_prior,phase,correction = tr
    n = len(views)
    joined = []
    for i,b in feedback.block_indices(np.arange(len(h))):
        a = [lag._windows(base,s,i) for s in views]
        x,m = (torch.cat(v,dim=0) for v in zip(*a,strict=True))
        j = lag.delayed_slots(q,i)[0]
        adjust = layer.patcher(phase[j].repeat(n,1,1),correction[j].repeat(n,1))
        hidden = []
        with_hooks = (base.embedding.register_forward_pre_hook(lambda _m,a:(adjust(a[0]),)),
                      base.risk_head.register_forward_pre_hook(lambda _m,a:hidden.append(a[0])))
        try:
            base(x,m)
        finally:
            for handle in with_hooks:
                handle.remove()
        v = hidden[0].unflatten(0,(n,len(i)))
        probability = (z[i]+((v-h[i])*beta).sum(-1)).sigmoid()
        joined.append(probability[:,:b].cpu().numpy())
    return np.concatenate(joined,axis=1)


def track(signal,args,dec,path,arch,k,rank):
    base = args[0]
    tr = group.parent_track(signal,args,dec,path)
    n = len(signal['values'])
    true_gain = gain_clock(n,rank)
    mask = signal['available']
    fault, shared = (inject(signal,true_gain,common_mode=m) for m in (False,True))
    fit,static,coupled,clean = (estimate(s,k,fixed=fixed) for s,fixed in (
                               (fault,False),(fault,True),(shared,False),(signal,False)))
    views = (fault,repair(fault,static),repair(fault,fit),reconstruct(fault,k),
             repair(fault,true_gain),shared,repair(shared,coupled),repair(signal,clean))
    curve = dict(zip(MODES,encode_views(base,views,tr),strict=True))
    oracle_error = float(abs(curve['oracle_gain']-arch['bias_only']).max())
    if oracle_error>4e-6 or not np.isfinite(np.stack(list(curve.values()))).all():
        raise ValueError(f'clean observer must replay the same risk: {oracle_error:g}')
    v = signal['values'][:,BT].astype(np.float64)
    relative = np.stack([abs(v-s['values'][:,BT])/np.maximum(abs(v),1.e-12) for s in views])
    return curve,dict(gain=true_gain,estimate=fit,estimate_shared=coupled,estimate_clean=clean,
                      relative_error=relative,valid=mask[:,BT] & (v!=0.)),oracle_error


def seal(out,only):
    # The normal risk pathway is the CUDA project runner. The inline gain
    # estimator itself is NumPy on the streaming measurement coordinate.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the frozen parent replay')
    start = monotonic()
    lock,p,s,args,decoder,path = history.inputs()
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    files,witness,rat,max_err = [],{},[],0.
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcome before counterfactuals freeze')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('no outcome before counterfactuals freeze'))):
        k,values = fit_k(p,lock)
        write(out/'protocol.json',dict(**{n:p[n] for n in history.LOCK},arms=ARMS,only=only,
            reference='causal intervention on the measured Bt embedding input; all parent side paths fixed',
            cold_coeff=k,cold_coefficients=values,phase=dict(onsets_zero_based=[CHANGE,2*CHANGE],
            factors=GAINS,estimate_window=W,update_stride=MIN,unchanged_tolerance=DEADBAND),
            comparator='first_gain_only freezes the first non-identity estimate; '
                       'static_qa replaces the observed Bt value by QA*Ip/k whenever defined',
            scope='known synthetic measurement faults, not observational performance or a diagnosis of real faults'))
        for rank,sid in enumerate(p['order'],1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,s)
            if rank<=p['startup']:
                v = arc['bias_only']
                arms = dict.fromkeys(MODES,v)
                detail = {}
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signals changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                arms,detail,e = track(sig,args,decoder,path,arc,k,rank)
                max_err = max(max_err,e)
                if rank in (21,120,360) or sid in CASES or only is not None:
                    n = min(650,len(sig['values']))
                    past,old = feedback.trimmed(sig,arc,n)
                    a,d,_ = track(past,args,decoder,path,old,k,rank)
                    err = max(float(abs(v-arms[m][:n]).max()) for m,v in a.items())
                    for name,vs in a.items():
                        for threshold in p['calibration'].values():
                            np.testing.assert_array_equal(vs>=threshold,arms[name][:n]>=threshold)
                    if err>3e-6:
                        raise ValueError(f'prefix probability changed: {err:g}')
                    for key in ('estimate','estimate_shared','estimate_clean'):
                        np.testing.assert_array_equal(d[key],detail[key][:n])
                    witness[str(sid)] = dict(points=n,max_error=err)
                good = (detail['gain']!=1.) & detail['valid']
                if good.any():
                    rat.append(dict(shot=sid,points=int(good.sum()),
                        relative_Bt_L1=dict(zip(MODES,detail['relative_error'][:,good].mean(1).tolist(),strict=True)),
                        estimated_gain_relative_error=float(abs(detail['estimate'][good]/detail['gain'][good]-1.).mean()),
                        shared_gain_relative_error=float(abs(detail['estimate_shared'][good]/detail['gain'][good]-1.).mean()),
                        changed_clean_points=int((detail['estimate_clean']!=1.).sum()),
                        changed_shared_points=int((detail['estimate_shared']!=1.).sum())))
            name = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,**{k:arc[k] for k in feedback.embed.OLD},**arms,**detail)
            files.append(name)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(rank=rank,elapsed_s=round(monotonic()-start,1)),flush=True)
    write(out/'seal.json',dict(files=files,prefix=witness,oracle_error=max_err,seconds=monotonic()-start))
    write(out/'gain_recovery.json',rat)


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if p['only'] is not None or len(s['files'])!=360:
        raise ValueError('only the sealed full stream may be scored')
    rows = {c:{k:[] for k in ARMS} for c in p['calibration']}
    for rank,(sid,name) in enumerate(zip(p['order'],s['files'],strict=True),1):
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            for c,table in rows.items():
                tau = p['original'] if rank<=p['startup'] else p['calibration'][c]
                for k in ARMS:
                    table[k].append(dict(shot=sid,disruptive=y,down_s=d,**score_curve(t,f[k],y,d,tau)))
    result = {c:{k:lead.summarize(a) for k,a in table.items()} for c,table in rows.items()}
    comp = {c:{f'{a}__to__{b}':flips(v[a],v[b]) for a,b in (
        ('bias_only',RAW),(RAW,FIXED),(RAW,'shot_gain'),('static_qa','shot_gain'),
        ('bias_only','shot_gain'),('bias_only','clean_sham'),
        ('shared_fault','shared_shot_gain'))} for c,v in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=result,paired=comp,raw_rows=360*len(rows)*len(ARMS)))
    print({c:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in table.items()}
           for c,table in result.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--smoke-rank',action='append',type=int)
    mode.add_argument('--score',action='store_true')
    a = cli.parse_args()
    if a.score:
        score(a.output.resolve())
    else:
        seal(a.output.resolve(),a.smoke_rank)
