"""Pair sealed held-window forecasts with actual missed and false-alarm decisions."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit

from . import commissioning_excursion_order_20261111 as matching
from . import commissioning_observation_inverse_20261114 as probe
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .history_gradient_metric_20261102.run import read, write


def classify_pairs(pos, neg, names):
    if not len(pos):
        return dict(count=0)
    inverse, shifted, older = (names.index(x) for x in (
        'inverse_current','inverse_shifted','inverse_without_latest'))
    u = (pos[:,inverse,2]>pos[:,0,2]) & (neg[:,inverse,2]<neg[:,0,2])
    beats_previous = (pos[:,inverse,2]>pos[:,older,2]) & (neg[:,inverse,2]<neg[:,older,2])
    predicts = (pos[:,inverse,1]<pos[:,0,1]) & (neg[:,inverse,1]<neg[:,0,1])
    beats_shift = (pos[:,inverse,1]<pos[:,shifted,1]) & (neg[:,inverse,1]<neg[:,shifted,1])
    return dict(count=len(pos),correct_risk_signs=int(u.sum()),
        improves_both_forecasts=int(predicts.sum()),beats_both_shifted_forecasts=int(beats_shift.sum()),
        joint=int((u&predicts).sum()),joint_beats_shifted=int((u&predicts&beats_shift).sum()),
        joint_beats_shifted_and_delayed_risk=int((u&predicts&beats_shift&beats_previous).sum()))


def run(out):
    p,s,r=(read(out/f'{k}.json') for k in ('protocol','seal','per_shot'))
    if len(s['curves'])!=360:
        raise ValueError('completed prediction seal required')
    lock=probe.lag.load_inputs()
    if lock['ids']!=p['order'] or p['startup']!=20:
        raise ValueError('original ordered cohort required')
    cases=[]
    positive,negative=[],[]
    tau=p['calibration']['0.2']
    for rank,sid in enumerate(p['order'],1):
        if rank<=p['startup']:
            continue
        row=r['0.2']['current_5'][rank-1]
        if (row['disruptive'] and row['timely']) or (not row['disruptive'] and not row['alarm']):
            continue
        y,down,t=probe.group.hdf._raw_truth(sid)
        if y!=row['disruptive'] or down!=row['down_s']:
            raise ValueError('post-seal raw outcome differs from scored shot')
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as f:
            frame, witness=f['frames'],f['follow']
            i=frame[:,1]
            if y:
                lead=np.round((down-t[i])*1000.,9)
                keep=(lead>30.)&(lead<=125.)
                settled=np.round((down-t[frame[:,-1]])*1000.,9)>30.
            else:
                first=int(np.flatnonzero(f['current_5']>=tau)[0])
                keep=i<=first
                settled=frame[:,-1]<=first
            i,w,release=i[keep],witness[keep],settled[keep]
            if not len(i):
                continue
            if len(i)!=len(np.unique(i)):
                raise ValueError('repeated witness action')
        raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
        signal=probe.lag.prepare_actual(raw,rank,lock['mapping'])
        cols=[io23.RAW23.index(k) for k in ('ip','Bt')]
        current=signal['values'][:,cols].copy()
        # Same issue/load coordinates for positive and negative cases.
        mask=signal['available'][:,cols]
        now,n=_window_mean(current,mask,i,-15,16)
        old,m=_window_mean(current[:,:1],mask[:,:1],i,-31,16)
        usable=(n>=8).all(1)&(m[:,0]>=8)
        context=np.c_[w[:,0,2]-logit(tau),now[:,0],now[:,1],now[:,0]-old[:,0]]
        bucket=positive if y else negative
        bucket.extend(dict(shot=sid,action=int(a),context=c,values=v,verified_before_decision=bool(f))
                      for a,c,v,f in zip(i[usable],context[usable],w[usable],release[usable],strict=True))

    # Preserve the raw-risk caliper. Use the preparation map's physical
    # ordering summaries for fixed units of Ip/Bt; no fit on post-20 outcomes.
    def convert(rows):
        design=np.asarray([r['context'] for r in rows],dtype=float).reshape(-1,4)
        ids=np.array([r['shot'] for r in rows],dtype=int)
        design[:,1]=np.arcsinh(design[:,1]/1e5)
        design[:,2]=np.arcsinh(design[:,2])
        design[:,3]=np.arcsinh(design[:,3]/1e4)
        return design,ids

    x,a=convert(positive)
    z,b=convert(negative)
    caliper=np.array([.50,.35,.35,.50])
    for key,pi,ni in (
        ('at_action',np.arange(len(positive)),np.arange(len(negative))),
        ('held_before_decision',np.flatnonzero([r['verified_before_decision'] for r in positive]),
         np.flatnonzero([r['verified_before_decision'] for r in negative]))):
        idx=matching.disjoint_pairs(x[pi],z[ni],a[pi],b[ni],caliper)
        pos=[positive[pi[i]] for i in idx[:,0]]
        neg=[negative[ni[j]] for j in idx[:,1]]
        pv,nv=(np.array([r['values'] for r in group]) for group in (pos,neg))
        cases.append(dict(scope=key,eligible_late_shots=len(set(a[pi])),eligible_safe_shots=len(set(b[ni])),
             counts=classify_pairs(pv,nv,p['witness_arms']),
             pair_ids=[[u['shot'],u['action'],v['shot'],v['action']] for u,v in zip(pos,neg,strict=True)]))
    write(out/'postseal_hard_attribution.json',dict(scope='diagnostic matched pairs, not a fitted action policy',
        columns=('same-parent margin','asinh(Ip / 100 kA)','asinh(Bt in file coordinates)',
                'asinh(Ip change / 10 kA)'),calipers=caliper.tolist(),cases=cases))
    print(cases,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    run(parser.parse_args().run.resolve())
