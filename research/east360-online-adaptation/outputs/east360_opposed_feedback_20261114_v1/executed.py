"""Compare two finite internal responses using delayed, fully observed diagnostics."""

import argparse
import os
from pathlib import Path
from shutil import copyfile
from time import monotonic, monotonic_ns
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit
from torch.nn import functional as F

from joint_lora_hindsight_capacity_20261104 import write_new
from layer0_query_protocol_20261105 import reveal_block

from . import commissioning_attention_reweight_20261114 as locked
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot


history, group, lag, layer = locked.history, locked.group, locked.lag, locked.layer
QUERY = io23.ROOT/'outputs/east360_conflict_query_20261114_v2'
OFFSET = lag.H+1+history.WAIT
PATCH = history.PATCH
# Budget for issuing the two hypothetical forecasts. Actual write size uses
# the old per-point token L2. The two issues have identical norms.
ISSUE_L2 = .89072
RIDGE = .10
BASES = ('fixed','bias_only','current_5','earlier_5')
ARMS = (*BASES,'pos','neg','observation_current','observation_earlier','predicted_only',
        'history_fixed','joint_current','joint_earlier','joint_reversed')
DIRECTIONS = ARMS[len(BASES):]
CASES = (90874,70320,73405,73743)
EPS = 1e-8


def centered_unit(x,m):
    clean = x.masked_fill(~m[...,None],0.)
    count = m.sum(1).clamp_min(1)
    mean = clean.sum(1)/count[:,None]
    raw = (clean-mean[:,None]).masked_fill(~m[...,None],0.)
    return F.normalize(raw.flatten(1),dim=1,eps=EPS).reshape_as(x)


def action_vectors(sign,dose,direction):
    return sign[...,None,None]*dose[None,:,None,None]*direction[None]


def signed_fit(positive,negative,target,*,dim):
    """Normalized squared-loss difference, computed around the shared midpoint."""
    center = (positive+negative)*.5
    contrast = positive-negative
    error = target-center
    scale = error.square().mean(dim).sqrt()*contrast.square().mean(dim).sqrt()
    return (error*contrast).mean(dim)/scale.clamp_min(.01)


def earlier_qualified(q,good):
    prior = np.maximum.accumulate(np.r_[-1,np.where(good,np.arange(len(q)),-1)])
    return prior[np.searchsorted(q+OFFSET,q,side='right')]


def fit_vectors(positive,negative,observed):
    # Each diagnostic family's relative trajectory chooses a sign, not a new
    # score. Use five bins of the full issued 25-point forecasts.
    p,n,y = (v-v.mean(1,keepdim=True) for v in (positive,negative,observed))
    return torch.stack([signed_fit(p[...,list(c)],n[...,list(c)],y[...,list(c)],dim=(1,2))
                        for c in FEATURE_GROUPS],dim=1)


FEATURE_GROUPS = tuple(tuple(group.USE.index(c) for c in g) for g in group.POSITIONS)
RAW_COLS = list(np.asarray(lag.CHANNELS)[group.USE])


def direction_at(base,x,mask,adjust,beta):
    store = []

    def tap(_module,args):
        z = args[0].detach().requires_grad_(True)
        store.append(z)
        return (z,*args[1:])

    hook = base.encoder.layers[0].register_forward_pre_hook(tap)
    patcher = base.embedding.register_forward_pre_hook(lambda _m,p:(adjust(p[0]),))
    pickup = base.risk_head.register_forward_pre_hook(lambda _m,p:store.append(p[0]))
    try:
        base(x,mask)
    finally:
        for h in (hook,patcher,pickup):
            h.remove()
    g, = torch.autograd.grad((store[1]*beta).sum(),store[0])
    valid = (mask & x.isfinite()).reshape(len(x),base.cfg.window//PATCH,-1).any(-1)
    return centered_unit(g[:,:-1],valid)


def forward_states(base,x,mask,adjust,direction,codes,dose,beta):
    n,b = codes.shape
    captured = []
    moves = action_vectors(codes,dose,direction).flatten(0,1)
    parity = []

    def transform(_m,a):
        return (adjust(a[0][:b]).repeat(n,1,1),)

    def write(_m,a):
        old = a[0]
        new = layer.add_to_patches(old,moves)
        actual = (new[:,:-1]-old[:,:-1]).flatten(1).norm(dim=1).reshape(n,b)
        parity.append(float((actual-codes.abs()*dose).abs().max()))
        return (new,*a[1:])

    h = (base.embedding.register_forward_pre_hook(transform),
         base.encoder.layers[0].register_forward_pre_hook(write),
         base.risk_head.register_forward_pre_hook(lambda _m,p:captured.append(p[0])))
    try:
        base(x.repeat(n,1,1),mask.repeat(n,1,1))
    finally:
        for a in h:
            a.remove()
    end = captured[0].reshape(n,b,-1)
    return end,end@beta,parity[0]


def proposed_feedback(signal,base,track,decoder,fields,truth,i):
    q,_,_,beta,prior,phase,correction = track
    t = q[i]
    j = lag.delayed_slots(q,t)[0]
    x,m = lag._windows(base,signal,t)
    adjust = layer.patcher(phase[j],correction[j])
    direction = direction_at(base,x,m,adjust,beta)
    recent,center,spread,noise = fields
    with torch.no_grad():
        r = direction.new_tensor([1.,-1.])[:,None].expand(-1,len(i))
        h,_,error = forward_states(base,x,m,adjust,direction,r,direction.new_full((len(i),),ISSUE_L2),beta)
        forecasts = [group.signed_levels(lag._predict(v,prior[i],decoder)*noise+recent[i,None],
                                         center[i],spread[i]) for v in h]
        actual = group.signed_levels(truth[i],center[i],spread[i])
        current = fit_vectors(*forecasts,actual)
        # "Predicted only" supplies the observed recent trend projected over
        # five bins. No prediction is fed back to simulate a sample arrival.
        x0 = ((x[:,-16:,RAW_COLS]-base.center[RAW_COLS])/base.scale[RAW_COLS]).asinh()
        mean0,mean1 = x0.unflatten(1,(2,8)).mean(2).unbind(1)
        recent_valid = m[:,-16:,RAW_COLS].all(dim=(1,2))
        # Compare the same forecast directions to a no-new-measurement
        # extrapolation, in its own earlier issued forecast units.
        linear = ((mean1[:,None]-mean0[:,None])*torch.linspace(-1.,1.,5,device=x.device)[None,:,None]
                  /spread[i][:,None,group.USE]).asinh()
        prior_obs = torch.where(recent_valid[:,None],fit_vectors(*forecasts,linear),torch.zeros_like(current))
        reversed_ = fit_vectors(*forecasts,actual.flip(1))
    return torch.stack((current,prior_obs,reversed_)),error


def feedback(signal,args,decoder,path,arc):
    track = group.parent_track(signal,args,decoder,path)
    q,_,z,*_ = track
    release = q+lag.H+1
    np.testing.assert_array_equal(release,arc['release'])
    eligible,fields,truth = group.feedback_waveform(signal,q,args[2],z.device)
    required = signal['available'][q[:,None]+np.arange(1,lag.H+1)][:,:,RAW_COLS]
    possible = eligible & required.all((1,2)) & arc['eligible']
    earlier = earlier_qualified(q,possible)
    avail = possible & (earlier>=0)
    a = z.new_zeros((3,len(q),len(FEATURE_GROUPS)))
    positions = np.flatnonzero(possible)
    e = 0.
    for j in range(0,len(positions),group.BATCH):
        i = positions[j:j+group.BATCH]
        a[:,i],err = proposed_feedback(signal,args[0],track,decoder,fields,truth,i)
        e = max(e,err)
    x = a[0].detach().cpu().numpy()
    older = np.zeros_like(x)
    older[avail] = x[earlier[avail]]
    if not np.isfinite(x).all() or e>1e-5:
        raise ValueError('issued directions or forecast responses are invalid')
    return track,np.stack((x,older,*a[1:].detach().cpu().numpy())),avail,e


def teacher(bank,rank):
    """Use only finite-step outcomes that are visible at this block boundary."""
    rows = [(y,x,d) for visible,y,x,d in bank if visible<=rank and len(x)]
    count = [sum(int(y)==i for y,_,_ in rows) for i in (0,1)]
    if not all(count):
        return np.zeros(len(FEATURE_GROUPS))
    x = np.concatenate([x for _,x,_ in rows]).astype(np.float64)
    y = np.concatenate([d for _,_,d in rows])
    weights = np.concatenate([np.full(len(x),.5/(len(x)*count[int(b)])) for b,x,_ in rows])
    r = x.T@(weights[:,None]*x)
    fit = np.linalg.solve(r+RIDGE*np.eye(r.shape[1]),x.T@(y*weights))
    # Separate the history-only intercept from the feedback-dependent part.
    # History-free conduct is preserved when supervision has not been released.
    return fit


def hard(x):
    # Ties in a matched signed control take the same predeclared option.
    return np.where(x>=0.,1.,-1.)


def route(packets,calibration):
    a,b,c,d = packets
    direct = [hard(v.mean(1)) for v in (a,b,c)]
    on = np.any(calibration)
    # Whole-shot outcome is not a scoring input. These coefficients are
    # frozen within the block, whether history is enabled or not.
    common = np.full(len(a),1. if calibration.mean()>=0. else -1.)
    gained = [hard(v@(np.full(len(calibration),1./len(calibration))+calibration))
              for v in (a,b,d)]
    constant = common if on else np.zeros(len(a))
    return np.stack((np.ones(len(a)),-np.ones(len(a)),*direct,constant,*gained))


def render(signal,base,clock,packets,legal,arc,weight):
    q,h,z,beta,_prior,phase,correction = clock
    t = np.arange(len(z))
    r = q+OFFSET
    s = np.searchsorted(r,t,side='right')
    g = np.r_[False,legal]
    support = (arc['dose']>0.) & g[s]
    directions = route(packets,weight)
    directions = np.pad(directions,((0,0),(1,0)))
    curves = np.broadcast_to(arc['bias_only'],(len(DIRECTIONS),len(t))).copy()
    spent = np.zeros(len(t),np.float32)
    detail = np.zeros(3)
    active = np.flatnonzero(support)
    for j in range(0,len(active),group.BATCH):
        e = active[j:j+group.BATCH]
        use = s[e]
        p = use-1
        f = group.handoff_fraction(r[p],e,PATCH)
        # Share the absolute dose even when an arm switches sign. To avoid
        # an abrupt full-strength flip, use the smallest ramp across arms.
        a = directions[:,use]
        old = directions[:,p]*g[p]
        ramp = f*a+(1.-f)*old
        participating = np.r_[np.arange(5), np.arange(6,9),5] if weight.any() else np.r_[np.arange(5),np.arange(6,9)]
        cap = arc['dose'][e]*abs(ramp[participating]).min(0)
        codes = np.sign(ramp)
        x,m = lag._windows(base,signal,e)
        j0 = lag.delayed_slots(q,e)[0]
        adjust = layer.patcher(phase[j0],correction[j0])
        axis = direction_at(base,x,m,adjust,beta)
        cap = z.new_tensor(cap)*(axis.flatten(1).norm(dim=1)>0.)
        with torch.no_grad():
            out = np.vstack((np.zeros((1,len(e))),codes))
            _,r0,err = forward_states(base,x,m,adjust,axis,z.new_tensor(out),cap,beta)
            prob = (z[e]+(r0-h[e]@beta)).sigmoid().cpu().numpy()
            detail = np.maximum(detail,(abs(prob[0]-arc['bias_only'][e]).max(),err,
                                         max(float((cap-z.new_tensor(arc['dose'][e])).max()),0.)))
            spent[e] = cap.cpu().numpy()
            curves[:,e] = np.where(spent[e][None]>0.,prob[1:],arc['bias_only'][e])
    if detail[0]>3e-6 or detail[1]>1e-5 or detail[2]>1e-6:
        raise ValueError(f'equal-time internal intervention diverged from parent or dose: {detail}')
    return curves,spent,detail


def teacher_points(times,parent,available,disruptive,down):
    if disruptive:
        when = np.round((down-times)*1000.,9)
        available = available & (when>30.) & (when<=125.)
    ends = np.flatnonzero(available)
    return np.array([part[np.argmax(parent[part])] for part in np.array_split(ends,4)
                     if len(part)],dtype=np.int64)


def teaching_entry(t,base,curves,spent,q,obs,y,down,tau):
    points = teacher_points(t,base,spent>0.,y,down)
    if not len(points):
        return obs[:0],np.empty(0)
    i = np.searchsorted(q+OFFSET,points,side='right')-1
    if (i<0).any():
        raise ValueError('teaching read a packet before its arrival')
    a,b = logit(np.clip(curves[:2,points],1e-7,1.-1e-7))
    unchanged = logit(np.clip(base[points],1e-7,1.-1e-7))
    threshold = logit(tau)
    s = -1. if y else 1.
    # Difference of the two actual, finite changes to the locked risk task.
    # y=1 near the warning deadline, y=0 on a safe discharge.
    ramp = np.logaddexp(0.,s*(b-threshold)/.25)-np.logaddexp(0.,s*(a-threshold)/.25)
    # Do not heavily fit an already-confident endpoint where both proposals agree.
    gap = np.abs(a-unchanged)+np.abs(b-unchanged)
    utility = np.tanh(ramp/np.maximum(1.,4.*gap))
    return obs[i],utility


def read_choice(p,block):
    prior = QUERY/'blocks'/f'{block:02d}_commit.json'
    row = p['choices'][block-1]
    if block==1 or block==18:
        return []
    k = read(prior)
    sid = k['methods']['alternating']
    if (row['rank_end']!=block*20 or row['choice']['alternating']!=sid
            or sid not in p['order'][(block-1)*20:block*20] or sid not in k['selected']):
        raise ValueError('alternating query schedule does not match the sealed records')
    return [sid]


def syncdir(path):
    fd = os.open(path,os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(out):
    start = monotonic()
    lock,p,arch,args,decoder,path = history.inputs()
    base = args[0]
    snapshots = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    order = p['order']
    locked_queries = read(QUERY/'seal.json')
    if len(order)!=360 or order!=locked_queries['order'] or p['startup']!=20:
        raise ValueError('fixed donor schedule and shot order required')
    (out/'shots').mkdir(parents=True)
    (out/'blocks').mkdir()
    copyfile(__file__,out/'executed.py')
    write_new(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},arms=ARMS,
        query_scope='fixed alternating16, current block labels only after durable lookahead forecasts',
        site='zero-mean first-attention input tokens; frozen risk gradient sets one sign-symmetric axis',
        issue_l2=ISSUE_L2,release_offset=OFFSET,hand_off=PATCH,
        teach='fit one four-feature orientation correction from realized plus/minus effects',
        independent='observation_current does not depend on EAST labels'))
    oracle = io23.read_outcome
    files, bank, witnesses, support = [], [], {}, []
    maxima = np.zeros(4)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed shot label')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('unsealed HDF5 label'))):
        for block in range(1,19):
            ids = read_choice(locked_queries,block)
            first = (block-1)*20+1
            weight = teacher(bank,first)
            queried, receipt = {}, []
            for rank,sid in enumerate(order[first-1:first+19],first):
                arc = history.archived_shot(rank,sid,arch)
                if rank>20:
                    if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                        raise ValueError('fixed EAST input has changed')
                    raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                    t = raw['times']
                    signal = lag.prepare_actual(raw,rank,lock['mapping'])
                    track,e,valid,err = feedback(signal,args,decoder,path,arc)
                    curves,dose,diag = render(signal,base,track,e,valid,arc,weight)
                    maxima = np.maximum(maxima,(*diag,err))
                    if rank in (21,120,360) or sid in CASES:
                        end = min(550,len(t))
                        s = arc['release']<end
                        a = {k:v[s] if k in ('release','eligible') else v[:end] for k,v in arc.items()}
                        cut = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                        short_track,short_packets,short_valid,_ = feedback(cut,args,decoder,path,a)
                        after,applied,_ = render(cut,base,short_track,short_packets,short_valid,a,weight)
                        np.testing.assert_allclose(applied,dose[:end],atol=3e-6,rtol=0.)
                        probe = locked.probe
                        probe.check_thresholds(after,curves[:,:end],tuple(p['calibration'].values()))
                        witnesses[str(sid)] = dict(points=end,error=float(abs(after-curves[:,:end]).max()))
                    if sid in ids:
                        queried[sid] = (t,arc['bias_only'],curves,dose,track[0],e[0])
                else:
                    n = len(arc['bias_only'])
                    curves = np.broadcast_to(arc['bias_only'],(len(DIRECTIONS),n))
                    dose = np.zeros(n)
                    valid = arc['eligible']
                fn = f'{rank:04d}_{sid}.npz'
                locked.save_shot(out/'shots'/fn,dict(
                    **{k:arc[k] for k in BASES},**dict(zip(DIRECTIONS,curves,strict=True)),dose=dose))
                files.append(fn)
                receipt.append(fn)
                support.append(dict(shot=sid,writes=int(np.count_nonzero(dose)),feedback=int(valid.sum())))
            syncdir(out/'shots')
            commit = out/'blocks'/f'{block:02d}_commit.json'
            write_new(commit,dict(block=block,rank_end=first+19,selected=ids,
                                  sealed=receipt,committed_ns=monotonic_ns()))
            syncdir(commit.parent)
            reveal = reveal_block(commit,ids,oracle)
            write_new(out/'blocks'/f'{block:02d}_reveal.json',reveal)
            learned = []
            for r in reveal['outcomes']:
                y,down = r['disruptive'],r['down']
                x,d = teaching_entry(*queried[r['shot']],y,down,p['calibration']['0.2'])
                bank.append((first+20,y,x,d))
                learned.append(dict(shot=r['shot'],disruptive=y,available_from=first+20,
                                    targets=d.tolist(),packets=x.tolist()))
            write_new(out/'blocks'/f'{block:02d}_learning.json',dict(
                applied=weight.tolist(),query=learned,active_history=len(bank)-len(learned)))
            if block in (1,3,6,10,14,18):
                print(dict(stage='opposed_feedback_sealed',shots=first+19,history=len(bank),
                           seconds=round(monotonic()-start,2)),flush=True)
    for k,v in base.state_dict().items():
        np.testing.assert_array_equal(v.detach().cpu().numpy(),snapshots[k].numpy())
    if len(bank)!=16:
        raise ValueError('fixed 16 historical queries required')
    write_new(out/'seal.json',dict(order=order,curves=files,prefix=witnesses,support=support,
                                   maximum=maxima.tolist(),seconds=monotonic()-start))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    ref = read(history.ARCHIVE/'per_shot.json')
    required = [f'{i:04d}_{sid}.npz' for i,sid in enumerate(p['order'],1)]
    if len(required)!=360 or required!=s['curves']:
        raise ValueError('score only a completed common stream')
    matrix = {b:{k:[] for k in ARMS} for b in p['calibration']}
    suffix = {k:[] for k in ARMS}
    potential, hard_rows = [], []
    prevented = {k:[] for k in ('new_actions_lower_envelope','same_parent_and_current_5')}
    cold_fp = 0
    for rank,(sid,name) in enumerate(zip(p['order'],required,strict=True),1):
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as q:
            archive = dict(q)
        a = archive['dose']>0.
        for k in DIRECTIONS:
            np.testing.assert_array_equal(archive[k][~a],archive['bias_only'][~a])
        for k,v in archive.items():
            if k=='dose':
                continue
            if rank>p['startup']:
                suffix[k].append(dict(shot=sid,disruptive=y,peak=peak_for_shot(t,v,y,down)))
            for b,tau in p['calibration'].items():
                th = p['original'] if rank<=p['startup'] else tau
                c = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,v,y,down,th))
                if k in BASES and c!=ref[b][k][rank-1]:
                    raise ValueError('archive does not rescore to the fixed prior')
                matrix[b][k].append(c)
        th = p['original'] if rank<=p['startup'] else p['calibration']['0.2']
        if y and not ref['0.2']['current_5'][rank-1]['timely']:
            d = np.round((down-t)*1e3,9)
            near = a & (d>30.) & (d<=125.)
            witness = dict(shot=sid,rank=rank,available=int(near.sum()),kind=(
                'late' if ref['0.2']['current_5'][rank-1]['alarm'] else 'never'),
                maximum={k:float(archive[k][near].max()) if near.any() else None for k in ('bias_only',*DIRECTIONS)},
                near_hits={k:int(np.count_nonzero(near & (archive[k]>=th))) for k in ('bias_only',*DIRECTIONS)})
            hard_rows.append(witness)
            if np.any(near & ((archive['pos']>=th)|(archive['neg']>=th))):
                potential.append(sid)
        if not y:
            if rank<=p['startup']:
                cold_fp += bool((archive['bias_only']>=th).any())
            for k,arms in (('new_actions_lower_envelope',('bias_only','pos','neg')),
                           ('same_parent_and_current_5',('bias_only','current_5','pos','neg'))):
                v = np.stack([archive[k] for k in arms])
                if (np.min(v,axis=0)>=th).any():
                    prevented[k].append(sid)
    metrics = {b:{k:lead.summarize(r) for k,r in v.items()} for b,v in matrix.items()}
    pairs = (('bias_only','observation_current'),('observation_earlier','observation_current'),
             ('predicted_only','observation_current'),('current_5','observation_current'),
             ('history_fixed','joint_current'),('joint_earlier','joint_current'),
             ('joint_reversed','joint_current'),('observation_current','joint_current'))
    rank = {str(n):{k:{i:g[i] for i in ('tp','fp')} for k,r in suffix.items()
                    for g in [best_at_count(r,n-cold_fp)]} for n in (18,27,36)}
    write_new(out/'per_shot.json',matrix)
    write_new(out/'result.json',dict(metrics=metrics,raw_rescored=360*len(ARMS)*len(matrix),
        postseal=dict(near_potential=potential,hard=hard_rows,irreducible_fp=prevented,
                      suffix_ranking_only=rank,cold_false=cold_fp),
        paired={b:{f'{i}_to_{j}':dict(flips=flips(a[i],a[j]),lead=lead.paired(a[i],a[j])) for i,j in pairs}
                for b,a in matrix.items()}))
    for b,a in metrics.items():
        print(b,{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in a.items()},
              'timely_near_union',potential,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--score',action='store_true')
    cfg = cli.parse_args()
    (score if cfg.score else run)(cfg.output.resolve())
