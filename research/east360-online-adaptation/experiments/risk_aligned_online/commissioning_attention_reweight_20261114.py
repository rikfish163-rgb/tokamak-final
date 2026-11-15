"""Delayed, shot-local attention recency training on the locked EAST stream."""

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

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha
from .lifecycle_stream_20261025.audit import flips


history, group, lag, layer = probe.history, probe.group, probe.lag, probe.layer
QUERY = io23.ROOT/'probe_outputs/layer0_direct_transfer360_20261112_v2'
# Four head offsets relative to the most recent 32 measurements, not score offsets.
PATCHES, HEADS = 16, 4
LOGIT_RADIUS = .5
NEW = ('attention_current','attention_earlier','attention_reversed',
       'inter_only','joint_current','joint_earlier','joint_reversed')
ARMS = ('fixed','bias_only','current_5','earlier_5',*NEW)
EPS = 1e-8


def age_basis(valid, dtype):
    # Each of two time contrasts has zero mean over observed keys. CLS does
    # not enter this centering. Missing keys keep the original padding mask.
    p = torch.linspace(-1.,1.,PATCHES,device=valid.device,dtype=dtype)
    x = torch.stack((p, .5*(3.*p.square()-1.)),dim=1)
    x = (x[None]*valid[...,None]).expand(len(valid),-1,-1)
    x = x-x.sum(1,keepdim=True)/valid.sum(1).clamp_min(1)[:,None,None]
    return x.masked_fill(~valid[...,None],0.)


def key_bias(valid, value):
    x = torch.einsum('bpc,bhc->bhp',age_basis(valid,value.dtype),value)
    return F.pad(x,(0,1))


def attention_bias(valid, state):
    x = key_bias(valid,state)
    return x[:,:,None,:].expand(-1,-1,PATCHES+1,-1).reshape(-1,PATCHES+1,PATCHES+1)


def _valid(mask):
    return mask.reshape(len(mask),PATCHES,-1).any(-1)


def hidden(base,x,mask,transform,state):
    """Use PyTorch's frozen Q/K/V and attention with an additive temporal key prior."""
    if (base.cfg.window,base.cfg.patch,base.cfg.heads)!=(128,8,HEADS):
        raise ValueError('requires the locked 16-token, four-head encoder')
    n = len(x)
    valid = _valid(mask & x.isfinite())
    added = attention_bias(valid,state)
    saved = []

    def tilt(_module,args,kw):
        if kw['attn_mask'] is not None:
            raise ValueError('unexpected original attention bias')
        return args,dict(kw,attn_mask=added)

    handles = (
        base.embedding.register_forward_pre_hook(lambda _m,a:(transform(a[0][:n]),)),
        base.encoder.layers[0].self_attn.register_forward_pre_hook(tilt,with_kwargs=True),
        base.risk_head.register_forward_pre_hook(lambda _m,a:saved.append(a[0])))
    try:
        base(x,mask)
    finally:
        for handle in handles:
            handle.remove()
    return saved[0]


def unit(x):
    return x/x.flatten(-2).norm(dim=-1).clamp_min(EPS)[...,None,None]


def scale_key_logits(valid,state):
    """Match the realized, centered key-prior L2 over heads and observed time keys."""
    return key_bias(valid,state).flatten(1).norm(dim=1).reshape(-1,1,1)


def historical_prior(policy, like):
    baseline = torch.zeros_like(like)
    baseline[...,0] = policy
    return baseline


def realized_tilts(current,previous,prior_known,fraction,clock,observed,policy,*,scouts=False):
    """One clock and one key-logit norm, including inter-only and all shams."""
    a,b = unit(current),unit(previous)
    f = fraction[None,:,None,None]
    intra = b.lerp(a,f)
    prior = historical_prior(policy,a[0])
    current_joint = unit(a*(1.+policy[None,None,:,None])+prior)
    previous_joint = unit(b*(1.+policy[None,None,:,None])+prior)
    known = torch.as_tensor(prior_known,device=current.device)[None,:,None,None]
    joint = (previous_joint*known).lerp(current_joint,f)
    arm = torch.cat((intra,unit(prior)[None],joint))
    active = [0,1,2,4,5,6]
    if bool(policy.any()):
        active.append(3)
    if scouts:
        # Equal-width one-head pressure directions from a fixed library.
        eye = F.pad(torch.eye(HEADS,device=arm.device,dtype=arm.dtype)[...,None],(0,1))
        arm = torch.cat((arm,torch.cat((eye,-eye))[:,None].expand(-1,len(observed),-1,-1)))
        active += list(range(len(NEW),len(arm)))
    repeated = observed.repeat(len(arm),1)
    norm = scale_key_logits(repeated,arm.flatten(0,1)).unflatten(0,(len(arm),len(observed)))
    measured = (norm[active,:,0,0]>EPS).all(0)
    ramp = torch.where(known[0,:,0,0],1.,fraction)
    # Old token L2 and new logit L2 use different units. Only the old zero/nonzero
    # support is reused. All new nonzero arms use a fixed max key-logit L2.
    dose = LOGIT_RADIUS*ramp*measured*(clock>0.)
    arm = arm*(dose[None,:,None,None]/norm.clamp_min(EPS))
    realized = scale_key_logits(repeated,arm.flatten(0,1)).view(len(arm),len(observed))
    return arm,dose,float((realized[active]-dose).abs().max())


def teacher(bank, rank):
    """Use actual past ±head effects; apply no target-label policy before both types."""
    visible = [(int(d),x,y) for release,d,x,y in bank if release<=rank]
    sides = [sum(d==j for d,_,_ in visible) for j in range(2)]
    if not all(sides):
        return np.zeros(HEADS,np.float32)
    x = np.concatenate([x for _,x,_ in visible])
    y = np.concatenate([y for _,_,y in visible])
    mass = np.concatenate([np.full(len(x),.5/(len(x)*sides[d])) for d,x,_ in visible])
    a = x.T@(mass[:,None]*x)
    w = np.linalg.solve(a+.01*np.eye(HEADS),x.T@(mass*y))
    if not np.isfinite(w).all():
        raise ValueError('nonfinite historical attention sensitivity')
    # Directional projection, with at most the base per-head scaling.
    scale = abs(w).max()
    return (w/max(.05,scale)).astype(np.float32)


def _soft_risk(v, threshold):
    return np.logaddexp(0.,(logit(np.clip(v,1e-7,1.-1e-7))-threshold)/.25)


def preference(t, base, candidate, disruptive, down, threshold):
    """Positive if the whole-trajectory fixed-line objective improves."""
    if disruptive:
        ms = np.round((down-t)*1e3,9)
        allowed = (ms>30.) & (ms<=125.)
    else:
        allowed = np.ones(len(t),bool)
    if not allowed.any():
        return 0.
    a,b = (float(x[allowed].max()) for x in (base,candidate))
    tau = logit(threshold)
    if disruptive:
        a,b,tau = 1.-a,1.-b,-tau
    return float(_soft_risk(a,tau)-_soft_risk(b,tau))


def measured_directions(signal,args,decoder,path,weights,old):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    net = args[0]
    eligible,fields,waveform = group.feedback_waveform(signal,q,args[2],z.device)
    recent,center,spread,noise = fields
    issued_slot = lag.delayed_slots(q,q)[0]
    release = q+lag.H+1
    np.testing.assert_array_equal(release,old['release'])
    # First seal the actual forecasts from the frozen issue-time graph.
    force = z.new_zeros((len(q),lag.H//group.BIN,len(group.USE)))
    required = np.asarray(lag.CHANNELS)[group.USE]
    observed_mask = signal['available'][q[:,None]+np.arange(1,lag.H+1)][:,:,required]
    eligible = eligible & observed_mask.all(axis=(1,2))
    supported = np.flatnonzero(eligible)
    for a in range(0,len(supported),probe.BATCH):
        i = supported[a:a+probe.BATCH]
        s = issued_slot[i]
        x,m = lag._windows(net,signal,q[i])
        with torch.no_grad():
            transform = layer.patcher(phase[s],correction[s])
            issued = hidden(net,x,m,transform,z.new_zeros((len(i),HEADS,2)))
            whole = lag._predict(issued,prior[i],decoder)*noise+recent[i,None]
            forecast = group.signed_levels(whole,center[i],spread[i])
        observed = group.signed_levels(waveform[i],center[i],spread[i])
        force[i] = ((observed-forecast)/group.BETA).clamp(-1.,1.)
    _,now,earlier,common = group.past_force_parts(q,force,eligible)
    valid = common & old['eligible']
    d = z.new_zeros((3,len(q)+1,HEADS,2))
    j = np.flatnonzero(valid)
    for a in range(0,len(j),probe.BATCH):
        i = j[a:a+probe.BATCH]
        s = issued_slot[i]
        x,m = lag._windows(net,signal,q[i])
        with torch.no_grad():
            issued = hidden(net,x,m,layer.patcher(phase[s],correction[s]),z.new_zeros((len(i),HEADS,2)))
        issued.requires_grad_(True)
        whole = lag._predict(issued,prior[i],decoder)*noise+recent[i,None]
        pred = group.signed_levels(whole,center[i],spread[i])
        xr,mr = lag._windows(net,signal,release[i])
        a0 = z.new_zeros((len(i),HEADS,2),requires_grad=True)
        arrived = hidden(net,xr,mr,layer.patcher(phase[i+1],correction[i+1]),a0)
        candidates = (now[i],earlier[i],now[i].flip(1))
        d[:,i+1] = group.signed_directions(issued,arrived,a0,
            [(pred,group.shape_force(c,weights)) for c in candidates])
    if not torch.isfinite(d).all():
        raise FloatingPointError('nonfinite attention feedback')
    return (q,h,z,beta,phase,correction),d,valid


@torch.no_grad()
def render(signal,base,track,states,valid,arc,learned,*,witness=False):
    q,h,z,beta,phase,correction = track
    r = q+lag.H+1+history.WAIT
    t = np.arange(len(z))
    all_slot = np.searchsorted(r,t,side='right')
    candidate = (arc['dose']>0.) & np.r_[False,valid][all_slot]
    active = np.flatnonzero(candidate)
    curves = np.broadcast_to(arc['bias_only'],(len(NEW),len(t))).copy()
    diag_curves = np.broadcast_to(arc['bias_only'],(2*HEADS,len(t))).copy() if witness else None
    spent = np.zeros(len(t),np.float32)
    op_err, dose_err, aux_err = 0.,0.,0.
    policy = z.new_tensor(np.asarray(learned))
    for k in range(0,len(active),probe.BATCH):
        e = active[k:k+probe.BATCH]
        slot = all_slot[e]
        fraction = group.handoff_fraction(np.r_[0,r][slot],e,base.cfg.patch)
        s = lag.delayed_slots(q,e)[0]
        x,m = lag._windows(base,signal,e)
        observed = _valid(m)
        prev = np.maximum(slot-1,0)
        # Inter-only uses the learned direction with the same key-prior
        # amplitude and deadline as the other arms. Joint compares a
        # feedback-dependent direction to that persistent direction.
        tilt,dose,norm_error = realized_tilts(states[:,slot],states[:,prev],np.r_[False,valid][prev],
            z.new_tensor(fraction),z.new_tensor(arc['dose'][e]),observed,policy,scouts=witness)
        dose_err = max(dose_err,norm_error)
        controls = z.new_zeros((1,len(e),HEADS,2))
        arms = torch.cat((controls,tilt))
        n = len(arms)
        transform = layer.patcher(phase[s],correction[s])
        def repeated(raw):
            return transform(raw[:len(e)]).repeat(n,1,1)
        outputs = hidden(base,x.repeat(n,1,1),m.repeat(n,1,1),repeated,arms.flatten(0,1))
        out = (z[e]+((outputs.reshape(n,len(e),-1)-h[e])*beta).sum(-1)).sigmoid().cpu().numpy()
        op_err = max(op_err,float(np.max(abs(out[0]-arc['bias_only'][e]))))
        peak = float(key_bias(observed.repeat(n,1),arms.flatten(0,1)).abs().max())
        aux_err = max(aux_err,peak)
        pred = out[1:1+len(NEW)]
        pred[:,dose.cpu().numpy()==0.] = arc['bias_only'][e[dose.cpu().numpy()==0.]]
        curves[:,e] = pred
        if witness:
            diag_curves[:,e] = out[1+len(NEW):]
            diag_curves[:,e[dose.cpu().numpy()==0.]] = arc['bias_only'][e[dose.cpu().numpy()==0.]]
        spent[e] = dose.cpu().numpy()
    if op_err>3e-6 or dose_err>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'parent/dose changed: {op_err:.4g}/{dose_err:.4g}')
    return curves,diag_curves,spent,(op_err,dose_err,aux_err)


def first_use(block):
    return block*20+1


def checked_selection(block,order,prior):
    p = QUERY/'blocks'/f'{block:02d}_commit.json'
    if sha(p)!=prior[block-1]['commit_sha256']:
        raise ValueError('changed prior query commitment')
    x = read(p)
    ids = order[(block-1)*20:block*20]
    want = 20 if block==1 else int(block<18)
    if (x['rank_end']!=block*20 or x['block']!=block or not set(x['selected'])<=set(ids)
            or len(set(x['selected']))!=want or len(x['selected'])!=want):
        raise ValueError('fixed historical query manifest differs')
    return x['selected']


def save_shot(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
        stream.flush()
        os.fsync(stream.fileno())


def run(out):
    begun = monotonic()
    lock,p,archive,args,decoder,path = history.inputs()
    order = p['order']
    old_q = read(QUERY/'protocol.json')
    query_seal = read(QUERY/'seal.json')
    if (p['startup']!=20 or old_q['cold']!=20 or old_q['query']!=16 or old_q['order']!=order
            or len(query_seal['blocks'])!=18 or sha(QUERY/'protocol.json')!=query_seal['protocol_sha256']):
        raise ValueError('the existing cold20 and exact 16-query EAST360 stream are required')
    (out/'shots').mkdir(parents=True)
    (out/'blocks').mkdir()
    copyfile(__file__,out/'executed.py')
    teaching = probe.precision.delay.HISTORY/'teaching.json'
    if sha(teaching)!=read(teaching.parent/'protocol.json')['teaching_sha256']:
        raise ValueError('fixed source feedback coefficients changed')
    net = args[0]
    initial = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    weights = torch.as_tensor(read(teaching)['weights'],device=net.center.device,dtype=torch.float32)
    protocol = dict(**{k:p[k] for k in history.LOCK},arms=ARMS,query_path=str(QUERY),query_count=16,
        attention=dict(site='first attention block, additive key logit, centered over visible time tokens',
            basis='linear and quadratic age contrast/head',key_logit_l2_cap=LOGIT_RADIUS,
            applied='next windows only; 5 recorded samples wait and 8 sample handoff'),
        diagnostic=dict(issue_stride=lag.STRIDE,horizon=lag.H,states='no labels and clear at shot end'),
        teach='previous authorized block labels fit one signed gain per attention head from ±scouts; no posterior label on a current shot')
    write_new(out/'protocol.json',protocol)
    oracle = io23.read_outcome
    labels,files,prefix,tall = [],[],{},np.zeros(3)
    coverage = []
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('target outcome before query commit')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('target truth forbidden during play'))):
        for block in range(1,19):
            ids = checked_selection(block,order,query_seal['blocks'])
            begins = (block-1)*20
            use = teacher(labels,begins+1)
            pending = {}
            blockfiles = []
            for rank,sid in enumerate(order[begins:begins+20],begins+1):
                arc = history.archived_shot(rank,sid,archive)
                t = None
                if rank<=20:
                    n = len(arc['bias_only'])
                    curves = np.broadcast_to(arc['bias_only'],(len(NEW),n))
                    dose = np.zeros(n,np.float32)
                    scouts = None
                    valid = arc['eligible']
                else:
                    if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                        raise ValueError('locked signal changed')
                    raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                    t = raw['times']
                    sig = lag.prepare_actual(raw,rank,lock['mapping'])
                    track,d,valid = measured_directions(sig,args,decoder,path,weights,arc)
                    curves,scouts,dose,err = render(sig,net,track,d,valid,arc,use,witness=(sid in ids))
                    tall = np.maximum(tall,err)
                    if rank in (21,120,360):
                        n = min(550,len(t))
                        keep = arc['release']<n
                        short = dict(release=arc['release'][keep],eligible=arc['eligible'][keep],
                                     **{k:arc[k][:n] for k in ('bias_only','dose')})
                        ss = {k:sig[k][:n] for k in ('values','available')}
                        tr,dd,vv = measured_directions(ss,args,decoder,path,weights,short)
                        replay,_,a,_ = render(ss,net,tr,dd,vv,short,use)
                        np.testing.assert_allclose(a,dose[:n],atol=3e-6,rtol=0.)
                        probe.check_thresholds(replay,curves[:,:n],tuple(p['calibration'].values()))
                        prefix[str(sid)] = dict(points=n,probability_error=float(abs(replay-curves[:,:n]).max()))
                name = f'{rank:04d}_{sid}.npz'
                dst = out/'shots'/name
                save_shot(dst,dict(**{k:arc[k] for k in ARMS[:4]},**dict(zip(NEW,curves,strict=True)),dose=dose))
                files.append(name)
                blockfiles.append(name)
                coverage.append(dict(shot=sid,rank=rank,feedback=int(valid.sum()),writes=int((dose>0.).sum()),
                    history=bool(use.any()),prior_queried=len(labels)))
                if sid in ids:
                    pending[sid]=(t,arc['bias_only'],scouts)
            commit = out/'blocks'/f'{block:02d}_commit.json'
            for directory in (out/'shots',out/'blocks'):
                fd = os.open(directory,os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            write_new(commit,dict(block=block,rank_end=begins+20,selected=ids,curves=blockfiles,
                                  committed_ns=monotonic_ns()))
            fd = os.open(commit.parent,os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            reveal = reveal_block(commit,ids,oracle)
            write_new(out/'blocks'/f'{block:02d}_reveals.json',reveal)
            records = []
            for item in reveal['outcomes']:
                sid = item['shot']
                clock,parent,scouts = pending[sid]
                # Cold20 outcomes are already in the locked parent. Do not
                # fit our new policy on uncomputed counterfactual curves.
                if scouts is None:
                    continue
                y,down = item['disruptive'],item['down']
                x = np.concatenate((np.eye(HEADS),-np.eye(HEADS)))
                effect = np.array([preference(clock,parent,v,y,down,p['calibration']['0.2'])
                                   for v in scouts])
                labels.append((first_use(block),y,x,effect))
                records.append(dict(shot=sid,available_from=first_use(block),disruptive=y,
                                    preference=effect.tolist()))
            write_new(out/'blocks'/f'{block:02d}_learning.json',dict(used=use.tolist(),query_effect=records))
            if block in (1,3,6,10,14,18):
                print(dict(stage='attention_sealed',shots=begins+20,feedback=len(labels),
                           seconds=round(monotonic()-begun,2)),flush=True)
    for k,v in net.state_dict().items():
        np.testing.assert_array_equal(v.detach().cpu().numpy(),initial[k].numpy())
    write_new(out/'seal.json',dict(order=order,curves=files,prefix=prefix,coverage=coverage,
        maxima=tall.tolist(),seconds=monotonic()-begun))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    old = read(history.ARCHIVE/'per_shot.json')
    names = [f'{rank:04d}_{sid}.npz' for rank,sid in enumerate(p['order'],1)]
    if len(names)!=360 or names!=s['curves'] or p['order']!=s['order']:
        raise ValueError('complete same EAST stream must be sealed')
    rows = {band:{k:[] for k in ARMS} for band in p['calibration']}
    for rank,(sid,name) in enumerate(zip(p['order'],names,strict=True),1):
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as stored:
            v = dict(stored)
        untouched = v['dose']<=0.
        for k in NEW:
            np.testing.assert_array_equal(v[k][untouched],v['bias_only'][untouched])
        for b,tau in p['calibration'].items():
            if rank<=p['startup']:
                tau = p['original']
            for k in ARMS:
                a = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,v[k],y,down,tau))
                if k in ARMS[:4] and a!=old[b][k][rank-1]:
                    raise AssertionError('unchanged reference first alarm differs')
                rows[b][k].append(a)
    stats = {b:{k:lead.summarize(v) for k,v in a.items()} for b,a in rows.items()}
    pairs = (('bias_only','attention_current'),('attention_earlier','attention_current'),
             ('attention_reversed','attention_current'),('current_5','attention_current'),
             ('bias_only','inter_only'),('inter_only','joint_current'),
             ('joint_earlier','joint_current'),('joint_reversed','joint_current'),
             ('attention_current','joint_current'))
    write_new(out/'per_shot.json',rows)
    write_new(out/'result.json',dict(metrics=stats,raw_rescored=360*len(ARMS)*len(rows),
        paired={b:{f'{i}_to_{j}':dict(flips=flips(a[i],a[j]),lead=lead.paired(a[i],a[j])) for i,j in pairs}
                for b,a in rows.items()}))
    for band,a in stats.items():
        print(band,{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in a.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--score',action='store_true')
    p = cli.parse_args()
    (score if p.score else run)(p.output.resolve())
