"""Delayed, per-shot updates to measured-value columns of the EAST embedding."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit
from torch.nn import functional as F

from . import commissioning_history_span_20261114 as history
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot


group, lag, layer = history.group, history.lag, history.layer
OLD = history.OLD
PATCH, WINDOW = history.PATCH, history.WINDOW
OFFSET = lag.H+1+history.WAIT
VALUES = tuple(sorted(c for part in group.TRUSTED for c in part))
ALL_AVAILABLE_VALUES = tuple(i for i,k in enumerate(io23.RAW23) if k!='MNM')
TARGETS = tuple(lag.CHANNELS[c] for c in group.USE)
MODES = ('value_current', 'value_earlier', 'value_reversed',
         'protected_current', 'protected_earlier', 'protected_reversed')
ARMS = (*OLD, 'legacy_common', *MODES)
CASES = (90874, 74970, 81406, 70320, 85422, 73743)
EPS = 1e-8


def measured_values(patches, values=VALUES):
    lanes = patches.unflatten(-1, (PATCH, 46))
    v = lanes[..., list(values)]
    mask = lanes[..., [23+c for c in values]].bool()
    return v.masked_fill(~mask, 0.).flatten(-2)


def reference_span(patches):
    """A rank-revealing basis in the numerical data coordinate, not a safety label."""
    _, s, vh = torch.linalg.svd(patches.double(), full_matrices=False)
    return vh[s > s.max()*1e-6]


def protect_reference(delta, basis):
    d,v = delta.double(),basis.double()
    return (d-(d@v.mT)@v).to(delta.dtype)


def stable_direction(delta, original):
    d = delta.flatten(-2)
    norm = d.norm(dim=-1,keepdim=True)
    # Projection of a vector entirely in the protected span is a zero update,
    # not a direction obtained by normalizing floating-point cancellation.
    floor = (original.flatten(-2).norm(dim=-1,keepdim=True)*1e-6).clamp_min(EPS)
    return (d/norm.clamp_min(EPS)).masked_fill(norm<=floor,0.).reshape_as(delta)


def feedback_support(q, observed):
    prev = np.arange(len(q))-1
    common = observed & (prev>=0)
    if len(q):
        near = prev.clip(min=0)
        common &= observed[near] & (q[near]+lag.STRIDE==q) & (q[near]+OFFSET<=q)
    return np.where(common,prev,-1),common


def action_clock(q, good, ends):
    r = np.r_[-1,q+OFFSET]
    slot = np.searchsorted(r, ends, side='right')-1
    legal = np.r_[False,good]
    f = group.handoff_fraction(r[slot], ends, PATCH)
    older = legal[(slot-1).clip(min=0)]
    scale = np.where(older,abs(1.-2.*f),f)*legal[slot]
    chosen = np.where(older & (f<.5),slot-1,slot)
    return chosen,scale


def forward_embed(base, x, mask, adjust, delta, *, requested=None, values=VALUES):
    """Equivalent to adding a window-specific matrix to the measured part of W."""
    n, b = delta.shape[:2]
    raw, state, report = [], [], []

    def prepare(_module, args):
        x = adjust(args[0][:b])
        raw.append(measured_values(x,values))
        return (x.repeat(n+1,1,1),)

    def change(_module, _args, out):
        move = torch.einsum('abdk,bpk->abpd',delta,raw[0])
        if requested is not None:
            radius = move.flatten(2).norm(dim=2)
            wanted = requested*(radius>EPS).all(0)
            move = move*(wanted[None]/radius.clamp_min(EPS))[...,None,None]
            old = out[b:].unflatten(0,(n,b))
            realized = ((old+move)-old).flatten(2).norm(dim=2)
            report.extend((wanted, (realized-wanted).abs().amax()))
        return torch.cat((out[:b],out[b:]+move.flatten(0,1)),dim=0)

    handles = (base.embedding.register_forward_pre_hook(prepare),
               base.embedding.register_forward_hook(change),
               base.risk_head.register_forward_pre_hook(lambda _m,a:state.append(a[0])))
    try:
        base(x.repeat(n+1,1,1),mask.repeat(n+1,1,1))
    finally:
        for h in handles:
            h.remove()
    return state[0].unflatten(0,(n+1,b)),report


def ref_values(base, signal):
    x = torch.as_tensor(signal['values'][:WINDOW], device=base.center.device)
    ok = torch.as_tensor(signal['available'][:WINDOW], device=x.device) & torch.isfinite(x)
    x = torch.where(ok,x,base.center)
    x = ((x-base.center)/base.scale).asinh()
    return x[:, list(VALUES)].reshape(-1,PATCH*len(VALUES))


def waveform(prediction, center, spread):
    """Full 25-sample evolution; remove each channel's packet mean."""
    scale = spread[:, list(group.USE)]
    origin = center[:, list(group.USE)]
    v = ((prediction[..., list(group.USE)]-origin[:, None])/scale[:, None]).asinh()
    return v-v.mean(dim=-2,keepdim=True)


def decode(hidden, prior, fields, decoder, index):
    recent, center, spread, noise = fields
    wave = lag._predict(hidden,prior[index],decoder)*noise+recent[index,None]
    return waveform(wave,center[index],spread[index])


def parameters(base, signal, track, fields, truth, decoder, index):
    q, _, z, _beta, prior, phase, correction = track
    endpoints = q[index]
    slots = lag.delayed_slots(q,endpoints)[0]
    x,m = lag._windows(base,signal,endpoints)
    d = z.new_zeros((len(index),base.cfg.dim,PATCH*len(VALUES)),requires_grad=True)
    hidden, _ = forward_embed(base,x,m,layer.patcher(phase[slots],correction[slots]),d[None])
    pred = decode(hidden[1],prior,fields,decoder,index)
    observed = waveform(truth[index],fields[1][index],fields[2][index])
    dl = ((observed-pred.detach())/group.BETA).clamp(-1.,1.)
    dr = ((observed.flip(1)-pred.detach())/group.BETA).clamp(-1.,1.)
    # These are negative Huber gradients for the same issued forecast and
    # the observed or reversed target, respectively; each example owns W.
    cur, = torch.autograd.grad(pred,d,dl,retain_graph=True)
    alt, = torch.autograd.grad(pred,d,dr)
    return cur,alt


def estimate(signal, args, decoder, path, arc):
    base = args[0]
    track = group.parent_track(signal,args,decoder,path)
    q,_,z,*_ = track
    np.testing.assert_array_equal(q+lag.H+1, arc['release'])
    good, fields, truth = group.feedback_waveform(signal,q,args[2],z.device)
    raw = signal['available'][q[:,None]+np.arange(1,lag.H+1)][:,:,list(TARGETS)]
    observed = good & raw.all((1,2))
    old, eligible = feedback_support(q,observed)
    eligible &= arc['eligible']
    shape = (len(q)+1,base.cfg.dim,PATCH*len(VALUES))
    now, rev = (z.new_zeros(shape) for _ in range(2))
    selected = np.flatnonzero(eligible)
    used = np.unique(np.r_[selected,old[selected]])
    for start in range(0,len(used),group.BATCH):
        ix = used[start:start+group.BATCH]
        now[ix+1],rev[ix+1] = parameters(base,signal,track,fields,truth,decoder,ix)
    prior = now[np.r_[0, old+1]]
    if len(used):
        ref = ref_values(base,signal)
        basis = reference_span(ref)
    else:
        ref = z.new_zeros((0,shape[-1]))
        basis = ref
    local = torch.stack((now,prior,rev))
    constrained = protect_reference(local,basis)
    mass = local.flatten(2).norm(dim=2)
    remaining = constrained.flatten(2).norm(dim=2)
    ratio = (remaining/mass.clamp_min(EPS)).cpu().numpy()
    # Slot 0 is always a no-op. Ineligible rows are never consumed as states,
    # but their issued gradients can be the preceding observed comparison.
    updates = torch.cat((stable_direction(local,local),stable_direction(constrained,local)))
    preserve = float((updates[3:]@ref.mT).abs().max()) if ref.numel() else 0.
    if preserve>2e-5 or not torch.isfinite(updates).all():
        raise ValueError(f'invalid measured embedding or early-reference protection: {preserve}')
    return track, updates, eligible, ratio, preserve, int(len(basis)), fields, truth


@torch.no_grad()
def render(base, signal, track, weights, eligible, arc, *, modes=MODES, values=VALUES):
    q, h, z, beta, _prior, phase, correction = track
    if len(modes)!=len(weights):
        raise ValueError('one name required for each embedding update arm')
    eligible_ends = np.flatnonzero(arc['dose']>0.)
    slot,schedule = action_clock(q,eligible,eligible_ends)
    selected = schedule>0.
    parent = arc['bias_only']
    curves = np.broadcast_to(parent,(len(modes),len(parent))).copy()
    legacy = parent.copy()
    realized = np.zeros_like(parent)
    pe = de = 0.
    for start in range(0, int(selected.sum()), group.BATCH):
        sl = slice(start,start+group.BATCH)
        t,s = eligible_ends[selected][sl],slot[selected][sl]
        x,m = lag._windows(base,signal,t)
        j = lag.delayed_slots(q,t)[0]
        cap = z.new_tensor((arc['dose'][eligible_ends]*schedule)[selected][sl])
        v,(mass,error) = forward_embed(base,x,m,layer.patcher(phase[j],correction[j]),
                                      weights[:,s],requested=cap,values=values)
        prob = (z[t]+((v-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        pe = max(pe,float(abs(prob[0]-parent[t]).max()))
        de = max(de,float(error))
        valid = mass.cpu().numpy()>0.
        curves[:,t[valid]] = prob[1:,valid]
        legacy[t[valid]] = arc['current_5'][t[valid]]
        realized[t] = mass.cpu().numpy()
    if pe>3e-6 or de>1e-5 or not np.isfinite(curves).all():
        raise ValueError(f'parent or common embedding dose failure: {pe:g}, {de:g}')
    return dict(legacy_common=legacy,**dict(zip(modes,curves,strict=True))),realized,(pe,de)


@torch.no_grad()
def witness(base, signal, track, weights, eligible, fields, truth, decoder, spent):
    """Evaluate the same action on a later disjoint H25; never feeds selection."""
    q,_,z,_,prior,phase,correction = track
    from_slot = np.flatnonzero(eligible)
    k = from_slot+2  # Two original issue intervals, with no target overlap.
    keep = k<len(q)
    i,k = from_slot[keep],k[keep]
    arrival = q[i]+OFFSET+PATCH
    keep = ((q[k]==q[i]+2*lag.STRIDE) & (spent[arrival]>0.) &
         signal['available'][q[k,None]+np.arange(1,lag.H+1)][:,:,list(TARGETS)].all((1,2)))
    i,k,arrival = i[keep],k[keep],arrival[keep]
    parts = []
    for start in range(0,len(i),group.BATCH):
        sl = slice(start,start+group.BATCH)
        a,j = i[sl],k[sl]
        x,m = lag._windows(base,signal,q[j])
        slots = lag.delayed_slots(q,q[j])[0]
        cap = z.new_tensor(spent[arrival[sl]])
        h,(mass,_) = forward_embed(base,x,m,layer.patcher(phase[slots],correction[slots]),
                                    weights[:,a+1],requested=cap)
        actual = waveform(truth[j],fields[1][j],fields[2][j])
        fore = torch.stack([decode(v,prior,fields,decoder,j) for v in h])
        # All predictions are made at j with the same, previously chosen
        # state. Persisting a constant equals zero in this centered target.
        fore = torch.cat((fore,torch.zeros_like(fore[:1])),dim=0)
        error = F.smooth_l1_loss(fore,actual.expand_as(fore),beta=group.BETA,
                                reduction='none').mean(dim=(-2,-1))
        parts.append(np.c_[a,j,mass.cpu().numpy(),error.T.cpu().numpy()])
    frames = np.concatenate(parts) if parts else np.empty((0,len(MODES)+5))
    frames = frames[frames[:,2]>0.]
    i,k = frames[:,:2].astype(np.int64).T
    clock = np.c_[q[i], q[i]+OFFSET+PATCH,q[k],q[k]+lag.H+1]
    return clock,frames[:,3:].astype(np.float32)


def track_shot(signal,args,decoder,path,arc,*,follow):
    base = args[0]
    track, state, good, ratio, hold_err, rk, fields, truth = estimate(signal,args,decoder,path,arc)
    curves,spent,err = render(base,signal,track,state,good,arc)
    detail = dict(dose=spent,q=track[0],qualified=good,reference_rank=np.array(rk),
                  protected_gradient_fraction=ratio)
    if follow:
        clock,error = witness(base,signal,track,state,good,fields,truth,decoder,spent)
        detail.update(witness_clock=clock,witness_loss=error)
    return curves,detail,(*err,hold_err)


def seal(out, *, only_ranks=None):
    start = monotonic()
    lock,p,s,args,decoder,path = history.inputs()
    order = p['order']
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},arms=ARMS,only=only_ranks,
        feedback='full H25, 25 per-channel centered observations, signed Huber gradient',
        actions=dict(value_names=[io23.RAW23[k] for k in VALUES],site='embedding value matrix',
                     target_names=[io23.RAW23[k] for k in TARGETS],ref_points=WINDOW,
                     source='locked source, not trained archival or any new risk readout'),
        schedule=dict(offset=OFFSET, handoff=PATCH, age_control='immediate previous complete packet',
                      budget='archived pointwise dose times a shared handoff, independent of arm sign',
                      until='shot end, no persisted state'), labels='none until the completed stream is saved'))
    base = args[0]
    frozen = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    names,prefix,max_error = [],{},np.zeros(3)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no current shot label')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('no preseal ground truth'))):
        for rank,sid in enumerate(order,1):
            if only_ranks is not None and rank not in only_ranks:
                continue
            arc = history.archived_shot(rank,sid,s)
            if rank>p['startup']:
                # Reuse the binding already required by the archived stream.
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                change,d,err = track_shot(sig,args,decoder,path,arc,follow=True)
                max_error = np.maximum(max_error,err)
                if rank in (21,120,360) or sid in CASES or only_ranks is not None:
                    n = min(len(raw['times']),650)
                    take = arc['release']<n
                    sub = {k:(v[take] if k in ('release','eligible') else v[:n]) for k,v in arc.items()}
                    short = {k:v[:n] for k,v in sig.items() if k in ('values','available')}
                    short_curves,new,_ = track_shot(short,args,decoder,path,sub,follow=False)
                    np.testing.assert_allclose(new['dose'],d['dose'][:n],atol=2e-6,rtol=0.)
                    prefix[str(sid)] = dict(end=n,error=max(float(abs(v-change[k][:n]).max())
                                                           for k,v in short_curves.items()))
                    for k,v in short_curves.items():
                        np.testing.assert_allclose(v,change[k][:n],atol=3e-6,rtol=0.)
                        for t in p['calibration'].values():
                            np.testing.assert_array_equal(v>=t,change[k][:n]>=t)
            else:
                v = arc['bias_only']
                change = dict.fromkeys(ARMS[len(OLD):],v)
                d = dict(dose=np.zeros_like(v),q=np.empty(0,np.int64),qualified=np.empty(0,bool),
                         reference_rank=np.array(0),protected_gradient_fraction=np.empty((3,0)),
                         witness_clock=np.empty((0,4),np.int64),witness_loss=np.empty((0,len(MODES)+2)))
            name = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f, **{k:arc[k] for k in OLD}, **change, **d)
            names.append(name)
            if only_ranks is not None or rank in (20,60,120,180,240,300,360):
                print(dict(stage='measured_embedding_sealed',rank=rank,
                           seconds=round(monotonic()-start,2)),flush=True)
    for k,v in base.state_dict().items():
        if not torch.equal(v.detach().cpu(),frozen[k]):
            raise ValueError(f'persisted source parameter changed: {k}')
    write(out/'seal.json',dict(shots=names, prefix=prefix, max_error=max_error.tolist(),
                               seconds=monotonic()-start))


def metrics(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    order = p['order']
    if p['only'] is not None or s['shots']!=[f'{r:04d}_{v}.npz' for r,v in enumerate(order,1)] or len(order)!=360:
        raise ValueError('full replay required for scores')
    paired_old = read(history.ARCHIVE/'per_shot.json')
    tables = {c:{k:[] for k in ARMS} for c in p['calibration']}
    bands = {k:[] for k in ARMS}
    witness_records = {tag:[] for tag in ('all','late_before_decision','never_before_decision','safe_pre_fp')}
    coverage,cold_fp = [],0
    hard = []
    tau = p['calibration']['0.2']
    for rank,(sid,name) in enumerate(zip(order,s['shots'],strict=True),1):
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            z = dict(f)
        active = z['dose']>0.
        if rank<=p['startup'] and not y and (z['bias_only']>=p['original']).any():
            cold_fp += 1
        for a in ARMS:
            if a not in OLD:
                np.testing.assert_array_equal(z[a][~active],z['bias_only'][~active])
            for b,tau0 in p['calibration'].items():
                row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,z[a],y,down,
                            p['original'] if rank<=p['startup'] else tau0))
                if a in OLD and row!=paired_old[b][a][rank-1]:
                    raise ValueError('same-parent denominator or reference changed')
                tables[b][a].append(row)
            if rank>p['startup']:
                bands[a].append(dict(disruptive=y,peak=peak_for_shot(t,z[a],y,down)))
        clocks = z['witness_clock']
        def collect(name, where):
            if where.any():
                witness_records[name].append(z['witness_loss'][where])
        collect('all',np.ones(len(clocks),bool))
        rr = tables['0.2']['current_5'][-1]
        if rank>p['startup'] and y and not rr['timely']:
            d = np.round((down-t)*1e3,9)
            near = (d>30.) & (d<=125.)
            near_action = near & active
            prior = logit(np.clip(z['bias_only'][near_action],1e-6,1.-1e-6))
            hard.append(dict(shot=sid,kind='late' if rr['alarm'] else 'never',support=int(near_action.sum()),
                gap=float(logit(tau)-prior.max()) if len(prior) else None,
                uplift={k:float((logit(z[k][near_action].clip(1e-6,1.-1e-6))-prior).max())
                        for k in MODES} if len(prior) else {},
                near_hit={k:bool((z[k][near_action]>=tau).any()) for k in MODES}))
            collect('late_before_decision' if rr['alarm'] else 'never_before_decision',
                    near[clocks[:,1]] & (d[clocks[:,3]]>30.))
        elif rank>p['startup'] and not y and tables['0.2']['bias_only'][-1]['alarm']:
            first = int(np.flatnonzero(z['bias_only']>=tau)[0])
            collect('safe_pre_fp', clocks[:,3]<first)
        relfrac = z['protected_gradient_fraction'][0,1:][z['qualified']]
        coverage.append(dict(shot=sid,active=int(active.sum()),updates=int(z['qualified'].sum()),
                             rank_of_reference=int(z['reference_rank']),
                             protected_fraction=float(np.median(relfrac)) if len(relfrac) else None))
    score = {c:{k:lead.summarize(v) for k,v in vv.items()} for c,vv in tables.items()}
    want = [('bias_only', 'value_current'),('bias_only','protected_current'),
            ('value_earlier','value_current'),('value_reversed','value_current'),
            ('protected_earlier','protected_current'),('protected_reversed','protected_current'),
            ('value_current','protected_current'),('current_5','protected_current'),
            ('legacy_common','protected_current'),('legacy_common','value_current')]
    compare = {c:{f'{a}__to__{b}':dict(flips=flips(v[a],v[b]),lead=lead.paired(v[a],v[b]))
                  for a,b in want} for c,v in tables.items()}
    selected = {str(n):{k:{j:a[j] for j in ('tp','fp')} for k,v in bands.items()
                     for a in [best_at_count(v,n-cold_fp)]} for n in (18,27,36)}
    report = {}
    for g,v in witness_records.items():
        report[g] = dict(shots=len(v),packets=sum(map(len,v)),huber=None)
        if v:
            eq_shot = np.stack([a.mean(0,dtype=np.float64) for a in v])
            report[g]['huber'] = dict(zip(('parent',*MODES,'hold'),eq_shot.mean(0).tolist(),strict=True))
    write(out/'per_shot.json',tables)
    write(out/'result.json',dict(metrics=score,paired=compare,raw_verified=360*len(ARMS)*len(score),
           coverage=coverage,hard=hard,followup=report,suffix_ranking_only=selected,cold_false=cold_fp))
    print({c:{k:(row['by_deadline']['30']['tp'],row['fp']) for k,row in v.items()} for c,v in score.items()},
          flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    what = parser.add_mutually_exclusive_group()
    what.add_argument('--score',action='store_true')
    what.add_argument('--smoke-rank',type=int,action='append')
    cfg = parser.parse_args()
    if cfg.score:
        metrics(cfg.output.resolve())
    else:
        seal(cfg.output.resolve(),only_ranks=cfg.smoke_rank)
