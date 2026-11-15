"""EAST within-shot split release: pending estimates for inference, observations for learning."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_observation_clock_20261114 as clock
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_pending_view_20261114 import pending_views
from .commissioning_seed_replay_20261111 import capacity
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .paper131_alarm_audit import alarm_endpoints


group, lag, layer = clock.group, clock.lag, clock.layer
WAIT, AGE = clock.WAIT, clock.DELAY
REFERENCE = io23.ROOT/'outputs/east360_observation_clock_20261114_v2'
MATURE = lag.H+1
GRADIENT_FLOOR = 1e-2
PREFIX_DOSE_TOL = 1.5e-4
BASELINES = ('fixed','bias_only','current_5','observation_parent',
             'observation_current','observation_earlier')
COPIED = tuple(k for k in BASELINES if k!='observation_parent')
VIEWS = ('masked','center','carry')
LEARN_VIEWS = ('masked','carry')
METHODS = ('synchronous','asynchronous')
FEEDBACKS = ('latest','earlier')
WRITES = tuple(f'{v}_{m}_{f}' for v in LEARN_VIEWS for m in METHODS for f in FEEDBACKS)
EARLY = tuple(f'{v}_early_only_{f}' for v in LEARN_VIEWS for f in FEEDBACKS)
ARMS = (*BASELINES, 'center_parent', 'carry_parent', *WRITES, *EARLY)
CHECK_IDS = (70320, 73405, 70470, 80746, 71076)


def release_schedule(q, age, *, asynchronous):
    extra = np.full(4, age, dtype=np.int64)
    if asynchronous:
        extra[[0, 1, 3]] = 0
    return q[:,None]+MATURE+extra


def packet_support(q, observed, anchor_known, release):
    """Per-family physical target support. A pending family cannot gate another."""
    ids = np.minimum(q[:,None]+np.arange(1,lag.H+1),len(observed)-1)
    eligible = observed[ids].all(1) & anchor_known
    return (release<len(observed)) & np.column_stack(
        [eligible[:,cols].all(1) for cols in group.POSITIONS])


def fixed_batches(indices, width):
    """Deterministic encoder/backward batch shape with independent repeated rows."""
    for j in range(0,len(indices),width):
        chunk = indices[j:j+width]
        yield np.pad(chunk,(0,width-len(chunk)),mode='edge'),len(chunk)


def blended_state(start, table, ends, width):
    indices = np.column_stack([np.searchsorted(start[:,g],ends,side='right') for g in range(4)])
    prev = np.maximum(indices-1,0)
    at = np.take_along_axis(np.pad(start,((1,0),(0,0)),constant_values=-1),indices,axis=0)
    alpha = table.new_tensor(group.handoff_fraction(at,ends[:,None],width))
    a,b = (torch.stack([table[:,ids[:,g],:,g] for g in range(4)],dim=-1) for ids in (prev,indices))
    return torch.lerp(a,b,alpha[None,:,None,:])


def normalize_direction(gradient):
    """Keep small nearly-cancelled derivatives small in the unit step budget."""
    return gradient/torch.linalg.vector_norm(gradient,dim=-1,keepdim=True).clamp_min(GRADIENT_FLOOR)


def risk_inputs(base,values,arrived,ends,age):
    # Both alternatives have the same encoder-view mask; only the values in
    # pending slots differ. The raw observation mask is still passed separately.
    carry,m,center = pending_views(values.detach().cpu().numpy(),arrived.cpu().numpy(),
                                   ends,clock.RADIATION,age,base.center.cpu().numpy())
    return values, torch.as_tensor(m,device=values.device), values.new_tensor(center), values.new_tensor(carry)


def forward_views(base, x, observed, views, masks, patcher, writes, max_norm=None, *, common_count=None):
    """Keep the base direction and both non-neural and neural paths identifiable.

    A view may estimate pending values for the encoder, but all token write
    vectors and the bound use the unestimated observed array only.
    """
    b = len(x)
    actual_count = 1+len(views)
    inputs = (x,*views)
    validity = (observed,*masks)
    expand = tuple(i for i,s in writes for _ in range(len(s)))
    a = torch.cat((*inputs,*(inputs[i] for i in expand)),dim=0)
    m = torch.cat((*validity,*(validity[i] for i in expand)),dim=0)
    hidden, geometry = [], []
    dose = x.new_zeros(b)
    error = x.new_zeros(())

    def prepare(_module,args):
        batch = args[0]
        raw = batch[:b]
        corrected = patcher(raw)
        move = corrected-raw
        if writes:
            cap = F.linear(move,base.embedding.weight).flatten(1).norm(dim=1)
            if max_norm is not None:
                cap = torch.broadcast_to(torch.as_tensor(max_norm,dtype=x.dtype,device=x.device),cap.shape)
            geometry.append((group.group_edges(corrected,base.embedding.weight),cap))
        repeats = len(a)//b-1
        return (torch.cat((corrected,batch[b:]+move.repeat(repeats,1,1))),)

    def inject(_module,args):
        nonlocal dose,error
        edge, cap = geometry[0]
        tokens = args[0]
        d = torch.cat([group.exchange(v,edge) for _,v in writes])
        # Normalize to the minimum proposed finite move and inherited cap.
        # Taking the minimum across both clocks also fixes the actual dose.
        norm = d.flatten(2).norm(dim=-1).reshape(len(writes), -1, b)
        common = norm[:,:common_count].flatten(0,1).amin(0)
        dose = torch.minimum(common,cap).masked_fill(common<=group.EPS,0.)
        nz = (norm>group.EPS).reshape(-1,b)
        target = torch.where(nz,dose,0.)
        r = norm.reshape(-1,b)
        if common_count is not None:
            in_matrix = (torch.arange(norm.shape[1],device=x.device)<common_count).repeat(len(writes))
            # An optional fast-only direction can nearly cancel across its
            # families. Instead of greatly magnifying it to match the full
            # factorial, record its lower finite dose in such a window.
            target = torch.where(in_matrix[:,None],target,torch.minimum(target,r))
        d *= (target/r.clamp_min(group.EPS))[...,None,None]
        if not torch.isfinite(d).all():
            raise ValueError('nonfinite token write')
        n = actual_count*b
        raw = tokens[n:,:-1].reshape_as(d)
        effect = ((raw+d)-raw).flatten(2).norm(dim=-1)
        error = max(float((effect-target).abs().max()),float((effect-cap).clamp_min(0.).max()))
        added = layer.add_to_patches(tokens[n:],d.flatten(0,1))
        return (torch.cat((tokens[:n],added)),*args[1:])

    hooks = [
        base.embedding.register_forward_pre_hook(prepare),
        base.risk_head.register_forward_pre_hook(lambda _m,a: hidden.append(a[0]))]
    if writes:
        hooks.append(base.encoder.layers[0].register_forward_pre_hook(inject))
    try:
        base(a,m)
    finally:
        for h in hooks:
            h.remove()
    return hidden[0].unflatten(0,(-1,b)),dose,float(error)


def family_directions(signal, args, decoder, coefficients, parents, age):
    """One direct issue-time H25 forecast and one derivative per mature family."""
    base,_,p = args[:3]
    h,z,_,_,rel_parent,phase,correction,_,tr = parents
    q = tr['q']
    q = q[q+MATURE<len(h)]
    release = release_schedule(q,age,asynchronous=True)
    values,observed,*_ = lag.direct.weak.shot_coordinates(signal,*p[:2])
    ref = group.recovery.rolling_reference(values,observed)
    recent,count = _window_mean(values,observed,q,-15,16)
    if age:
        recent[:,clock.FEEDBACK],count[:,clock.FEEDBACK] = _window_mean(
            values[:,clock.FEEDBACK],observed[:,clock.FEEDBACK],q,-15,16-age)
    known = ref['reference_known'][q] & (count>=8)
    eligible = packet_support(q,observed,known,release)
    device = z.device
    edges = base.cfg.window//base.cfg.patch-1
    states = z.new_zeros((2,len(q)+1,edges,4))
    support = np.zeros_like(eligible)
    packet_loss = np.full((len(q),4),np.nan,dtype=np.float32)
    prior = torch.as_tensor(lag.direct.base_forecast(tr,p[2])[:len(q)],device=device)
    noise = torch.as_tensor(p[3],device=device)
    recent, cen, std = (torch.as_tensor(x,dtype=z.dtype,device=device)
                        for x in (recent,ref['center'][q],ref['spread'][q]))

    def at(times):
        ix = np.searchsorted(rel_parent,times,side='right')
        v,m = clock._batch_windows(base,signal,times,age)
        theta = z.new_zeros((len(times),edges,4),requires_grad=True)
        view = group.forward_hidden(base,v,m,layer.patcher(phase[ix],correction[ix]),state=theta)
        return view,theta

    issue = np.flatnonzero(eligible.any(1))
    predictions = z.new_zeros((len(q),lag.H//group.BIN,len(group.USE)))
    hidden = z.new_zeros((len(q),h.shape[1]))

    def forecast(i,issued):
        wave = lag._predict(issued,prior[i],decoder)*noise+recent[i,None]
        return group.signed_levels(wave,cen[i],std[i])

    with torch.no_grad():
        for i,n in fixed_batches(issue,group.BATCH):
            view,_ = at(q[i])
            hidden[i[:n]] = view[:n]
            predictions[i[:n]] = forecast(i,view)[:n]
    for g,columns in enumerate(group.POSITIONS):
        indices = np.flatnonzero(eligible[:,g])
        c = np.asarray([group.USE.index(i) for i in columns],dtype=np.int64)
        residuals = z.new_zeros(predictions.shape)
        if len(indices):
            at_time = q[indices,None]+np.arange(1,lag.H+1)
            wave = z.new_tensor(values[at_time][...,columns])
            y = ((wave.unflatten(1,(-1,group.BIN)).mean(2)-cen[indices][:,None,columns])/
                 std[indices][:,None,columns]).asinh()
            pred = predictions[indices][:,:,c]
            error = y-pred
            packet_loss[indices,g] = error.abs().mean((1,2)).cpu().numpy()
            residuals[indices[:,None,None],np.arange(error.shape[1])[None,:,None],c[None,None,:]] = (
                error/group.BETA).clamp(-1.,1.)
        _,now,old,good = clock.innovation(q,release[:,g],residuals,eligible[:,g])
        support[:,g] = good
        idx = np.flatnonzero(good)
        for i,n in fixed_batches(idx,group.BATCH):
            view,theta = at(release[i,g])
            issued = hidden[i].detach().requires_grad_(True)
            pred = forecast(i,issued)
            specs = tuple((pred,group.shape_force(f[i],coefficients)/len(columns)) for f in (now,old))
            v = group.signed_directions(issued,view,theta,specs)[...,g]
            states[:,i[:n]+1,:,g] = normalize_direction(v[:,:n])
    if not torch.isfinite(states).all():
        raise ValueError('invalid family update')
    return q,release,states,support,packet_loss


@torch.no_grad()
def render(signal,args,parents,packs,age):
    base = args[0]
    h,z,beta,_,rel,phase,correction,_,_ = parents
    q,releases,table,support,loss = packs
    sync = release_schedule(q,age,asynchronous=False)
    plain = ('observation_parent','center_parent','carry_parent')
    keys = (*plain, *(a for v in LEARN_VIEWS for a in (*WRITES,*EARLY) if a.startswith(v+'_')))
    curves = {a:np.empty(len(z),np.float32) for a in keys}
    doses = np.zeros(len(z),np.float32)
    raw_estimates = 0
    error = 0.
    for j in range(0,len(z),group.BATCH):
        t = np.arange(j,min(j+group.BATCH,len(z)))
        i = np.searchsorted(rel,t,side='right')
        x,m = clock._batch_windows(base,signal,t,age)
        x,estimated,center,carry = risk_inputs(base,x,m,t,age)
        raw_estimates += int((estimated & ~m).sum())
        directions = tuple(blended_state(r+WAIT,table,t,base.cfg.patch) for r in (sync,releases))
        early = directions[1].clone()
        early[...,2] = 0.
        new = torch.cat((*directions,early),0)
        v,d,e = forward_views(base,x,m,(center,carry),(estimated,estimated),
            layer.patcher(phase[i],correction[i]),((0,new),(2,new)),common_count=4)
        p = (z[t]+((v-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        used = d.cpu().numpy()>0.
        for k, row in zip(keys,p,strict=True):
            curves[k][t] = row if k in plain else np.where(
                used,row,p[0 if k.startswith('masked') else 2])
        doses[t] = d.cpu().numpy()
        error = max(error,e)
    if not all(np.isfinite(x).all() for x in curves.values()) or error>1e-5:
        raise ValueError('finite alarm scores and matched realized L2 required')
    return curves,dict(dose=doses, q=q, releases=releases,
            support=support, packet_loss=loss, estimated_slots=raw_estimates),error


def evaluate(signal,args,decoder,path,teacher):
    parents = clock.prepared_parent(signal,args,decoder,path,AGE,AGE)
    packs = family_directions(signal,args,decoder,teacher,parents,AGE)
    return render(signal,args,parents,packs,AGE)


def check_prefix(signal,args,decoder,path,teacher,curves,trace,thresholds,n):
    part = {k:signal[k][:n] for k in ('values','available')}
    view, st, _ = evaluate(part,args,decoder,path,teacher)
    error = max(clock.prefix_check(p[:n],view[k],thresholds) for k,p in curves.items())
    eligible = trace['releases'][:len(st['q'])]<n
    np.testing.assert_array_equal(st['support'],trace['support'][:len(st['q'])]&eligible)
    np.testing.assert_array_equal(st['dose']>0.,trace['dose'][:n]>0.)
    # The inherited parent rebatched for a shorter prefix is accurate to
    # FP32, but residual-to-family normalization can amplify its last bits.
    # The bound is only for the trace; alarm, packet and nonzero support
    # checks above remain exact, and matched live doses are checked at 1e-5.
    z = trace['dose'][:n]
    np.testing.assert_allclose(st['dose'],z,atol=PREFIX_DOSE_TOL,rtol=0.)
    return dict(max_probability_error=error,max_dose_error=float(abs(st['dose']-z).max()))


def read_previous():
    p,s = (read(REFERENCE/f'{k}.json') for k in ('protocol','seal'))
    if len(s['curves'])!=360 or p['delays_samples']['late_observation']!=AGE:
        raise ValueError('sealed same-age observation replay required')
    return p,s


def seal(out, rank=None):
    now = time.monotonic()
    lock,protocol,_,args,decoder,path = clock.archive.inputs()
    p,old = read_previous()
    keys = clock.archive.LOCK
    if any(p[k]!=protocol[k] for k in keys):
        raise ValueError('source, threshold and stream must match the observation replay')
    asset = clock.HISTORY/'teaching.json'
    if sha(asset)!=read(clock.HISTORY/'protocol.json')['teaching_sha256']:
        raise ValueError('original diagnostic teaching weights changed')
    weights = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    (out/'shots').mkdir(parents=True)
    for source, target in ((__file__,'executed.py'),(clock.__file__,'executed_clock.py'),
                           (Path(__file__).with_name('commissioning_pending_view_20261114.py'),'executed_pending.py')):
        copyfile(source,out/target)
    write(out/'protocol.json',dict(scope='EAST360_PENDING_VIEW_ASYNCHRONOUS_FAMILY_FEEDBACK_DEVELOPMENT',
        **{k:p[k] for k in keys},arms=ARMS,rank_only=rank,reference=str(REFERENCE),
        reference_seal=sha(REFERENCE/'seal.json'),radiation_delay=AGE,
        group_names=[[io23.RAW23[c] for c in cols] for cols in group.TRUSTED],
        timing=dict(first=lag.FIRST,period=lag.STRIDE,direct_forecast=lag.H,
                    early_release=MATURE,radiation_release=MATURE+AGE,write_wait=WAIT,
                    interpolation=args[0].cfg.patch),
        views=dict(masked='only received',center='same expected lanes as carry, source center',
                   carry='take t-8 anchor only, only in the encoder lane; forecast and auxiliary losses use observations'),
        scores='same frozen teacher, backbone and risk coefficient, aligned time; no new outcome inputs',
        training='keep untouched issue-time H25 feature prediction, compute physical family residual against arrived truth, '
                 'freeze already released residual archive at issue; use diagonal diagnostic-group coefficient writes',
        scheduling='share per-family update packets learned on their own true arrival clock; only compare application clock '
                   'q+26 versus q+34 for early groups, then one common +5 point wait and 8-step handoff. '
                   'Radiation slots independent; early-only drops radiation writes but not observed inputs',
        dose='family step normalized above a common 1e-2 gradient floor, maximum inherited token radius '
             'and equal realized displacement within matched '
             'view and timing factorial; optional early-only cannot amplify its smaller proposed move',
        prefix_comparison=dict(probability_atol=clock.PREFIX_PROB_TOL,dose_atol=PREFIX_DOSE_TOL,
                               exact_alarm_and_write_support=True),
        state='first 20 predictions unmodified; fit assets frozen; no new between-shot state'))
    files,witness = {},{}
    th = tuple(p['calibration'].values())
    before = [state_digest(x) for x in args[:2]]
    parent_err = dose_err = 0.
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome before full-stream seal')):
        for i,sid in enumerate(p['order'],1):
            if rank is not None and rank!=i:
                continue
            f = f'{i:04d}_{sid}.npz'
            ref = REFERENCE/'shots'/f
            if sha(ref)!=old['curves'][f]:
                raise ValueError('locked comparison changed')
            with np.load(ref,allow_pickle=False) as arc:
                a = {k:arc[k] for k in BASELINES}
            if i<=p['startup']:
                n = len(a['fixed'])
                curves = dict.fromkeys(ARMS,a['fixed'])
                trace = dict(dose=np.zeros(n,np.float32),q=np.empty(0,np.int64),
                             releases=np.empty((0,4),np.int64),support=np.empty((0,4),bool),
                             packet_loss=np.empty((0,4),np.float32),estimated_slots=0)
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('changed authoritative signal')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,i,lock['mapping'])
                curves, trace, err = evaluate(signal,args,decoder,path,weights)
                for v in th:
                    np.testing.assert_array_equal(curves['observation_parent']>=v,a['observation_parent']>=v)
                np.testing.assert_allclose(curves['observation_parent'],a['observation_parent'],rtol=0.,atol=3e-6)
                parent_err = max(parent_err,float(abs(curves['observation_parent']-a['observation_parent']).max()))
                dose_err = max(dose_err,err)
                if i in (21,120,360) or sid in CHECK_IDS or rank==i:
                    n = min(622,len(raw['times']))
                    witness[str(sid)] = dict(points=n,**check_prefix(
                        signal,args,decoder,path,weights,curves,trace,th,n))
                # Keep the recomputed parent used by its matched write arms.
                curves.update({k:a[k] for k in COPIED})
            dest = out/'shots'/f
            with dest.open('xb') as fd:
                np.savez_compressed(fd,**curves,**trace)
            files[f] = sha(dest)
            if i in (21,60,120,180,240,300,360) or rank:
                print(dict(stage='sealed',rank=i,elapsed=round(time.monotonic()-now,2)),flush=True)
    if before!=[state_digest(x) for x in args[:2]]:
        raise ValueError('persistent source parameters changed')
    write(out/'seal.json',dict(curves=files,prefix=witness,max_parent_error=parent_err,
                              max_realized_dose_error=dose_err,elapsed_s=time.monotonic()-now))


def _maximum(t, scores, disrupted, down):
    d = np.round((down-t)*1000,9)>30. if disrupted else np.ones(len(t),bool)
    return float(scores[d].max()) if d.any() else None


def score(out):
    p,s = (read(out/f'{x}.json') for x in ('protocol','seal'))
    if p['rank_only'] is not None or len(s['curves'])!=360 or p['reference_seal']!=sha(REFERENCE/'seal.json'):
        raise ValueError('complete sealed stream required before HDF5 evaluation')
    old = read(REFERENCE/'per_shot.json')
    bands = p['calibration']
    per = {b:{a:[] for a in ARMS} for b in bands}
    peak = {a:[] for a in ARMS}
    summary = {}
    for rank,sid in enumerate(p['order'],1):
        file = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(file)!=s['curves'][file.name]:
            raise ValueError('sealed shot curve changed')
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(file,allow_pickle=False) as z:
            actual = z['dose']>0.
            if not len(t)==len(actual):
                raise ValueError('recorded physical time disagrees with seal')
            starts = z['releases']+WAIT
            known = z['support']
            for a in (*WRITES,*EARLY):
                parent = 'carry_parent' if a.startswith('carry') else 'observation_parent'
                x,v = z[a][~actual],z[parent][~actual]
                if not np.array_equal(x,v):
                    # The v2 seal copied the matched observation parent from
                    # its verified prior replay. Its FP32 re-evaluation was
                    # already checked at seal time; require both an error
                    # bound in those same units and identical alert decisions.
                    np.testing.assert_allclose(x,v,atol=3e-6,rtol=0.)
                    for tau in bands.values():
                        np.testing.assert_array_equal(x>=tau,v>=tau)
            # Count shared writes in the interval when only the early route
            # can use the new fast-family residual.
            wrote = np.flatnonzero(actual)
            window = np.searchsorted(starts[:,0],wrote,side='right')-1
            if (window<0).any():
                raise ValueError('wrote before a physical packet could reach the update')
            advanced = (wrote<starts[window,2]) & known[window][:,[0,1,3]].any(1)
            summary[str(sid)] = dict(estimated_slots=int(z['estimated_slots']),
                usable_packets=known.sum(0).tolist(),written=int(actual.sum()),split_interval_written=int(advanced.sum()))
            for a in ARMS:
                for b,tau in bands.items():
                    limit = p['original'] if rank<=p['startup'] else tau
                    raw = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,z[a],y,down,limit))
                    if a in BASELINES and raw!=old[b][a][rank-1]:
                        raise ValueError('baseline first alarm inconsistent with raw HDF5')
                    per[b][a].append(raw)
                    if b=='0.2':
                        peak[a].append(dict(raw,maximum=_maximum(t,z[a],y,down)))
    write(out/'per_shot.json',per)
    comparisons = (('observation_parent','center_parent'),('observation_parent','carry_parent'),
                   ('center_parent','carry_parent'),('current_5','carry_parent'),
                   *((f'{"observation" if v=="masked" else "carry"}_parent',f'{v}_{m}_{f}')
                     for v in LEARN_VIEWS for m in METHODS for f in FEEDBACKS),
                   *((f'{v}_synchronous_{f}',f'{v}_asynchronous_{f}') for v in LEARN_VIEWS for f in FEEDBACKS),
                   *((f'{v}_{m}_earlier',f'{v}_{m}_latest') for v in LEARN_VIEWS for m in METHODS),
                   *((f'{v}_early_only_{f}',f'{v}_asynchronous_{f}') for v in LEARN_VIEWS for f in FEEDBACKS))
    m = {b:{a:lead.summarize(rows) for a,rows in x.items()} for b,x in per.items()}
    paired = {b:{f'{a}__{c}':dict(flips=flips(rows[a],rows[c]),lead=lead.paired(rows[a],rows[c]))
                 for a,c in comparisons} for b,rows in per.items()}
    equal_fp = {str(cap):{a:{k:x[k] for k in ('tp','fp')} for a,v in peak.items()
                          if (x:=capacity(v,p['startup'],cap)) is not None} for cap in (.10,.15,.20)}
    write(out/'result.json',dict(status='COMPLETE_ASYNCHRONOUS_EAST360_DEVELOPMENT',n_scored=360*len(ARMS)*len(bands),
        metrics=m,paired=paired, postseal_equal_fp=equal_fp, packet_and_write_coverage=summary))
    print({b:{a:(r['by_deadline']['30']['tp'],r['fp']) for a,r in x.items()}
           for b,x in m.items()},flush=True)


def postseal(out):
    """Locate before-deadline writes and check two-point alarm confirmation."""
    p,s,m,per = (read(out/f'{x}.json') for x in ('protocol','seal','result','per_shot'))
    if len(s['curves'])!=360 or m['n_scored']!=360*len(ARMS)*len(p['calibration']):
        raise ValueError('complete scored replay required')
    arms = ('current_5','observation_parent','observation_current','center_parent','carry_parent',
            'carry_synchronous_latest','carry_asynchronous_latest','carry_asynchronous_earlier',
            'carry_early_only_latest')
    two = {b:{a:[] for a in arms} for b in p['calibration']}
    delta = []
    late = {}
    for rank,sid in enumerate(p['order'],1):
        f = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=s['curves'][f.name]:
            raise ValueError('sealed paired comparison has changed')
        y,down,t = group.hdf._raw_truth(sid)
        ld = np.round((down-t)*1000,9) if y else np.full(len(t),np.inf)
        valid = ld>=-1e-6
        with np.load(f,allow_pickle=False) as z:
            for b,tau in p['calibration'].items():
                tau = float(np.float32(p['original'] if rank<=p['startup'] else tau))
                for a in arms:
                    point = alarm_endpoints(t[valid],z[a][valid],tau,points=2)
                    advance = None if point is None or not y else float(ld[valid][point[1]])
                    two[b][a].append(dict(shot=sid,disruptive=y,alarm=point is not None,lead_ms=advance,
                                          timely=bool(advance is not None and advance>30.)))
            if rank<=p['startup']:
                continue
            a = per['0.2']['carry_synchronous_latest'][rank-1]
            c = per['0.2']['carry_asynchronous_latest'][rank-1]
            if (a['alarm'],a['lead_ms'])!=(c['alarm'],c['lead_ms']):
                tau = p['calibration']['0.2']
                starts = z['releases']+WAIT
                first = {}
                for k in ('carry_parent','carry_synchronous_latest','carry_asynchronous_latest',
                          'carry_asynchronous_earlier','carry_early_only_latest'):
                    hits = np.flatnonzero(valid & (z[k]>=tau))
                    first[k] = None if len(hits)==0 else int(hits[0])
                j = first['carry_asynchronous_latest']
                i = int(np.searchsorted(starts[:,0],j,side='right')-1) if j is not None else -1
                delta.append(dict(shot=sid,disrupted=y,indices=first,
                    full_write_dose=float(z['dose'][j]) if j is not None else None,
                    before_same_window_radiation_write=bool(i>=0 and j<starts[i,2]),
                    old_and_new_score=[float(z[k][j]) for k in ('carry_synchronous_latest',
                        'carry_asynchronous_latest','carry_asynchronous_earlier')] if j is not None else None,
                    synchronous_lead=a['lead_ms'],asynchronous_lead=c['lead_ms']))
            if y and not per['0.2']['carry_parent'][rank-1]['timely']:
                support = (z['dose']>0.)&(ld>30.)&(ld<=125.)
                if support.any():
                    tau = logit(p['calibration']['0.2'])
                    names = ('carry_parent','carry_asynchronous_latest','carry_asynchronous_earlier')
                    parent, latest, earlier = (logit(z[k][support].clip(1e-6,1.-1e-6)) for k in names)
                    late[str(sid)] = dict(n=int(support.sum()),gap=float(tau-parent.max()),
                        latest_lift=float((latest-parent).max()),earlier_lift=float((earlier-parent).max()),
                        latest_cross=bool((latest>=tau).any()),earlier_cross=bool((earlier>=tau).any()))
    detail = dict(two_point={b:{k:lead.summarize(rows) for k,rows in a.items()} for b,a in two.items()},
                  current_vs_sync_different_first_alarms=delta,carry_parent_late_near_feedback=late)
    write(out/'postseal_mechanism.json',detail)
    print({b:{k:(r['by_deadline']['30']['tp'],r['fp']) for k,r in a.items()} for b,a in detail['two_point'].items()},
          flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    ap.add_argument('--rank',type=int)
    stages = ap.add_mutually_exclusive_group()
    stages.add_argument('--postseal',action='store_true')
    stages.add_argument('--score',action='store_true')
    opt = ap.parse_args()
    dest = opt.out.resolve()
    if opt.postseal:
        postseal(dest)
    elif opt.score:
        score(dest)
    else:
        seal(dest,opt.rank)
        if opt.rank is None:
            score(dest)
