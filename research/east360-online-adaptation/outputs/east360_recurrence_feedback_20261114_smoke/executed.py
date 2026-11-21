"""Adapt the internal trajectory using departures from matched same-shot precedents."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_cross_diagnostic_20261113 as cross
from . import commissioning_history_span_20261114 as history
from . import commissioning_recurrence_core_20261114 as core
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_excursion_order_20261111 import disjoint_pairs
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer, precision = cross.group, cross.lag, cross.layer, cross.precision
OLD = history.OLD
NEW = ('network_innovation','matched_innovation','context_innovation','displaced_innovation',
       'earlier_matched_innovation','matched_explained')
ARMS = (*OLD, 'legacy_matched', *NEW)
FOLLOW = ('parent', *NEW)
PREDICTORS = ('hold_motion','network','matched','context','displaced')
CHANNELS = tuple(lag.CHANNELS[k] for k in group.USE)
FAMILY = np.asarray(group.SIGNED_GROUPS)
FAMILIES = np.arange(len(group.TRUSTED))
MATCH_PAST, RECENT_SLOTS = 48, 16
CALIPERS = np.array([.50,.35,.35,.50])
CASES = (70320,73405,79649,71076)


@dataclass(frozen=True)
class Observation:
    key: np.ndarray
    context: np.ndarray
    issue_ok: np.ndarray
    complete: np.ndarray
    location: np.ndarray
    spread: np.ndarray
    target: np.ndarray


def center(a):
    return a-a.mean(axis=-2, keepdims=True) if isinstance(a,np.ndarray) else a-a.mean(dim=-2,keepdim=True)


def window_view(x, mask, ctx, ctx_mask, q):
    """Issue-time keys and realized future targets have distinct validity."""
    before = q[:,None]+np.arange(1-MATCH_PAST,1)
    z = x[before].reshape(len(q),3,16,len(CHANNELS)).mean(axis=2,dtype=np.float64)
    u = ctx[before][:,-32:].reshape(len(q),2,16,2).mean(axis=2,dtype=np.float64)
    context = np.tanh(np.c_[u[:,-1],2.*(u[:,-1,0]-u[:,0,0])])
    key = np.column_stack((context,np.tanh(z[:,-1]),np.tanh(2.*(z[:,-1]-z[:,1])),
                           np.tanh(2.*(z[:,1]-z[:,0]))))
    ref = group.recovery.rolling_reference(x,mask)
    known = ref['reference_known'][q].all(1)
    issue_ok = mask[before].all((1,2))&ctx_mask[before][:,-32:].all((1,2))&known
    location, spread = ref['center'][q], ref['spread'][q]
    complete = issue_ok & (q+lag.H+1<len(x))
    valid = np.flatnonzero(complete)
    future = q[valid,None]+np.arange(1,lag.H+1)
    complete[valid] &= mask[future].all((1,2))
    target = np.zeros((len(q),lag.H//group.BIN,len(CHANNELS)),np.float32)
    waveform = x[future].reshape(len(valid),lag.H//group.BIN,group.BIN,len(CHANNELS)).mean(axis=2)
    standardized = np.arcsinh((waveform-location[valid,None])/spread[valid,None])
    target[valid] = center(standardized)
    return Observation(key,context,issue_ok,complete,location,spread,target)


def transplant(target, chosen):
    """Use only the chosen predecessor's measured future in each held group."""
    result = np.zeros_like(target)
    for g in FAMILIES:
        donor = chosen[g]
        i = np.flatnonzero(donor>=0)
        cols = FAMILY==g
        result[i[...,None],np.arange(target.shape[1]),:] += (
            target[donor[i]]*cols).reshape(len(i),target.shape[1],target.shape[2])
    return result


def previous_issue(q, ok):
    j = np.searchsorted(q,q-lag.STRIDE).clip(max=len(q)-1) if len(q) else q
    valid = ok[j]&(q[j]+lag.STRIDE==q)&(q[j]+lag.H+1<=q)
    return j, valid


def aligned_difference(target, forecast):
    """The part of this observed motion not present in the predicted motion."""
    return target-forecast


def observation_packet(signal, coordinates):
    x, mask, ctx, cm = lag.direct.weak.shot_coordinates(signal,*coordinates)
    q = np.arange(lag.FIRST,len(x),lag.STRIDE,dtype=np.int64)
    v = window_view(x[:,group.USE],mask[:,group.USE],ctx,cm,q)
    identity = np.r_[np.full(3,-1),np.tile(FAMILY,3)]
    per_family = dict(zip(*np.unique(identity,return_counts=True),strict=True))
    weight = np.array([1./per_family[i] for i in identity])
    a,b,c,d,e = core.find_precedents(q,v.key,identity,v.complete,stride=lag.STRIDE,horizon=lag.H,
        recent_slots=RECENT_SLOTS,weights=weight,min_separation=MATCH_PAST+lag.H,query_ok=v.issue_ok)
    sketches = np.stack([transplant(v.target,donor) for donor in (a,b,c)])
    return q,v,(a,b,c),sketches,(d,e)


def finite_track(signal,args,decoder,path,archive):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    base = args[0]
    eligible, fields, wave = group.feedback_waveform(signal,q,args[2],z.device)
    target_shape = group.signed_levels(wave,fields[1],fields[2])
    _,_,_,old_common = group.past_force_parts(q,torch.zeros_like(target_shape),eligible)
    np.testing.assert_array_equal(q+lag.H+1,archive['release'])
    np.testing.assert_array_equal(old_common,archive['eligible'])
    grid,v,donor,memory,distance = observation_packet(signal,args[2][:2])
    ix = np.searchsorted(grid,q)
    np.testing.assert_array_equal(grid[ix],q)
    matched = (donor[0]>=0).all(0)
    usable = v.complete & matched
    before, prior_ok = previous_issue(grid,usable)
    legal = old_common & eligible & usable[ix] & prior_ok[ix]
    # These diagnostics store only already available donors and current labels
    # in different arrays. Future observation data do not enter the lookup.
    y = torch.as_tensor(v.target[ix],device=z.device)
    remembered = torch.as_tensor(memory[:,ix],device=z.device)
    prediction = y.new_zeros(y.shape)
    comparison = np.zeros((len(q),len(PREDICTORS),len(FAMILIES)),np.float32)
    states = z.new_zeros((len(NEW),len(q)+1,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    response = y.new_zeros((len(grid), *y.shape[1:]))
    for start in range(0,len(q),group.BATCH):
        i = np.arange(start,min(start+group.BATCH,len(q)))
        if not len(i):
            continue
        with torch.no_grad():
            issued = encoder_issue(base,signal,q,q[i],phase,correction)
            p = center(cross.decode(issued,history.fields_at(fields,prior,i),decoder))
            prediction[i] = p
            response[ix[i]] = aligned_difference(y[i],remembered[0,i])
            estimates = torch.cat((torch.zeros_like(p)[None],p[None],remembered[:,i]))
            comparison[i] = cross.group_loss(estimates,y[i]).transpose(0,1).cpu().numpy()
    active = np.flatnonzero(legal)
    for start in range(0,len(active),group.BATCH):
        i = active[start:start+group.BATCH]
        issued, now, state, waveform, loc, spread = group.signed_window(
            base,signal,q,i,phase,correction,prior,fields,decoder)
        p = center(group.signed_levels(waveform,loc,spread))
        torch.testing.assert_close(p.detach(),prediction[i],atol=4e-6,rtol=0.)
        torch.testing.assert_close(y[i],center(target_shape[i]),atol=4e-6,rtol=0.)
        # All arms use the same current graph. Only the reference response
        # subtracted from the observation, or its clock slot, changes.
        force = [aligned_difference(y[i],p.detach()),*(aligned_difference(y[i],saved[i]) for saved in remembered),
                 response[before[ix[i]]],remembered[0,i]-p.detach()]
        specs = [(p, (f/group.BETA).clamp(-1.,1.)) for f in force]
        states[:,i+1] = group.signed_directions(issued,now,state,specs)
    legal &= (states[:,1:].flatten(2).norm(dim=-1)>group.EPS).all(0).cpu().numpy()
    trace = q,h,z,beta,phase,correction,states,legal
    receipt = dict(grid=grid,chosen=np.stack(donor),distances=np.stack(distance),issue_ok=v.issue_ok,
                   complete=v.complete,matched=matched,q=q,legal=legal,comparison=comparison,
                   context=v.context[ix])
    return trace, receipt, (fields,prior,y,v.complete[ix])


def encoder_issue(base,signal,q,ends,phase,correction):
    x,m = lag._windows(base,signal,ends)
    k = lag.delayed_slots(q,ends)[0]
    u = x.new_zeros((len(ends),base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    return group.forward_hidden(base,x,m,layer.patcher(phase[k],correction[k]),state=u)


@torch.no_grad()
def held_forecasts(signal,trace, packet,args,decoder,dose):
    q,h,z,beta,phase,correction,states,legal = trace
    fields,prior,y,complete = packet
    start = q+lag.H+1+cross.WAIT
    acted = start+args[0].cfg.patch
    following = q+2*lag.STRIDE
    j = np.searchsorted(q,following).clip(max=max(len(q)-1,0))
    enough = (acted<len(dose)) & legal & (q[j]==following) & complete[j]
    i = np.flatnonzero(enough)
    i = i[dose[acted[i]]>0.]
    out, movement, keep = [],[],[]
    for off in range(0,len(i),group.BATCH):
        k = i[off:off+group.BATCH]
        dest = j[k]
        # Carry only this write; the held target never teaches the action.
        x,m = lag._windows(args[0],signal,q[dest])
        slot = lag.delayed_slots(q,q[dest])[0]
        hidden, actual, error = group.forward_hidden(args[0],x,m,layer.patcher(phase[slot],correction[slot]),
                                                      writes=states[:,k+1])
        if error>1e-5:
            raise ValueError('held forecast budgets must match')
        p = center(cross.decode(hidden,history.fields_at(fields,prior,dest),decoder))
        why,_,delta = cross.risk_trial(signal,q,acted[k],args,h,z,beta,phase,correction,states[:,k+1],
                                       torch.as_tensor(dose[acted[k]],device=z.device))
        if delta>1e-5:
            raise ValueError('action witness differs from common-budget risk forward')
        mask = (actual>0.).cpu().numpy()
        out.append(cross.group_loss(p,y[dest]).transpose(0,1).cpu().numpy()[mask])
        movement.append(why.T.cpu().numpy()[mask])
        keep.append(k[mask])
    if not keep:
        return (np.empty((0,4),np.int64),
                np.empty((0,len(FOLLOW),len(FAMILIES)),np.float32),np.empty((0,len(FOLLOW)),np.float32))
    k = np.concatenate(keep)
    return (np.c_[q[k],acted[k],q[j[k]],q[j[k]]+lag.H+1],np.concatenate(out),np.concatenate(movement))


def predict(signal,args,decoder,path,archive):
    trace, data, packet = finite_track(signal,args,decoder,path,archive)
    q,*_,legal = trace
    parent, old_dose = archive['bias_only'], archive['dose']
    dose = np.zeros_like(old_dose)
    idx = np.flatnonzero(old_dose>0.)
    _, valid = history.action_clock(q,legal,idx)
    dose[idx[valid]] = old_dose[idx[valid]]
    values,pe,de = precision.render(args[0],signal,trace,parent,dose)
    frames,held,risk = held_forecasts(signal,trace,packet,args,decoder,dose)
    legacy = np.where(dose>0.,archive['current_5'],parent)
    values = dict(legacy_matched=legacy,**dict(zip(NEW,values,strict=True)))
    data.update(dose=dose,frames=frames,held=held,action_risk=risk)
    return values,data,(pe,de)


def empty_detail(n):
    return dict(grid=np.empty(0,np.int64),chosen=np.empty((3,len(FAMILIES),0),np.int64),
                distances=np.empty((2,len(FAMILIES),0),np.float64),issue_ok=np.empty(0,bool),
                complete=np.empty(0,bool),matched=np.empty(0,bool),q=np.empty(0,np.int64),
                legal=np.empty(0,bool),comparison=np.empty((0,len(PREDICTORS),len(FAMILIES)),np.float32),
                context=np.empty((0,3),np.float64),
                dose=np.zeros(n,np.float32),frames=np.empty((0,4),np.int64),
                held=np.empty((0,len(FOLLOW),len(FAMILIES)),np.float32),action_risk=np.empty((0,len(FOLLOW)),np.float32))


def seal_run(out,smoke_rank):
    start = time.monotonic()
    lock,p,s,args,decoder,path = history.inputs()
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    copyfile(core.__file__,out/'executed_core.py')
    write(out/'protocol.json',dict(scope='EAST360_SHOT_LOCAL_RECURRENCE_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK},arms=ARMS,forecast_arms=PREDICTORS,holdout_arms=FOLLOW,
        reference=str(history.ARCHIVE),reference_seal=sha(history.ARCHIVE/'seal.json'),
        history=dict(key_samples=MATCH_PAST,earliest_samples=RECENT_SLOTS*lag.STRIDE,
            donor_future_separation=MATCH_PAST+lag.H,release=lag.H+1,
            channels=[io23.RAW23[c] for c in CHANNELS],group_ids=FAMILY.tolist(),
            center='Each prediction/target removes its mean over 5 bins; frozen 125-point local scaling.'),
        updates=dict(wait=cross.WAIT,handoff=args[0].cfg.patch,holdout_issue_delay=2*lag.STRIDE,
            fit='frozen current shot only, complete direct 25-point measurement, no outcomes',
            comparators='same budget: frozen forecast, same-shot other-group match, Ip/Bt only, index displacement, '
                        'previous observed difference, and only the matched predictable component.'),
        smoke_rank=smoke_rank))
    frozen = [state_digest(m) for m in args[:2]]
    files,witnesses,maximum = {},{},np.zeros(2)
    for rank,sid in enumerate(p['order'],1):
        if smoke_rank is not None and rank!=smoke_rank:
            continue
        archive = history.archived_shot(rank,sid,s)
        parent = archive['bias_only']
        if rank<=p['startup']:
            values,data = dict.fromkeys(ARMS[len(OLD):],parent),empty_detail(len(parent))
        else:
            with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
                raw_path = io23.path_for('EAST',sid)
                if sha(raw_path)!=lock['hashes'][str(sid)]:
                    raise ValueError('recorded signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                values,data,error = predict(signal,args,decoder,path,archive)
                maximum = np.maximum(maximum,error)
                if rank in (21,120,360) or sid in CASES or smoke_rank==rank:
                    n = min(690,len(parent))
                    use = archive['release']<n
                    truncation = {k:v[:n] for k,v in signal.items() if k in ('values','available')}
                    clipped = {k:v[:n] for k,v in archive.items() if k not in ('release','eligible')}
                    clipped.update({k:archive[k][use] for k in ('release','eligible')})
                    fresh,details,_ = predict(truncation,args,decoder,path,clipped)
                    for k in fresh:
                        np.testing.assert_allclose(fresh[k],values[k][:n],atol=3e-6,rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(fresh[k]>=tau,values[k][:n]>=tau)
                    # Also compare query decisions whose current truth has not
                    # yet arrived: truncating the suffix must not change them.
                    np.testing.assert_array_equal(details['chosen'],data['chosen'][...,:len(details['grid'])])
                    use = data['frames'][:,-1]<n
                    np.testing.assert_array_equal(details['frames'],data['frames'][use])
                    np.testing.assert_allclose(details['held'],data['held'][use],atol=7e-6,rtol=0.)
                    witnesses[str(sid)] = dict(prefix=n,issued=len(details['grid']),held=len(details['frames']),
                        max_error=max(float(abs(fresh[k]-values[k][:n]).max()) for k in fresh))
        name = f'{rank:04d}_{sid}.npz'
        dest = out/'shots'/name
        with dest.open('xb') as f:
            np.savez_compressed(f,**{k:archive[k] for k in OLD},**values,**data)
        files[name] = sha(dest)
        if rank==21 or rank%60==0 or smoke_rank:
            print(dict(stage='seal_recurrence',rank=rank,active=int((data['dose']>0.).sum()),
                       elapsed_s=round(time.monotonic()-start,2)),flush=True)
    if frozen != [state_digest(m) for m in args[:2]]:
        raise ValueError('model changed across the replay')
    write(out/'seal.json',dict(curves=files,witnesses=witnesses,max_error=maximum.tolist(),
                              elapsed_s=time.monotonic()-start))


def summarize(samples,names):
    rows = [v for v in samples if len(v)]
    if not rows:
        return dict(shots=0,blocks=0)
    values = np.stack([v.mean(axis=0,dtype=np.float64) for v in rows])
    aggregate = values.mean(axis=-1)
    return dict(shots=len(rows),blocks=sum(map(len,rows)),
                names=names,motion_huber_by_group=values.mean(axis=0).tolist(),
                motion_huber=dict(zip(names,aggregate.mean(axis=0).tolist(),strict=True)),
                win_vs_index_0=dict(zip(names,(aggregate<aggregate[:,0:1]).sum(0).tolist(),strict=True)))


def hard_pairs(rows):
    if not rows[0] or not rows[1]:
        return dict(count=0)
    features,ids = ([np.asarray([r[k] for r in side]) for side in rows] for k in ('context','shot'))
    ix = disjoint_pairs(*features,*ids,CALIPERS)
    records = []
    for a,b in ix:
        p,s = rows[0][a], rows[1][b]
        records.append(dict(pair=(p['shot'],s['shot']),action=(p['t'],s['t']),
            forecast_preference=bool(p['gain']<s['gain']),
            gain=(p['gain'],s['gain']),risk=(p['risk'],s['risk']),
            direction_both=bool(p['risk']>0. and s['risk']<0.),
            both_future_beats_controls=bool(p['better'] and s['better'])))
    return dict(count=len(records),columns=('margin','Ip','Bt','Ip change'),calipers=CALIPERS.tolist(),
        forecast_preference=sum(r['forecast_preference'] for r in records),
        both_direction=sum(r['direction_both'] for r in records),
        both_direction_and_future=sum(r['direction_both'] and r['both_future_beats_controls'] for r in records),
        pairs=records)


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if p['smoke_rank'] is not None or len(s['curves'])!=360 or p['reference_seal']!=sha(history.ARCHIVE/'seal.json'):
        raise ValueError('a sealed complete stream is required')
    archive = read(history.ARCHIVE/'per_shot.json')
    rows = {b:{a:[] for a in ARMS} for b in p['calibration']}
    sets = ('all','late_before_deadline','safe_before_false_alarm')
    first, held = ({k:[] for k in sets} for _ in (0,1))
    matched, late, coverage = ([],[]), [], []
    tau = p['calibration']['0.2']
    for rank,sid in enumerate(p['order'],1):
        y,down,t = group.hdf._raw_truth(sid)
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path)!=s['curves'][name]:
            raise ValueError('sealed curve changed')
        with np.load(path,allow_pickle=False) as f:
            active = f['dose']>0.
            for b,threshold in p['calibration'].items():
                for a in ARMS:
                    curve = f[a]
                    if a not in OLD:
                        np.testing.assert_array_equal(curve[~active],f['bias_only'][~active])
                    row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,curve,y,down,
                                 p['original'] if rank<=p['startup'] else threshold))
                    if a in OLD and row!=archive[b][a][rank-1]:
                        raise ValueError('old reference changed')
                    rows[b][a].append(row)
            if rank<=p['startup']:
                continue
            ref = rows['0.2']['current_5'][-1]
            q,grid = f['q'], f['grid']
            match = np.searchsorted(grid,q)
            available = f['complete'][match] & f['matched'][match]
            comparison,frames,future,risk = (f[k] for k in ('comparison','frames','held','action_risk'))
            first['all'].append(comparison[available])
            held['all'].append(future)
            donor = f['chosen']
            age = grid[None,None,:]-grid[donor.clip(min=0)]
            coverage.append(dict(shot=sid,issued=len(grid),complete=int(f['complete'].sum()),
                queried=int(f['issue_ok'].sum()),matched=int(f['matched'].sum()),
                matched_complete=int(available.sum()),legal=int(f['legal'].sum()),active=int(active.sum()),
                match_changed_from_context=int(((donor[0]!=donor[1])&(donor[0]>=0)).sum()),
                donor_age=dict(zip(('matched','context','displaced'),
                    (float(np.median(a[j>=0])) if (j>=0).any() else None
                     for a,j in zip(age,donor,strict=True)),strict=True))))
            if y and not ref['timely']:
                dt = np.round((down-t)*1000.,9)
                near = (dt>30.)&(dt<=125.)
                actions = near&active
                # Exclude a later check that would be revealed too late.
                k = near[frames[:,1]]&(dt[frames[:,-1]]>30.)
                close = available & near[np.minimum(q+lag.H+1,len(t)-1)]
                g, cls = 'late_before_deadline', 0
                if actions.any():
                    b = logit(f['current_5'][actions].clip(1e-6,1.-1e-6))
                    late.append(dict(shot=sid,points=int(actions.sum()),gap=float(logit(tau)-b.max()),
                        max_lift={a:float((logit(f[a][actions].clip(1e-6,1.-1e-6))-b).max()) for a in NEW},
                        crossed={a:bool((f[a][actions]>=np.float32(tau)).any()) for a in NEW}))
            elif not y and ref['alarm']:
                hit = int(np.flatnonzero(f['current_5']>=np.float32(tau))[0])
                k = frames[:,-1]<=hit
                close = available & (q+lag.H+1<=hit)
                g, cls = 'safe_before_false_alarm', 1
            else:
                continue
            first[g].append(comparison[close])
            held[g].append(future[k])
            for q0,t0,z,e in zip(frames[k,0],frames[k,1],risk[k],future[k],strict=True):
                j = np.searchsorted(q,q0)
                f_cost = comparison[j].mean(axis=1)
                c = np.r_[z[0]-logit(tau),f['context'][j]]
                f_cost = np.log1p(f_cost/group.BETA)
                target_d = e.mean(1)
                matched[cls].append(dict(shot=sid,t=int(t0),context=c,
                    gain=float(f_cost[3]-f_cost[2]),risk=float(z[FOLLOW.index('matched_innovation')]-z[0]),
                    better=bool(target_d[FOLLOW.index('matched_innovation')]<min(
                        target_d[FOLLOW.index(a)] for a in ('context_innovation','displaced_innovation')))))
    comparisons = [('legacy_matched','current_5'),('legacy_matched','matched_innovation'),
                   ('bias_only','matched_innovation'),*((k,'matched_innovation') for k in NEW if k!='matched_innovation')]
    paired = {b:{f'{a}__to__{c}':dict(flips=flips(v[a],v[c]),lead=lead.paired(v[a],v[c]))
                 for a,c in comparisons} for b,v in rows.items()}
    diagnostics = {k:dict(at_release=summarize(first[k],PREDICTORS),held=summarize(held[k],FOLLOW))
                   for k in sets}
    metrics = {b:{a:lead.summarize(rr) for a,rr in v.items()} for b,v in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,paired=paired,diagnostics=diagnostics,
        hard=hard_pairs(matched),coverage=coverage,late=late,raw_verified=360*len(ARMS)*len(rows)))
    print({b:{a:(m['by_deadline']['30']['tp'],m['fp']) for a,m in v.items()} for b,v in metrics.items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument('--smoke-rank',type=int)
    mx.add_argument('--score-only',action='store_true')
    opts = ap.parse_args()
    if not opts.score_only:
        seal_run(opts.output.resolve(),opts.smoke_rank)
    if opts.smoke_rank is None:
        score(opts.output.resolve())
