"""Test finite shot-local writes on diagnostic targets omitted from the update."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_feedback_precision_20261113 as precision
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write


group, lag, layer = precision.group, precision.lag, precision.layer
MEMBERS = np.asarray(group.SIGNED_GROUPS)
PAIRS = ((0, 1, 'mag_to_position'), (1, 0, 'position_to_mag'),
         (2, 3, 'radiation_to_voltage'), (3, 2, 'voltage_to_radiation'))
ARMS = ('all_now', 'all_old', *(f'{name}_{age}' for _, _, name in PAIRS for age in ('now', 'old')))
WAIT = precision.WAIT
FOLLOW = 2*lag.STRIDE
LOCK_KEYS = ('order', 'startup', 'original', 'calibration', 'source_sha256')


def directions(pred, state, now, earlier, teacher):
    group_ids = torch.as_tensor(MEMBERS, device=pred.device)
    current, old = (group.shape_force(x, teacher).detach() for x in (now, earlier))
    specs = (current, old, *(x*(group_ids==a) for a, _, _ in PAIRS for x in (current, old)))
    return torch.stack([torch.autograd.grad(pred, state, grad_outputs=x,
                         retain_graph=j<len(specs)-1)[0] for j, x in enumerate(specs)])


def group_loss(pred, truth):
    r = (pred-truth[None]).abs()
    values = torch.where(r<group.BETA, .5*r.square()/group.BETA, r-.5*group.BETA)
    idx = torch.as_tensor(MEMBERS, device=pred.device)
    return torch.stack([values[..., idx==k].mean(dim=(-1,-2)) for k in range(4)], dim=-1)


def relative_gain(before, after):
    """Bound the fractional improvement when the frozen forecast was already accurate."""
    return (before-after)/(before+after+group.BETA)


def follow_window(q, origin, eligible):
    """Choose q+64, the first original issue after the 8-point handoff finishes."""
    if not len(origin):
        return origin, origin
    desired = q[origin]+FOLLOW
    j = np.searchsorted(q, desired)
    safe = np.minimum(j, len(q)-1)
    ok = (j<len(q)) & (q[safe]==desired) & np.asarray(eligible)[safe]
    return origin[ok], j[ok]


def analysis_windows(action, clock, *, down, horizon, deadline, first_alarm_s, dose):
    """Post-seal selection, with no feedback evidence from after the scoring deadline."""
    keep = dose>0.
    if down is not None:
        lead = np.round((down-clock[action])*1000.,9)
        keep &= (lead>deadline)&(lead<=horizon)
    elif first_alarm_s is not None:
        keep &= clock[action]<=first_alarm_s
    return keep


def source_view(raw, view):
    if view == 'native':
        return raw
    available = raw['available'].copy()
    unavailable = ('MNM',) if view=='mask_mnm' else ('MNM','CIII')
    available[:,[io23.RAW23.index(k) for k in unavailable]] = False
    return dict(raw, available=available)


def issued_forecasts(base, signal, q, positions, prior, phase, correction, fields, decoder):
    issue = q[positions]
    parent_slot, _ = lag.delayed_slots(q, issue)
    x, m = lag._windows(base,signal,issue)
    state = x.new_zeros((len(issue), base.cfg.window//base.cfg.patch-1, len(group.TRUSTED)),
                        requires_grad=torch.is_grad_enabled())
    port = layer.patcher(phase[parent_slot], correction[parent_slot])
    h = group.forward_hidden(base, x, m, port, state=state)
    recent, center, spread, noise = fields
    f = (prior[positions], recent[positions], center[positions], spread[positions], noise)
    return state, (x, m, port, f), decode(h, f, decoder)


def decode(hidden, fields, decoder):
    prior, recent, center, spread, noise = fields
    shape = hidden.shape[:-1]
    if hidden.ndim==2:
        wave = lag._predict(hidden, prior, decoder)*noise + recent[:, None]
        return group.signed_levels(wave, center, spread)
    w = lag._predict(hidden.flatten(0,1), prior.repeat(hidden.shape[0],1,1), decoder)
    wave = w*noise + recent.repeat(hidden.shape[0],1)[:, None]
    return group.signed_levels(wave, center.repeat(hidden.shape[0],1),
                               spread.repeat(hidden.shape[0],1)).reshape(*shape, lag.H//group.BIN, len(MEMBERS))


@torch.no_grad()
def forecast_trial(base, spec, directions_, decoder, target):
    """A real finite forward at the issued origin; no score is written retroactively."""
    x, m, port, fields = spec
    h, dose, err = group.forward_hidden(base, x, m, port, writes=directions_)
    pred = decode(h, fields, decoder)
    return group_loss(pred, target), dose, float(err)


@torch.no_grad()
def risk_trial(signal, q, ends, args, h, z, beta, phase, correction, updates, locked_dose):
    x, m = lag._windows(args[0], signal, ends)
    parent_slot, _ = lag.delayed_slots(q, ends)
    # This hook-local budget is set by the actual admissible move at this
    # endpoint. A tiny FP32 difference in a truncated window cannot increase
    # the move beyond either the inherited radius or the sealed target.
    port = layer.patcher(phase[parent_slot], correction[parent_slot])
    base = args[0]
    if locked_dose is None:
        view, dose, err = group.forward_hidden(base,x,m,port,writes=updates)
    else:
        view, dose, err = group.forward_hidden(base,x,m,port,writes=updates,target_norm=locked_dose)
    return z[ends]+((view-h[ends])*beta).sum(-1), dose, float(err)


def compare_signal(signal, args, decoder, path, teacher, *, source_threshold=None, archive=None):
    """Emit row statistics independent of any current or future shot outcome."""
    base = args[0]
    source = source_threshold is not None
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path, source=source)
    eligible, fields, waveform = group.feedback_waveform(signal, q, args[2], z.device)
    force = z.new_zeros((len(q),lag.H//group.BIN,len(MEMBERS)))
    _, _, _, common = group.past_force_parts(q, force, eligible)
    ends = q+lag.H+1+WAIT+base.cfg.patch
    if archive is not None:
        np.testing.assert_array_equal(q+lag.H+1, archive['release'])
        np.testing.assert_array_equal(common, archive['eligible'])
    context_raw = lag.direct.weak.shot_coordinates(signal, args[2][0], args[2][1])
    diagnostics, mask, context, c_mask = context_raw
    candidates = np.flatnonzero(common & (ends<len(z)))
    action = ends[candidates]
    archived_dose = archive['dose'][action] if archive is not None else np.ones(len(action), dtype=np.float32)
    candidates = candidates[archived_dose>0.]
    design, context = lag.direct.shape.inputs(diagnostics, mask, context, c_mask, ends[candidates])
    hist = {k:[] for k in ('issue','action','loss','follow_loss','follow_issue','dose_origin',
                            'dose_action','dose_follow','risk','context','context_valid','activity')}
    prior_end = -1
    max_error = 0.
    for start in range(0,len(candidates),group.BATCH):
        stop = candidates[start:start+group.BATCH]
        need = eligible & (q<=q[stop[-1]]) & (q>prior_end)
        prior_end = q[stop[-1]]
        i = np.flatnonzero(need)
        state, spec, pred = issued_forecasts(base,signal,q,i,prior,phase,correction,fields,decoder)
        center, spread = spec[-1][2:4]
        target = group.signed_levels(waveform[i],center,spread)
        f = ((target-pred.detach())/group.BETA).clamp(-1.,1.)
        force[i] = f
        _, now, old, _ = group.past_force_parts(q,force,eligible)
        u = directions(pred,state,now[i],old[i],teacher)
        ready = np.searchsorted(i, stop)
        np.testing.assert_array_equal(i[ready],stop)
        x, m, _, fs = spec
        origin_slot = lag.delayed_slots(q,q[stop])[0]
        sliced_port = layer.patcher(phase[origin_slot], correction[origin_slot])
        ready_fields = (*[v[ready] for v in fs[:-1]],fs[-1])
        loss,d0,e0 = forecast_trial(base,(x[ready],m[ready],sliced_port,ready_fields),u[:,ready],decoder,target[ready])
        a = ends[stop]
        risk,da,ea = risk_trial(signal,q,a,args,h,z,beta,phase,correction,u[:,ready],None)
        if archive is not None:
            fixed_dose = torch.as_tensor(archive['dose'][a],device=z.device)
            shared_cap = torch.minimum(da,fixed_dose)
            torch.testing.assert_close(da[da>0.],fixed_dose[da>0.],rtol=0.,atol=4e-6)
            risk,da,cap_error = risk_trial(signal,q,a,args,h,z,beta,phase,correction,u[:,ready],shared_cap)
            ea = max(ea,cap_error)
            np.testing.assert_allclose(risk[0].sigmoid().cpu().numpy(),archive['bias_only'][a],atol=3e-6,rtol=0.)
        f_src, f_dst = follow_window(q,stop,eligible)
        follow_loss = np.full((len(ARMS)+1,len(stop),4),np.nan,np.float32)
        second = np.full(len(stop),-1)
        d_follow = np.zeros(len(stop),np.float32)
        e1 = 0.
        if len(f_src):
            idx = np.searchsorted(stop, f_src)
            with torch.no_grad():
                _, fspec,_ = issued_forecasts(base,signal,q,f_dst,prior,phase,correction,fields,decoder)
                f_target = group.signed_levels(waveform[f_dst],fields[1][f_dst],fields[2][f_dst])
                later,d1,e1 = forecast_trial(base,fspec,u[:,ready[idx]],decoder,f_target)
            second[idx] = q[f_dst]
            follow_loss[:,idx] = later.cpu().numpy()
            d_follow[idx] = d1.cpu().numpy()
        max_error = max(max_error,e0,ea,e1)
        for name,value in dict(issue=q[stop],action=a,loss=loss.cpu().numpy(),
            follow_loss=follow_loss,follow_issue=second,dose_origin=d0.cpu().numpy(),dose_action=da.cpu().numpy(),
            dose_follow=d_follow,risk=risk.cpu().numpy(),
            context=np.tanh(context[start:start+len(a),:3]*np.array([1.,1.,4.])),
            context_valid=design[start:start+len(a),-3:].astype(bool).all(1),
            activity=context[start:start+len(a),3:]).items():
            hist[name].append(value)
    matrix = ('loss','follow_loss','risk')
    result = {}
    for k,vs in hist.items():
        if vs:
            result[k] = np.concatenate(vs,axis=1 if k in matrix else 0)
        elif k in matrix:
            result[k] = np.empty((len(ARMS)+1,0,4) if k.endswith('loss') else (len(ARMS)+1,0),
                                 np.float32)
        else:
            shape = {'context': (0,3),'activity': (0,4)}.get(k,(0,))
            dtype = {'context_valid':bool}.get(k,int if k in ('issue','action','follow_issue') else np.float32)
            result[k] = np.empty(shape,dtype)
    if source:
        alarms = np.flatnonzero(z.cpu().numpy()>=logit(source_threshold))
        result['base_alarm_index'] = np.array(alarms[0] if len(alarms) else -1)
    if max_error>1e-5 or not np.isfinite(result['risk']).all() or not np.isfinite(result['loss']).all():
        raise ValueError(f'invalid finite writes or equal-dose parity: {max_error:g}')
    return result, max_error


def seal(out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ref, _, args, decoder, path = group.load_reference(lock)
    old, sealed = (read(precision.ARCHIVE/f'{k}.json') for k in ('protocol','seal'))
    asset = precision.delay.HISTORY/'teaching.json'
    if (any(ref[k]!=old[k] for k in LOCK_KEYS) or len(sealed['curve_sha256'])!=360
            or old['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or read(precision.delay.HISTORY/'protocol.json')['teaching_sha256']!=sha(asset)):
        raise ValueError('the fixed readout/clock and complete original replay are required')
    teacher = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    source = io23.source_roles()
    out.mkdir()
    (out/'frames').mkdir()
    copyfile(Path(__file__),out/'executed.py')
    write(out/'protocol.json',dict(scope='CROSS_DIAGNOSTIC_FINITE_STEP_DEVELOPMENT',arms=('parent',*ARMS),
          **{k:old[k] for k in LOCK_KEYS},source={
              'train':source['train'], 'val':source['val'],
              'val_mask_mnm':source['val'], 'val_mask_mnm_ciii':source['val']},
          archive=str(precision.ARCHIVE),archive_seal=sha(precision.ARCHIVE/'seal.json'),
          groups=[[io23.RAW23[lag.CHANNELS[c]] for c in t] for t in group.POSITIONS],
          risk_clock=f'original clock; first full handoff q+{lag.H+1+WAIT+args[0].cfg.patch}; local probes only',
          held_window=f'original issue forecast replay A→B, plus disjoint original q+{FOLLOW} issue held in shadow',
          scope_detail='candidate gradients use only the issue-time graph, unlike the archived release-time gradient; '
                       'the held target is omitted from this graph and all candidate gradients for the held pair',
          finite_step='normalize all candidate actual token shifts at each comparison point; '
                      'a zero direction makes that point ineligible in every arm',
          relative_gain_floor=group.BETA, executed_sha256=sha(out/'executed.py')))
    roles = {'east':old['order'], **read(out/'protocol.json')['source']}
    verification = {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcomes inside feature replay')):
        for role,roster in roles.items():
            is_source = role!='east'
            raw_device = 'J-TEXT' if is_source else 'EAST'
            error, n_frames, support, following = 0.,0,0,0
            witnesses = {}
            for rank,sid in enumerate(roster,1):
                raw = simple_data.read_shot(raw_device,sid,rate_contract='strict' if is_source else 'nominal')
                record, archived = raw,None
                if is_source:
                    view = role.removeprefix('val_') if role.startswith('val_') else 'native'
                    record=source_view(raw,view)
                else:
                    if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                        raise ValueError('the fixed HDF5 signal file has changed')
                    record = lag.prepare_actual(raw,old['startup']+1,lock['mapping'])
                    if rank>old['startup']:
                        name=f'{rank:04d}_{sid}.npz'
                        f=precision.ARCHIVE/'shots'/name
                        if sha(f)!=sealed['curve_sha256'][name]:
                            raise ValueError('the committed risk/dose control has changed')
                        with np.load(f,allow_pickle=False) as z:
                            archived={k:z[k] for k in ('release','eligible','bias_only')}
                            archived['dose']=z['dose'][precision.delay.WAITS.index(WAIT)]
                data, err = compare_signal(record,args,decoder,path,teacher,
                    source_threshold=old['original'] if is_source else None,archive=archived)
                if not is_source and rank in (21,120,360):
                    end = min(550,len(raw['times']))
                    small={k:record[k][:end] for k in ('values','available')}
                    ok=archived['release']<end
                    locked={k:v[ok] if k in ('release','eligible') else v[:end] for k,v in archived.items()}
                    replay,_=compare_signal(small,args,decoder,path,teacher,archive=locked)
                    current = data['action']<end
                    np.testing.assert_array_equal(data['action'][current],replay['action'])
                    witness={}
                    for k in ('loss','risk'):
                        before=data[k][:,current]
                        np.testing.assert_allclose(before,replay[k],atol=3e-5,rtol=0.)
                        witness[k]=float(np.max(abs(before-replay[k]),initial=0.))
                    available=(data['follow_issue']>=0)&(data['follow_issue']+lag.H+1<end)
                    inside=(replay['follow_issue']>=0)
                    np.testing.assert_array_equal(data['follow_issue'][available],replay['follow_issue'][inside])
                    np.testing.assert_allclose(data['follow_loss'][:,available],replay['follow_loss'][:,inside],
                                               atol=3e-5,rtol=0.)
                    witnesses[str(sid)]=dict(samples=end,**witness)
                with (out/'frames'/f'{role}_{rank:04d}_{sid}.npz').open('xb') as f:
                    np.savez_compressed(f,**data)
                error=max(error,err)
                n_frames+=len(data['issue'])
                support+=int(((data['dose_origin']>0.)&(data['dose_action']>0.)).sum())
                following+=int((data['dose_follow']>0.).sum())
                if rank in (20,21) or rank%300==0 or rank==len(roster):
                    print(dict(stage='feature_seal',role=role,rank=rank,frames=n_frames,
                               seconds=round(time.monotonic()-started,1)),flush=True)
            verification[role]=dict(shots=len(roster),frames=n_frames,dose_common=support,
                dose_follow=following,max_dose_error=error,prefix_witnesses=witnesses)
    write(out/'seal.json',dict(verification=verification,elapsed_s=time.monotonic()-started))


def row_features(data):
    same, later = data['loss'], data['follow_loss']
    sg = relative_gain(same[:1],same[1:])
    lg = relative_gain(later[:1],later[1:])
    view = {}
    for j,(a,b,name) in enumerate(PAIRS):
        k=2+2*j
        view[name]=np.column_stack((sg[k,:,a],sg[k,:,b],sg[k+1,:,b],
            sg[0,:,b],lg[k,:,b],lg[k+1,:,b],lg[0,:,b]))
    return view


def stats(rows):
    """One mass per shot, then describe a within-shot aggregate."""
    a=np.array(rows,dtype=float)
    n=np.sum(np.isfinite(a),0)
    mean=np.divide(np.nansum(a,axis=0),n,out=np.full(7,np.nan),where=n>0)
    return dict(shots=len(rows),available_shots=n.tolist(),
        mean=[float(v) if np.isfinite(v) else None for v in mean])


def analyze(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    archived=read(precision.ARCHIVE/'per_shot.json')['0.2']
    advanced=io23.advanced_times()
    roles={'east':p['order'],**p['source']}
    if any(s['verification'][k]['shots']!=len(roster) for k,roster in roles.items()):
        raise ValueError('all requested shots must be sealed before scoring')
    inventory={}
    for role,roster in roles.items():
        summaries={name:{} for _,_,name in PAIRS}
        safe_refs, positives, matched_hard = [],[],[]
        counts={}
        positive_target=10. if role!='east' else 30.
        cases=[]
        for rank,sid in enumerate(roster,1):
            f=out/'frames'/f'{role}_{rank:04d}_{sid}.npz'
            with np.load(f,allow_pickle=False) as d:
                data=dict(d)
            if role=='east':
                disruptive,down,t=group.hdf._raw_truth(sid)
                horizon=125.
                original=archived['current_5'][rank-1]
                if (original['shot'],original['disruptive'],original['down_s'])!=(sid,disruptive,down):
                    raise ValueError('raw outcome and original full-stream scoring differ')
                with np.load(precision.ARCHIVE/'shots'/f'{rank:04d}_{sid}.npz') as z:
                    old=z['current_5']
                threshold=p['original'] if rank<=p['startup'] else p['calibration']['0.2']
                if dict(shot=sid,disruptive=disruptive,down_s=down,**score_curve(
                        t,old,disruptive,down,threshold))!=original:
                    raise ValueError('archived reference first alarm failed HDF5 recount')
                crossings=np.flatnonzero(old>=threshold)
                safe_time=float(t[crossings[0]]) if not disruptive and len(crossings) else None
            else:
                raw=simple_data.read_shot('J-TEXT',sid)
                disruptive,down=io23.read_outcome('J-TEXT',sid)
                t=raw['times']
                horizon=io23.effective_horizon_ms(sid,advanced)[0]
                threshold=p['original']
                first=int(data['base_alarm_index'])
                is_alarm=first>=0 and (not disruptive or t[first]<=down+1e-9)
                original=dict(alarm=is_alarm,timely=bool(disruptive and is_alarm and
                    round(float((down-t[first])*1000.),9)>10.))
                safe_time=float(t[first]) if (not disruptive and is_alarm) else None
            key = ('cold' if rank<=p['startup'] else 'post') if role=='east' else role
            cutoff=logit(threshold)
            action=data['action']
            risk=data['risk'][0]
            support=(data['dose_action']>0.)&(data['dose_origin']>0.)
            keep=analysis_windows(action,t,down=down,horizon=horizon,deadline=positive_target,
                                  first_alarm_s=safe_time,dose=support.astype(float))
            follow=(data['dose_follow']>0.)&(data['follow_issue']>=0)
            check_time=t[np.where(follow,data['follow_issue']+lag.H+1,0)]
            follow_terminal = follow.copy()
            if disruptive:
                follow_lead = np.round((down-check_time)*1000.,9)
                follow_terminal &= follow_lead>=0.
                follow &= follow_lead>positive_target
            elif safe_time is not None:
                follow &= check_time<=safe_time
                follow_terminal=follow
            tags=[key+'_positive' if disruptive else key+'_safe']
            if original is not None:
                if disruptive and not original['timely']:
                    tags.append(key+'_late')
                if not disruptive and original['alarm']:
                    tags.append(key+'_false_alarm')
            for tag in tags:
                c=counts.setdefault(tag,dict(shots=0,selected_shots=0,selected_frames=0,
                    follow_before_deadline=0,follow_before_terminal=0))
                c['shots']+=1
                c['selected_shots']+=int(keep.any())
                c['selected_frames']+=int(keep.sum())
                c['follow_before_deadline']+=int((keep&follow).any())
                c['follow_before_terminal']+=int((keep&follow_terminal).any())
            if not keep.any():
                continue
            features=row_features(data)
            for tag in tags:
                for name,v in features.items():
                    x=v[keep].copy()
                    f=follow[keep]
                    x[~f,4:]=np.nan
                    n=np.sum(np.isfinite(x),axis=0)
                    summaries[name].setdefault(tag,[]).append(
                        np.divide(np.nansum(x,axis=0),n,out=np.full(7,np.nan),where=n>0))
            picks=np.flatnonzero(keep & data['context_valid'])
            ix=np.concatenate([a[np.argsort(-risk[a],kind='stable')[:2]]
                               for a in np.array_split(picks,3)])
            for j in ix:
                row=dict(shot=sid,rank=rank,action=int(action[j]),match=np.r_[risk[j]-cutoff,data['context'][j]],
                    scores={name:v[j,:4] for name,v in features.items()},delta=data['risk'][1:,j]-risk[j],
                    dose=float(data['dose_action'][j]))
                if disruptive:
                    positives.append(row)
                else:
                    safe_refs.append(row)
                if f'{key}_late' in tags or f'{key}_false_alarm' in tags:
                    matched_hard.append(row|{'positive':disruptive})
                if sid in (70320,73405,79649,71076,67900):
                    cases.append(dict(shot=sid,rank=rank,action=int(action[j]),down=down,
                        lead_ms=float((down-t[action[j]])*1000) if disruptive else None,
                        margin=float(risk[j]-cutoff),risk_delta=row['delta'].tolist(),
                        gain={n:v.tolist() for n,v in row['scores'].items()}))
        def pair_report(pos, negative):
            matches=group.teacher.match_shots(pos,negative)
            pairs=[(pos[i],negative[j]) for i,j in matches]
            d={}
            for index,(_,_,name) in enumerate(PAIRS):
                u,v=(np.array([r[side]['scores'][name] for r in pairs]).reshape(-1,4) for side in (0,1))
                plus,minus=(np.array([r[side]['delta'][2+2*index] for r in pairs]) for side in (0,1))
                wx = (u[:,1]>0.)&(u[:,1]>u[:,2])&(v[:,1]>0.)&(v[:,1]>v[:,2])
                d[name]=dict(current_held_win=int((u[:,1]>v[:,1]).sum()),
                    previous_held_win=int((u[:,2]>v[:,2]).sum()),
                    current_minus_previous_win=int(((u[:,1]-u[:,2])>(v[:,1]-v[:,2])).sum()),
                    both_hold_improve_and_current_wins=int(wx.sum()),
                    both_risk_sign_correct=int(((plus>0)&(minus<0)).sum()),
                    joint=int((wx&(plus>0)&(minus<0)).sum()))
            refs=[dict(positive=[a['shot'],a['action']],safe=[b['shot'],b['action']],
                       positive_margin=float(a['match'][0]),safe_margin=float(b['match'][0])) for a,b in pairs]
            return dict(pairs=len(pairs),matches=refs,groups=d)
        inventory[role]=dict(counts=counts, held={g:{tag:stats(v) for tag,v in by.items()}
            for g,by in summaries.items()},
            matched=pair_report(positives,safe_refs) if role!='east' else {
                key:pair_report([r for r in positives if selected(r)], [r for r in safe_refs if selected(r)])
                for key,selected in (('cold',lambda r:r['rank']<=p['startup']),
                                     ('post',lambda r:r['rank']>p['startup']))},
            hard_matched=pair_report(
                [r for r in matched_hard if r['positive'] and (role!='east' or r['rank']>p['startup'])],
                [r for r in matched_hard if not r['positive'] and (role!='east' or r['rank']>p['startup'])]),
            cases=cases)
    write(out/'result.json',dict(scope='SEALED_MECHANISM_PROBE_NOT_A_NEW_ALARM_POLICY',analysis=inventory,
        held_columns=('current_same_group','current_held','earlier_held','all_groups_fitted_same_window',
                      'current_disjoint_held','earlier_disjoint_held','all_groups_fitted_disjoint'),
        interpretation='Retrospective reforecast and new shadow forecast only; no shot accuracy claim. '
            'Held-channel evidence may be correlated through the diagnostics themselves or the inherited parent.'))
    print({role:d['counts'] for role,d in inventory.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    a=ap.parse_args()
    seal(a.out.resolve())
    analyze(a.out.resolve())
