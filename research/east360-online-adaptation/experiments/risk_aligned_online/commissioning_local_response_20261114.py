"""Measure recovery and persistence against diagnostics not used to define them."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import commissioning_history_span_20261114 as history
from . import commissioning_recovery_clock_20261111 as recovery
from . import io23, simple_data
from .commissioning_excursion_order_20261111 import disjoint_pairs
from .commissioning_roughness_20261110 import coordinates
from .commissioning_transition_data_20261110 import _window_mean
from .history_gradient_metric_20261102.run import read, sha, write
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


group, lag = history.group, history.lag
DIAGNOSTICS = tuple(lag.CHANNELS[k] for k in group.USE)
MAGNETIC, POSITION, RADIATION, VOLTAGE = (
    [DIAGNOSTICS.index(c) for c in cols] for cols in group.TRUSTED)
DRIVERS = {'magnetic': MAGNETIC, 'position': POSITION}
HELD = {'radiation': RADIATION, 'loop_voltage': VOLTAGE}
IP, BT = (io23.RAW23.index(k) for k in ('ip', 'Bt'))
CHECK_SHOTS = (70320, 73743, 79649, 71076, 73405)
OBSERVE, HORIZON, BIN = lag.H, lag.H, group.BIN
ARMS = ('hold', 'context', 'peak_and_tail', 'local_status')
SETTINGS = tuple((g, h) for g in DRIVERS for h in HELD)
# Workpoint, elapsed since first recorded sample, initial and held amplitudes,
# and change of the held signal before issue; fixed before reading outcomes.
TOL = np.array([.35, .35, .50, .35, .25, .25, .20])
RIDGE = .10


def radius(x, center, spread):
    return np.sqrt(np.square((x-center)/spread).mean(axis=-1))


def events(ref, columns, *, slots=False):
    """Fixed-age status known from the same observed duration in both classes."""
    v, center, spread = [ref[k][:,columns] for k in ('value','center','spread')]
    observed, known = [ref[k][:,columns].all(1) for k in ('observed_now','reference_known')]
    n = len(v)
    trace = recovery.episode_trace(v,observed,center,spread,known)
    onset, stop, status = trace['episodes'].T
    if slots:
        # At the legacy H25 feedback clock, use the most recent onset
        # already old enough for a fixed-age decision. Future onsets don't
        # change the choice; never replace a recent incomplete observation
        # with an older event outside the already fixed 125-point memory.
        grid=np.arange(lag.FIRST+lag.H, n-1, lag.STRIDE)
        prior=np.searchsorted(onset+OBSERVE,grid,side='right')-1
        j=np.flatnonzero(prior>=0)
        use=prior[j]
        valid=grid[j]-onset[use]<recovery.MEMORY
        onset,stop,status=(a[use[valid]] for a in (onset,stop,status))
        landmark=grid[j[valid]]
    else:
        landmark = onset+OBSERVE
    known_at = landmark+1
    observed_prefix = np.pad(np.cumsum(~observed), (1,0))
    idx = np.flatnonzero(known_at<n)
    idx = idx[observed_prefix[landmark[idx]+1]==observed_prefix[onset[idx]]]
    anchor = onset[idx]-(recovery.CONFIRM-1)
    loc, dev = center[anchor],spread[anchor]
    tail = radius(v[landmark[idx,None]-np.arange(recovery.CONFIRM)],loc[:,None],dev[:,None])
    returned = ((stop[idx]<=landmark[idx]) & (status[idx]==recovery.RETURNED)
                & np.all(tail<=recovery.LOW,axis=1))
    persistent = (stop[idx]>=landmark[idx]) & np.all(tail>recovery.HIGH,axis=1)
    # Freeze the reference at the first above-high point. A gap, a partial
    # recovery, or an epilogue that cannot reach this age does not get a label.
    use = returned | persistent
    return dict(onset=onset[idx][use],landmark=landmark[idx][use],known_at=known_at[idx][use],
        anchor=anchor[use],response=np.where(persistent[use],1,-1).astype(np.int8),
        initial=np.log1p(radius(v[onset[idx]],loc,dev))[use],
        tail=np.log1p(tail.mean(1))[use])


def held_packet(values, mask, issue, anchor, *, reference=None):
    """Make one direct five-bin held target; eligibility never uses its numeric values."""
    ref = recovery.rolling_reference(values,mask) if reference is None else reference
    c, sd = ref['center'][anchor],ref['spread'][anchor]
    recent, before = (np.log1p(radius(ref['value'][end],c,sd)) for end in (issue,issue-BIN))
    support = (ref['reference_known'][anchor] & ref['observed_now'][issue]
               & ref['observed_now'][issue-BIN]).all(1)
    mature = np.zeros(len(issue),bool)
    idx = np.flatnonzero(support & (issue+HORIZON+1<len(values)))
    new = issue[idx,None]+np.arange(1,HORIZON+1)
    valid = mask[new].all((1,2))
    future = np.full((len(issue),HORIZON//BIN),np.nan)
    idx,new = idx[valid],new[valid]
    binned = values[new].reshape(len(idx),HORIZON//BIN,BIN,values.shape[1]).mean(2)
    future[idx] = np.log1p(radius(binned,c[idx,None],sd[idx,None]))
    mature[idx] = True
    return dict(support=support,mature=mature,recent=recent,trend=recent-before,future=future)


def shot_rows(signal, raw, center, scale, *, rank, parent, slots=False):
    x, m = coordinates(signal['values'][:,DIAGNOSTICS],signal['available'][:,DIAGNOSTICS],
                       center[list(DIAGNOSTICS)],scale[list(DIAGNOSTICS)])
    ref = recovery.rolling_reference(x,m)
    oper = signal['values'][:,[IP,BT]]
    oper_mask = signal['available'][:,[IP,BT]]
    clock = raw['times']
    for driver,columns in DRIVERS.items():
        e = events(ref,columns,slots=slots)
        issue, known, anchor = (e[k] for k in ('landmark','known_at','anchor'))
        v,m1 = _window_mean(oper,oper_mask,issue,-15,16)
        u,m0 = _window_mean(oper,oper_mask,issue,-31,16)
        workpoint = np.arcsinh(np.c_[abs(v[:,0])/1e5,abs(v[:,1]),(abs(v[:,0])-abs(u[:,0]))/1e4])
        common = ((m1>=8) & (m0>=8)).all(1)
        elapsed = np.log1p((clock[known]-clock[0])*1000.)
        for held,cols in HELD.items():
            packet = held_packet(x[:,cols],m[:,cols],issue,anchor,
                                 reference={k:a[:,cols] for k,a in ref.items()})
            available = common & packet['support']
            design = np.c_[workpoint, elapsed,e['initial'],packet['recent'],packet['trend']]
            # The full state arm knows whether a return has been confirmed,
            # not the local trajectory after the current issue.
            n = len(issue)
            context = np.c_[design,e['tail']]
            yield (driver,held),dict(
                shot=np.full(n,int(raw['shot'])),rank=np.full(n,rank),
                **e,usable=available,mature=common & packet['mature'],
                context=context,future=packet['future'],source_parent=np.full(n,np.nan) if parent is None
                else parent[known])


def bind():
    lock=lag.load_inputs()
    p,s=(read(history.ARCHIVE/f'{k}.json') for k in ('protocol','seal'))
    if (p['order']!=lock['ids'] or p['startup']!=20 or p['source_sha256']!=PARENT_SHA256
            or len(s['curve_sha256'])!=360):
        raise ValueError('the locked parent and complete 360-shot reference are required')
    return lock,p,s


def observe(out, *, slots=False):
    """Extract all roles from signal-only reads; EAST outcomes are never opened here."""
    start=time.monotonic()
    lock,p,s=bind()
    base,_ = load_parent(DEFAULT_PARENT)
    center,scale = (a.detach().numpy() for a in (base.center,base.scale))
    source=simple_data.source_roles()
    roles={'train':source['train'],'val':source['val'],'east':p['order']}
    out.mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(scope='LOCAL_DIAGNOSTIC_RESPONSE_AND_CENSORING_DEVELOPMENT',
        order=p['order'],startup=p['startup'],original=p['original'],calibration=p['calibration'],
        reference=str(history.ARCHIVE),reference_seal=sha(history.ARCHIVE/'seal.json'),
        source_sha256=p['source_sha256'],data_roles={k:len(v) for k,v in roles.items()},
        definition=dict(observe=OBSERVE,horizon=HORIZON,bin=BIN,low=recovery.LOW,
            high=recovery.HIGH,confirm=recovery.CONFIRM,reference=recovery.REFERENCE_POINTS,
            reference_lag=recovery.REFERENCE_LAG,
            landmark='onset+25 or legacy packet grid with previous episode age [25,125)' if slots
                else 'onset+25',
            clock='status at landmark+1, held target at landmark+26',
            drivers={k:[io23.RAW23[DIAGNOSTICS[i]] for i in v] for k,v in DRIVERS.items()},
            held={k:[io23.RAW23[DIAGNOSTICS[i]] for i in v] for k,v in HELD.items()}),
        comparison=ARMS,fit='J-TEXT TRAIN only; outcome-free; VAL and full EAST labeled only postseal'))
    prefixes={}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before seal')):
        for role,ids in roles.items():
            parts = {pair:[] for pair in SETTINGS}
            for rank,sid in enumerate(ids,1):
                target=role=='east'
                raw=simple_data.read_shot('EAST' if target else 'J-TEXT',sid,
                                           rate_contract='nominal' if target else 'strict')
                signal=lag.prepare_actual(raw,rank,lock['mapping']) if target else raw
                a=history.archived_shot(rank,sid,s)['bias_only'] if target else None
                r=dict(shot_rows(signal,raw,center,scale,rank=rank,parent=a,slots=slots))
                for k,v in r.items():
                    parts[k].append(v)
                if target and (rank in (21,120,360) or sid in CHECK_SHOTS):
                    end=min(len(raw['times']),620)
                    # Keep a real event when it appears only after the fixed
                    # short prefix, rather than passing an all-empty witness.
                    ends=np.concatenate([v['known_at'] for v in r.values()])
                    if len(ends):
                        end=min(len(raw['times']),max(end,int(ends.min())+HORIZON+12))
                    cut=dict(values=signal['values'][:end],available=signal['available'][:end])
                    short=dict(shot_rows(cut,dict(shot=sid,times=raw['times'][:end]),center,scale,
                                         rank=rank,parent=a[:end],slots=slots))
                    exist, complete=0,0
                    for k,v in r.items():
                        j=v['known_at']<end
                        for name in v:
                            if name in ('future','mature'):
                                allowed=(v['landmark']+HORIZON+1)[j]<end
                                np.testing.assert_equal(short[k][name][allowed],v[name][j][allowed])
                            else:
                                np.testing.assert_equal(short[k][name],v[name][j])
                        exist+=int(j.sum())
                        complete+=int(short[k]['mature'].sum())
                    prefixes[str(sid)]=dict(points=end,status_rows=exist,complete_witness=complete)
                if rank%350==0 or rank==len(ids):
                    print(dict(stage=role,rank=rank,total=len(ids),seconds=round(time.monotonic()-start,2)),flush=True)
            dest=out/f'{role}.npz'
            with dest.open('xb') as stream:
                np.savez_compressed(stream,**{f'{g}__{h}__{k}':np.concatenate([a[k] for a in rows])
                                             for (g,h),rows in parts.items() for k in rows[0]})
    write(out/'seal.json',dict(prefixes=prefixes,seconds=time.monotonic()-start,
        assets={k:sha(out/k) for k in ('train.npz','val.npz','east.npz','protocol.json','executed.py')}))


def decode_parts(out, role):
    with np.load(out/f'{role}.npz',allow_pickle=False) as f:
        return {pair:{k.split('__')[2]:f[k] for k in f.files if k.startswith('__'.join(pair)+'__')}
                for pair in SETTINGS}


def shot_weights(ids):
    _, j, n = np.unique(ids,return_inverse=True,return_counts=True)
    return 1./n[j]/len(n)


def design(row):
    base=row['context']
    return (base[:,:-1],base,np.c_[base,row['response']])


def fit_and_apply(train, test):
    sel=train['mature']
    w=shot_weights(train['shot'][sel])
    recent=train['context'][sel,5,None]
    y=train['future'][sel]-recent
    out=[np.broadcast_to(test['context'][:,5,None],test['future'].shape)]
    params={}
    for name,x,v in zip(ARMS[1:],design(train),design(test),strict=True):
        x=x[sel]
        center=w@x
        spread=np.sqrt(w@np.square(x-center)).clip(.05)
        z=np.c_[np.clip((x-center)/spread,-5.,5.),np.ones(len(x))]
        penalty=np.diag([RIDGE]*len(center)+[0.])
        elements=np.linalg.solve(z.T@(z*w[:,None])+penalty,z.T@(y*w[:,None]))
        v=np.c_[np.clip((v-center)/spread,-5.,5.),np.ones(len(v))]
        out.append(v@elements+test['context'][:,5,None])
        params[name]=dict(center=center.tolist(),spread=spread.tolist(),weights=elements.tolist())
    return np.stack(out,axis=1),params


def mae(row, predictions, use):
    if not use.any():
        return dict(shots=0,events=0,mae={})
    err=np.abs(predictions[use]-row['future'][use,None]).mean(2)
    scores=shot_weights(row['shot'][use])@err
    return dict(shots=len(np.unique(row['shot'][use])),events=int(use.sum()),
                mae=dict(zip(ARMS,scores.tolist(),strict=True)))


def same_shot_pairs(row, accept):
    """One pair per discharge; match strictly on information other than driver status/tail."""
    ids=row['shot']
    ctx=row['context'][:,:len(TOL)]/TOL
    pairs=[]
    for sid in np.unique(ids[accept]):
        keys=accept & (ids==sid)
        a,b=(np.flatnonzero(keys & (row['response']==k)) for k in (1,-1))
        if len(a) and len(b):
            d=ctx[a,None]-ctx[b]
            cost=np.where((np.abs(d)<=1.).all(2),np.square(d).sum(2),np.inf)
            i,j=np.unravel_index(cost.argmin(),cost.shape)
            if np.isfinite(cost[i,j]):
                pairs.append((a[i],b[j]))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2)


def pair_summary(row, predictions, pairs):
    i,j=pairs.T
    recent=row['context'][:,5]
    z=(row['future'][:,-2:].mean(1)-recent)
    error = z-(predictions[:,1,-2:].mean(1)-recent)
    gain=z[i]-z[j]
    residual=error[i]-error[j]
    status=row['response'][i]-row['response'][j]
    feature_wins = ((predictions[i,-1,-2:].mean(1)-predictions[j,-1,-2:].mean(1))>0.)
    return dict(pairs=len(i),observed_growth_wins=int((gain>0.).sum()),status_wins=int((status>0.).sum()),
        context_residual_wins=int((residual>0.).sum()),local_predicted_witness_wins=int(feature_wins.sum()),
        observed_growth_median=float(np.median(gain)) if len(i) else None,
        shot_pairs=np.c_[row['shot'][i],row['landmark'][i],row['shot'][j],row['landmark'][j]].tolist())


def postseal_case_inventory(out):
    """Union of usable events by shot; failure coverage must not count views twice."""
    p,s=(read(out/f'{k}.json') for k in ('protocol','seal'))
    if any(sha(out/k)!=s['assets'][k] for k in ('east.npz','protocol.json')):
        raise ValueError('sealed event archive changed')
    data=decode_parts(out,'east')
    timing,outcomes=target_clock(p,data)
    detail={}
    for group_name,outcome_name in (('late','late_suffix'),('fp','false_alarm_suffix')):
        cases={}
        for pair,row in data.items():
            flags=timing[pair]
            good=(flags['late'] & flags['near'] if group_name=='late' else flags['fp'] & flags['safe'])
            family='__'.join(pair)
            for i in np.flatnonzero(good & row['usable']):
                sid=int(row['shot'][i])
                cases.setdefault(sid,[]).append(dict(driver_and_held=family,
                    onset=int(row['onset'][i]), known_at=int(row['known_at'][i]),
                    signal_recovered=bool(row['response'][i]<0),
                    held_mature=bool(row['mature'][i]),
                    held_before_deadline=bool(flags['witness_before'][i])))
        detail[group_name]=dict(outcome_shots=outcomes[outcome_name],
            driver_status_shots=sorted(cases), missing_status_shots=sorted(set(outcomes[outcome_name])-cases.keys()),
            timely_witness_shots=sorted(k for k,v in cases.items() if any(r['held_before_deadline'] for r in v)),
            individual=cases)
    write(out/'postseal_case_inventory.json',detail)


def target_clock(p, data):
    old=read(history.ARCHIVE/'per_shot.json')['0.2']['current_5']
    archived=read(history.ARCHIVE/'seal.json')
    rows = {pair:dict(disruptive=np.zeros(len(v['shot']),bool),late=np.zeros(len(v['shot']),bool),
        safe=np.zeros(len(v['shot']),bool),fp=np.zeros(len(v['shot']),bool),
        near=np.zeros(len(v['shot']),bool), witness_before=np.zeros(len(v['shot']),bool))
        for pair,v in data.items()}
    late_shots,false_shots=[],[]
    for i,sid in enumerate(p['order'],1):
        y,down,t=group.hdf._raw_truth(sid)
        leg=old[i-1]
        if (y,down,sid)!=(leg['disruptive'],leg['down_s'],leg['shot']):
            raise ValueError('individual HDF5 outcomes differ from archived alarms')
        if i<=p['startup']:
            continue
        late=bool(y and not leg['timely'])
        fp=bool(not y and leg['alarm'])
        if late:
            late_shots.append(sid)
        if fp:
            false_shots.append(sid)
        if fp:
            z=history.archived_shot(i,sid,archived)['current_5']
            stop=int(np.flatnonzero(z>=p['calibration']['0.2'])[0])
        for pair,v in data.items():
            use=v['rank']==i
            flags=rows[pair]
            a=v['known_at'][use]
            mature=v['mature'][use]
            e=v['landmark'][use][mature]+HORIZON+1
            if y:
                delay=np.round((down-t[a])*1000.,9)
                near=(delay>30.) & (delay<=125.)
                witness=np.zeros(len(a),bool)
                witness[mature]=np.round((down-t[e])*1000.,9)>30.
                flags['near'][use]=near
                flags['witness_before'][use]=witness
                flags['late'][use]=late
            else:
                flags['safe'][use]=(a<=stop) if fp else True
                witness=np.zeros(len(a),bool)
                witness[mature]=(e<=stop) if fp else True
                flags['witness_before'][use]=witness
                flags['fp'][use]=fp
            flags['disruptive'][use]=y
    return rows,dict(late_suffix=late_shots,false_alarm_suffix=false_shots)


def warning_pairs(row,predictions,positive,safe,tau):
    a,b=(np.flatnonzero(k) for k in (positive,safe))
    margin=logit(row['source_parent'].clip(1e-6,1.-1e-6))-logit(tau)
    v=np.c_[margin,row['context'][:,:len(TOL)]]
    a0,b0=disjoint_pairs(v[a],v[b],row['shot'][a],row['shot'][b],np.r_[.50,TOL]).T
    i,j=a[a0],b[b0]
    out=pair_summary(row,predictions,np.c_[i,j])
    out.update(positive_shots=len(np.unique(row['shot'][a])),safe_shots=len(np.unique(row['shot'][b])),
        parent_wins=int((margin[i]>margin[j]).sum()),status_ties=int((row['response'][i]==row['response'][j]).sum()))
    return out


def count(row, mask):
    return dict(events=int(mask.sum()),shots=len(np.unique(row['shot'][mask])),
                returned=int((mask & (row['response']<0)).sum()),
                persistent=int((mask & (row['response']>0)).sum()))


def analyze(out):
    p,s=(read(out/f'{k}.json') for k in ('protocol','seal'))
    if (any(sha(out/k)!=v for k,v in s['assets'].items())
            or sha(history.ARCHIVE/'seal.json')!=p['reference_seal']):
        raise ValueError('sealed event observations or parent changed')
    tr,val,east=(decode_parts(out,k) for k in ('train','val','east'))
    timing,outcomes=target_clock(p,east)
    summary={}
    models={}
    for pair,rows in east.items():
        name='__'.join(pair)
        cold=rows['rank']<=p['startup']
        va,_=fit_and_apply(tr[pair],val[pair])
        actual,models[name]=fit_and_apply(tr[pair],rows)
        m=rows['mature']
        flags=timing[pair]
        late=flags['late'] & flags['near'] & rows['usable']
        fp=flags['fp'] & flags['safe'] & rows['usable']
        strict=flags['witness_before'] & m
        e_usable=rows['usable'] & ~cold
        summary[name]=dict(
            fit=count(tr[pair],tr[pair]['mature']),
            val=mae(val[pair],va,val[pair]['mature']),
            cold=mae(rows,actual,m & cold),
            suffix=mae(rows,actual,m & ~cold),
            near_old_late=mae(rows,actual,late & m),
            before_old_fp=mae(rows,actual,fp & m),
            status_suffix=count(rows,e_usable),late_status=count(rows,late),fp_status=count(rows,fp),
            late_witness_before_deadline=count(rows,late & strict),
            fp_witness_before_alarm=count(rows,fp & strict),
            within_shot=pair_summary(rows,actual,same_shot_pairs(rows,m & ~cold)),
            hard_pairs=warning_pairs(rows,actual,late & m,fp & m,p['calibration']['0.2']),
            hard_pairs_both_witnesses_in_time=warning_pairs(rows,actual,late & strict,fp & strict,
                                                            p['calibration']['0.2']),
            all_pairs=warning_pairs(rows,actual,flags['near'] & m,flags['safe'] & m,
                                    p['calibration']['0.2']))
    write(out/'model.json',models)
    write(out/'result.json',dict(kind='postseal diagnostic; no new alarms',settings=summary,
                                 matched_tolerances=TOL.tolist(),ridge=RIDGE,**outcomes))
    for name,v in summary.items():
        show=('suffix','near_old_late','before_old_fp','late_status','fp_status',
              'within_shot','hard_pairs_both_witnesses_in_time')
        print(name,{k:{j:u for j,u in v[k].items() if j!='shot_pairs'} for k in show},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    stages=parser.add_mutually_exclusive_group()
    stages.add_argument('--analyze',action='store_true')
    stages.add_argument('--case-inventory',action='store_true')
    stages.add_argument('--legacy-slots',action='store_true')
    a=parser.parse_args()
    out=a.output.resolve()
    if a.analyze:
        analyze(out)
    elif a.case_inventory:
        postseal_case_inventory(out)
    else:
        observe(out,slots=a.legacy_slots)
