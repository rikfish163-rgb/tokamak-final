"""Compare workpoint-conditioned diagnostic continuations before adapting a shot."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_history_span_20261114 as history
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_excursion_order_20261111 import CALIPERS, disjoint_pairs
from .commissioning_history_span_attribution_20261114 import confirmation
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = history.group, history.lag, history.layer
precision = history.cross.precision
WAIT, PATCH, OLD = history.WAIT, history.PATCH, history.OLD
RELEASE = lag.H+1+WAIT
READY = RELEASE+PATCH
TARGET_WIDTH = lag.H//group.BIN * len(group.USE)
DIAGNOSTIC_COLUMNS = [lag.CHANNELS[c] for c in group.USE]
RIDGE, COLD_PRIOR_SHOTS, SAFE_WINDOWS = .1, 8, 4
FAMILIES = ('magnetic', 'position', 'radiation', 'loop_voltage')
ARMS = ('competition_now', 'competition_earlier', 'competition_hold', 'source_competition_now')
FEATURES = ('target_now', 'target_earlier', 'target_hold', 'source_target_now',
            'issued', 'source_issued')
CASES = (70320, 73405, 69813, 73743, 71076)


def precursor_rows(q, times, down, horizon, deadline):
    """Select finalized historical training examples by the actual action clock."""
    lead_ms = np.round((down-times[q+READY])*1000., 9)
    return (lead_ms > deadline) & (lead_ms <= horizon+1e-6)


def previous_packet(q, support):
    if not len(q):
        return np.empty(0, dtype=np.int64)
    old = np.searchsorted(q, q-lag.STRIDE)
    return np.where((q[old] == q-lag.STRIDE) & support[old] & (q[old]+lag.H+1 <= q), old, -1)


def issue_design(signal, q, parameters, center, spread, recent):
    """Pre-issue workpoint, recent displacement and local drift in diagnostic coordinates."""
    v, m, ctx, cm = lag.direct.weak.shot_coordinates(signal, *parameters[:2])
    use = group.USE
    older, nr = _window_mean(v[:, use], m[:, use], q, -31, 16)
    raw = signal['values'][:, (io23.RAW23.index('ip'), io23.RAW23.index('Bt'))]
    cp, count = _window_mean(raw, cm, q, -15, 16)
    pp, old_count = _window_mean(raw, cm, q, -31, 16)
    workpoint = np.arcsinh(np.c_[np.abs(cp[:, 0])/1e5, np.abs(cp[:, 1]),
                                (np.abs(cp[:, 0])-np.abs(pp[:, 0]))/1e4])
    current = recent[:, use]
    initial = np.arcsinh((current-center[:, use])/spread[:, use])
    motion = np.arcsinh((current-older)/spread[:, use])
    enough = ((nr >= 8).all(1) & (count >= 8).all(1) & (old_count >= 8).all(1))
    return np.c_[workpoint, initial, motion], enough


def relative_fit(level, normal, adverse, scale):
    """Shared noise scale and family weights: advantage for the adverse forecast."""
    loss = F.smooth_l1_loss(level/scale, normal/scale, beta=group.BETA, reduction='none')
    other = F.smooth_l1_loss(level/scale, adverse/scale, beta=group.BETA, reduction='none')
    advantage = torch.stack([(loss-other)[..., np.asarray(group.SIGNED_GROUPS)==i].mean((-1, -2))
                             for i in range(len(FAMILIES))], dim=-1)
    return advantage.tanh()


@dataclass(frozen=True)
class Regressors:
    center: np.ndarray
    spread: np.ndarray
    source: np.ndarray
    scale: np.ndarray

    def design(self, x):
        return np.c_[np.clip((x-self.center)/self.spread, -5., 5.), np.ones(len(x))]

    def forecast(self, x, weights):
        return self.design(x) @ weights

    def fit_cold(self, packets):
        """Cold labels only correct intercepts, shrunk by eight prior shots per class."""
        weights = self.source.copy()
        for label in (0, 1):
            residual = [(y-self.forecast(x, self.source)[s]).mean(0) for s,x,y in packets if s==label]
            if not residual:
                raise ValueError('cold fit requires usable observations in both outcome strata')
            weights[label,-1] += np.sum(residual, axis=0)/(len(residual)+COLD_PRIOR_SHOTS)
        return weights


def fit_source(packets):
    """Two regularized direct H25 fits; each class and each shot gets equal weight."""
    groups = [[(x,y) for k,x,y in packets if k==s] for s in (0, 1)]
    center = np.mean([np.mean([x.mean(0) for x,_ in g], axis=0) for g in groups], axis=0)
    spread = np.sqrt(np.mean([np.mean([np.square(x-center).mean(0) for x,_ in g], axis=0)
                              for g in groups], axis=0)).clip(.05)
    transformer = Regressors(center, spread, np.empty(0), np.empty(0))
    weights, variances = [], []
    reg = np.diag([RIDGE]*len(center)+[0.])
    for g in groups:
        x, y, w = zip(*[(transformer.design(a), b, 1./len(a)) for a,b in g], strict=True)
        gram = sum(wi*(a.T@a) for wi,a in zip(w,x,strict=True))/len(g)+reg
        cov = sum(wi*(a.T@b) for wi,a,b in zip(w,x,y,strict=True))/len(g)
        weight = np.linalg.solve(gram,cov)
        weights.append(weight)
        variances.append(np.mean([np.square(a@weight-b).reshape(-1, len(group.USE)).mean(0)
                                  for a,b in zip(x,y,strict=True)], axis=0))
    scale = np.sqrt(np.mean(variances, axis=0)).clip(group.BETA)
    return Regressors(center, spread, np.stack(weights), scale)


def historical_packets(ids, device, parameters, advanced, mapping=None):
    packets, inventory = [], []
    for sid in ids:
        raw = simple_data.read_shot(device, sid, rate_contract='strict' if mapping is None else 'nominal')
        target = simple_data.attach_outcome(raw, device, advanced)
        signal = raw if mapping is None else lag.prepare_actual(raw, 21, mapping)
        q = np.arange(lag.FIRST, len(raw['times'])-READY, lag.STRIDE)
        ok, fields, wave = group.feedback_waveform(signal, q, parameters, 'cpu')
        recent,c,m = (v.numpy() for v in fields[:3])
        x, has_context = issue_design(signal, q, parameters, c, m, recent)
        move = group.signed_levels(wave, fields[0], fields[2]).numpy().reshape(len(q), TARGET_WIDTH)
        mask = signal['available'][:, DIAGNOSTIC_COLUMNS]
        full = mask[q[:,None]+np.arange(1,lag.H+1)].all((1,2))
        rows = np.flatnonzero(ok & has_context & full)
        k = int(target['is_disrupt'])
        if k:
            pos = precursor_rows(q, raw['times'], target['down'], target['horizon'],
                                 10. if device=='J-TEXT' else 30.)
            rows = rows[pos[rows]]
        else:
            # Select large *pre-issue* excursions on historical safe shots,
            # rather than letting quiet windows dominate the safe comparator.
            activity = np.square(x[:, 3:]).mean(1)
            take = np.argsort(-activity[rows], kind='stable')[:SAFE_WINDOWS]
            rows = rows[take]
        inventory.append(dict(shot=sid, label=k, origins=q[rows].tolist()))
        if len(rows):
            packets.append((k,x[rows],move[rows]))
    return packets, inventory


def compare_history(model, packets, weights_per_packet):
    result = defaultdict(list)
    c = len(group.USE)
    scale = torch.from_numpy(model.scale)
    for (s,x,y), weights in zip(packets,weights_per_packet,strict=True):
        normal, adverse = torch.from_numpy(model.forecast(x, weights).reshape(2,len(x),-1,c))
        actual = torch.from_numpy(y.reshape(len(x),-1,c))
        advant = relative_fit(actual,normal,adverse,scale).mean(-1).numpy()
        result[s].append((float(advant.mean()), float((advant>0).mean()), len(x)))
    return {s:dict(shots=len(r),windows=sum(v[2] for v in r),
                   adverse_advantage=float(np.mean([v[0] for v in r])),
                   adverse_preference_fraction=float(np.mean([v[1] for v in r])))
            for s,r in result.items()}


def fit_archive(out):
    lock,p,_,args,_,_=history.inputs()
    roles = simple_data.source_roles()
    train, val, cold = roles['train'], roles['val'], p['order'][:p['startup']]
    if (len(train),len(val),len(cold))!=(1353,160,20) or set(train)&set(val):
        raise ValueError('use fixed source TRAIN/VAL and only the established EAST cold 20')
    advanced = simple_data.advanced_times()
    for sid in cold:
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('EAST cold feedback differs from the fixed protocol')
    a,ai = historical_packets(train,'J-TEXT',args[2],advanced)
    b,bi = historical_packets(val,'J-TEXT',args[2],advanced)
    t,ti = historical_packets(cold,'EAST',args[2],{},lock['mapping'])
    model = fit_source(a)
    weights = model.fit_cold(t)
    stats = dict(train=compare_history(model,a,[model.source]*len(a)),
                 val=compare_history(model,b,[model.source]*len(b)),
                 cold_source=compare_history(model,t,[model.source]*len(t)),
                 cold_leave_one_shot_out=compare_history(model,t,
                     [model.fit_cold(t[:i]+t[i+1:]) for i in range(len(t))]))
    out.mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'forecast_model.json',dict(source_sha256=p['source_sha256'], cold_ids=cold,
        ridge=RIDGE, cold_prior_shots=COLD_PRIOR_SHOTS, parameters={k:v.tolist() for k,v in vars(model).items()},
        cold_weights=weights.tolist(), inventory=dict(train=ai,val=bi,cold=ti), stats=stats))
    print(dict(stage='fit',stats=stats),flush=True)


def compose(legacy, auxiliary, qualified):
    u = F.normalize(legacy.flatten(-2), dim=-1, eps=group.EPS)
    v = auxiliary.flatten(-2)
    normalized = v/v.norm(dim=-1,keepdim=True).clamp_min(.01)
    combined = (u[None]+.5*normalized).reshape_as(auxiliary)
    return torch.where(qualified[None,:,None,None],combined,legacy[None])


def estimate(signal, args, decoder, path, teacher, model, cold_weights, archive):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    np.testing.assert_array_equal(q+lag.H+1, archive['release'])
    base = args[0]
    eligible,fields,waveform = group.feedback_waveform(signal,q,args[2],z.device)
    recent,centers,spreads,_ = fields
    x, valid = issue_design(signal,q,args[2],centers.cpu().numpy(),spreads.cpu().numpy(),recent.cpu().numpy())
    v = group.signed_levels(waveform,recent,spreads)
    earlier = previous_packet(q,eligible & valid)
    expected = [model.forecast(x,w).reshape(2,len(q),lag.H//group.BIN,len(group.USE))
                for w in (cold_weights,model.source)]
    a, b = (torch.as_tensor(w,device=z.device,dtype=z.dtype) for w in expected)
    scale = torch.as_tensor(model.scale,device=z.device,dtype=z.dtype)
    legacy = v.new_zeros(v.shape)
    _,_,_,common = group.past_force_parts(q,legacy,eligible)
    np.testing.assert_array_equal(common,archive['eligible'])
    qualified = common & valid & (earlier>=0)
    family = v.new_zeros((len(q),len(FEATURES),len(FAMILIES)))
    cosines = v.new_zeros((len(q),len(ARMS)))
    states = v.new_zeros((len(ARMS)+1,len(q)+1,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    indices = np.flatnonzero(eligible)
    for start in range(0,len(indices),group.BATCH):
        i = indices[start:start+group.BATCH]
        issued,current,state,wave,c,spread = group.signed_window(
            base,signal,q,i,phase,correction,prior,fields,decoder)
        pred,legacy[i] = precision.signed_residual(wave,waveform[i],c,spread)
        _,now,_,_ = group.past_force_parts(q,legacy,eligible)
        use = torch.as_tensor(common[i],device=z.device)
        good = torch.as_tensor(qualified[i],device=z.device)
        p = group.signed_levels(wave,recent[i],spread)
        d = relative_fit(p,*a[:,i],scale)
        src = relative_fit(p,*b[:,i],scale)
        t = torch.stack((relative_fit(v[i],*a[:,i],scale),
                         relative_fit(v[np.maximum(earlier[i],0)],*a[:,i],scale),
                         relative_fit(torch.zeros_like(v[i]),*a[:,i],scale),
                         relative_fit(v[i],*b[:,i],scale)))
        family[i] = torch.cat((t,d.detach()[None],src.detach()[None])).transpose(0,1)
        d_ref = torch.stack((d,d,d,src)).detach()
        force = ((t-d_ref)/group.BETA).clamp(-1.,1.)*good[None,:,None]
        directions = group.signed_directions(issued,current,state,(
            (pred,group.shape_force(now[i],teacher)*use[:,None,None]),
            (d,force[0]), (d,force[1]), (d,force[2]), (src,force[3])))
        states[0,i+1] = directions[0]
        states[1:,i+1] = compose(directions[0],directions[1:],good)
        cosines[i] = F.cosine_similarity(directions[1:].flatten(2), directions[1:2].flatten(2), dim=-1).T
    if not torch.isfinite(states).all():
        raise ValueError('nonfinite continuation update')
    trace = dict(q=q,qualified=qualified,context=x[:, :3],
                 excitation=np.tanh(np.sqrt(np.square(x[:,3:]).mean(1))),
                 fit=family.cpu().numpy(),cosines=cosines.cpu().numpy())
    return (q,h,z,beta,phase,correction,states,common),trace


def render(base, signal, track, archive, thresholds):
    parent, planned = archive['bias_only'], archive['dose']
    curves, representation_error, dose_error = precision.render(base,signal,track,parent,planned)
    reference = archive['current_5']
    np.testing.assert_allclose(curves[0],reference,atol=3e-6,rtol=0.)
    for tau in thresholds:
        np.testing.assert_array_equal(curves[0]>=tau,reference>=tau)
    error = max(float(np.abs(curves[0]-reference).max()), representation_error)
    return dict(zip(ARMS,curves[1:],strict=True)), (error,dose_error)


def replay(out, *, rank_only=None):
    start = time.monotonic()
    lock,p,s,args,decoder,path = history.inputs()
    fitted = read(out/'forecast_model.json')
    if (fitted['source_sha256']!=p['source_sha256'] or fitted['cold_ids']!=p['order'][:p['startup']]
            or len(fitted['inventory']['train'])!=1353):
        raise ValueError('the same pre-stream source and cold fit are required')
    if (out/'executed.py').read_bytes()!=Path(__file__).read_bytes():
        raise ValueError('the replay differs from the source fitting implementation')
    model = Regressors(**{k:np.asarray(v) for k,v in fitted['parameters'].items()})
    weights = np.asarray(fitted['cold_weights'])
    teaching = precision.delay.HISTORY/'teaching.json'
    if sha(teaching)!=read(teaching.parent/'protocol.json')['teaching_sha256']:
        raise ValueError('inherited diagnostic teacher changed')
    base = args[0]
    teacher = torch.as_tensor(read(teaching)['weights'],device=base.center.device,dtype=torch.float32)
    thresholds = tuple(p['calibration'].values())
    frozen = [state_digest(k) for k in args[:2]]
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(scope='EAST360_COMPETING_DIAGNOSTIC_CONTINUATIONS_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK}, arms=(*OLD,*ARMS), reference=str(history.ARCHIVE),
        reference_seal=sha(history.ARCHIVE/'seal.json'), fit_sha256=sha(out/'forecast_model.json'),
        rank_only=rank_only, fit='J-TEXT TRAIN-only balanced ridge; cold20 intercepts fit before shot21',
        purpose='change the first-layer time-token state by matching the relative merit of two '
                'forecasts after observation; no term is added to the risk score',
        feedback_clock=dict(horizon=lag.H,maturity=lag.H+1,wait=WAIT,handoff=PATCH),
        risk='existing encoder, risk readout, thresholds and first crossing; no inter-shot updating'))
    files, witnesses, maxima = {}, {}, np.zeros(2)

    def predict(signal, arc):
        track, trace = estimate(signal,args,decoder,path,teacher,model,weights,arc)
        curves, err = render(base,signal,track,arc,thresholds)
        return curves,trace,err

    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome')):
        for rank,sid in enumerate(p['order'],1):
            if rank_only is not None and rank!=rank_only:
                continue
            arc = history.archived_shot(rank,sid,s)
            if rank<=p['startup']:
                curves, trace = dict.fromkeys(ARMS,arc['current_5']), {}
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('EAST signal changed from the fixed input')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                curves,trace,err = predict(signal,arc)
                maxima = np.maximum(maxima,err)
                if rank in (21,120,360) or sid in CASES or rank_only==rank:
                    hit = np.flatnonzero(curves[ARMS[0]]>=p['calibration']['0.2'])
                    n = min(len(raw['times']),max(550,int(hit[0])+2) if len(hit) else 620)
                    prefix = {k:signal[k][:n] for k in ('values','available')}
                    cut = {k:arc[k][:n] for k in (*OLD,'dose')}
                    keep = arc['release']<n
                    cut.update(release=arc['release'][keep],eligible=arc['eligible'][keep])
                    short, earlier, _ = predict(prefix,cut)
                    np.testing.assert_array_equal(earlier['qualified'],trace['qualified'][:sum(keep)])
                    diff = 0.
                    for a in ARMS:
                        np.testing.assert_allclose(short[a],curves[a][:n],atol=5e-6,rtol=0.)
                        for th in thresholds:
                            np.testing.assert_array_equal(short[a]>=th,curves[a][:n]>=th)
                        diff = max(diff,float(np.abs(short[a]-curves[a][:n]).max()))
                    witnesses[str(sid)] = dict(points=n, max_curve_error=diff)
            name=f'{rank:04d}_{sid}.npz'
            dest=out/'shots'/name
            with dest.open('xb') as stream:
                np.savez_compressed(stream,**{k:arc[k] for k in (*OLD,'dose')},**curves,**trace)
            files[name]=sha(dest)
            if rank in (21,60,120,180,240,300,360) or rank_only:
                print(dict(stage='sealed',rank=rank,elapsed_s=round(time.monotonic()-start,2)),flush=True)
    if frozen!=[state_digest(k) for k in args[:2]]:
        raise ValueError('persistent source or parent tensors changed')
    write(out/'seal.json',dict(curves=files,prefix=witnesses,maximum=maxima.tolist(),
                              elapsed_s=time.monotonic()-start))


def pair_attributions(positive, safe):
    calipers = CALIPERS[:5]
    rows = []
    for misses in (False,True):
        a,b = [[r for r in group if not misses or r['wrong']] for group in (positive,safe)]
        pairs = []
        p,s = [np.asarray([(*r['match'],r['excitation']) for r in g]) for g in (a,b)]
        p_id,s_id = [np.asarray([r['shot'] for r in g]) for g in (a,b)]
        for i,j in disjoint_pairs(p,s,p_id,s_id,calipers):
            u,v = a[i],b[j]
            pairs.append(dict(positive=u,safe=v))
        name = 'miss_and_false_alarm' if misses else 'all_positive_and_safe'
        wins = {}
        for k in ('parent','old',*ARMS):
            wins[k] = sum(r['positive']['risk'][k]>r['safe']['risk'][k] for r in pairs)
        for k in FEATURES:
            wins[k] = sum(r['positive']['relative_fit'][k]>r['safe']['relative_fit'][k] for r in pairs)
        rows.append(dict(subset=name,positive_shots=len(set(p_id)), calipers=calipers.tolist(),
                         safe_shots=len(set(s_id)), wins=wins,pairs=pairs))
    return rows


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if (p['rank_only'] is not None or len(s['curves'])!=360
            or sha(out/'forecast_model.json')!=p['fit_sha256']
            or sha(history.ARCHIVE/'seal.json')!=p['reference_seal']):
        raise ValueError('score only the complete locked predictions')
    previous = read(history.ARCHIVE/'per_shot.json')
    per = {t:{a:[] for a in (*OLD,*ARMS)} for t in p['calibration']}
    candidates = ([],[])
    misses, traces, windows = [], defaultdict(list), 0
    for rank,sid in enumerate(p['order'],1):
        name=f'{rank:04d}_{sid}.npz'
        dest=out/'shots'/name
        if sha(dest)!=s['curves'][name]:
            raise ValueError('sealed prediction changed')
        y,down,times = group.hdf._raw_truth(sid)
        tau = p['calibration']['0.2'] if rank>p['startup'] else p['original']
        with np.load(dest,allow_pickle=False) as f:
            for b,th in p['calibration'].items():
                for a in per[b]:
                    row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(
                        times,f[a],y,down,th if rank>p['startup'] else p['original']))
                    if a in OLD and row!=previous[b][a][rank-1]:
                        raise ValueError('locked baseline scoring mismatch')
                    per[b][a].append(row)
            if rank<=p['startup']:
                for a in ARMS:
                    np.testing.assert_array_equal(f[a],f['current_5'])
                continue
            q = f['q']
            action = q+READY
            eligible = np.flatnonzero(f['qualified'] & (action<len(times)))
            qualified = f['qualified']
            use = np.flatnonzero(f['dose']>0.)
            slots = np.searchsorted(q+RELEASE,use,side='right')
            idx = np.r_[-1,q+RELEASE][slots]
            can = np.r_[False,qualified]
            seen = (can[slots] & (use>idx)) | (can[np.maximum(slots-1,0)] & (use<idx+PATCH))
            only_legacy = np.ones(len(times),bool)
            only_legacy[use[seen]] = False
            for a in ARMS:
                np.testing.assert_allclose(f[a][only_legacy],f['current_5'][only_legacy],
                                           atol=3e-6,rtol=0.)
                np.testing.assert_array_equal(f[a][only_legacy]>=tau,f['current_5'][only_legacy]>=tau)
            windows += int(qualified.sum())
            j = eligible[f['dose'][action[eligible]]>0.]
            first = np.flatnonzero(f['current_5']>=tau)
            old = per['0.2']['current_5'][-1]
            wrong = bool(not old['timely']) if y else bool(old['alarm'])
            view = f['fit']
            if y:
                ld = np.round((down-times)*1000.,9)
                j = j[(ld[action[j]]>30.) & (ld[action[j]]<=125.)]
                watch = use[seen]
                watch = watch[(ld[watch]>30.)&(ld[watch]<=125.)]
                if wrong and len(watch):
                    logits = logit(f['current_5'][watch].clip(1e-6,1.-1e-6))
                    misses.append(dict(shot=sid,endpoints=len(watch),original_gap=float(logit(tau)-logits.max()),
                        max_lift={a:float((logit(f[a][watch].clip(1e-6,1.-1e-6))-logits).max()) for a in ARMS},
                        crosses={a:bool((f[a][watch]>=tau).any()) for a in ARMS}))
            elif len(first):
                j = j[action[j]<=first[0]]
            if wrong and len(j):
                traces[int(y)].extend(f['cosines'][j,1].tolist())
            margin = logit(f['bias_only'][action[j]].clip(1e-6,1.-1e-6))-logit(tau)
            fit = view[j].mean(-1)
            workpoints = f['context'][j]
            for i,k in enumerate(j):
                endpoint = action[k]
                candidates[int(not y)].append(dict(shot=sid,wrong=wrong,issue=int(q[k]),endpoint=int(endpoint),
                    match=[float(margin[i]),*workpoints[i].tolist()],
                    excitation=float(f['excitation'][k]),
                    relative_fit=dict(zip(FEATURES,fit[i].tolist(),strict=True)),
                    risk=dict(parent=float(f['bias_only'][endpoint]),old=float(f['current_5'][endpoint]),
                              **{a:float(f[a][endpoint]) for a in ARMS})))
    for table in per.values():
        for arm in table.values():
            if sum(x['disruptive'] for x in arm)!=180:
                raise ValueError('EAST 180/180 class contract changed')
    paired = {b:{f'{a}__to__{k}':dict(flips=flips(table[a],table[k]),lead=lead.paired(table[a],table[k]))
                 for a in (*OLD,ARMS[1],ARMS[2],ARMS[3]) for k in ARMS[:1]} for b,table in per.items()}
    metrics = {b:{a:lead.summarize(x) for a,x in table.items()} for b,table in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(metrics=metrics,paired=paired,near_miss=misses,
        diagnostic_pairs=pair_attributions(*candidates),qualified=windows,
        true_old_direction_cosine_wrong_cases={k:float(np.median(v)) for k,v in traces.items() if v},
        hdf5_scored=360*len(per)*len((*OLD,*ARMS))))
    write(out/'postseal_confirmation.json',confirmation(out,names=('bias_only','current_5',*ARMS)))
    print({b:{a:(v['by_deadline']['30']['tp'],v['fp']) for a,v in m.items()} for b,m in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('fit','replay','score'),required=True)
    parser.add_argument('--rank-only',type=int)
    a=parser.parse_args()
    if a.stage=='fit':
        fit_archive(a.output.resolve())
    elif a.stage=='replay':
        replay(a.output.resolve(),rank_only=a.rank_only)
    else:
        score(a.output.resolve())
