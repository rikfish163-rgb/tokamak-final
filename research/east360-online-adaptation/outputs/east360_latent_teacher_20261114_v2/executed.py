"""A cold-start latent forecast teaches shot-local numerical embedding updates."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_embedding_fast_20261114 as embed
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot


history, group, lag, layer = embed.history, embed.group, embed.lag, embed.layer
MODES = ('latent_current', 'latent_earlier', 'latent_reversed', 'latent_hold', 'orthogonal_current')
ORTHOGONAL_CONTROLS = ('orthogonal_earlier', 'orthogonal_reversed', 'orthogonal_hold', 'readout_current')
RIDGE = .05
CASES = (90874, 74970, 81406, 70320, 85422, 73743, 73405)


def teacher_views(values, masks):
    """Compare observed/reversed values against persistence, with no new mask bits."""
    h = lag.H
    cols = list(embed.VALUES)
    old = values[:, -h-1:-h]
    targets = values[:, -h:, cols]
    views = values.expand(3, *values.shape).clone()
    views[:, :, -h:] = old
    views[0, :, -h:, cols] = targets
    views[1, :, -h:, cols] = targets.flip(1)
    available = masks.clone()
    available[:, -h:] = masks[:, -h-1:-h]
    available[:, -h:, cols] = masks[:, -h:, cols]
    return views, available


@torch.no_grad()
def teacher_increment(base, values, masks, transform):
    x, m = teacher_views(values, masks)
    n = x.shape[1]
    seen = []
    hook = base.risk_head.register_forward_pre_hook(lambda _m,a:seen.append(a[0]))
    # Only the issued historical correction is shared, not the input.
    inp = base.embedding.register_forward_pre_hook(
        lambda _m,a:(torch.cat([transform(v) for v in a[0].split(n)]),))
    try:
        base(x.flatten(0,1),m.repeat(3,1,1))
    finally:
        hook.remove()
        inp.remove()
    y = seen[0].unflatten(0,(3,n))
    return y[:2]-y[2:]


def statistics(x, y):
    """One equally weighted moment tuple per completed shot."""
    x,y = (np.asarray(v, dtype=np.float64) for v in (x,y))
    if len(x)<1 or len(x)!=len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('nonempty aligned latent forecast examples required')
    return x.mean(0),y.mean(0),(x.T@x)/len(x),(x.T@y)/len(x)


def fit(bank):
    x,y,xx,xy = (np.mean(v,axis=0) for v in zip(*bank,strict=True))
    cov = xx-np.outer(x,x)
    scale = np.trace(cov)/len(x)
    if not np.isfinite(scale) or scale<=0.:
        raise ValueError('cold-start latent inputs must vary')
    w = np.linalg.solve(cov+RIDGE*scale*np.eye(len(x)),xy-np.outer(x,y))
    return w,y-x@w


def risk_component(values, beta):
    e = F.normalize(beta,dim=0)
    return (values@e)[...,None]*e


def support_examples(signal, track):
    q,*_ = track
    v = signal['available'][:, list(embed.VALUES)]
    # At release both the incremental teacher inputs and their persistence
    # baseline are well observed. Every compared method uses this same support.
    ready = v[q[:,None]+np.arange(lag.H+1)].all((1,2))
    return q, ready


def origin_graph(base, signal, track, i):
    q,_,z,_,_,phase,correction = track
    x,m = lag._windows(base,signal,q[i])
    j = lag.delayed_slots(q,q[i])[0]
    return x,m,layer.patcher(phase[j],correction[j]),z


def actual_targets(base, signal, q, i, transform):
    x,m = lag._windows(base,signal,q[i]+lag.H)
    return teacher_increment(base,x,m,transform)


def device_readout(base, fitted):
    return tuple(torch.as_tensor(v, device=base.center.device, dtype=base.center.dtype) for v in fitted)


@torch.no_grad()
def cold_examples(base, signal, track):
    q,ok = support_examples(signal,track)
    ids = np.flatnonzero(ok)
    out, target, reversed_target = [],[],[]
    for start in range(0,len(ids),group.BATCH):
        i = ids[start:start+group.BATCH]
        x,m,f,z = origin_graph(base,signal,track,i)
        t = actual_targets(base,signal,q,i,f)
        d = z.new_zeros((1,len(i),base.cfg.dim,embed.PATCH*len(embed.VALUES)))
        h,_ = embed.forward_embed(base,x,m,f,d)
        out.append(h[0].cpu().numpy())
        target.append(t[0].cpu().numpy())
        reversed_target.append(t[1].cpu().numpy())
    if not out:
        raise ValueError('cold-start shot has no observed H25 for the latent forecast')
    return tuple(np.concatenate(v) for v in (out,target,reversed_target))


def fit_cold(lock,p,args,decoder,path):
    base, records = args[0], []
    for sid in p['order'][:p['startup']]:
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        # The same frozen numerical map is available after the cold segment.
        sig = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
        track = group.parent_track(sig,args,decoder,path)
        records.append(cold_examples(base,sig,track))
    stats = [statistics(x,y) for x,y,_ in records]
    tests = []
    for i,(x,y,r) in enumerate(records):
        w,b = fit(stats[:i]+stats[i+1:])
        pred = x@w+b
        # All quantities are auxiliary, not outcome labels. The third error
        # scores the very same predictor against the reversed observed block.
        tests.append(dict(shot=p['order'][i],packets=len(x),values=dict(
            forecast_mse=float(np.mean((y-pred)**2)), hold_mse=float(np.mean(y**2)),
            reversed_mse=float(np.mean((r-pred)**2)))))
    return fit(stats),tests


def directions(base,signal,track,i,readout,*,extended):
    q,beta = track[0],track[3]
    x,m,f,z = origin_graph(base,signal,track,i)
    with torch.no_grad():
        y,rev = actual_targets(base,signal,q,i,f)
    d = z.new_zeros((len(i),base.cfg.dim,embed.PATCH*len(embed.VALUES)),requires_grad=True)
    h,_ = embed.forward_embed(base,x,m,f,d[None])
    prediction = h[1]@readout[0]+readout[1]
    # Four negative gradients of latent MSE, up to a shared scalar factor.
    error = y-prediction.detach()
    permuted, held = rev-prediction.detach(), -prediction.detach()
    parallel = risk_component(error,beta)
    force = (error,permuted,held,error-parallel)
    if extended:
        force += (permuted-risk_component(permuted,beta),held-risk_component(held,beta),parallel)
    return torch.stack([torch.autograd.grad(prediction,d,g,retain_graph=j+1<len(force))[0]
                        for j,g in enumerate(force)])


def assemble(table,prev,modes):
    a,b,c,d,*extra = table.unbind()
    lagged = np.r_[0,prev+1]
    raw = (a,a[lagged],b,c,d)
    if modes!=MODES:
        rev,hold,parallel = extra
        raw += (d[lagged],rev,hold,parallel)
    return torch.stack(raw)


def estimate(signal,args,decoder,path,readout,arc,modes):
    base = args[0]
    track = group.parent_track(signal,args,decoder,path)
    q,observed = support_examples(signal,track)
    np.testing.assert_array_equal(q+lag.H+1,arc['release'])
    prev,eligible = embed.feedback_support(q, observed)
    eligible &= arc['eligible']
    extended = modes!=MODES
    table = base.center.new_zeros((7 if extended else 4,len(q)+1,base.cfg.dim,embed.PATCH*len(embed.VALUES)))
    ids = np.flatnonzero(eligible)
    used = np.unique(np.r_[ids,prev[ids]])
    for start in range(0,len(used),group.BATCH):
        i = used[start:start+group.BATCH]
        table[:,i+1] = directions(base,signal,track,i,readout,extended=extended)
    raw = assemble(table,prev,modes)
    if not torch.isfinite(raw).all():
        raise FloatingPointError('nonfinite measured latent feedback direction')
    return track,embed.stable_direction(raw,raw),eligible,observed


@torch.no_grad()
def witness(base,signal,track,weights,elig,observed,readout,dose):
    q,_,_,beta,*_ = track
    take = np.flatnonzero(elig)
    j = take+2
    ok = j<len(q)
    i,j = take[ok],j[ok]
    action = q[i]+embed.OFFSET+embed.PATCH
    ok = (q[j]-q[i]==2*lag.STRIDE) & observed[j] & (dose[action]>0.)
    i,j,action = i[ok],j[ok],action[ok]
    clocks,rows,parallel_rows = [],[],[]
    for start in range(0,len(i),group.BATCH):
        a,b,t = (v[start:start+group.BATCH] for v in (i,j,action))
        x,m,f,z = origin_graph(base,signal,track,b)
        y,_ = actual_targets(base,signal,q,b,f)
        h,(mass,_) = embed.forward_embed(base,x,m,f,weights[:,a+1],requested=z.new_tensor(dose[t]))
        pred = h@readout[0]+readout[1]
        pred = torch.cat((pred,torch.zeros_like(pred[:1])))
        error = pred-y
        e = F.normalize(beta,dim=0)
        valid = mass.cpu().numpy()>0.
        clocks.append(np.c_[q[a],t,q[b],q[b]+lag.H+1][valid])
        rows.append(error.square().mean(dim=-1).T.cpu().numpy()[valid])
        parallel_rows.append((error@e).square().T.cpu().numpy()[valid])
    empty = np.empty((0,len(weights)+2),np.float32)
    return (np.concatenate(clocks), np.concatenate(rows), np.concatenate(parallel_rows)) if clocks else (
        np.empty((0,4),np.int64),empty,empty)


def predict(signal,args,decoder,path,readout,arc,*,modes):
    track,states,eligible,observed = estimate(signal,args,decoder,path,readout,arc,modes)
    arm,dose,error = embed.render(args[0],signal,track,states,eligible,arc,modes=modes)
    out = dict(dose=dose,q=track[0],eligible=eligible)
    out['clock'],out['loss'],out['parallel_loss'] = witness(
        args[0],signal,track,states,eligible,observed,readout,dose)
    return arm,out,error


def seal(out, only=None, *, orthogonal_controls=False):
    start = monotonic()
    lock,p,ref,args,decoder,path = history.inputs()
    order = p['order']
    modes = (*MODES,*ORTHOGONAL_CONTROLS) if orthogonal_controls else MODES
    arms = (*embed.OLD,'legacy_common',*modes)
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},arms=arms,only=only,
        aux_readout=dict(fit_ids=order[:p['startup']], labels=False,method='shot equal-weight ridge',
                         ridge_as_mean_input_variance=RIDGE,target='future CLS difference vs held observed levels'),
        movable_channels=[io23.RAW23[c] for c in embed.VALUES],
        update='negative latent MSE gradient on eight measured lanes only; fresh from zero',
        clock=dict(horizon=lag.H,release=lag.H+1,offset=embed.OFFSET,handoff=embed.PATCH,
                   dose='same archived cap times a common transition; no persisted state'),
        labels='no outcomes read before the full stream is sealed'))
    base = args[0]
    old_state = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    files,checks,max_error = [],{},np.zeros(2)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no preseal shot labels')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('no preseal HDF5 truth'))):
        fitted,cold = fit_cold(lock,p,args,decoder,path)
        readout = device_readout(base,fitted)
        with (out/'readout.npz').open('xb') as f:
            np.savez_compressed(f,weight=fitted[0],bias=fitted[1])
        write(out/'cold_crossfit.json',cold)
        print(dict(stage='cold_latent_crossfit',packets=sum(r['packets'] for r in cold),
                   error={k:float(np.mean([r['values'][k] for r in cold])) for k in cold[0]['values']}),
              flush=True)
        for rank,sid in enumerate(order,1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,ref)
            if rank<=p['startup']:
                parent = arc['bias_only']
                arm = dict.fromkeys(arms[len(embed.OLD):],parent)
                empty = np.empty((0,len(modes)+2),np.float32)
                trace = dict(dose=np.zeros_like(parent),q=np.empty(0,np.int64),eligible=np.empty(0,bool),
                             clock=np.empty((0,4),np.int64),loss=empty,parallel_loss=empty)
            else:
                # Original invariant: fixed files and parent predictions on this stream.
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                arm,trace,err = predict(sig,args,decoder,path,readout,arc,modes=modes)
                max_error = np.maximum(err,max_error)
                if only is not None or rank in (21,120,360) or sid in CASES:
                    n = min(len(raw['times']),650)
                    mature = arc['release']<n
                    small = {k:(v[mature] if k in ('release','eligible') else v[:n]) for k,v in arc.items()}
                    trimmed = {k:v[:n] for k,v in sig.items() if k in ('values','available')}
                    a,t,_ = predict(trimmed,args,decoder,path,readout,small,modes=modes)
                    seen = trace['clock'][:,-1]<n
                    np.testing.assert_array_equal(t['clock'],trace['clock'][seen])
                    np.testing.assert_allclose(t['loss'],trace['loss'][seen],rtol=0.,atol=8e-6)
                    np.testing.assert_allclose(t['parallel_loss'],trace['parallel_loss'][seen],rtol=0.,atol=8e-6)
                    np.testing.assert_array_equal(t['eligible'],trace['eligible'][:len(t['eligible'])])
                    np.testing.assert_allclose(t['dose'],trace['dose'][:n],rtol=0.,atol=2e-6)
                    for k,v in a.items():
                        np.testing.assert_allclose(v,arm[k][:n],rtol=0.,atol=3e-6)
                        for th in p['calibration'].values():
                            np.testing.assert_array_equal(v>=th,arm[k][:n]>=th)
                    checks[str(sid)] = dict(points=n,witnesses=int(seen.sum()),
                                          probability=max(float(abs(v-arm[k][:n]).max()) for k,v in a.items()))
            name = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,**{k:arc[k] for k in embed.OLD},**arm,**trace)
            files.append(name)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(stage='latent_stream_sealed',rank=rank,seconds=round(monotonic()-start,2)),
                      flush=True)
    if any(not torch.equal(v.detach().cpu(),old_state[k]) for k,v in base.state_dict().items()):
        raise ValueError('frozen parent weights changed')
    write(out/'seal.json',dict(shots=files,prefix=checks,max_error=max_error.tolist(),
                               seconds=monotonic()-start))


def summarize_witness(groups,modes):
    report = {}
    for k,g in groups.items():
        report[k] = dict(shots=len(g),windows=sum(len(v) for v in g))
        if g:
            mean = np.stack([v.mean(0,dtype=np.float64) for v in g]).mean(0)
            for name,line in zip(('latent_mse','readout_direction_mse'),mean,strict=True):
                report[k][name] = dict(zip(('parent',*modes,'zero_increment'),line.tolist(),strict=True))
    return report


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    order = p['order']
    arms = tuple(p['arms'])
    modes = arms[len(embed.OLD)+1:]
    if arms[:len(embed.OLD)+1]!=(*embed.OLD,'legacy_common') or modes not in (
            MODES,(*MODES,*ORTHOGONAL_CONTROLS)):
        raise ValueError('unexpected sealed arm selection')
    if p['only'] is not None or len(order)!=360 or s['shots']!=[f'{i:04d}_{sid}.npz'
            for i,sid in enumerate(order,1)]:
        raise ValueError('score only from a complete sealed stream')
    archived = read(history.ARCHIVE/'per_shot.json')
    tab = {c:{k:[] for k in arms} for c in p['calibration']}
    ranking = {k:[] for k in arms}
    frames = {g:[] for g in ('all','late_before_decision','never_before_decision','safe_pre_fp')}
    hard, support, cold_fp = [],[],0
    for r,(sid,fn) in enumerate(zip(order,s['shots'],strict=True),1):
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/fn,allow_pickle=False) as f:
            z = dict(f)
        cold = r<=p['startup']
        common = z['dose']>0.
        support.append(dict(shot=sid,updates=int(z['eligible'].sum()),active=int(common.sum())))
        for cap,rows in tab.items():
            th = p['original'] if cold else p['calibration'][cap]
            for k in arms:
                row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,z[k],y,down,th))
                if k in embed.OLD and row!=archived[cap][k][r-1]:
                    raise ValueError('original comparator changed')
                if k in modes:
                    np.testing.assert_array_equal(z[k][~common],z['bias_only'][~common])
                rows[k].append(row)
        ref = tab['0.2']['bias_only'][-1]
        if cold:
            cold_fp += int(not y and ref['alarm'])
            continue
        for k in arms:
            ranking[k].append(dict(disruptive=y,peak=peak_for_shot(t,z[k],y,down)))
        clock = z['clock']
        both = np.stack((z['loss'],z['parallel_loss']),axis=1)
        if len(both):
            frames['all'].append(both)
        tau = p['calibration']['0.2']
        old = tab['0.2']['current_5'][-1]
        if y and not old['timely']:
            ms = np.round((down-t)*1000.,9)
            timely = (ms>30.) & (ms<=125.)
            which = 'late' if old['alarm'] else 'never'
            same = timely & common
            hard.append(dict(shot=sid,kind=which,active_near=int(same.sum()),
                crossing={k:bool((z[k][same]>=tau).any()) for k in modes}))
            use = timely[clock[:,1]] & (ms[clock[:,3]]>30.)
            if use.any():
                frames[f'{which}_before_decision'].append(both[use])
        elif not y and ref['alarm']:
            use = clock[:,3]<np.flatnonzero(z['bias_only']>=tau)[0]
            if use.any():
                frames['safe_pre_fp'].append(both[use])
    metrics = {c:{k:lead.summarize(v) for k,v in m.items()} for c,m in tab.items()}
    pairs = [('bias_only', k) for k in modes]+[(k,'latent_current')
        for k in ('current_5','legacy_common','latent_earlier','latent_reversed','latent_hold','orthogonal_current')]
    if modes!=MODES:
        pairs += [(k,'orthogonal_current') for k in (*ORTHOGONAL_CONTROLS,'current_5')]
    paired = {c:{f'{a}__to__{b}':dict(flips=flips(v[a],v[b]),lead=lead.paired(v[a],v[b]))
                 for a,b in pairs} for c,v in tab.items()}
    bands = {str(n):{k:{i:r[i] for i in ('tp','fp')} for k,v in ranking.items()
                     for r in [best_at_count(v,n-cold_fp)]} for n in (18,27,36)}
    write(out/'per_shot.json',tab)
    write(out/'result.json',dict(metrics=metrics,paired=paired,witness=summarize_witness(frames,modes),
        hard=hard,support=support,postseal_ranking_only=bands,raw_verified=360*len(arms)*len(tab)))
    print({c:{k:(r['by_deadline']['30']['tp'],r['fp']) for k,r in v.items()} for c,v in metrics.items()},
          flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--score',action='store_true')
    mode.add_argument('--smoke-rank',type=int,action='append')
    parser.add_argument('--orthogonal-controls',action='store_true')
    cfg = parser.parse_args()
    if cfg.score:
        score(cfg.output.resolve())
    else:
        seal(cfg.output.resolve(),cfg.smoke_rank,orthogonal_controls=cfg.orthogonal_controls)
