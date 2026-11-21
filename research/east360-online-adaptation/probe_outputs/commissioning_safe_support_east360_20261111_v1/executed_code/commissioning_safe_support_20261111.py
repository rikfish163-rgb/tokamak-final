"""Isolate the effect of near-boundary safe training support for signed feedback."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


FAR, FAR_SHIFT, NEAR, NEAR_SHIFT = NEW = (
    'safe_far_true', 'safe_far_shifted', 'safe_near_true', 'safe_near_shifted',
)
ARMS = (*guard.ARMS, *NEW)
PAIRS = ((lag.BASES[-1], NEAR), (guard.PATH, FAR), (guard.PATH, NEAR),
         (FAR, NEAR), (FAR_SHIFT, NEAR_SHIFT), (FAR_SHIFT, FAR), (NEAR_SHIFT, NEAR))
BAND = .50
MATCH_TOLERANCE = np.array([.35, .35, .50, 1., .15])


def reachable_anchor(margin, origins, valid):
    """A safe training shot must be below threshold even before adaptation."""
    if not len(origins) or margin.max() >= 0.:
        return None
    ends = np.flatnonzero(valid & (np.arange(len(margin)) >= origins[0]+lag.H+1))
    if not len(ends):
        return None
    t = int(ends[margin[ends].argmax()])
    slot, _ = lag.delayed_slots(origins, [t])
    return int(origins[slot[0]-1]), t


def match_safety(candidates, allowed, excluded):
    ids = {r['shot'] for r in candidates}
    if len(ids) != len(candidates) or not ids <= allowed-excluded:
        raise ValueError('invalid or reused safety role')
    if any(not np.isfinite(r['live_margin']) or r['live_margin'] >= 0. for r in candidates):
        raise ValueError('safe matching requires an originally negative margin')
    near = [r for r in candidates if r['live_margin'] >= -BAND]
    far = [r for r in candidates if r['live_margin'] < -BAND]
    if not near or not far:
        return []
    keys = [np.asarray([r['key'] for r in rows]) for rows in (near,far)]
    gap = (keys[0][:,None]-keys[1][None])/MATCH_TOLERANCE
    feasible = (abs(gap)<=1.).all(-1)
    # The feasible cost is at most five; the dummy makes coverage primary.
    dummy = 6.*(len(near)+1)
    cost = np.concatenate((np.where(feasible,np.square(gap).sum(-1),2.*dummy),
                           np.full((len(near),len(near)),dummy)),axis=1)
    a,b = linear_sum_assignment(cost)
    return [(near[i],far[j]) for i,j in zip(a,b,strict=True) if j<len(far)]


def extra_safe_loss(pair_delta, safe_margin, safe_delta):
    up, down = pair_delta.T
    direct = .5*F.softplus(-up/.10).mean()+.25*F.softplus((down-up)/.20).mean()
    safe = F.softplus(safe_delta/.10)+F.relu(safe_delta-(-safe_margin).clamp_min(0.))/.10
    return direct+.5*safe.mean()


@torch.no_grad()
def _scan(role, authorized, excluded, args, lock, threshold):
    rows, counts = [], dict(safe=0, originally_alarmed=0, no_reachable_anchor=0, near=0, far=0)
    device = 'EAST' if role=='cold' else 'J-TEXT'
    for sid in authorized:
        if sid in excluded or io23.read_outcome(device,sid)[0]:
            continue
        counts['safe'] += 1
        if role=='cold' and sha(io23.path_for(device,sid))!=lock['hashes'][str(sid)]:
            raise ValueError('cold-start signal changed')
        raw = simple_data.read_shot(device,sid,rate_contract='nominal' if role=='cold' else 'strict')
        signal = lag.prepare_actual(raw,lag.STARTUP+1,lock['mapping']) if role=='cold' else raw
        tr, _, z, _ = guard._track(signal,args,source=role!='cold')
        m = z-logit(threshold)
        if not np.isfinite(m).all():
            raise FloatingPointError('nonfinite frozen margin')
        if m.max()>=0.:
            counts['originally_alarmed'] += 1
            continue
        q = tr['q']
        q = q[tr['observed'].all((1,2)) & (q+lag.H+1<tr['length'])]
        found = reachable_anchor(m,q,tr['matchable'])
        if found is None:
            counts['no_reachable_anchor'] += 1
            continue
        origin,t = found
        context = tr['context'][t,:3].astype(float)
        context[2] *= 4.
        key = [*np.tanh(context).tolist(),float(np.log2(len(z))),
               float(signal['available'][:,list(lag.CHANNELS)].mean())]
        rows.append(dict(shot=sid,role=role,origin=origin,action=t,lead_ms=None,
                         live_margin=float(m[t]),full_margin=float(m.max()),key=key,
                         match=[float(m[t]),*key[:3]]))
        counts['near' if m[t]>=-BAND else 'far'] += 1
    return rows, counts


def _pack_safety(rows, args, decoder, lock, roles, ids, original, cutoff):
    data, paths, manifest = [], [], []
    allowed = {'train':set(roles['train']),'val':set(roles['val']),'cold':set(ids[:lag.STARTUP])}
    offset = 0
    for r in rows:
        role = r['role']
        item, _, record = guard._record(role,r,positive=False,args=args,decoder=decoder,
            lock=lock,allowed=allowed[role],advanced={},
            threshold=cutoff if role=='cold' else original)
        if abs(float(item['m'].max())-r['full_margin'])>2e-6:
            raise ValueError('full safety peak changed in packed training support')
        paths.append(torch.arange(offset,offset+len(item['m']),device=item['m'].device))
        offset += len(item['m'])
        data.append(item)
        manifest.append(record)
    columns = {k:torch.cat([d[k] for d in data]) for k in data[0]}
    return columns,paths,manifest


def _fit(base, paired, support, tag):
    data, anchors, paths, _ = paired
    safe, extra, _ = support
    params = torch.nn.Parameter(data['key'].new_zeros((teacher.GROUPS,data['key'].shape[1])))
    opt = torch.optim.Adam([params],lr=.025)
    rng = np.random.default_rng(teacher.SEED)
    additional = np.random.default_rng(teacher.SEED+1)
    for step in range(teacher.STEPS):
        pair = rng.choice(len(anchors),min(len(anchors),24),replace=False)
        added = additional.choice(len(extra),min(len(extra),24),replace=False)
        opt.zero_grad(set_to_none=True)
        direction = guard._gain(base,data,params,anchors[pair].flatten()).reshape(-1,2)
        old, delta = guard._path_gains(base,data,params,[paths[i] for i in pair])
        ex_m, ex_d = guard._path_gains(base,safe,params,[extra[i] for i in added])
        loss = extra_safe_loss(direction,torch.cat((old,ex_m)),torch.cat((delta,ex_d)))
        loss = loss+.025*params.square().mean()
        loss.backward()
        opt.step()
        if (step+1)%96==0:
            print(dict(stage=tag,step=step+1,loss=float(loss.detach())),flush=True)
    if not torch.isfinite(params).all():
        raise FloatingPointError('nonfinite support teacher')
    return params.detach()


@torch.no_grad()
def _safe_summary(base, data, paths, methods):
    out = {}
    for name, p in methods.items():
        m, delta = guard._path_gains(base,data,p,paths)
        out[name] = dict(shots=len(m),old_alarm=int((m>=0.).sum()),
            new_alarm=int(((m<0.) & (m+delta>=0.)).sum()),
            removed_alarm=int(((m>=0.) & (m+delta<0.)).sum()),
            mean_peak_shift=float(delta.mean()),median_peak_shift=float(delta.median()))
    return out


def run(previous, output):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    ids, roles = lock['ids'], io23.source_roles()
    p0, seal, fitted = (read(previous/f) for f in ('protocol.json','seal.json','teacher.json'))
    prev_teacher = Path(p0['previous_teacher'])
    tp = read(prev_teacher/'protocol.json')
    mp = read(Path(tp['previous_lag'])/'protocol.json')
    archived = read(prev_teacher/'teacher.json')
    coords = next(Path(a).parent for a in p0['assets']
                  if Path(a).parent.name.startswith('commissioning_local_cross_response_'))
    p, _, args, assets = lag.parent._frozen(Path(mp['reference']),coords,lock)
    if (p0['scope']!='EAST360_SIGNED_POINT_VS_PATH_GUARD_DEVELOPMENT'
            or tuple(p0['arms'])!=guard.ARMS or p0['order']!=ids or p0['assets']!=assets
            or p0['original']!=p['original'] or p0['calibration']!=p['calibration']
            or p0['fit_roles']!={'J-TEXT/train':roles['train'],'EAST/cold':ids[:lag.STARTUP]}
            or p0['held_source']!=roles['val'] or len(seal['curve_sha256'])!=360
            or sha(previous/'teacher.json')!=p0['teacher_sha256']
            or sha(prev_teacher/'teacher.json')!=p0['previous_pairs_sha256']
            or sha(prev_teacher/'seal.json')!=p0['previous_seal']):
        raise ValueError('sealed signed trajectory reference required')
    for rank,sid in enumerate(ids[:lag.STARTUP],1):
        name = f'{rank:04d}_{sid}.npz'
        if sha(previous/'shots'/name)!=seal['curve_sha256'][name]:
            raise ValueError('cold start must seal before outcomes')
    decoder = lag._bind_forecast(Path(mp['source_decoder']).parent,p)
    producer = Path(__file__)
    code = dict(p0['code_sha256'],**{producer.name:sha(producer)})
    (output/'shots').mkdir(parents=True)
    (output/'executed_code').mkdir()
    for name,digest in code.items():
        src = producer.parent/name
        if sha(src)!=digest:
            raise ValueError('frozen dependency changed: '+name)
        copyfile(src,output/'executed_code'/name)
    base = args[0]
    frozen = tuple(state_digest(m) for m in args[:2])
    original, thresholds = p['original'],p['calibration']
    cutoff = thresholds['0.2']
    exclude = {k:{r['shot'] for pair in archived['pairs'][k] for r in pair} for k in ('train','cold')}
    source, source_counts = _scan('train',roles['train'],exclude['train'],args,lock,original)
    cold, cold_counts = _scan('cold',ids[:lag.STARTUP],exclude['cold'],args,lock,cutoff)
    matches = match_safety(source,set(roles['train']),exclude['train'])
    selection = dict(near_band=BAND,tolerance=MATCH_TOLERANCE.tolist(),
        key=['scaled_current','scaled_field','scaled_current_change','log2_samples','coverage'],
        source_counts=source_counts,cold_counts=cold_counts,
        source_pairs=[dict(near=a,far=b) for a,b in matches],shared_cold=cold)
    write(output/'selection.json',selection)
    print(dict(stage='support_locked',matches=len(matches),source=source_counts,cold=cold_counts,
               seconds=round(time.monotonic()-start,2)),flush=True)
    if not matches:
        raise ValueError('no matched near/far safety support in authorized history')
    pack = (args,decoder,lock,roles,ids,original,cutoff)
    pairs = guard._pack(('train','cold'),archived,*pack)
    np.testing.assert_equal(pairs[-1],fitted['fit_windows'])
    far_set = _pack_safety([*cold,*(b for _,b in matches)],*pack)
    near_set = _pack_safety([*cold,*(a for a,_ in matches)],*pack)
    far, near = (_fit(base,pairs,data,name) for name,data in (('fit_far',far_set),('fit_near',near_set)))
    previous_path, context = (torch.as_tensor(fitted[k],device=base.center.device,
                                              dtype=base.center.dtype) for k in ('path','context'))
    methods = {guard.PATH:previous_path,FAR:far,NEAR:near}
    safety_fit = {k:_safe_summary(base,data[0],data[1],methods)
                  for k,data in (('far',far_set),('near',near_set))}
    extra_fit_paths = {'far':far_set[-1],'near':near_set[-1]}
    del pairs,far_set,near_set
    held = guard._pack(('val',),archived,*pack)
    np.testing.assert_equal(held[-1],fitted['val_windows'])
    val_diag = guard._describe(base,held,far,near,context)
    val_diag = {dict(zip(guard.NEW[:4],NEW,strict=True)).get(k,k):v for k,v in val_diag.items()}
    del held
    val_pool,val_counts = _scan('val',roles['val'],set(),args,lock,original)
    val_near = [r for r in val_pool if r['live_margin']>=-BAND]
    if val_near:
        val_data,val_paths,val_manifest = _pack_safety(val_near,*pack)
        val_safety = _safe_summary(base,val_data,val_paths,methods)
        del val_data,val_paths
    else:
        val_safety,val_manifest = {},[]
    write(output/'teacher.json',dict(far=far.cpu().tolist(),near=near.cpu().tolist(),
        seed=teacher.SEED,steps=teacher.STEPS,selected_source_pairs=len(matches),shared_cold=len(cold),
        same_paired_positives=True,held_source_pairs=val_diag,held_source_counts=val_counts,
        held_near_rows=val_manifest,held_near_safety=val_safety,fit_safety=safety_fit,
        extra_fit_paths=extra_fit_paths))
    write(output/'protocol.json',dict(scope='EAST360_MATCHED_SAFE_SUPPORT_DEVELOPMENT',
        order=ids,arms=ARMS,new=NEW,startup=lag.STARTUP,original=original,calibration=thresholds,
        source_sha256=p['source_sha256'],assets=assets,code_sha256=code,previous_guard=str(previous),
        previous_seal=sha(previous/'seal.json'),selection_sha256=sha(output/'selection.json'),
        teacher_sha256=sha(output/'teacher.json'),fit_roles=p0['fit_roles'],held_source=roles['val'],
        comparisons=PAIRS,radius=lag.RADIUS,input_rms=lag.INPUT_RMS,
        first_effect=lag.FIRST_EFFECT,stride=lag.STRIDE,persistent_updates=0,
        safety_loss='same total weight; original paired safety plus matched supplemental safety',
        causal='no suffix HDF5 outcome until all predictions seal'))
    print(dict(stage='fit_sealed',held_near=val_safety,seconds=round(time.monotonic()-start,2)),flush=True)
    hashes,witnesses,updates = {},{},0
    names = dict(zip(guard.NEW[:4],NEW,strict=True))
    old_seal = read(prev_teacher/'seal.json')
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            path, old = previous/'shots'/name,prev_teacher/'shots'/name
            if sha(path)!=seal['curve_sha256'][name] or sha(old)!=old_seal['curve_sha256'][name]:
                raise ValueError('previous forecast changed')
            with np.load(path,allow_pickle=False) as ref, np.load(old,allow_pickle=False) as rg:
                times = ref['times']
                if rank<=lag.STARTUP:
                    curves, trace = dict.fromkeys(NEW,rg['fixed']),{}
                else:
                    if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                        raise ValueError('locked signal changed')
                    raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                    np.testing.assert_array_equal(raw['times'],times)
                    signal = lag.prepare_actual(raw,rank,lock['mapping'])
                    prior, scored, release, radius = guard._replay(signal,args,decoder,far,near,context)
                    np.testing.assert_array_equal(prior,rg[lag.BASES[-1]])
                    np.testing.assert_array_equal(release,ref['release'])
                    np.testing.assert_allclose(radius,ref['radius'],atol=3e-7,rtol=1e-6)
                    np.testing.assert_array_equal(scored[guard.CONTEXT],ref[guard.CONTEXT])
                    stop = int(release[0]) if len(release) else len(times)
                    curves = {k:scored[j] for j,k in names.items()}
                    for v in curves.values():
                        np.testing.assert_array_equal(v[:stop],prior[:stop])
                    if rank in (21,120,360):
                        end = lag.FIRST_EFFECT+3*lag.STRIDE+5
                        prefix = {k:signal[k][:end] for k in ('values','available')}
                        _, short, rel, _ = guard._replay(prefix,args,decoder,far,near,context)
                        for a,b in names.items():
                            np.testing.assert_allclose(short[a],curves[b][:end],atol=3e-6,rtol=0.)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(short[a]>=tau,curves[b][:end]>=tau)
                        np.testing.assert_array_equal(rel,release[release<end])
                        witnesses[str(sid)] = end
                    trace = dict(release=release,radius=radius[:,:4])
                    updates += int((radius[:,0]>0.).sum())
            if not all(np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite risk curve')
            target = output/'shots'/name
            with target.open('xb') as fd:
                np.savez_compressed(fd,times=times,**curves,**trace)
            hashes[name] = sha(target)
            if rank==21 or rank%40==0:
                print(dict(stage='sealed',rank=rank,seconds=round(time.monotonic()-start,2)),flush=True)
    if tuple(state_digest(m) for m in args[:2])!=frozen:
        raise ValueError('frozen model changed')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses))
    per = read(previous/'per_shot.json')
    for table in per.values():
        table.update({name:[] for name in NEW})
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        path = output/'shots'/name
        if sha(path)!=hashes[name]:
            raise ValueError('unsealed risk curve')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z:
            for band,table in per.items():
                tau = original if rank<=lag.STARTUP else thresholds[band]
                for k in NEW:
                    table[k].append(dict(shot=sid,disruptive=d,down_s=down,
                                         **score_curve(z['times'],z[k],d,down,tau)))
    write(output/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    paired = {c:{f'{a}__to__{b}':flips(g[a],g[b]) for a,b in PAIRS} for c,g in per.items()}
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],ARMS))
    write(output/'result.json',dict(status='COMPLETED_MATCHED_SAFE_SUPPORT_DEVELOPMENT',
        metrics=metrics,paired=paired,updates=updates,seconds=time.monotonic()-start))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items() if k in NEW}
           for c,g in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    cli = parser.parse_args()
    run(cli.previous.resolve(),cli.output.resolve())
