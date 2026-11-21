"""Separate cross-device linear readback error from frozen-encoder bottlenecks."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from . import commissioning_patch_retention_20261114 as layers
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, write
from .operating_point_audit import best_at_count


probe = layers.probe
SOURCE = io23.ROOT/'outputs/east360_patch_retention_20261114_v1'
MODES = ('source', 'cold', 'cold_local_offset', 'cold_local_true', 'cold_local_shifted')
ARMS = tuple(f'{k}_{m}' for k in layers.STAGES for m in MODES)
SPACING = probe.WIDTH
RIDGE = .05


def calibration_slots(ends, updates):
    """Fit only on disjoint earlier windows, with no target/input overlap."""
    return np.searchsorted(updates+SPACING,ends,side='right')


def local_corrections(x,residual,ends,eligible):
    """Paired and half-window rolled residual targets share exactly the same features."""
    select = ((ends-(SPACING-1))%SPACING==0) & eligible
    j = np.flatnonzero(select)
    slots = calibration_slots(ends,ends[j])
    x = np.asarray(x,np.float64)
    residual = np.asarray(residual,np.float64)
    gram = np.eye(x.shape[-1])*RIDGE
    cross = np.zeros((2,x.shape[-1],residual.shape[-1]))
    result = np.zeros((3,*residual.shape),np.float32)
    common = np.zeros(residual.shape[-1])
    for i,k in enumerate(j,1):
        s = x[k]
        target = residual[k]
        average = target.mean(0)
        centered = target-average
        common += average
        gram += s.T@s/len(s)
        for p,v in enumerate((centered,np.roll(centered, layers.SLOTS//2,axis=0))):
            cross[p] += s.T@v/len(s)
        use = np.flatnonzero(slots==i)
        if len(use):
            solved = np.linalg.solve(gram,np.concatenate(tuple(cross),axis=1))
            solved = np.stack(np.hsplit(solved,len(cross)))
            shift = common/(i+2.)
            result[0,use] = shift
            result[1:,use] = x[use]@solved[:,None]+shift
    return result,slots


def centers(view):
    raw = layers.temporal_coordinates(view['raw'])
    return raw,raw.reshape(len(raw),layers.SLOTS,layers.PATCH*len(layers.COLS))


def pooling(lock, order, base):
    records = {}
    for rank,sid in enumerate(order,1):
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        signal = probe.lag.prepare_actual(raw,rank,lock['mapping'])
        ends = np.arange(SPACING-1,len(raw['times']),SPACING,dtype=np.int64)
        packet = layers.encoder_views(base,signal,ends)
        valid = packet.pop('supported')
        records[sid] = {k:v[valid] for k,v in packet.items()}
    return records


def cold_models(shots,source_models):
    rows = [v for v in shots if len(v['raw'])]
    x = {k:np.concatenate([v[k] for v in rows]) for k in (*layers.STAGES,'raw')}
    _,goal = centers(x)
    mass = np.concatenate([np.full(len(r['raw']),1./(len(rows)*len(r['raw']))) for r in rows])
    models = {}
    for k in layers.STAGES:
        # Map only the source decoder's errors; source targets/labels never change.
        err = goal-layers.decode(x[k],source_models[k])
        models[k] = layers.fit_decoder(x[k],err,mass,ridge=RIDGE)
    return models


def errors(v,truth):
    return np.stack([layers.recovery_error(x.reshape(truth.shape),truth) for x in v],axis=1).astype(np.float32)


def cold_oof(cold,source_models):
    result=[]
    for sid,v in cold.items():
        if not len(v['raw']):
            continue
        active = cold_models([v for k,v in cold.items() if k!=sid],source_models)
        truth,_ = centers(v)
        stage = {}
        for k in layers.STAGES:
            source = layers.decode(v[k],source_models[k])
            cold_predict = source+layers.decode(v[k],active[k])
            stage[k] = errors([source,cold_predict],truth).mean(0).tolist()
        result.append(dict(shot=sid,frames=len(v['raw']),recovery=stage))
    return result


def views(v,ends,source_models,cold_model,head):
    truth,flat = centers(v)
    traces,metrics = [],[]
    support = 0
    for k in layers.STAGES:
        source = layers.decode(v[k],source_models[k])
        cold = source+layers.decode(v[k],cold_model[k])
        z = v[k].astype(np.float64)
        # Separate estimating an all-patch offset from learning the residual's
        # order given content. Scaling uses the fixed decoder to define units.
        spread = cold_model[k]['xscale']
        local = (z-z.mean(1,keepdims=True))/spread
        corrections,support = local_corrections(local,flat-cold,ends,v['supported'])
        curves = [source,cold,*(cold[None]+corrections)]
        traces.extend(probe.apply_probe(v['op'],layers.flatten(t),head) for t in curves)
        metrics.append(errors(curves,truth))
    return np.column_stack(traces).astype(np.float32),np.concatenate(metrics,axis=1),support


def scale_cold(fit,records):
    for k,v in fit.items():
        # Each shot has one total contribution. This scale only conditions the
        # 64-dimensional residual solve, not the frozen risk model or input.
        per = [(a[k]-a[k].mean(1,keepdims=True)).astype(np.float64)**2 for a in records if len(a[k])]
        v['xscale'] = np.sqrt(np.maximum(np.mean([a.mean((0,1)) for a in per],axis=0),.05**2))
    return fit


def observe(out):
    begun = monotonic()
    original = read(SOURCE/'seal.json')
    s = read(SOURCE/'source.json')
    source_models = {k:{a:np.asarray(b) for a,b in v.items()} for k,v in s['decoders'].items()}
    lock,p,_,args,decoder,path = probe.history.inputs()
    order = p['order']
    if order!=original['order'] or len(original['completed'])!=360 or p['startup']!=20:
        raise ValueError('same already sealed EAST development stream required')
    base = args[0]
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    completed,prefix = [],{}
    parity = 0.
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no EAST label in readback')),
          patch.object(probe.group.hdf,'_raw_truth',side_effect=RuntimeError('no EAST label in readback'))):
        cold = pooling(lock,order[:20],base)
        oof = cold_oof(cold,source_models)
        model = scale_cold(cold_models(cold.values(),source_models),cold.values())
        write(out/'cold.json',dict(fit_ids=order[:20],leave_one_shot=oof,
            model={k:{a:b.tolist() for a,b in v.items()} for k,v in model.items()}))
        head = s['heads']['raw_exact']
        for rank,sid in enumerate(order[20:],21):
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            ends = probe.end_grid(len(raw['times']))
            signal = probe.lag.prepare_actual(raw,rank,lock['mapping'])
            name = f'{rank:04d}_{sid}.npz'
            with np.load(SOURCE/'shots'/name,allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored['ends'],ends)
                prior_scores = stored['scores'][:,2:5]
                prior_errors = stored['error'][:,1:4]
                prior_available = stored['available']
            at = layers.transform_schedule(signal,args,decoder,path)
            v = layers.encoder_views(base,signal,ends,at)
            x,e,n = views(v,ends,source_models,model,head)
            known = v['supported']
            original_columns = np.arange(0,len(ARMS),len(MODES))
            np.testing.assert_array_equal(known,prior_available)
            np.testing.assert_allclose(x[:,original_columns],prior_scores,atol=5e-6,rtol=0.)
            np.testing.assert_allclose(e[:,original_columns],prior_errors,atol=5e-6,rtol=0.)
            if len(ends):
                parity = max(parity,float(abs(x[:,original_columns]-prior_scores).max()))
            if rank in (21,120,360):
                cut = min(704,len(raw['times']))
                signal_prefix = {k:v[:cut] for k,v in signal.items() if k in ('values','available')}
                pre = ends<cut
                short = layers.encoder_views(base,signal_prefix,ends[pre],
                    layers.transform_schedule(signal_prefix,args,decoder,path))
                a,b,c = views(short,ends[pre],source_models,model,head)
                np.testing.assert_allclose(a,x[pre],atol=4e-5,rtol=0.)
                np.testing.assert_allclose(b,e[pre],atol=3e-5,rtol=0.)
                np.testing.assert_array_equal(c,n[pre])
                prefix[str(sid)] = dict(points=cut,earlier_disjoint_frames=int(c[-1]))
            with (out/'shots'/name).open('xb') as file:
                np.savez_compressed(file,ends=ends,available=known,scores=x,error=e,earlier_packets=n)
            completed.append(name)
            if rank in (21,120,240,360):
                print(dict(stage='readback_sealed',rank=rank,seconds=round(monotonic()-begun,2)),flush=True)
    write(out/'seal.json',dict(order=order,cold=20,arms=ARMS,completed=completed,prefix_checks=prefix,
        same_source_readback_max=parity,checkpoint='no change to frozen source or target risk',
        train='no target outcomes; update only on completed 128-point windows disjoint from current window',
        comparison='bias-only trajectory; fit only auxiliary decoder, never deployed to alarm',
        seconds=monotonic()-begun))


def estimate_means(shots):
    a = np.stack(shots).mean(0)
    return {k:dict(zip(layers.ERRORS,x.tolist(),strict=True)) for k,x in zip(ARMS,a,strict=True)}


def score(out):
    p = read(out/'seal.json')
    old = read(probe.history.ARCHIVE/'per_shot.json')['0.2']['current_5']
    reference = read(probe.history.ARCHIVE/'seal.json')
    threshold = read(SOURCE/'seal.json')['calibration']['0.2']
    if p['order']!=[r['shot'] for r in old] or len(p['completed'])!=340:
        raise ValueError('before truth require the same 20 preparation plus 340 sealed discharges')
    recovered = {k:[] for k in ('whole','disjoint_history','never_near','late_near','false_before_alarm')}
    active = {k:0 for k in recovered}
    peaks = {k:[] for k in ARMS}
    pair_scores = {}
    for rank,sid in enumerate(p['order'][20:],21):
        name = f'{rank:04d}_{sid}.npz'
        if p['completed'][rank-21]!=name:
            raise ValueError('shot order mismatch')
        y,down,t = probe.group.hdf._raw_truth(sid)
        prior = old[rank-1]
        if (y,down)!=(prior['disruptive'],prior['down_s']):
            raise ValueError('archived truth mismatch')
        with np.load(out/'shots'/name,allow_pickle=False) as archive:
            v = dict(archive)
        pair_scores[sid] = v
        e,g,era = (v[k] for k in ('ends','available','earlier_packets'))
        np.testing.assert_array_equal(e,probe.end_grid(len(t)))
        lead = np.round((down-t[e])*1000.,9) if y else np.full(len(e),np.inf)
        indices = dict(whole=g,disjoint_history=g & (era>0))
        if y and not prior['timely']:
            indices['late_near' if prior['alarm'] else 'never_near'] = g & (era>0) & (lead>30.) & (lead<=125.)
        elif not y and prior['alarm']:
            curve = probe.history.archived_shot(rank,sid,reference)['current_5']
            first = np.flatnonzero(curve>=threshold)[0]
            indices['false_before_alarm'] = g & (era>0) & (e<=first)
        for k,i in indices.items():
            if i.any():
                recovered[k].append(v['error'][i].mean(0))
                active[k] += int(i.sum())
        use = g & (lead>30.)
        if not y and not use.any():
            raise ValueError('common observation required for safe peak')
        for k,x in zip(ARMS,v['scores'].T,strict=True):
            peaks[k].append(dict(shot=sid,disruptive=y,peak=float(expit(np.float64(x[use].max())))
                if use.any() else None))
    results = {k:dict(n_shots=len(n),n_frames=active[k],mae=estimate_means(n))
               for k,n in recovered.items() if len(n)}
    ranking = {str(n):{k:{m:a[m] for m in ('tp','fp')} for k,row in peaks.items()
                       for a in [best_at_count(row,n)]} for n in (12,21,30)}
    # Same precommitted paired anchors; no fitting or ranking chooses them.
    comparisons = read(layers.PAIRED)['comparisons']
    pair_results = {}
    for name,c in comparisons.items():
        groups = pair_results[name] = {}
        for group in ('no_first_alarm','late_first_alarm'):
            rows=[]
            for pair in c['pairs']:
                if pair['positive_group']!=group:
                    continue
                a,b = [pair_scores[pair[k]] for k in ('positive_shot','safe_shot')]
                clocks = [(pair['anchors'][k]['positive_index'],pair['anchors'][k]['safe_index'])
                          for k in ('earlier_32','current')]
                clock = [(int(np.searchsorted(a['ends'],i)),int(np.searchsorted(b['ends'],j))) for i,j in clocks]
                if any((a['ends'][i],b['ends'][j])!=expected for (i,j),expected in zip(clock,clocks,strict=True)):
                    raise ValueError('pair time is not on the locked score grid')
                if all(a['available'][i] and b['available'][j] for i,j in clock):
                    rows.append(dict(positive=pair['positive_shot'],safe=pair['safe_shot'],
                        margin=[(a['scores'][i]-b['scores'][j]).tolist() for i,j in clock],
                        past_packets=[[int(a['earlier_packets'][i]),int(b['earlier_packets'][j])] for i,j in clock]))
            margins = np.asarray([r['margin'] for r in rows]).reshape(-1,2,len(ARMS))
            groups[group] = dict(n=len(rows),both=dict(zip(ARMS,(margins>0.).all(1).sum(0).tolist(),strict=True)),
                                 pairs=rows)
    write(out/'result.json',dict(errors=results, suffix_equal_false_peak_ranking=ranking, pairs=pair_results,
        meaning='only waveforms read from frozen existing features; no risk or alarm update'))
    print(dict(shot_support={k:(v['n_shots'],v['n_frames']) for k,v in results.items()},
               recovered_radiation_detail={k:{a:v['mae'][a]['radiation_within_patch'] for a in ARMS}
                    for k,v in results.items()},false30=ranking['30']),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--score',action='store_true')
    args = cli.parse_args()
    (score if args.score else observe)(args.output.resolve())
