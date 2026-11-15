"""Recompute the complete fixed stream and challenge the causal first alarms."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_write_delay_20261112 as delay
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group,lag = delay.group,delay.lag
FOLDER = io23.ROOT/'probe_outputs/commissioning_write_delay_east360_20261112_v2'
CHECK = ('fixed','bias_only','current_0','earlier_0','current_5','earlier_5')


def score(out):
    p,s,ref = (read(FOLDER/f'{k}.json') for k in ('protocol','seal','per_shot'))
    if (len(p['order'])!=360 or len(s['curve_sha256'])!=360
            or p['history_seal']!=sha(delay.HISTORY/'seal.json')
            or p['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or p['executed_sha256']!=sha(FOLDER/'executed_probe.py')
            or p['group_sha256']!=sha(FOLDER/'executed_group.py')):
        raise ValueError('fixed 360 replay seal or its references changed')
    o,ss = (read(group.REFERENCE/f'{k}.json') for k in ('protocol','seal'))
    rows = {b:{k:[] for k in CHECK} for b in p['calibration']}
    active = np.zeros(len(delay.WAITS),np.int64)
    releases,eligible = 0,0
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        a,b = FOLDER/'shots'/name,group.REFERENCE/'shots'/name
        if sha(a)!=s['curve_sha256'][name] or sha(b)!=ss['curve_sha256'][name]:
            raise ValueError('changed sealed curve')
        d,down,t = group.hdf._raw_truth(sid)
        with np.load(a,allow_pickle=False) as f,np.load(b,allow_pickle=False) as clock:
            np.testing.assert_array_equal(clock['times'],t)
            np.testing.assert_array_equal(f['release'],clock['release'])
            rel,e,dose = f['release'],f['eligible'],f['dose']
            # Cold preparation uses the same unmodified scored forecast for all arms.
            if rank<=p['startup'] and np.any(dose):
                raise ValueError('cold preparation was scored after updating')
            if rank>p['startup']:
                np.testing.assert_array_equal((rel-lag.H-1-lag.FIRST)%lag.STRIDE,np.zeros_like(rel))
                releases += len(rel)
                eligible += int(e.sum())
            for j,w in enumerate(delay.WAITS):
                changed = dose[j]>0.
                slots = np.searchsorted(rel+w,np.arange(len(t)),side='right')
                if (changed & ~np.r_[False,e][slots]).any():
                    raise ValueError('temporary state applied before allowed arrival')
                active[j] += changed.sum()
                for k in (f'current_{w}',f'earlier_{w}'):
                    np.testing.assert_array_equal(f[k][~changed],f['bias_only'][~changed])
            for band,tau in p['calibration'].items():
                for k in CHECK:
                    value = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,f[k],d,down,
                                    p['original'] if rank<=p['startup'] else tau))
                    if ref[band][k][rank-1]!=value:
                        raise ValueError('HDF5 first alarm differs from archived score')
                    rows[band][k].append(value)
    result = {}
    for b,v in rows.items():
        result[b] = dict(metrics={k:lead.summarize(a) for k,a in v.items()},paired={
            f'{a}__to__{c}':dict(flips=flips(v[a],v[c]),lead=lead.paired(v[a],v[c]))
            for a,c in (('fixed','current_0'),('bias_only','current_0'),('earlier_0','current_0'),
                        ('bias_only','current_5'),('earlier_5','current_5'))})
    if o['order']!=p['order']:
        raise ValueError('reference roster mismatch')
    write(out,dict(scope='POSTSEAL_FIXED360_FIRST_ALARM_INDEPENDENT_SCORE',scores=360*len(CHECK)*len(rows),
        thresholds=p['calibration'],references={'delay':sha(FOLDER/'seal.json'),'clock':sha(group.REFERENCE/'seal.json')},
        samples=dict(mature_windows=releases,eligible_windows=eligible,written_endpoints=active.tolist()),bands=result))
    print({b:{k:(a['by_deadline']['30']['tp'],a['fp']) for k,a in v['metrics'].items()} for b,v in result.items()},
          flush=True)


def witness(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,ref,args,decoder,path = group.load_reference(lock)
    s = read(FOLDER/'seal.json')
    h = read(delay.HISTORY/'protocol.json')
    teach = delay.HISTORY/'teaching.json'
    if sha(teach)!=h['teaching_sha256']:
        raise ValueError('changed frozen signed-force teacher')
    w = torch.tensor(read(teach)['weights'],device='cuda',dtype=torch.float32)
    source = [state_digest(m) for m in args[:2]]
    trace = {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcomes during prefix prediction')):
        for sid in (70320,73405):
            rank = p['order'].index(sid)+1
            key = f'{rank:04d}_{sid}.npz'
            a,b = FOLDER/'shots'/key,group.REFERENCE/'shots'/key
            if (sha(a)!=s['curve_sha256'][key] or sha(b)!=ref['curve_sha256'][key]
                    or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]):
                raise ValueError('input or comparison changed')
            tau = p['calibration']['0.2']
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            with np.load(a,allow_pickle=False) as archive,np.load(b,allow_pickle=False) as clock:
                np.testing.assert_array_equal(raw['times'],clock['times'])
                n = int(np.flatnonzero(archive['current_5']>=tau)[0])
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                rel,elig = archive['release'],archive['eligible']
                cuts = []
                for stop in (n,n+1,min(n+34,len(raw['times']))):
                    prefix = {k:signal[k][:stop] for k in ('values','available')}
                    keep = rel<stop
                    with torch.enable_grad():
                        q,h0,z,beta,prior,phase,correction = group.parent_track(prefix,args,decoder,path)
                        available,fields,wave = group.feedback_waveform(prefix,q,args[2],z.device)
                        states,common = group._estimate_innovation(args[0],prefix,q,available,phase,correction,
                                                                    prior,fields,wave,decoder,w)
                    np.testing.assert_array_equal(q+lag.H+1,rel[keep])
                    np.testing.assert_array_equal(common,elig[keep])
                    cur,ear,dose,_,err,parent = delay.render(args[0],prefix,q,h0,z,beta,phase,correction,states,common)
                    compare = dict(bias_only=parent,current_0=cur[0],earlier_0=ear[0],current_5=cur[-1],earlier_5=ear[-1])
                    errors = {}
                    for k,v in compare.items():
                        refv = archive[k][:stop]
                        np.testing.assert_allclose(v,refv,atol=3e-6,rtol=0.)
                        for threshold in p['calibration'].values():
                            np.testing.assert_array_equal(v>=threshold,refv>=threshold)
                        errors[k] = float(np.max(abs(v-refv)))
                    np.testing.assert_array_equal(dose>0.,archive['dose'][:,:stop]>0.)
                    np.testing.assert_allclose(dose,archive['dose'][:,:stop],atol=3e-6,rtol=1e-6)
                    hit = np.flatnonzero(cur[-1]>=tau)
                    cuts.append(dict(prefix_points=stop,max_probability_error=max(errors.values()),
                        max_equal_dose_error=float(err),latest_first_alarm_index=int(hit[0]) if len(hit) else None))
                over = archive['current_5']>=tau
                edge = np.flatnonzero(~over[n:])
                slots = []
                for wait in (0,5):
                    j = int(np.searchsorted(rel+wait,n,side='right'))-1
                    r = int(rel[j])
                    slots.append(dict(wait_samples=wait,slot_release=r,forecast_origin=r-lag.H-1,
                        last_target_index=r-1,new_weight=float(group.handoff_fraction(r+wait,n,args[0].cfg.patch)),
                        previous_slot_release=int(rel[j-1]) if j else None))
                trace[str(sid)] = dict(rank=rank,first_index=n,time_s=float(raw['times'][n]),
                    probability=float(archive['current_5'][n]),parent_probability=float(archive['bias_only'][n]),
                    earlier_probability=float(archive['earlier_5'][n]),crossing_run=int(edge[0]) if len(edge) else len(over)-n,
                    prefix_witnesses=cuts,clock_at_alarm=slots)
    if source!=[state_digest(m) for m in args[:2]]:
        raise ValueError('witness changed frozen model')
    copyfile(Path(__file__),out/'executed.py')
    write(out/'prefix_at_first_alarm.json',dict(scope='SIGNAL_ONLY_PREFIX_AT_ACTUAL_FIRST_ALARMS',
        delay_seal=sha(FOLDER/'seal.json'),shots=trace,seconds=time.monotonic()-start))
    print(trace,flush=True)


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage',choices=('score','witness'))
    a.add_argument('out',type=Path)
    p = a.parse_args()
    dest = p.out.resolve()
    dest.mkdir(exist_ok=True)
    if p.stage=='score':
        score(dest/'independent_scores.json')
    else:
        witness(dest)
