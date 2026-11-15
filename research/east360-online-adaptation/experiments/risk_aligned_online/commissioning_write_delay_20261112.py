"""Replay the same measured gradient with an extra first-block write delay."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .commissioning_group_carry_20261112 import HISTORY
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


lag,layer = group.lag,group.layer
WAITS = (0,2,5)
ARMS = (*group.OLD,*(f'{source}_{d}' for source in ('current','earlier') for d in WAITS))
PAIRS = ((group.bias.BIAS,'current_0'), *((f'{s}_0',f'{s}_{d}') for s in ('current','earlier') for d in WAITS[1:]),
         *((f'earlier_{d}',f'current_{d}') for d in WAITS))


@torch.no_grad()
def render(base,signal,q,h,z,beta,phase,correction,states,eligible,parent=None,*,waits=WAITS,logits=False,
           hidden_trace=None):
    release = q+lag.H+1
    ends = np.arange(len(z))
    slots, _ = lag.delayed_slots(q,ends)
    reconstruct = parent is None
    if logits and not reconstruct:
        raise ValueError('raw logits require reconstructing the parent')
    def output(value):
        return (value if logits else value.sigmoid()).cpu().numpy()
    if reconstruct:
        parent = output(z).copy()
    later = [np.searchsorted(release+d,ends,side='right') for d in waits]
    curves = np.broadcast_to(parent,(len(waits),2,len(parent))).copy()
    dose = np.zeros((len(waits),len(parent)),np.float32)
    # Extra delay only affects a short band at each release; reuse all other
    # forwards, including the common parent with its original clock.
    compare = []
    for j,d in enumerate(waits):
        delayed = later[j]
        if j==0:
            i = ends[slots>0]
        else:
            curves[j] = curves[0]
            dose[j] = dose[0]
            i = ends[(slots>0) & (ends-release[np.maximum(slots-1,0)]<base.cfg.patch+d)] if len(q) else ends[:0]
        pe,de = 0.,0.
        for start in range(0,len(i),group.BATCH):
            k = i[start:start+group.BATCH]
            s = delayed[k]
            r = np.r_[0,release+d][s]
            frac = group.handoff_fraction(r,k,base.cfg.patch)
            x,m = lag._windows(base,signal,k)
            enabled = torch.as_tensor(np.r_[False,eligible][s],device=z.device)
            previous = states[:,np.maximum(s-1,0)]*enabled[None,:,None,None]
            view,n,err = group.forward_hidden(base,x,m,layer.patcher(phase[slots[k]],correction[slots[k]]),
                writes=states[:,s],handoff=(previous,torch.as_tensor(frac,device=z.device,dtype=z.dtype)))
            if hidden_trace is not None:
                hidden_trace(j,k,view[0],view[-2:],n>0.)
            values = output(z[k]+((view-h[k])*beta).sum(-1))
            if reconstruct and j==0:
                parent[k] = values[0]
            pe = max(pe,float(np.abs(values[0]-parent[k]).max()))
            de = max(de,float(err))
            dose[j,k] = n.cpu().numpy()
            curves[j,:,k] = np.where((dose[j,k]>0.)[:,None],values[-2:].T,parent[k,None])
        compare.append((pe,de))
    pe,de = np.max(compare,axis=0)
    if pe>3e-6 or de>1e-5 or not np.isfinite(curves).all():
        raise ValueError('parent or within-arm dose parity failed')
    return curves[:,0],curves[:,1],dose,pe,de,parent


def predict_shot(signal,args,decoder,path,weights,parent,release,eligible):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    base = args[0]
    available,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
    states,common = group._estimate_innovation(base,signal,q,available,phase,correction,prior,fields,
                                                wave,decoder,weights)
    np.testing.assert_array_equal(q+lag.H+1,release)
    np.testing.assert_array_equal(common,eligible)
    return render(base,signal,q,h,z,beta,phase,correction,states,eligible,parent)[:-1]


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,ref,args,decoder,path = group.load_reference(lock)
    h,s = (read(HISTORY/k) for k in ('protocol.json','seal.json'))
    keys = ('order','startup','original','calibration','source_sha256')
    asset = HISTORY/'teaching.json'
    if (any(h[k]!=p[k] for k in keys) or h['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or h['teaching_sha256']!=sha(asset) or len(s['curve_sha256'])!=360):
        raise ValueError('the original release and gradient archive is required')
    base = args[0]
    weights = torch.as_tensor(read(asset)['weights'],device=base.center.device,dtype=torch.float32)
    before = [state_digest(m) for m in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    copyfile(Path(group.__file__),out/'executed_group.py')
    write(out/'protocol.json',dict(scope='EAST360_DELAYED_FIRST_BLOCK_WRITE_SENSITIVITY',
        **{k:p[k] for k in keys},arms=ARMS,additional_wait_samples=WAITS,
        history_seal=sha(HISTORY/'seal.json'),reference_seal=h['reference_seal'],
        executed_sha256=sha(out/'executed_probe.py'),group_sha256=sha(out/'executed_group.py'),
        clock='same issued gradients and original parent clock; wait additional samples before '
              'switching to the new coefficient; then same patch-length handoff',
        dose='original instantaneous token L2 ceiling at each endpoint; '
             'jointly match all original directions inside each delay arm; dose schedule follows handoff',
        interpretation='partial timing sensitivity at nominal sample clock, not measured latency or retuned alarm'))
    digests,witnesses = {},{}
    maxima = np.zeros(3)
    prefix_dose_error = 0.
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before sealing')):
        for rank,sid in enumerate(p['order'],1):
            name = f'{rank:04d}_{sid}.npz'
            a,b = group.REFERENCE/'shots'/name,HISTORY/'shots'/name
            if (sha(a)!=ref['curve_sha256'][name] or sha(b)!=s['curve_sha256'][name]
                    or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]):
                raise ValueError('changed data or reference')
            with np.load(a,allow_pickle=False) as f,np.load(b,allow_pickle=False) as g:
                t,fixed,parent = (f[k] for k in ('times',*group.OLD))
                release,eligible,old_dose = (g[k] for k in ('release','eligible','dose'))
                old = np.stack([g[k] for k in group.INNOVATION[-2:]])
                np.testing.assert_array_equal(release,f['release'])
            if rank<=p['startup']:
                a = b = np.broadcast_to(parent,(len(WAITS),len(t)))
                dose = np.zeros_like(a)
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(t,raw['times'])
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                a,b,dose,pe,de = predict_shot(sig,args,decoder,path,weights,parent,release,eligible)
                current = np.stack((a[0],b[0]))
                np.testing.assert_allclose(current,old,atol=3e-6,rtol=0.)
                np.testing.assert_allclose(dose[0],old_dose,atol=2e-6,rtol=0.)
                for tau in p['calibration'].values():
                    np.testing.assert_array_equal(current>=tau,old>=tau)
                maxima = np.maximum(maxima,(pe,de,float(np.abs(current-old).max())))
                if rank in (21,120,360):
                    n = min(550,len(t))
                    i = release<n
                    prefix = {k:sig[k][:n] for k in ('values','available')}
                    xa,xb,xd,*_ = predict_shot(prefix,args,decoder,path,weights,parent[:n],release[i],eligible[i])
                    np.testing.assert_allclose(xa,a[:,:n],atol=3e-6,rtol=0.)
                    np.testing.assert_allclose(xb,b[:,:n],atol=3e-6,rtol=0.)
                    np.testing.assert_array_equal(xd>0.,dose[:,:n]>0.)
                    prefix_dose_error = max(prefix_dose_error,float(np.abs(xd-dose[:,:n]).max()))
                    # The inherited FP32 parent path is batched differently after truncation.
                    np.testing.assert_allclose(xd,dose[:,:n],atol=3e-6,rtol=1e-6)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(xa>=tau,a[:,:n]>=tau)
                        np.testing.assert_array_equal(xb>=tau,b[:,:n]>=tau)
                    witnesses[str(sid)] = n
            dest = out/'shots'/name
            with dest.open('xb') as stream:
                np.savez_compressed(stream,fixed=fixed,bias_only=parent,release=release,eligible=eligible,dose=dose,
                    **dict(zip(ARMS[2:],np.concatenate((a,b)),strict=True)))
            digests[name] = sha(dest)
            if rank%60==0 or rank==21:
                print(dict(stage='seal_write_delay',rank=rank,seconds=round(time.monotonic()-start,2)),flush=True)
    if before!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen source or inherited path changed')
    write(out/'seal.json',dict(curve_sha256=digests,prefix_witnesses=witnesses,max_parent_error=maxima[0],
        max_token_norm_error=maxima[1],max_no_delay_difference=maxima[2],
        max_prefix_dose_error=prefix_dose_error,seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    if (len(s['curve_sha256'])!=360 or p['history_seal']!=sha(HISTORY/'seal.json')
            or p['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or p['executed_sha256']!=sha(out/'executed_probe.py')
            or p['group_sha256']!=sha(out/'executed_group.py')):
        raise ValueError('sealed delay replay incomplete')
    old = read(HISTORY/'per_shot.json')
    expected = dict(fixed='fixed',bias_only='bias_only',current_0='innovation_current',earlier_0='innovation_lagged')
    per = {band:{k:[] for k in ARMS} for band in p['calibration']}
    endpoints = np.zeros(len(WAITS),np.int64)
    near_support = np.zeros_like(endpoints)
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        file = out/'shots'/name
        if sha(file)!=s['curve_sha256'][name]:
            raise ValueError('changed prediction seal')
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(file,allow_pickle=False) as z:
            dose = z['dose']
            endpoints += (dose>0.).sum(1)
            if y and rank>p['startup'] and not old['0.2']['innovation_current'][rank-1]['timely']:
                lead = np.round((d-t)*1000.,9)
                near_support += ((dose>0.)&(lead>30.)&(lead<=125.)).any(1)
            for j,wait in enumerate(WAITS):
                legal = np.r_[False,z['eligible']][np.searchsorted(z['release']+wait,np.arange(len(t)),side='right')]
                if ((dose[j]>0.) & ~legal).any():
                    raise ValueError('write before arrival or without original feedback support')
                for k in (f'current_{wait}',f'earlier_{wait}'):
                    np.testing.assert_array_equal(z[k][dose[j]==0.],z['bias_only'][dose[j]==0.])
            for band,tau in p['calibration'].items():
                for k in ARMS:
                    row = dict(shot=sid,disruptive=y,down_s=d,**score_curve(
                        t,z[k],y,d,p['original'] if rank<=p['startup'] else tau))
                    if k in expected and row!=old[band][expected[k]][rank-1]:
                        raise ValueError('no-delay control or parent first alarm changed')
                    per[band][k].append(row)
    write(out/'per_shot.json',per)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {b:{k:timing_decomposition(v) for k,v in rows.items()} for b,rows in per.items()}
    write(out/'result.json',dict(status='COMPLETED_WRITE_DELAY_SENSITIVITY_DEVELOPMENT',metrics=metrics,
        raw_verified=360*len(ARMS)*len(metrics),writable_endpoints=dict(zip(WAITS,endpoints.tolist(),strict=True)),
        original_missed_near_support=dict(zip(WAITS,near_support.tolist(),strict=True)),
        paired={b:{f'{a}__to__{k}':flips(rows[a],rows[k]) for a,k in PAIRS} for b,rows in per.items()}))
    print({b:{k:(v['tp'],v['fp']) for k,v in rows.items()} for b,rows in metrics.items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    a = ap.parse_args()
    run(a.out.resolve())
    score(a.out.resolve())
