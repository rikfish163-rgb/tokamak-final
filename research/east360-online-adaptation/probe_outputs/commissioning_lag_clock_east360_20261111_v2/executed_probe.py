"""Replay the archived signed lag adapter with its full H25 target on a second clock."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


REFERENCE = eos.REFERENCE
HALF = lag.STRIDE//2
STATIC, INHERITED, OLD, OLD_SHAM = 'fixed', lag.BASES[-1], guard.PATH, guard.PATH_SHIFT
PHASE, PHASE_SHAM, DENSE, DENSE_SHAM = 'phase16_true','phase16_shifted','dense16_true','dense16_shifted'
ARMS = (STATIC,INHERITED,OLD,OLD_SHAM,PHASE,PHASE_SHAM,DENSE,DENSE_SHAM)
PAIRS = ((OLD,DENSE),(OLD_SHAM,DENSE_SHAM),(DENSE_SHAM,DENSE),(PHASE_SHAM,PHASE),
         (OLD,PHASE),(INHERITED,DENSE),(STATIC,DENSE),(OLD_SHAM,OLD))


def offset_origins(length):
    return np.arange(lag.FIRST+HALF,length-lag.H-1,lag.STRIDE,dtype=np.int64)


def offset_active(original, offset, endpoints):
    before = np.r_[-1,original][np.searchsorted(original,endpoints,side='right')]
    after = np.r_[-1,offset][np.searchsorted(offset,endpoints,side='right')]
    return after>before


def read_archived_curves(reference, previous, name, guard_hashes, parent_hashes, prepared):
    """Guard archives store only their new arms; inherited curves live in the teacher archive."""
    a,b = (root/'shots'/name for root in (previous,reference))
    if sha(a)!=parent_hashes[name] or sha(b)!=guard_hashes[name]:
        raise ValueError('archived curve changed')
    with np.load(a,allow_pickle=False) as parent, np.load(b,allow_pickle=False) as guard_curves:
        times = parent['times']
        np.testing.assert_array_equal(times,guard_curves['times'])
        curves = {k:parent[k] for k in (STATIC,INHERITED)}
        curves.update({k:guard_curves[k] for k in (OLD,OLD_SHAM)})
        releases = np.empty(0,dtype=np.int64) if prepared else guard_curves['release']
    return times,curves,releases


def offset_feedback(signal, tr, params):
    center,scale,forecast,noise,_ = params
    v,m,_,_ = lag.direct.weak.shot_coordinates(signal,center,scale)
    q = offset_origins(len(v))
    target,observed = lag.direct.direct_targets(v,m,q)
    ok = observed.all((1,2))
    q, target = q[ok],target[ok]
    recent,n = _window_mean(v,m,q,-15,16)
    earlier,b = _window_mean(v,m,q,-31,16)
    velocity = np.where((n>=8)&(b>=8),recent-earlier,0.)
    frozen_pre = torch.from_numpy(tr['pre'][q]).tanh().numpy()
    design = np.column_stack((frozen_pre,np.tanh(tr['context'][q]),np.tanh(velocity))).astype(np.float32)
    prior = lag.direct.base_forecast({'design':design},forecast)
    target = lag.direct.normalized_target({'target':target},noise)
    return q,tr['key'][q],prior,target


def _phase(signal, args, decoder, params):
    base = args[0]
    tr,h,z,beta = guard._track(signal,args)
    q,key,prior,target = offset_feedback(signal,tr,args[2])
    z = torch.from_numpy(z).cuda()
    unchanged = z.sigmoid().cpu().numpy()
    if not len(q):
        return unchanged,unchanged,unchanged,q
    real,shift = guard._bases(base,signal,q,prior,target,decoder)
    k = torch.from_numpy(key).to(beta.device)
    matrices = [(-lag.RADIUS*guard.signed_direction(*a,k,params)).detach().cpu().numpy()
                for a in (real,shift)]
    slots,_ = lag.delayed_slots(q,np.arange(len(z)))
    outputs = []
    for a in matrices:
        effective = np.concatenate((np.zeros_like(a[:1]),a),0)[slots]
        h1 = lag._embed_active(base,signal,h,effective)
        dz = (torch.from_numpy(h1-h).cuda()*beta).sum(-1)
        pred = (z+dz).sigmoid().cpu().numpy()
        inactive = ~effective.any((1,2))
        pred[inactive] = unchanged[inactive]
        outputs.append(pred)
    return unchanged,*outputs,q+lag.H+1


def predict(reference, out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p,prev_seal,previous,args,decoder = eos._load(reference,lock)
    fixed = read(reference/'seal.json')['curve_sha256']
    if sha(reference/'teacher.json')!=p['teacher_sha256'] or p['stride']!=lag.STRIDE:
        raise ValueError('the archived signed-path controller changed')
    path = torch.tensor(read(reference/'teacher.json')['path'],device='cuda',dtype=torch.float32)
    frozen = [state_digest(a) for a in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_ARCHIVED_PATH_CLOCK_CONTROL_DEVELOPMENT',
        reference=str(reference),reference_seal=sha(reference/'seal.json'),teacher_sha256=p['teacher_sha256'],
        producer_sha256=sha(Path(__file__)),source_sha256=p['source_sha256'],order=lock['ids'],
        startup=lag.STARTUP,calibration=p['calibration'],original=p['original'],arms=ARMS,
        legacy_origin=lag.FIRST,extra_origin=lag.FIRST+HALF,legacy_stride=lag.STRIDE,
        horizon=lag.H,first_extra_effect=lag.FIRST_EFFECT+HALF,
        update_contract='same exact frozen signed-path controller; each complete 25 point target '
                        'first acts at origin+26; use newest completed state, no accumulation',
        comparison='phase16 uses only the offset clock (no higher feedback count); dense16 alternates '
                   'archived old and newly computed offset actions, with identical inherited paths',
        dose='same legacy matrix norm and input RMS cap; actual input RMS not matched arm-to-arm',
        training='none; no EAST suffix outcome access before all predictions seal'))
    hashes,witnesses = {},{}
    original_updates,extra_updates = 0,0
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes are not observable')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('locked HDF5 changed')
            t,curves,r_old = read_archived_curves(reference,previous,name,fixed,
                                                prev_seal['curve_sha256'],rank<=lag.STARTUP)
            if rank<=lag.STARTUP:
                curves.update(dict.fromkeys(ARMS[4:],curves[STATIC]))
                r_mid = np.empty(0,dtype=int)
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(t,raw['times'])
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                parent,a,b,r_mid = _phase(signal,args,decoder,path)
                np.testing.assert_array_equal(parent,curves[INHERITED])
                take = offset_active(r_old,r_mid,np.arange(len(t)))
                curves.update({PHASE:a,PHASE_SHAM:b,DENSE:np.where(take,a,curves[OLD]),
                               DENSE_SHAM:np.where(take,b,curves[OLD_SHAM])})
                original_updates += len(r_old)
                extra_updates += len(r_mid)
                if rank in (21,120,360):
                    end = min(len(t),lag.FIRST_EFFECT+3*lag.STRIDE+5)
                    prefix = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                    _,short,other,rr = _phase(prefix,args,decoder,path)
                    np.testing.assert_allclose(short,a[:end],atol=3e-6,rtol=0.)
                    np.testing.assert_allclose(other,b[:end],atol=3e-6,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(short>=tau,a[:end]>=tau)
                        np.testing.assert_array_equal(other>=tau,b[:end]>=tau)
                    np.testing.assert_array_equal(rr,r_mid[r_mid<end])
                    witnesses[str(sid)] = end
            if not all(np.isfinite(a).all() for a in curves.values()):
                raise FloatingPointError('nonfinite alarm replay')
            f = out/'shots'/name
            with f.open('xb') as h:
                np.savez_compressed(h,times=t,release_legacy=r_old,release_extra=r_mid,**curves)
            hashes[name] = sha(f)
            if rank in (20,21) or rank%60==0:
                print(dict(stage='causal_clock_predictions',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    if [state_digest(a) for a in args[:2]]!=frozen:
        raise ValueError('the frozen source encoder or transition net changed')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
        original_updates=original_updates,extra_updates=extra_updates,seconds=time.monotonic()-started))


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    reference = Path(p['reference'])
    if (sha(reference/'seal.json')!=p['reference_seal'] or sha(out/'executed_probe.py')!=p['producer_sha256']
            or len(s['curve_sha256'])!=360):
        raise ValueError('sealed experiment identity changed')
    old = read(reference/'seal.json')['curve_sha256']
    protocol = read(reference/'protocol.json')
    previous = Path(protocol['previous_teacher'])
    if sha(previous/'seal.json')!=protocol['previous_seal']:
        raise ValueError('sealed teacher archive changed')
    previous_hashes = read(previous/'seal.json')['curve_sha256']
    original = read(reference/'per_shot.json')
    results = {b:{k:[] for k in ARMS} for b in p['calibration']}
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        f = out/'shots'/name
        if sha(f)!=s['curve_sha256'][name]:
            raise ValueError('changed curves during independent raw-score check')
        prepared = rank<=p['startup']
        times,old_curves,old_r = read_archived_curves(reference,previous,name,old,previous_hashes,prepared)
        d,down,t = hdf._raw_truth(sid)
        np.testing.assert_array_equal(times,t)
        with np.load(f,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['times'],t)
            for k in ARMS[:4]:
                np.testing.assert_array_equal(a[k],old_curves[k])
            curves = {k:a[k] for k in ARMS}
            r,r2 = a['release_legacy'],a['release_extra']
            if prepared:
                for k in ARMS:
                    np.testing.assert_array_equal(curves[k],curves[STATIC])
            else:
                np.testing.assert_array_equal(r,old_r)
                if not (np.isin(r2,offset_origins(len(t))+lag.H+1).all() and np.all(np.diff(r2)>0.)):
                    raise ValueError('early or disordered offset feedback')
                effect = offset_active(r,r2,np.arange(len(t)))
                before = np.arange(len(t))<(r2[0] if len(r2) else len(t))
                for k,j,c in ((PHASE,DENSE,OLD),(PHASE_SHAM,DENSE_SHAM,OLD_SHAM)):
                    np.testing.assert_array_equal(curves[k][before],curves[INHERITED][before])
                    np.testing.assert_array_equal(curves[j],np.where(effect,curves[k],curves[c]))
        for cap,tau in p['calibration'].items():
            if rank<=p['startup']:
                tau = p['original']
            for k,a in curves.items():
                row = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,a,d,down,tau))
                if k in ARMS[:4] and row!=original[cap][k][rank-1]:
                    raise ValueError('archived parent or reference first alarm did not reproduce')
                results[cap][k].append(row)
    write(out/'per_shot.json',results)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {b:{k:timing_decomposition(v) for k,v in arms.items()} for b,arms in results.items()}
    paired = {b:{f'{k}__to__{j}':flips(arms[k],arms[j]) for k,j in PAIRS} for b,arms in results.items()}
    write(out/'result.json',dict(status='COMPLETED_EAST360_FROZEN_PATH_CLOCK_ABLATION',metrics=metrics,paired=paired,
        raw_verified_scores=len(p['order'])*len(ARMS)*len(results),
        original_updates=s['original_updates'],extra_updates=s['extra_updates']))
    print({b:{k:(m['tp'],m['fp']) for k,m in arms.items()} for b,arms in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    a = parser.parse_args()
    predict(a.reference.resolve(),a.out) if a.stage=='predict' else score(a.out)
