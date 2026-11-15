"""Apply the frozen grouped EAST warning experiment to all other local discharges."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_write_delay_20261112 as delay
from . import io23, simple_data
from . import warning_lead_metrics as lead
from .commissioning_suppression_reach_20261112 import REFERENCE
from .commissioning_transfer_20261108 import contrast
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition


group, lag = delay.group, delay.lag
PREVIOUS = io23.ROOT/'probe_outputs/layer0_conditional835_20261106_v1'
CHECK_RANKS = (21,99,118,120,217,360)
ARMS = delay.ARMS
PRIMARY = (('fixed',group.bias.BIAS),(group.bias.BIAS,'current_5'),(group.bias.BIAS,'current_0'),
           ('earlier_5','current_5'),('earlier_0','current_0'),('current_0','current_5'))
BALANCE_SEED = 131


def inventory(root, original, prior):
    ids = [int(p.stem) for p in root.glob('*/*.hdf5') if p.is_file()]
    original, prior = set(original), set(prior)
    if len(ids)!=len(set(ids)) or original&prior or not (original|prior)<=set(ids):
        raise ValueError('ambiguous local inventory or missing archived cohort members')
    return [dict(shot=sid,old835=sid in prior) for sid in sorted(set(ids)-original)]


def balanced_indices(disruptive, old835):
    """Retain equal counts within each historical group, without reading any predictions."""
    y, old = np.asarray(disruptive,dtype=bool),np.asarray(old835,dtype=bool)
    rng = np.random.default_rng(BALANCE_SEED)
    selected = []
    for g in (True,False):
        d, s = (np.flatnonzero((old==g)&(y==k)) for k in (True,False))
        n = min(len(d),len(s))
        for i in (d,s):
            selected.extend(i if len(i)==n else rng.choice(i,n,replace=False))
    return np.sort(np.asarray(selected,dtype=np.int64))


def predict(signal,args,decoder,path,weights):
    base = args[0]
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    # The static arm in the original 360 uses the same inherited 24-dim
    # frozen feature extractor, without its shot-local state corrections.
    bare = lag.direct.track(args[1],signal,*args[2][:2],np.arange(len(z)))['pre']
    matrix = np.concatenate((h.cpu().numpy(),contrast(signal['values'],signal['available']),
                             torch.from_numpy(bare).tanh().numpy()),axis=1)
    fixed = (torch.from_numpy(matrix).to(z.device)@args[6][:-1]+args[6][-1]).sigmoid().cpu().numpy()
    avail,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
    states,common = group._estimate_innovation(base,signal,q,avail,phase,correction,prior,fields,wave,decoder,weights)
    current,earlier,dose,_,err,parent = delay.render(base,signal,q,h,z,beta,phase,correction,states,common)
    values = dict(zip(ARMS,(fixed,parent,*current,*earlier),strict=True))
    return values,q+lag.H+1,common,dose,err


def parity(curves, archived, thresholds):
    error = 0.
    for k,v in curves.items():
        a = archived[k][:len(v)]
        error = max(error,float(np.abs(a-v).max()))
        for tau in thresholds.values():
            np.testing.assert_array_equal(v>=tau,a>=tau,err_msg=f'first-alarm mismatch for {k}')
    if error>3e-6:
        raise ValueError(f'probability mismatch: {error:g}')
    return error


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,ref,args,decoder,path = group.load_reference(lock)
    history, original, seal, prior, prior_seal = (read(k) for k in (
        delay.HISTORY/'protocol.json',REFERENCE/'protocol.json',REFERENCE/'seal.json',
        PREVIOUS/'protocol.json',PREVIOUS/'completed.json'))
    asset = delay.HISTORY/'teaching.json'
    keys = ('order','startup','original','calibration','source_sha256')
    if (any(original[k]!=p[k] for k in keys) or original['history_seal']!=sha(delay.HISTORY/'seal.json')
            or original['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or history['teaching_sha256']!=sha(asset) or original['group_sha256']!=sha(Path(group.__file__))
            or tuple(original['arms'])!=ARMS or tuple(original['additional_wait_samples'])!=delay.WAITS
            or len(seal['curve_sha256'])!=360 or len(prior['order'])!=835
            or [r['shot'] for r in prior_seal['sealed']]!=prior['order']):
        raise ValueError('expected the archived original 360 and disjoint earlier 835')
    order = inventory(io23.DATA_ROOT/'EAST/unified_hdf5',p['order'],prior['order'])
    weights = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    before = tuple(state_digest(m) for m in args[:2])
    (out/'shots').mkdir(parents=True)
    for mod in (Path(__file__),Path(delay.__file__)):
        copyfile(mod,out/mod.name)
    write(out/'protocol.json',dict(scope='EAST_ALL_LOCAL_DISJOINT_FROZEN_RETROSPECTIVE',
        order=order,prior_360=p['order'],historical_preparation=p['order'][:p['startup']],
        calibration=p['calibration'],original=p['original'],source_sha256=p['source_sha256'],arms=ARMS,
        acquisition_dates_verified=False,clock='original HDF5 clock, nominal 1kHz contract without resampling',
        order_rule='all other inventory shots sorted by integer ID; shot-local reset; not acquisition chronology',
        selection='exclude only the original 360; no signal or label filtering',
        cold='reuse already prepared original 20, none of the extension is used for cold preparation',
        outcomes='read raw HDF5 only after sealing all new predictions; no tuning or persistent updates',
        balance='all eligible shots primary; secondary deterministic 1:1 without replacement within each '
                'older-835 membership stratum after scoring seal; selection reads class and stratum only',
        balance_seed=BALANCE_SEED,main_contrasts=PRIMARY,original_witness_ranks=CHECK_RANKS,
        delay_reference_seal=sha(REFERENCE/'seal.json'),prior_protocol_sha256=sha(PREVIOUS/'protocol.json'),
        teaching_sha256=sha(asset),code_sha256={k.name:sha(out/k.name) for k in (Path(__file__),Path(delay.__file__))}))
    digests, errors, witnesses, check = {},[],{},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome is not a prediction input')):
        for rank in CHECK_RANKS:
            sid = p['order'][rank-1]
            name = f'{rank:04d}_{sid}.npz'
            archived = REFERENCE/'shots'/name
            if (sha(archived)!=seal['curve_sha256'][name] or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]
                    or sha(group.REFERENCE/'shots'/name)!=ref['curve_sha256'][name]):
                raise ValueError('changed original witness')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            signal = lag.prepare_actual(raw,rank,lock['mapping'])
            curves,release,eligible,dose,_ = predict(signal,args,decoder,path,weights)
            with np.load(archived,allow_pickle=False) as f:
                check[str(sid)] = parity(curves,f,p['calibration'])
                np.testing.assert_array_equal(release,f['release'])
                np.testing.assert_array_equal(eligible,f['eligible'])
                np.testing.assert_allclose(dose,f['dose'],atol=3e-6,rtol=1e-6)
        print(dict(stage='archive_parity',shots=len(check),max_error=max(check.values())),flush=True)
        for rank,entry in enumerate(order,1):
            sid = entry['shot']
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
            curves,release,eligible,dose,err = predict(signal,args,decoder,path,weights)
            if rank in (1,(len(order)+1)//2,len(order)):
                stop = min(len(raw['times']),550)
                cut = {k:signal[k][:stop] for k in ('values','available')}
                short,r,e,d,_ = predict(cut,args,decoder,path,weights)
                np.testing.assert_array_equal(r,release[release<stop])
                np.testing.assert_array_equal(e,eligible[release<stop])
                np.testing.assert_array_equal(d>0.,dose[:,:stop]>0.)
                np.testing.assert_allclose(d,dose[:,:stop],atol=3e-6,rtol=1e-6)
                witnesses[str(sid)] = dict(points=stop,error=parity(short,curves,p['calibration']))
            dest = out/'shots'/f'{rank:04d}_{sid}.npz'
            with dest.open('xb') as f:
                np.savez_compressed(f,times=raw['times'],release=release,eligible=eligible,dose=dose,**curves)
            digests[dest.name] = sha(dest)
            errors.append(err)
            if rank==1 or rank%200==0 or rank==len(order):
                print(dict(stage='sealed_extended',rank=rank,total=len(order),seconds=round(time.monotonic()-start,2)),flush=True)
    if before!=tuple(state_digest(m) for m in args[:2]):
        raise ValueError('frozen state changed')
    write(out/'seal.json',dict(curve_sha256=digests,old_archive_parity=check,prefix_witnesses=witnesses,
        max_token_norm_error=max(errors),seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    if (p['delay_reference_seal']!=sha(REFERENCE/'seal.json') or len(s['curve_sha256'])!=len(p['order'])
            or any(sha(out/k)!=v for k,v in p['code_sha256'].items())):
        raise ValueError('whole extension must be sealed before scoring')
    per = {band:{k:[] for k in ARMS} for band in p['calibration']}
    labels,old = [],[]
    update_endpoints = np.zeros(len(delay.WAITS),np.int64)
    for rank,entry in enumerate(p['order'],1):
        sid = entry['shot']
        file = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(file)!=s['curve_sha256'][file.name]:
            raise ValueError('changed extension curve')
        y,d,t = group.hdf._raw_truth(sid)
        labels.append(y)
        old.append(entry['old835'])
        with np.load(file,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            for j,wait in enumerate(delay.WAITS):
                dormant = f['dose'][j]==0.
                update_endpoints[j] += int((~dormant).sum())
                for name in (f'current_{wait}',f'earlier_{wait}'):
                    np.testing.assert_array_equal(f[name][dormant],f[group.bias.BIAS][dormant])
            data = {k:f[k] for k in ARMS}
            for band,tau in p['calibration'].items():
                for k,v in data.items():
                    per[band][k].append(dict(shot=sid,disruptive=y,down_s=d,**score_curve(t,v,y,d,tau)))
    write(out/'per_shot.json',per)
    balance = balanced_indices(labels,old)
    write(out/'balanced_cohort.json',dict(seed=BALANCE_SEED,order=[p['order'][i] for i in balance]))
    summary = {}
    for group_name,idx in (('all',np.arange(len(old))),('old835',np.flatnonzero(old)),
                          ('other',np.flatnonzero(~np.asarray(old))),('balanced_1to1',balance)):
        if not len(idx):
            continue
        summary[group_name] = {}
        for band,rows in per.items():
            sub = {k:[v[i] for i in idx] for k,v in rows.items()}
            metrics = {k:timing_decomposition(v) for k,v in sub.items()}
            for k,v in sub.items():
                metrics[k]['warning_times'] = lead.summarize(v)
            summary[group_name][band] = dict(metrics=metrics, paired={f'{a}__to__{b}':dict(
                flips=flips(sub[a],sub[b]),timing=lead.paired(sub[a],sub[b])) for a,b in PRIMARY})
    write(out/'result.json',dict(status='COMPLETE_FROZEN_EXTENDED_EAST_DEVELOPMENT',raw_verified=len(old)*len(ARMS)*len(per),
          actual_write_endpoints=update_endpoints.tolist(),groups=summary))
    for k,g in summary.items():
        print(k,{b:{a:(m['tp'],m['fp']) for a,m in v['metrics'].items()} for b,v in g.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.out.resolve())
    score(args.out.resolve())
