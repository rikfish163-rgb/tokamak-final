"""Factorize inherited EAST risk and omit its state contributions at frozen feedback."""

import argparse
from collections import Counter
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_frozen_extension_20261113 as extension
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_transfer_20261108 import contrast
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition


group, lag = extension.group, extension.lag
delay = extension.delay
WIDE = io23.ROOT/'probe_outputs/commissioning_frozen_extension_east2593_20261113_v1'
SMALL = extension.REFERENCE
STAGES = ('fixed','inherited','legacy_current4','signed_path_true','bias_only','current_5')
OMIT = {
    'bias_no_states': ('bias_only',0),
    'current_no_old': ('current_5',2),
    'current_no_latest': ('current_5',1),
    'current_no_states': ('current_5',0),
    'earlier_no_states': ('earlier_5',0),
}
ARMS = (*STAGES,'earlier_5',*OMIT)
GENERATED = (*STAGES[1:4],*OMIT)
PAIRS = (*zip(STAGES,STAGES[1:]), *((v[0],k) for k,v in OMIT.items()),
         ('bias_no_states','current_no_states'),('earlier_no_states','current_no_states'),
         ('fixed','current_no_states'))


def readout_levels(x,pre,inherited,latest,coef):
    """Keep the original full matrix multiply and its feature ordering."""
    return tuple(torch.from_numpy(np.column_stack((x,v.tanh().numpy()))).to(coef.device)@coef[:-1]+coef[-1]
                 for v in (pre,inherited,pre+latest,inherited+latest))


@torch.no_grad()
def risk_views(signal,args,decoder,path,weights):
    base,net,params,mapper0,quiet,mapper1,coef,_ = args
    with torch.enable_grad():
        q,h,z,beta,prior_h,phase,correction = group.parent_track(signal,args,decoder,path)
        avail,fields,wave = group.feedback_waveform(signal,q,params,z.device)
        gradients,common = group._estimate_innovation(base,signal,q,avail,phase,correction,prior_h,
                                                       fields,wave,decoder,weights)
    e = np.arange(len(signal['values']))
    tr,inherited,ready = lag.history.observed_track(net,signal,e,params,mapper0)
    states,_ = lag.persistent.replay(*ready,quiet)
    pre = torch.from_numpy(tr['pre'])
    # Only the minus-old view needs the additional direct state term.
    new = torch.einsum('...k,...f,kfw->...w',torch.from_numpy(tr['key']),
                       4.*torch.from_numpy(states['current_equal_norm']),mapper1)
    x = np.column_stack((h.cpu().numpy(),contrast(signal['values'],signal['available'])))
    fixed,prior,minus_old,legacy = readout_levels(x,pre,inherited,new,coef)
    torch.testing.assert_close(legacy,z,atol=5e-6,rtol=0.)
    cur,ear,dose,_,error,parent = delay.render(base,signal,q,h,z,beta,phase,correction,gradients,common,
                                               waits=(5,),logits=True)
    raw = dict(bias_only=parent,current_5=cur[0],earlier_5=ear[0])
    signed = z.clone()
    release = q+lag.H+1
    slot = np.searchsorted(release,e,side='right')
    idx = np.flatnonzero(slot)
    for start in range(0,len(idx),256):
        i = idx[start:start+256]
        s = slot[i]
        xw,mw = lag._windows(base,signal,i)
        view = lag._features(base,xw,mw,phase[s])
        signed[i] = z[i]+((view-h[i])*beta).sum(-1)
    curves = {k:v.sigmoid().cpu().numpy() for k,v in zip(STAGES,(fixed,prior,legacy,signed))}
    curves.update({k:torch.from_numpy(v).to(z.device).sigmoid().cpu().numpy() for k,v in raw.items()})
    # The full matrix multiply keeps the original readout summation order.
    shift = torch.stack((fixed,prior,minus_old))-legacy
    for k,(a,i) in OMIT.items():
        curves[k] = (torch.from_numpy(raw[a]).to(z.device)+shift[i]).sigmoid().cpu().numpy()
    return curves,release,common,dose[0],error


def predict(signal,args,decoder,path,weights,archived,thresholds):
    nested,release,common,dose,norm_error = risk_views(signal,args,decoder,path,weights)
    err = extension.parity({k:nested[k] for k in ('fixed','bias_only','current_5','earlier_5')},archived,thresholds)
    np.testing.assert_array_equal(release,archived['release'])
    np.testing.assert_array_equal(common,archived['eligible'])
    np.testing.assert_array_equal(dose>0.,archived['dose']>0.)
    np.testing.assert_allclose(dose,archived['dose'],atol=3e-6,rtol=1e-6)
    zero = dose==0.
    for k in ('current_no_states','earlier_no_states'):
        np.testing.assert_array_equal(nested[k][zero],nested['bias_no_states'][zero])
    return {k:nested[k] for k in GENERATED},err,norm_error


def bind(folder,name,seal):
    p = folder/'shots'/name
    if sha(p)!=seal['curve_sha256'][name]:
        raise ValueError(f'changed curve {p}')
    return p


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,r,args,decoder,path = group.load_reference(lock)
    large,large_seal = (read(WIDE/f'{k}.json') for k in ('protocol','seal'))
    small,small_seal = (read(SMALL/f'{k}.json') for k in ('protocol','seal'))
    asset = delay.HISTORY/'teaching.json'
    weights = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    order = [dict(shot=s,cohort='fixed360',rank=i+1) for i,s in enumerate(p['order'])]
    order += [dict(row,cohort='extension',rank=i+1) for i,row in enumerate(large['order'])]
    if (large['prior_360']!=p['order'] or len(order)!=2953 or len({row['shot'] for row in order})!=2953
            or len(large_seal['curve_sha256'])!=2593 or len(small_seal['curve_sha256'])!=360
            or large['teaching_sha256']!=sha(asset)
            or small['group_sha256']!=sha(Path(group.__file__))
            or any(large[k]!=small[k] or small[k]!=p[k] for k in ('calibration','original','source_sha256'))):
        raise ValueError('fixed original and disjoint extension seals required')
    sources = {'fixed360': (SMALL,small_seal), 'extension': (WIDE,large_seal)}
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed.py')
    copyfile(Path(delay.__file__),out/'executed_delay.py')
    write(out/'protocol.json',dict(
        scope='EAST_FROZEN_READOUT_COMPONENTS_NOT_NEW_FEEDBACK',order=order,arms=ARMS,stages=STAGES,
        original=p['original'],calibration=p['calibration'],startup=p['startup'],source_sha256=p['source_sha256'],
        references={k:dict(path=str(d),seal_sha256=sha(d/'seal.json')) for k,(d,_) in sources.items()},
        balanced_manifest=sha(WIDE/'balanced_cohort.json'),executed_sha256=sha(out/'executed.py'),
        executed_delay_sha256=sha(out/'executed_delay.py'),
        analysis='nested frozen stages and omit state terms before tanh only at risk readout',
        online='original data, parent state generation, feedback, token dose and clock; no new fit or calibration',
        outcomes='score only after all new curves seal'))
    before = [state_digest(a) for a in args[:2]]
    hashes,witnesses,old_checks = {},{},{}
    maxima = {'fixed360': 0.,'extension': 0.}
    norm_error = 0.
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before sealing')):
        for i,row in enumerate(order,1):
            sid,rank,cohort = (row[k] for k in ('shot','rank','cohort'))
            name = f'{rank:04d}_{sid}.npz'
            folder,seal = sources[cohort]
            f = bind(folder,name,seal)
            with np.load(f,allow_pickle=False) as saved:
                archived = {k:saved[k] for k in ('fixed','bias_only','current_5','earlier_5')}
                archived.update(release=saved['release'],eligible=saved['eligible'],dose=saved['dose'][-1])
                t = saved['times'] if cohort=='extension' else None
            if cohort=='fixed360':
                with np.load(bind(group.REFERENCE,name,r),allow_pickle=False) as saved:
                    t = saved['times']
                    stage = {k:saved[k] for k in ('legacy_current4','signed_path_true')}
            if cohort=='fixed360' and rank<=p['startup']:
                curves = dict.fromkeys(GENERATED,archived['fixed'])
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
                curves,err,e1 = predict(signal,args,decoder,path,weights,archived,p['calibration'])
                maxima[cohort] = max(maxima[cohort],err)
                norm_error = max(norm_error,e1)
                if cohort=='fixed360':
                    old_checks[str(sid)] = extension.parity({k:curves[k] for k in stage},stage,p['calibration'])
                if i in (21,360,361,len(order)):
                    stop = min(550,len(t))
                    cut = {k:signal[k][:stop] for k in ('values','available')}
                    keep = archived['release']<stop
                    prefix = {k:v[keep] if k in ('release','eligible') else v[:stop] for k,v in archived.items()}
                    short,e0,_ = predict(cut,args,decoder,path,weights,prefix,p['calibration'])
                    witnesses[str(sid)] = dict(points=stop,max_error=max(e0,extension.parity(short,curves,p['calibration'])))
            dest = out/'shots'/f'{i:04d}_{sid}.npz'
            with dest.open('xb') as file:
                np.savez_compressed(file,**curves)
            hashes[dest.name] = sha(dest)
            if i==21 or i%200==0 or i==len(order):
                print(dict(stage='sealed_parent_readout',rank=i,total=len(order),seconds=round(time.monotonic()-start,2)),
                      flush=True)
    if before!=[state_digest(a) for a in args[:2]]:
        raise ValueError('frozen model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,max_parent_error=maxima,
                             max_token_norm_error=norm_error,
                             original_nested_max_error=max(old_checks.values()),seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    sources = {k:(Path(v['path']),read(Path(v['path'])/'seal.json')) for k,v in p['references'].items()}
    if (len(s['curve_sha256'])!=len(p['order']) or sha(out/'executed.py')!=p['executed_sha256']
            or sha(out/'executed_delay.py')!=p['executed_delay_sha256']
            or any(sha(d/'seal.json')!=p['references'][k]['seal_sha256'] for k,(d,_) in sources.items())
            or sha(WIDE/'balanced_cohort.json')!=p['balanced_manifest']):
        raise ValueError('incomplete frozen readout seal')
    per = {b:{a:[] for a in ARMS} for b in p['calibration']}
    signatures = []
    for i,row in enumerate(p['order'],1):
        sid,rank,cohort = (row[k] for k in ('shot','rank','cohort'))
        d,down,t = group.hdf._raw_truth(sid)
        path,seal = sources[cohort]
        f = bind(path,f'{rank:04d}_{sid}.npz',seal)
        new = bind(out,f'{i:04d}_{sid}.npz',s)
        with np.load(f,allow_pickle=False) as old,np.load(new,allow_pickle=False) as extra:
            a = {k:(extra[k] if k in GENERATED else old[k]) for k in ARMS}
            no_write = old['dose'][-1]==0.
        for band,tau in p['calibration'].items():
            tau = p['original'] if cohort=='fixed360' and rank<=p['startup'] else tau
            for k in ARMS:
                per[band][k].append(dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,a[k],d,down,tau)))
            if band not in ('0.15','0.2') or d:
                continue
            above = np.stack([a[k]>=tau for k in STAGES])
            any_hit = above.any(1)
            if not any_hit.any():
                continue
            hit = np.flatnonzero(above[-1])
            signatures.append(dict(shot=sid,band=band,cohort=cohort,old835=row.get('old835'),
                stage_alarm=any_hit.tolist(),crossings=above.sum(1).tolist(),
                candidate_first=int(hit[0]) if len(hit) else None,
                immutable=bool((above[-1]&no_write).any()),
                stage_at_candidate_first=above[:,hit[0]].tolist() if len(hit) else None))
    write(out/'per_shot.json',per)
    balanced = {v['shot'] for v in read(WIDE/'balanced_cohort.json')['order']}
    tests = {
        'fixed360': lambda r:r['cohort']=='fixed360',
        'fixed340': lambda r:r['cohort']=='fixed360' and r['rank']>p['startup'],
        'extension': lambda r:r['cohort']=='extension',
        'old835': lambda r:r.get('old835') is True,
        'other1758': lambda r:r.get('old835') is False,
        'balanced': lambda r:r['shot'] in balanced,
    }
    groups = {}
    for name,keep in tests.items():
        idx = [i for i,r in enumerate(p['order']) if keep(r)]
        groups[name] = {}
        for band,data in per.items():
            rows = {k:[v[i] for i in idx] for k,v in data.items()}
            metrics = {k:dict(timing_decomposition(v),warning_times=lead.summarize(v)) for k,v in rows.items()}
            groups[name][band] = dict(metrics=metrics,paired={f'{a}__to__{b}':dict(
                flips=flips(rows[a],rows[b]),lead=lead.paired(rows[a],rows[b])) for a,b in PAIRS})
    write(out/'false_alarm_stages.json',dict(stages=STAGES,rows=signatures,extension_patterns={
        b:dict(Counter(''.join('1' if x else '0' for x in r['stage_alarm']) for r in signatures
                       if r['cohort']=='extension' and r['band']==b)) for b in ('0.2','0.15')}))
    write(out/'result.json',dict(status='COMPLETE_FROZEN_PARENT_READOUT_AUDIT',
          raw_verified=len(p['order'])*len(ARMS)*len(per),groups=groups))
    for name in ('fixed360','extension','balanced'):
        print(name,{b:{a:(v['tp'],v['fp']) for a,v in x['metrics'].items()}
                    for b,x in groups[name].items() if b!='0.1'},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    args = ap.parse_args()
    run(args.out.resolve())
    score(args.out.resolve())
