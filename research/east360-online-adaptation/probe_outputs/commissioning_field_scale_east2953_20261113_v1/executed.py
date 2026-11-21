"""Fixed static EAST sensitivity to field-related signal magnitude, without refitting."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_hidden_background_20261113 as bg
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .operating_point_audit import best_at_count


BT,QA = (io23.RAW23.index(k) for k in ('Bt','qa_proxy'))
N = 100
VIEWS = {'bt_only':(BT,), 'proxy_only':(QA,), 'both':(BT,QA)}
par,group,lag = bg.prior,bg.group,bg.lag


def reference_levels(medians):
    v = np.sort(np.asarray(medians,dtype=np.float64))
    if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or (v<=0.).any() or v[0]==v[-1]:
        raise ValueError('preparation must contain two positive reading clusters')
    i = int(np.argmax(v[1:]/v[:-1]))+1
    return float(np.sqrt(v[i]*v[i-1])),float(np.median(v[i:])/np.median(v[:i]))


def direction(bt, boundary):
    if len(bt)<N or not np.isfinite(bt[:N]).all():
        return 0
    v = float(np.median(abs(bt[:N])))
    return (1 if v<boundary else -1) if v>0. else 0


def rescale(signal, columns, factor):
    x,m = signal['values'].copy(),signal['available']
    for j in columns:
        x[m[:,j],j] *= factor
    return dict(signal,values=x)


@torch.no_grad()
def predict(signal, factor, args, reference=None):
    _,z = bg.static(signal,args)
    fixed = z.sigmoid().cpu().numpy()
    base = fixed if reference is None else reference
    curves = {'fixed':fixed}
    for name,columns in VIEWS.items():
        if factor==1.:
            curves[name] = base.copy()
        else:
            _,z = bg.static(rescale(signal,columns,factor),args)
            v = z.sigmoid().cpu().numpy()
            v[:N-1] = base[:N-1]
            curves[name] = v
    return curves


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,_,args,_,_ = group.load_reference(lock)
    prior,ps = (read(bg.PARENT/f'{k}.json') for k in ('protocol','seal'))
    paths = {k:Path(v['path']) for k,v in prior['references'].items()}
    refs = {k:read(d/'seal.json') for k,d in paths.items()}
    if (prior['calibration']!=p['calibration'] or prior['original']!=p['original']
            or len(ps['curve_sha256'])!=2953 or any(sha(d/'seal.json')!=prior['references'][k]['seal_sha256']
                                                  for k,d in paths.items())
            or [r['shot'] for r in prior['order'][:360]]!=p['order'] or p['startup']!=20):
        raise ValueError('same frozen full historical roster and thresholds required')
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before seal')):
        cold = []
        for sid in p['order'][:p['startup']]:
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('cold signal changed')
            a = simple_data.read_shot('EAST',sid,rate_contract='nominal')['planning_Bt']
            cold.append(float(np.median(abs(a[:N]))))
        cut,gain = reference_levels(cold)
        before = [state_digest(a) for a in args[:2]]
        (out/'shots').mkdir(parents=True)
        copyfile(Path(__file__),out/'executed.py')
        roster = prior['order']
        write(out/'protocol.json',dict(scope='STATIC_RAW_FIELD_COUNTERFACTUAL',order=roster,
            parent_seal=sha(bg.PARENT/'seal.json'),original=p['original'],calibration=p['calibration'],
            cutoff_from_cold=cut,gain_from_cold=gain,preparation=p['order'][:20],cold_medians=cold,
            startup=p['startup'],first=N-1,views=VIEWS,executed_sha256=sha(out/'executed.py'),
            input_intervention='original channels only; paired positive gain or reciprocal; fixed mask and no fit'))
        hashes,signs,witnesses = {},[],{}
        max_error = 0.
        for i,row in enumerate(roster,1):
            sid,rank,c = (row[k] for k in ('shot','rank','cohort'))
            source = par.bind(paths[c],f'{rank:04d}_{sid}.npz',refs[c])
            with np.load(source,allow_pickle=False) as f:
                fixed = f['fixed']
                t = f['times'] if c=='extension' else None
            if i<=p['startup']:
                curves = dict.fromkeys(VIEWS,fixed)
                sign = direction(np.array([cold[i-1]]*N),cut)
            else:
                if c=='fixed360' and sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('changed original 360 HDF5')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                if t is not None:
                    np.testing.assert_array_equal(raw['times'],t)
                sign = direction(raw['planning_Bt'],cut)
                signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
                curves = predict(signal,gain**sign,args,fixed)
                max_error = max(max_error,par.extension.parity({'fixed':curves.pop('fixed')},
                                        {'fixed':fixed},p['calibration']))
                for v in curves.values():
                    np.testing.assert_array_equal(v[:N-1],fixed[:N-1])
                if i in (21,360,361,len(roster)):
                    stop = min(550,len(fixed))
                    short_signal = {k:signal[k][:stop] for k in ('values','available')}
                    short_sign = direction(raw['planning_Bt'][:stop],cut)
                    if short_sign!=sign:
                        raise ValueError('short signal changed past magnitude estimate')
                    short = predict(short_signal,gain**short_sign,args,fixed[:stop])
                    short.pop('fixed')
                    witnesses[str(sid)] = dict(points=stop,max_error=par.extension.parity(short,curves,p['calibration']))
            name = f'{i:04d}_{sid}.npz'
            dest = out/'shots'/name
            with dest.open('xb') as f:
                np.savez_compressed(f,**curves)
            hashes[name] = sha(dest)
            signs.append(sign)
            if i==21 or i%200==0 or i==len(roster):
                print(dict(stage='seal_static_field_counterfactual',rank=i,total=len(roster),
                           seconds=round(time.monotonic()-start,2)),flush=True)
        if before!=[state_digest(a) for a in args[:2]]:
            raise ValueError('frozen model changed')
        write(out/'seal.json',dict(curve_sha256=hashes,direction=signs,witnesses=witnesses,
                                 max_reference_error=max_error,seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if (len(s['curve_sha256'])!=2953 or len(s['direction'])!=2953 or p['parent_seal']!=sha(bg.PARENT/'seal.json')
            or p['executed_sha256']!=sha(out/'executed.py')):
        raise ValueError('complete static counterfactual seal required')
    old = read(bg.PARENT/'per_shot.json')
    all_rows = {b:{'fixed':old[b]['fixed'],**{k:[] for k in VIEWS}} for b in p['calibration']}
    peaks = {k:[] for k in VIEWS}
    balance = read(bg.PARENT/'postseal_bt_scale_balance.json')
    if balance['seal_sha256']!=p['parent_seal']:
        raise ValueError('fixed balance artifact changed')
    matched = set(balance['matched_ids'])
    historic = {r['shot'] for r in read(par.WIDE/'balanced_cohort.json')['order']}
    for i,row in enumerate(p['order'],1):
        sid = row['shot']
        d,down,t = group.hdf._raw_truth(sid)
        if (sid,d,down)!=tuple(old['0.2']['fixed'][i-1][k] for k in ('shot','disruptive','down_s')):
            raise ValueError('HDF5 and reference disagree')
        valid = np.round((down-t)*1000.,9)>30. if d else np.ones(len(t),bool)
        with np.load(par.bind(out,f'{i:04d}_{sid}.npz',s),allow_pickle=False) as f:
            for k in VIEWS:
                v = f[k]
                peaks[k].append(dict(disruptive=d,peak=float(v[valid].max()) if valid.any() else None))
                for b,tau in p['calibration'].items():
                    all_rows[b][k].append(dict(shot=sid,disruptive=d,down_s=down,
                        **score_curve(t,v,d,down,p['original'] if i<=p['startup'] else tau)))
    write(out/'per_shot.json',all_rows)
    masks = dict(fixed360=lambda i,r:r['cohort']=='fixed360',extension=lambda i,r:r['cohort']=='extension',
                 small=lambda i,r:r['cohort']=='extension' and s['direction'][i]>0,
                 large=lambda i,r:r['cohort']=='extension' and s['direction'][i]<0,
                 historical_balance=lambda i,r:r['shot'] in historic,raw_scale_balance=lambda i,r:r['shot'] in matched,
                 old835=lambda i,r:r.get('old835') is True,other1758=lambda i,r:r.get('old835') is False)
    result = {}
    for name,select in masks.items():
        idx = [i for i,row in enumerate(p['order']) if select(i,row)]
        result[name] = {}
        for b,rows in all_rows.items():
            v = {k:[a[i] for i in idx] for k,a in rows.items()}
            result[name][b] = {k:dict(metrics=lead.summarize(a),
                paired=dict(flips=flips(v['fixed'],a),lead=lead.paired(v['fixed'],a))) for k,a in v.items()}
        if name!='fixed360':
            n = result[name]['0.2']['fixed']['metrics']['n_safe']
            result[name]['postseal_matched_fp'] = {k:{str(c):{x:r[x] for x in ('tp','fp','max_fp')}
                for c,r in ((c,best_at_count([v[i] for i in idx],int(c*n))) for c in (.10,.15,.20))}
                for k,v in peaks.items()}
        print(name,{b:{k:(v['metrics']['by_deadline']['30']['tp'],v['metrics']['fp'])
                        for k,v in result[name][b].items()} for b in ('0.2','0.15')},flush=True)
    write(out/'result.json',dict(scope='POSTSEAL_STATIC_FIELD_MAGNITUDE_SENSITIVITY',
                               groups=result,raw_scored=2953*len(VIEWS)*len(all_rows)))


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('out',type=Path)
    opt = a.parse_args()
    run(opt.out.resolve())
    score(opt.out.resolve())
