"""Turn off the grouped write with the same frozen readout and corrected parent."""

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


ARMS = ('bias_no_states','level_parent','drift_parent')
par,group,lag,delay = bg.prior,bg.group,bg.lag,bg.delay


@torch.no_grad()
def predict(signal,args,decoder,path,coef):
    base = args[0]
    with torch.enable_grad():
        q,h,z,beta,_,phase,correction = group.parent_track(signal,args,decoder,path)
    h,z_static = bg.static(signal,args,h)
    corrected = h.clone()
    def capture(_j,k,p,_x,_enabled):
        corrected[k] = p
    empty = h.new_zeros((len(group.INNOVATION),len(q)+1,base.cfg.window//base.cfg.patch-1,len(group.TRUSTED)))
    _,_,dose,_,_,parent = delay.render(base,signal,q,h,z,beta,phase,correction,empty,np.zeros(len(q),bool),
                                       waits=(5,),logits=True,hidden_trace=capture)
    if dose.any():
        raise ValueError('no grouped writes may be applied in this arm')
    fixed = torch.from_numpy(parent).to(z.device)+(z_static-z)
    arms = dict(bias_no_states=fixed.sigmoid().cpu().numpy())
    level,drift = (torch.from_numpy(v).to(z.device) for v in bg.background_features(h.cpu().numpy()))
    enabled = torch.arange(len(h),device=z.device)>=bg.FIRST
    for key,x in (('level',level),('drift',drift)):
        c = coef[key]
        change = x@c[:-1]+c[-1]+(corrected-h)@c[:-1]
        arms[f'{key}_parent'] = (fixed+enabled*change).sigmoid().cpu().numpy()
    return arms,q+lag.H+1


def run(folder,out):
    lag.configure()
    start = time.monotonic()
    p,seal,fit = (read(folder/f'{k}.json') for k in ('protocol','seal','fit'))
    pp,ps = (read(bg.PARENT/f'{k}.json') for k in ('protocol','seal'))
    lock = lag.load_inputs()
    fixed,_,args,decoder,path = group.load_reference(lock)
    if (p['order']!=pp['order'] or len(seal['curve_sha256'])!=2953 or p['parent_seal']!=sha(bg.PARENT/'seal.json')
            or sha(folder/'fit.json')!=p['fit_sha256'] or p['first']!=bg.FIRST
            or p['calibration']!=pp['calibration'] or p['calibration']!=fixed['calibration']
            or p['original']!=fixed['original'] or p['startup']!=20
            or fit['prepared']!=fixed['order'][:20] or sha(Path(bg.__file__))!=p['executed_sha256']
            or sha(Path(delay.__file__))!=p['executed_delay_sha256']):
        raise ValueError('unchanged completely sealed background experiment is required')
    coef = {k:torch.tensor(fit['coef'][k],device='cuda',dtype=torch.float32) for k in ('level','drift')}
    paths = {k:Path(v['path']) for k,v in pp['references'].items()}
    refs = {k:read(f/'seal.json') for k,f in paths.items()}
    if any(sha(f/'seal.json')!=pp['references'][k]['seal_sha256'] for k,f in paths.items()):
        raise ValueError('changed fixed reference')
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed.py')
    write(out/'protocol.json',dict(scope='FROZEN_BACKGROUND_MATCHED_NO_GROUP_WRITE',order=p['order'],
        original=p['original'],calibration=p['calibration'],arms=ARMS,startup=p['startup'],
        background_seal_sha256=sha(folder/'seal.json'),fit_sha256=p['fit_sha256'],
        executed_sha256=sha(out/'executed.py')))
    hashes,witnesses = {},{}
    ref_error=zero_error=0.
    before = [state_digest(a) for a in args[:2]]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('holdout outcome forbidden before seal')):
        for i,row in enumerate(p['order'],1):
            sid,rank,c = (row[k] for k in ('shot','rank','cohort'))
            name,original = f'{i:04d}_{sid}.npz',f'{rank:04d}_{sid}.npz'
            with np.load(par.bind(bg.PARENT,name,ps),allow_pickle=False) as f, \
                    np.load(par.bind(paths[c],original,refs[c]),allow_pickle=False) as g:
                old,release,dose = f['bias_no_states'],g['release'],g['dose'][-1]
            if i<=p['startup']:
                curves = dict.fromkeys(ARMS,old)
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
                curves,r = predict(signal,args,decoder,path,coef)
                ref_error = max(ref_error,par.extension.parity({'bias_no_states':curves['bias_no_states']},
                                      {'bias_no_states':old},p['calibration']))
                np.testing.assert_array_equal(r,release)
                for k in ARMS[1:]:
                    np.testing.assert_array_equal(curves[k][:bg.FIRST],curves['bias_no_states'][:bg.FIRST])
                with np.load(par.bind(folder,name,seal),allow_pickle=False) as f:
                    zero = dose==0.
                    controls = {k:curves[k][zero] for k in ARMS[1:]}
                    for side in ('current','earlier'):
                        z = {k:f[k.replace('parent',side)][zero] for k in ARMS[1:]}
                        if zero.any():
                            zero_error = max(zero_error,par.extension.parity(controls,z,p['calibration']))
                if i in (21,360,361,len(p['order'])):
                    end = min(550,len(old))
                    short,_ = predict({k:signal[k][:end] for k in ('values','available')},args,decoder,path,coef)
                    witnesses[str(sid)] = dict(points=end,max_error=par.extension.parity(short,curves,p['calibration']))
            dest = out/'shots'/name
            with dest.open('xb') as file:
                np.savez_compressed(file,**{k:curves[k] for k in ARMS[1:]})
            hashes[name] = sha(dest)
            if i==21 or i%200==0 or i==len(p['order']):
                print(dict(stage='seal_matched_no_group_write',rank=i,total=len(p['order']),
                           seconds=round(time.monotonic()-start,2)),flush=True)
    if before!=[state_digest(a) for a in args[:2]]:
        raise ValueError('frozen model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,witnesses=witnesses,max_parent_ref_error=ref_error,
        max_zero_dose_error=zero_error,seconds=time.monotonic()-start))


def score(folder,out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    balance = read(folder/'postseal_bt_scale_balance.json')
    matched = set(balance['matched_ids'])
    historic = {r['shot'] for r in read(par.WIDE/'balanced_cohort.json')['order']}
    if (len(s['curve_sha256'])!=len(p['order']) or sha(folder/'seal.json')!=p['background_seal_sha256']
            or balance['seal_sha256']!=p['background_seal_sha256'] or p['executed_sha256']!=sha(out/'executed.py')):
        raise ValueError('complete paired seals required')
    ref = read(folder/'per_shot.json')
    old = read(bg.PARENT/'per_shot.json')
    rows = {b:{**{k:[] for k in ARMS[1:]},ARMS[0]:old[b][ARMS[0]],**ref[b]} for b in p['calibration']}
    peaks = {k:[] for k in ARMS[1:]}
    for i,r in enumerate(p['order'],1):
        sid = r['shot']
        d,down,t = group.hdf._raw_truth(sid)
        if (sid,d,down)!=tuple(ref['0.2']['fixed'][i-1][k] for k in ('shot','disruptive','down_s')):
            raise ValueError('HDF5 and background evaluation disagree')
        valid = np.round((down-t)*1000.,9)>30. if d else np.ones(len(t),bool)
        with np.load(par.bind(out,f'{i:04d}_{sid}.npz',s),allow_pickle=False) as f:
            for k in ARMS[1:]:
                peaks[k].append(dict(shot=sid,disruptive=d,peak=float(f[k][valid].max()) if valid.any() else None))
                for b,tau in p['calibration'].items():
                    rows[b][k].append(dict(shot=sid,disruptive=d,down_s=down,
                        **score_curve(t,f[k],d,down,p['original'] if i<=p['startup'] else tau)))
    write(out/'per_shot.json',{b:{k:x[k] for k in ARMS[1:]} for b,x in rows.items()})
    cohorts = dict(fixed360=lambda r:r['cohort']=='fixed360',extension=lambda r:r['cohort']=='extension',
                   historical_balance=lambda r:r['shot'] in historic,raw_scale_balance=lambda r:r['shot'] in matched,
                   old835=lambda r:r.get('old835') is True,other1758=lambda r:r.get('old835') is False)
    summary = {}
    for name,select in cohorts.items():
        idx = [i for i,r in enumerate(p['order']) if select(r)]
        result = {}
        for b,grouped in rows.items():
            table = {k:[v[i] for i in idx] for k,v in grouped.items()}
            result[b] = {f'{h}_{v}':dict(metrics=lead.summarize(table[f'{h}_{v}']),
                paired_vs_parent=dict(flips=flips(table[f'{h}_parent'],table[f'{h}_{v}']),
                                      lead=lead.paired(table[f'{h}_parent'],table[f'{h}_{v}'])))
                for h in ('level','drift') for v in ('parent','current','earlier')}
        summary[name] = result
        if name!='fixed360':
            n = result['0.2']['drift_parent']['metrics']['n_safe']
            result['postseal_matched_fp'] = {k:{str(c):{x:r[x] for x in ('tp','fp','max_fp')}
                for c,r in ((c,best_at_count([v[i] for i in idx],int(c*n))) for c in (.10,.15,.20))}
                for k,v in peaks.items()}
        print(name,{b:{k:(v['metrics']['by_deadline']['30']['tp'],v['metrics']['fp']) for k,v in result[b].items()}
                    for b in ('0.2','0.15')},flush=True)
    write(out/'result.json',dict(scope='POSTSEAL_MATCHED_GROUP_WRITE_EFFECT',groups=summary,
        raw_scored=len(p['order'])*len(p['calibration'])*2))


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('background',type=Path)
    a.add_argument('out',type=Path)
    opts = a.parse_args()
    run(opts.background.resolve(),opts.out.resolve())
    score(opts.background.resolve(),opts.out.resolve())
