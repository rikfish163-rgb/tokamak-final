"""Cold-prepared whole-shot risk readouts with a causal encoder background."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_parent_readout_20261113 as prior
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_transfer_20261108 import contrast
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .operating_point_audit import best_at_count
from .source_anticipation_20261104.static_stream import timing_decomposition


group,lag,delay = prior.group,prior.lag,prior.delay
PARENT = io23.ROOT/'probe_outputs/commissioning_parent_readout_east2953_20261113_v1'
INPUT, REFERENCE_SPAN = 128,128
FIRST = INPUT+REFERENCE_SPAN+INPUT-2
STEPS, RIDGE, LR = 256,.05,.02
VIEWS = ('static','current','earlier')
HEADS = ('offset','level','drift')
BASES = ('fixed','current_no_states','earlier_no_states')
ARMS = (*BASES,'offset_static',*(f'{h}_{v}' for h in HEADS[1:] for v in VIEWS))
PAIRS = (('fixed','offset_static'),('fixed','level_static'),('fixed','drift_static'),
         ('offset_static','level_static'),('offset_static','drift_static'),('level_static','drift_static'),
         ('current_no_states','drift_current'),('level_current','drift_current'),
         *((f'{h}_static',f'{h}_{v}') for h in HEADS[1:] for v in VIEWS[1:]),
         *((f'{h}_earlier',f'{h}_current') for h in HEADS[1:]))


def background_features(hidden):
    """Current embedding vs embeddings ending before its entire input window."""
    n = len(hidden)
    t = np.arange(FIRST,n)
    total = np.vstack((np.zeros((1,hidden.shape[1])),np.cumsum(hidden,axis=0,dtype=np.float64)))
    hi = t-INPUT+1
    lo = hi-REFERENCE_SPAN
    level,drift = np.zeros_like(hidden),np.zeros_like(hidden)
    level[t] = hidden[t]
    drift[t] = hidden[t]-(total[hi]-total[lo])/REFERENCE_SPAN
    return level,drift


def shot_loss(curves, labels, timely, n_positive):
    """Per-shot point loss plus a timely-vs-safety peak comparison."""
    point = torch.stack([F.binary_cross_entropy_with_logits(z,y.to(z.dtype)) for z,y in zip(curves,labels,strict=True)])
    positive = torch.stack([z[m].amax() for z,m in zip(curves[:n_positive],timely[:n_positive],strict=True)])
    negative = torch.stack([z.amax() for z in curves[n_positive:]])
    return .5*(point[:n_positive].mean()+point[n_positive:].mean()) + .5*F.softplus(
        negative[None,:]-positive[:,None]).mean() + .25*(F.softplus(-positive).mean()+F.softplus(negative).mean())


@torch.no_grad()
def static(signal,args,h=None):
    base,net,params,_,_,_,coef,_ = args
    if h is None:
        raw,_ = lag.features(base,signal,'cuda')
        h = torch.from_numpy(raw[:,:base.cfg.dim]).to(coef.device)
    else:
        raw = np.column_stack((h.cpu().numpy(),contrast(signal['values'],signal['available'])))
    tr = lag.direct.track(net,signal,*params[:2],np.arange(len(h)))
    design = np.column_stack((raw,torch.from_numpy(tr['pre']).tanh().numpy()))
    z = torch.from_numpy(design).to(coef.device)@coef[:-1]+coef[-1]
    return h,z


def fit(rows,head,cutoff):
    dev = rows[0]['z'].device
    pos = [r for r in rows if r['disruptive']]
    neg = [r for r in rows if not r['disruptive']]
    ordered = pos+neg
    if not pos or not neg or any(not r['timely'].any() for r in pos):
        raise ValueError('both classes and timely training support are required')
    co = torch.nn.Parameter(torch.zeros(1 if head=='offset' else ordered[0][head].shape[1]+1,device=dev))
    opt = torch.optim.Adam([co],lr=LR)
    groups = [r['z']-cutoff for r in ordered]
    eligible = [r['eligible'] for r in ordered]
    features = [r[head] for r in ordered] if head!='offset' else None
    labels = [r['label'] for r in ordered]
    times = [r['timely'] for r in ordered]
    def objective():
        z = [base+ok*(co[-1] if features is None else features[i]@co[:-1]+co[-1])
             for i,(base,ok) in enumerate(zip(groups,eligible,strict=True))]
        return shot_loss(z,labels,times,len(pos)) + RIDGE*co.square().sum()
    before = float(objective().detach())
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        opt.step()
    after = float(objective().detach())
    if not np.isfinite(after):
        raise ValueError('nonfinite prepared readout')
    return co.detach(),dict(before=before,after=after,n_disruptive=len(pos),n_safe=len(neg))


def fit_cold(args,lock,p,reference):
    rows = []
    for rank,sid in enumerate(p['order'][:p['startup']],1):
        name = f'{rank:04d}_{sid}.npz'
        prior.bind(prior.SMALL,name,reference)
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('prepared data changed')
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        y,down = io23.read_outcome('EAST',sid)
        signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
        h,z = static(signal,args)
        x = background_features(h.cpu().numpy())
        t = raw['times']
        label,valid = io23.labels_at(t,y,down,125.)
        eligible = np.arange(len(t))>=FIRST
        timely = (label>0.)&(np.round((down-t)*1000.,9)>30.) if y else np.zeros(len(t),bool)
        def v(a):
            return torch.from_numpy(a[valid]).to(z.device)
        rows.append(dict(shot=sid,disruptive=y,z=z[torch.from_numpy(valid).to(z.device)],level=v(x[0]),drift=v(x[1]),
                         label=v(label),timely=v(timely),eligible=v(eligible)))
    if sum(r['disruptive'] for r in rows)!=8:
        raise ValueError('the locked preparation has eight disruptive and twelve safe shots')
    coeff,details = {},{}
    for name in HEADS:
        coeff[name],details[name] = fit(rows,name,float(logit(p['calibration']['0.2'])))
    return coeff,details


@torch.no_grad()
def predict(signal,args,decoder,path,weights,coeff):
    base = args[0]
    with torch.enable_grad():
        q,h,z,beta,forecast,phase,correction = group.parent_track(signal,args,decoder,path)
        avail,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
        states,common = group._estimate_innovation(base,signal,q,avail,phase,correction,forecast,
                                                    fields,wave,decoder,weights)
    h,z_static = static(signal,args,h)
    hidden = h[None].repeat(2,1,1)
    def capture(_j,k,p,x,valid):
        hidden[:,k] = torch.where(valid[None,:,None],x,p[None])
    cur,ear,dose,_,err,_ = delay.render(base,signal,q,h,z,beta,phase,correction,states,common,
                                      waits=(5,),logits=True,hidden_trace=capture)
    output = dict(fixed=z_static.sigmoid().cpu().numpy())
    # Preserve the original phase/bias and grouped logit differences while
    # keeping both inherited readout-state terms out of the new head.
    zz = (z_static,*(torch.from_numpy(v).to(z.device)+(z_static-z) for v in (cur[0],ear[0])))
    output.update({k:v.sigmoid().cpu().numpy() for k,v in zip(BASES[1:],zz[1:],strict=True)})
    output['offset_static'] = (z_static+coeff['offset'][-1]*
                               (torch.arange(len(h),device=z.device)>=FIRST)).sigmoid().cpu().numpy()
    level,drift = (torch.from_numpy(v).to(z.device) for v in background_features(h.cpu().numpy()))
    eligible = torch.arange(len(h),device=z.device)>=FIRST
    for name,x in (('level',level),('drift',drift)):
        w = coeff[name]
        common_shift = x@w[:-1]+w[-1]
        for i,(key,zv) in enumerate(zip(VIEWS,zz,strict=True)):
            # Same fixed background from unmodified history in every arm.
            # Only the current encoder representation receives the original write.
            relative = 0. if i==0 else (hidden[i-1]-h)@w[:-1]
            output[f'{name}_{key}'] = (zv+eligible*(common_shift+relative)).sigmoid().cpu().numpy()
    return output,q+lag.H+1,common,dose[0],err


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,raw_ref,args,decoder,path = group.load_reference(lock)
    protocol,seal = (read(PARENT/f'{k}.json') for k in ('protocol','seal'))
    folders = {k:Path(v['path']) for k,v in protocol['references'].items()}
    bound = {k:read(f/'seal.json') for k,f in folders.items()}
    asset = delay.HISTORY/'teaching.json'
    weights = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    if (len(seal['curve_sha256'])!=2953 or [r['shot'] for r in protocol['order'][:360]]!=p['order']
            or any(sha(f/'seal.json')!=protocol['references'][k]['seal_sha256'] for k,f in folders.items())
            or protocol['calibration']!=p['calibration'] or protocol['original']!=p['original']
            or read(prior.WIDE/'protocol.json')['teaching_sha256']!=sha(asset)
            or p['startup']!=20 or args[0].cfg.window!=INPUT):
        raise ValueError('expected the sealed original and expansion forecasts')
    before = [state_digest(a) for a in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed.py')
    copyfile(Path(delay.__file__),out/'executed_delay.py')
    coeff,details = fit_cold(args,lock,p,bound['fixed360'])
    write(out/'fit.json',dict(coef={k:v.cpu().tolist() for k,v in coeff.items()},details=details,
                              prepared=p['order'][:20],steps=STEPS,ridge=RIDGE,lr=LR))
    write(out/'protocol.json',dict(scope='EAST_FROZEN_BACKGROUND_RISK_DEVELOPMENT',order=protocol['order'],
        startup=p['startup'],original=p['original'],calibration=p['calibration'],arms=ARMS,
        parent_seal=sha(PARENT/'seal.json'),original_reference_seals={k:sha(v/'seal.json') for k,v in folders.items()},
        fit_sha256=sha(out/'fit.json'),executed_sha256=sha(out/'executed.py'),executed_delay_sha256=sha(out/'executed_delay.py'),
        first=FIRST,background_points=REFERENCE_SPAN,embedding_window=INPUT,
        objective='class-balanced per-shot BCE plus all pairs of safe maxima and eligible timely positive maxima',
        feedback='unchanged grouping, direction, clock and dose; same unmodified history background; omit inherited readout states',
        access='EAST cold20 only for fit, all held predictions sealed before opening outcomes; fixed thresholds'))
    hashes,witnesses = {},{}
    max_score=max_norm=0.
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('holdout outcome forbidden before seal')):
        for i,row in enumerate(protocol['order'],1):
            sid,rank,c = (row[k] for k in ('shot','rank','cohort'))
            n = f'{rank:04d}_{sid}.npz'
            source = prior.bind(folders[c],n,bound[c])
            parent_path = prior.bind(PARENT,f'{i:04d}_{sid}.npz',seal)
            with np.load(source,allow_pickle=False) as f,np.load(parent_path,allow_pickle=False) as g:
                archived = {k:(f[k] if k=='fixed' else g[k]) for k in BASES}
                release,eligible,dose = f['release'],f['eligible'],f['dose'][-1]
                t = f['times'] if c=='extension' else None
            if c=='fixed360':
                with np.load(prior.bind(group.REFERENCE,n,raw_ref),allow_pickle=False) as f:
                    t = f['times']
            if i<=p['startup']:
                curves = dict.fromkeys(ARMS,archived['fixed'])
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                signal = lag.prepare_actual(raw,p['startup']+1,lock['mapping'])
                curves,r,e,d,norm = predict(signal,args,decoder,path,weights,coeff)
                max_score = max(max_score,prior.extension.parity({k:curves[k] for k in BASES},archived,p['calibration']))
                max_norm = max(max_norm,norm)
                np.testing.assert_array_equal(r,release)
                np.testing.assert_array_equal(e,eligible)
                np.testing.assert_array_equal(d>0.,dose>0.)
                np.testing.assert_allclose(d,dose,atol=3e-6,rtol=1e-6)
                for k in ARMS[3:]:
                    ref = 'fixed' if k.endswith('_static') else k.split('_')[-1]+'_no_states'
                    np.testing.assert_array_equal(curves[k][:FIRST],curves[ref][:FIRST])
                if i in (21,360,361,len(protocol['order'])):
                    end = min(550,len(t))
                    short,*_ = predict({k:signal[k][:end] for k in ('values','available')},
                                       args,decoder,path,weights,coeff)
                    witnesses[str(sid)] = dict(points=end,max_error=prior.extension.parity(short,curves,p['calibration']))
            dest = out/'shots'/f'{i:04d}_{sid}.npz'
            with dest.open('xb') as f:
                np.savez_compressed(f,**{k:curves[k] for k in ARMS[3:]})
            hashes[dest.name] = sha(dest)
            if i==21 or i%200==0 or i==len(protocol['order']):
                print(dict(stage='seal_background_readout',rank=i,total=len(protocol['order']),
                           seconds=round(time.monotonic()-start,2)),flush=True)
    if before!=[state_digest(a) for a in args[:2]]:
        raise ValueError('frozen model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,witnesses=witnesses,max_archive_error=max_score,
                              max_norm_error=max_norm,seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    frozen = read(PARENT/'per_shot.json')
    if (len(s['curve_sha256'])!=len(p['order']) or p['parent_seal']!=sha(PARENT/'seal.json')
            or p['fit_sha256']!=sha(out/'fit.json') or p['executed_sha256']!=sha(out/'executed.py')
            or p['executed_delay_sha256']!=sha(out/'executed_delay.py')):
        raise ValueError('full frozen score seal required')
    per = {b:{a:[] for a in ARMS} for b in p['calibration']}
    peaks = {a:[] for a in ARMS[3:]}
    for i,row in enumerate(p['order'],1):
        sid = row['shot']
        d,down,t = group.hdf._raw_truth(sid)
        timely = np.round((down-t)*1000.,9)>30. if d else np.ones(len(t),bool)
        with np.load(prior.bind(out,f'{i:04d}_{sid}.npz',s),allow_pickle=False) as f:
            for b,tau in p['calibration'].items():
                for k in BASES:
                    m = frozen[b][k][i-1]
                    if (m['shot'],m['disruptive'],m['down_s'])!=(sid,d,down):
                        raise ValueError('reference score or raw HDF5 changed')
                    per[b][k].append(m)
                for k in ARMS[3:]:
                    m = score_curve(t,f[k],d,down,p['original'] if i<=p['startup'] else tau)
                    per[b][k].append(dict(shot=sid,disruptive=d,down_s=down,**m))
            for k in peaks:
                peaks[k].append(dict(shot=sid,disruptive=d,peak=float(f[k][timely].max()) if timely.any() else None))
    write(out/'per_shot.json',per)
    b_ids = {r['shot'] for r in read(prior.WIDE/'balanced_cohort.json')['order']}
    cohorts = dict(fixed360=lambda r:r['cohort']=='fixed360',extension=lambda r:r['cohort']=='extension',
                   balanced=lambda r:r['shot'] in b_ids,old835=lambda r:r.get('old835') is True,
                   other1758=lambda r:r.get('old835') is False)
    result = {}
    for g,keep in cohorts.items():
        idx = [j for j,row in enumerate(p['order']) if keep(row)]
        summary = {b:{} for b in p['calibration']}
        for b,rows in per.items():
            data = {k:[v[j] for j in idx] for k,v in rows.items()}
            summary[b] = dict(metrics={k:dict(timing_decomposition(v),warning_times=lead.summarize(v))
                                      for k,v in data.items()},paired={f'{a}__to__{c}':dict(
                    flips=flips(data[a],data[c]),lead=lead.paired(data[a],data[c])) for a,c in PAIRS})
        result[g] = summary
        if g!='fixed360':
            n = summary['0.2']['metrics']['fixed']['n_safe']
            summary['postseal_matched_fp'] = {k:{str(cap):{m:v[m] for m in ('tp','fp','max_fp')}
                    for cap,v in ((c,best_at_count([z[j] for j in idx],int(c*n))) for c in (.10,.15,.20))}
                    for k,z in peaks.items()}
        print(g,{b:{k:(v['tp'],v['fp']) for k,v in summary[b]['metrics'].items()} for b in ('0.2','0.15')},
              flush=True)
    write(out/'result.json',dict(status='COMPLETE_FROZEN_BACKGROUND_RISK_DEVELOPMENT',
                               raw_scored=len(p['order'])*len(ARMS[3:])*len(per),groups=result))


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('out',type=Path)
    o = a.parse_args().out.resolve()
    run(o)
    score(o)
