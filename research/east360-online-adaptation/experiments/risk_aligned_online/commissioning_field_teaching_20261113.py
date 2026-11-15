"""Train a frozen-encoder EAST risk residual under paired diagnostic gain perturbations."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_field_scale_20261113 as field
from . import commissioning_hidden_background_20261113 as bg
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_extension_order_budget_20261113 import PREFIX
from .commissioning_transfer_20261108 import contrast
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .operating_point_audit import best_at_count


par,group,lag,delay = bg.prior,bg.group,bg.lag,bg.delay
FAMILIES = ('shot_balanced','point_balanced','gain_augmented','gain_consistent')
PATHS = ('static','parent','current','earlier')
REF = dict(static='fixed',parent='bias_no_states',current='current_no_states',earlier='earlier_no_states')
ARMS = (*REF.values(),*(f'{a}_{v}' for a in FAMILIES for v in PATHS))
STEPS,LR,RIDGE,CONSISTENCY = 256,.02,.05,1.


def point_weights(shot, label, disruptive):
    """Same records: equal outcome-class mass per shot vs per endpoint class."""
    group_count = np.bincount(shot)
    shot_weight = .5/np.bincount(disruptive.astype(int))[disruptive.astype(int)]
    by_shot = (shot_weight/group_count)[shot]
    by_point = np.zeros(len(shot),np.float64)
    for target in (False,True):
        mask = label==target
        counts = np.bincount(shot[mask],minlength=len(disruptive))
        if not mask.any():
            raise ValueError('both original endpoint labels are required')
        by_point[mask] = .5/(np.count_nonzero(counts)*counts[shot[mask]])
    return by_shot,by_point


def warning(z, groups, timely, pos, safe, n):
    """Recompute the safety maximum over every eligible record point."""
    idx = groups[None].expand(len(z),-1)
    blank = z.new_full((len(z),n),-torch.inf)
    p = blank.scatter_reduce(1,idx,z.masked_fill(~timely,-torch.inf),'amax')[:,pos]
    s = blank.scatter_reduce(1,idx,z,'amax')[:,safe]
    return .5*F.softplus(s[:,:,None]-p[:,None,:]).mean() + .25*(F.softplus(-p).mean()+F.softplus(s).mean())


@torch.no_grad()
def features(signal,args,h=None):
    base,net,params,_,_,_,coef,_ = args
    if h is None:
        raw,_ = lag.features(base,signal,'cuda')
    else:
        raw = np.column_stack((h.cpu().numpy(),contrast(signal['values'],signal['available'])))
    pre = lag.direct.track(net,signal,*params[:2],np.arange(len(raw)))['pre']
    x = torch.from_numpy(np.column_stack((raw,torch.from_numpy(pre).tanh().numpy()))).to(coef.device)
    z = x@coef[:-1]+coef[-1]
    return x,z


def prepare_rows(ids, device, args, lock, fit_cut, gain):
    src = device=='J-TEXT'
    advanced = io23.advanced_times() if src else {}
    packed = []
    for i,sid in enumerate(ids,1):
        r = simple_data.read_shot(device,sid,rate_contract='strict' if src else 'nominal')
        y = simple_data.attach_outcome(r,device,advanced)
        if src:
            a = r['available'].copy()
            a[:,io23.RAW23.index('MNM')] = False
            signal = dict(values=r['values'],available=a)
        else:
            if sha(io23.path_for(device,sid))!=lock['hashes'][str(sid)]:
                raise ValueError('changed cold HDF5')
            signal = lag.prepare_actual(r,21,lock['mapping'])
        sign = field.direction(r['planning_Bt'],fit_cut)
        altered = field.rescale(signal,(field.BT,field.QA),gain**sign)
        x,z = features(signal,args)
        xx,zz = features(altered,args)
        valid = y['valid']
        label = y['y'][valid].astype(bool)
        timely = label & (np.round((y['down']-r['times'][valid])*1000.,9)>(10. if src else 30.)) \
            if y['is_disrupt'] else np.zeros(len(label),bool)
        observed = np.flatnonzero(valid)
        kk = torch.from_numpy(observed>=field.N-1).to(x.device)
        # The arm has no contribution before its shared release.
        x,xx,z,zz = (v[torch.from_numpy(observed).to(v.device)] for v in (x,xx,z,zz))
        xx[~kk] = x[~kk]
        zz[~kk] = z[~kk]
        packed.append((sid,y['is_disrupt'],label,timely,kk.cpu().numpy(),x,xx,z,zz))
        if src and (i%300==0 or i==len(ids)):
            print(dict(stage='frozen_features',device=device,shots=i,total=len(ids)),flush=True)
    sid,d,label,timely,ok,x,xx,z,zz = zip(*packed,strict=True)
    label,timely,ok = (np.concatenate(v) for v in (label,timely,ok))
    g = np.repeat(np.arange(len(d)),[len(v[2]) for v in packed])
    d = np.asarray(d)
    p = np.unique(g[timely])
    n = np.flatnonzero(~d)
    w0,w1 = point_weights(g,label,d)
    dev = x[0].device
    def t(a):
        return torch.from_numpy(a).to(dev)
    return dict(ids=sid,disruptive=d,label=t(label),timely=t(timely),eligible=t(ok),groups=t(g),
                positive=t(p),safe=t(n),w0=t(w0.astype(np.float32)),w1=t(w1.astype(np.float32)),
                x=torch.stack((torch.cat(x),torch.cat(xx))),z=torch.stack((torch.cat(z),torch.cat(zz))),
                support=dict(shots=len(d),disruptive=int(d.sum()),safe=len(n),positive_points=int(label.sum()),
                             timely_points=int(timely.sum()),timely_shots=len(p),points=len(label),
                             point_positive_mass_original=float(w0[label].sum()),
                             point_positive_mass_balanced=float(w1[label].sum())))


def risk(coef,domain,views,cut):
    x,z = domain['x'][:views],domain['z'][:views]
    return z+(x@coef[:-1]+coef[-1])*domain['eligible']-cut


def domain_loss(z,domain,weight):
    t = domain['label'].expand_as(z).to(z.dtype)
    point = (F.binary_cross_entropy_with_logits(z,t,reduction='none')@domain[weight]).mean()
    w = warning(z,domain['groups'],domain['timely'],domain['positive'],domain['safe'],len(domain['ids']))
    return point+w


def fit(source,target,source_tau,east_tau):
    coef = {}
    details = {}
    for i,name in enumerate(FAMILIES):
        c = torch.nn.Parameter(source['x'].new_zeros(source['x'].shape[-1]+1))
        optim = torch.optim.Adam([c],lr=LR)
        weight = 'w0' if i==0 else 'w1'
        views = 1 if i<2 else 2
        def objective():
            a = risk(c,source,views,source_tau)
            b = risk(c,target,views,east_tau)
            loss = .5*(domain_loss(a,source,weight)+domain_loss(b,target,weight))+RIDGE*c.square().sum()
            if i==3:
                # Give positive and negative endpoints equal mass in the paired penalty as well.
                loss = loss + .5*CONSISTENCY*((a[1]-a[0]).square()@source['w1']+
                                              (b[1]-b[0]).square()@target['w1'])
            return loss
        initial = float(objective().detach())
        for _ in range(STEPS):
            optim.zero_grad(set_to_none=True)
            objective().backward()
            optim.step()
        with torch.no_grad():
            final = float(objective())
            if not np.isfinite(final):
                raise FloatingPointError('nonfinite fitted readout')
            coef[name] = c.detach()
            details[name] = dict(before=initial,after=final,weight_norm=float(c[:-1].norm()),
                bias=float(c[-1]),train_gain_gap={k:float((((v[1]-v[0]).square())@d['w1']).sqrt())
                    for k,d,tau in (('source',source,source_tau),('east',target,east_tau))
                    for v in (risk(c,d,2,tau),)})
            print(dict(stage='fitted_readout',arm=name,**details[name]),flush=True)
    return coef,details


def val_diagnostic(args,lock,source_val,fit_cut,gain,coef,threshold):
    """Once fitted, audit native and paired perturbations on the fixed source VAL."""
    r = prepare_rows(source_val,'J-TEXT',args,lock,fit_cut,gain)
    c = dict(fixed=r['x'].new_zeros(r['x'].shape[-1]+1),**coef)
    out = {}
    for arm,value in c.items():
        z = risk(value,r,2,float(logit(threshold)))
        with torch.no_grad():
            group_index = r['groups'][None].expand(2,-1)
            blank = z.new_full((2,len(source_val)),-torch.inf)
            timely = blank.scatter_reduce(1,group_index,z.masked_fill(~r['timely'],-torch.inf),'amax')[:,r['positive']]
            safe = blank.scatter_reduce(1,group_index,z,'amax')[:,r['safe']]
            out[arm] = dict(positive_window_hit=(timely>=0).sum(1).tolist(),fp=(safe>=0).sum(1).tolist(),
                            gain_rms=float(((z[1]-z[0]).square()@r['w1']).sqrt()))
    return dict(support=r['support'],threshold=threshold,arms=out,
                positive_window_hit_is_not_timely_first_alarm=True)


@torch.no_grad()
def predict(signal,args,decoder,path,weights,coef):
    base = args[0]
    with torch.enable_grad():
        q,h,z,beta,forecast,phase,correction = group.parent_track(signal,args,decoder,path)
        avail,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
        states,common = group._estimate_innovation(base,signal,q,avail,phase,correction,forecast,fields,wave,decoder,weights)
    x,static = features(signal,args,h)
    modified = h[None].repeat(3,1,1)
    def capture(_j,k,p,u,enabled):
        modified[0,k] = p
        modified[1:,k] = torch.where(enabled[None,:,None],u,p[None])
    cur,ear,dose,_,err,parent = delay.render(base,signal,q,h,z,beta,phase,correction,states,common,
                                             waits=(5,),logits=True,hidden_trace=capture)
    shifts = static-z
    base_z = (static,*(torch.from_numpy(v).to(z.device)+shifts for v in (parent,cur[0],ear[0])))
    arms = {k:v.sigmoid().cpu().numpy() for k,v in zip(REF.values(),base_z,strict=True)}
    gate = torch.arange(len(h),device=z.device)>=field.N-1
    delta = modified-h
    for name,c in coef.items():
        dz = x@c[:-1]+c[-1]
        for i,(k,v) in enumerate(zip(PATHS,base_z,strict=True)):
            a = dz if i==0 else dz+delta[i-1]@c[:h.shape[-1]]
            arms[f'{name}_{k}'] = (v+gate*a).sigmoid().cpu().numpy()
    return arms,q+lag.H+1,common,dose[0],err


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,origin,args,decoder,path = group.load_reference(lock)
    pp,ps = (read(bg.PARENT/f'{k}.json') for k in ('protocol','seal'))
    paths = {k:Path(v['path']) for k,v in pp['references'].items()}
    refs = {k:read(f/'seal.json') for k,f in paths.items()}
    roles = io23.source_roles()
    asset = delay.HISTORY/'teaching.json'
    weights = torch.tensor(read(asset)['weights'],device='cuda',dtype=torch.float32)
    if (pp['calibration']!=p['calibration'] or len(ps['curve_sha256'])!=2953 or p['startup']!=20
            or [r['shot'] for r in pp['order'][:360]]!=p['order'] or pp['original']!=p['original']
            or any(sha(f/'seal.json')!=pp['references'][k]['seal_sha256'] for k,f in paths.items())
            or read(par.WIDE/'protocol.json')['teaching_sha256']!=sha(asset)):
        raise ValueError('required frozen source, feedback and historical references changed')
    cold = p['order'][:20]
    medians = [float(np.median(abs(simple_data.read_shot('EAST',sid,rate_contract='nominal')['planning_Bt'][:field.N])))
               for sid in cold]
    cut,gain = field.reference_levels(medians)
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed.py')
    write(out/'protocol.json',dict(scope='FROZEN_ENCODER_GAIN_TRAINING_PAIRED_EAST',order=pp['order'],startup=20,
        original=p['original'],calibration=p['calibration'],arms=ARMS,parent_seal=sha(bg.PARENT/'seal.json'),
        source_train=roles['train'],source_val=roles['val'],target_fit=cold,cut_from_cold=cut,gain_from_cold=gain,
        first=field.N-1,fit=dict(steps=STEPS,lr=LR,ridge=RIDGE,consistency=CONSISTENCY),
        executed_sha256=sha(out/'executed.py'),feedback_asset_sha256=sha(asset),
        deployment='original signals only; same frozen diagnostic feedback, phase, bias, release, dose and thresholds'))
    state = [state_digest(m) for m in args[:2]]
    # Bind label access during fitting to the declared source TRAIN and sealed cold preparation.
    allowed = {('J-TEXT',i) for i in roles['train']}|{('EAST',i) for i in cold}
    original_read = io23.read_outcome
    def fit_read(device,sid):
        if (device,sid) not in allowed:
            raise ValueError('outcome outside declared training support')
        return original_read(device,sid)
    for i,sid in enumerate(cold,1):
        par.bind(paths['fixed360'],f'{i:04d}_{sid}.npz',refs['fixed360'])
    with patch.object(io23,'read_outcome',side_effect=fit_read):
        src = prepare_rows(roles['train'],'J-TEXT',args,lock,cut,gain)
        target = prepare_rows(cold,'EAST',args,lock,cut,gain)
        if (src['support']['disruptive'],target['support']['disruptive'])!=(188,8):
            raise ValueError('source TRAIN or cold support changed')
        coef,details = fit(src,target,float(logit(p['original'])),float(logit(p['calibration']['0.2'])))
    write(out/'fit.json',dict(weights={k:v.tolist() for k,v in coef.items()},details=details,
                              support={'source':src['support'],'cold':target['support']}))
    del src,target
    allowed = {('J-TEXT',i) for i in roles['val']}
    with patch.object(io23,'read_outcome',side_effect=fit_read):
        write(out/'source_val.json',val_diagnostic(args,lock,roles['val'],cut,gain,coef,p['original']))
    hashes,witnesses = {},{}
    archived_error=dose_error=0.
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('current outcome forbidden before sealing')):
        for i,row in enumerate(pp['order'],1):
            sid,rank,c = (row[k] for k in ('shot','rank','cohort'))
            name,n = f'{i:04d}_{sid}.npz',f'{rank:04d}_{sid}.npz'
            with np.load(par.bind(paths[c],n,refs[c]),allow_pickle=False) as f, \
                    np.load(par.bind(bg.PARENT,name,ps),allow_pickle=False) as g:
                ref = {k:f['fixed'] if k=='fixed' else g[k] for k in REF.values()}
                release,eligible,dose = f['release'],f['eligible'],f['dose'][-1]
                times = f['times'] if c=='extension' else None
            if i<=20:
                values = dict.fromkeys(ARMS,ref['fixed'])
            else:
                if c=='fixed360':
                    with np.load(par.bind(group.REFERENCE,n,origin),allow_pickle=False) as f:
                        times = f['times']
                r = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(r['times'],times)
                sig = lag.prepare_actual(r,21,lock['mapping'])
                values,rr,ee,dd,err = predict(sig,args,decoder,path,weights,coef)
                archived_error = max(archived_error,par.extension.parity({k:values[k] for k in ref},ref,p['calibration']))
                dose_error = max(dose_error,err)
                np.testing.assert_array_equal(rr,release)
                np.testing.assert_array_equal(ee,eligible)
                np.testing.assert_array_equal(dd>0.,dose>0.)
                np.testing.assert_allclose(dd,dose,atol=3e-6,rtol=1e-6)
                for k in FAMILIES:
                    for v,a in REF.items():
                        np.testing.assert_array_equal(values[f'{k}_{v}'][:field.N-1],values[a][:field.N-1])
                    for v in ('current','earlier'):
                        np.testing.assert_array_equal(values[f'{k}_{v}'][dd==0.],values[f'{k}_parent'][dd==0.])
                if i in (21,360,361,len(pp['order'])):
                    stop = min(550,len(times))
                    short,*_ = predict({k:sig[k][:stop] for k in ('values','available')},args,decoder,path,weights,coef)
                    witnesses[str(sid)] = dict(points=stop,max_error=par.extension.parity(short,values,p['calibration']))
            file = out/'shots'/name
            with file.open('xb') as f:
                np.savez_compressed(f,**{k:v for k,v in values.items() if k not in ref})
            hashes[name] = sha(file)
            if i==21 or i%200==0 or i==len(pp['order']):
                print(dict(stage='seal_gain_training',rank=i,total=len(pp['order']),seconds=round(time.monotonic()-start,2)),flush=True)
    if state!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen source/diagnostics were modified')
    write(out/'seal.json',dict(curve_sha256=hashes,witnesses=witnesses,max_reference_error=archived_error,
        max_dose_error=dose_error,fit_sha256=sha(out/'fit.json'),seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if (len(s['curve_sha256'])!=2953 or p['parent_seal']!=sha(bg.PARENT/'seal.json')
            or p['executed_sha256']!=sha(out/'executed.py') or s['fit_sha256']!=sha(out/'fit.json')):
        raise ValueError('complete paired seal required')
    old = read(bg.PARENT/'per_shot.json')
    b = read(bg.PARENT/'postseal_bt_scale_balance.json')
    matched = set(b['matched_ids'])
    historical = {r['shot'] for r in read(par.WIDE/'balanced_cohort.json')['order']}
    if (b['seal_sha256']!=p['parent_seal']
            or sha(PREFIX/'order.json')!=read(PREFIX/'roster.json')['order_sha256']):
        raise ValueError('postseal raw strata or fixed balance changed')
    profiles = {r['shot']:r['initial_abs_Bt_raw'] for r in read(PREFIX/'order.json')}
    new = ARMS[len(REF):]
    rows = {b:{**{k:old[b][k] for k in REF.values()},**{k:[] for k in new}} for b in p['calibration']}
    peaks = {k:[] for k in new}
    for i,r in enumerate(p['order'],1):
        sid = r['shot']
        d,down,t = group.hdf._raw_truth(sid)
        if (sid,d,down)!=tuple(rows['0.2']['fixed'][i-1][k] for k in ('shot','disruptive','down_s')):
            raise ValueError('raw HDF5 outcome or identity changed')
        valid = np.round((down-t)*1000.,9)>30. if d else np.ones(len(t),bool)
        with np.load(par.bind(out,f'{i:04d}_{sid}.npz',s),allow_pickle=False) as f:
            for k in new:
                v = f[k]
                peaks[k].append(dict(disruptive=d,peak=float(v[valid].max()) if valid.any() else None))
                for band,tau in p['calibration'].items():
                    rows[band][k].append(dict(shot=sid,disruptive=d,down_s=down,
                        **score_curve(t,v,d,down,p['original'] if i<=20 else tau)))
    write(out/'per_shot.json',rows)
    selectors = dict(fixed360=lambda r:r['cohort']=='fixed360',extension=lambda r:r['cohort']=='extension',
        balanced_history=lambda r:r['shot'] in historical,balanced_field=lambda r:r['shot'] in matched,
        old835=lambda r:r.get('old835') is True,other1758=lambda r:r.get('old835') is False,
        smaller=lambda r:r['cohort']=='extension' and profiles[r['shot']]<b['raw_split'],
        larger=lambda r:r['cohort']=='extension' and profiles[r['shot']]>=b['raw_split'])
    pairs = (*((REF[v],f'{a}_{v}') for a in FAMILIES for v in PATHS),
             *((f'{a}_parent',f'{a}_{v}') for a in FAMILIES for v in ('current','earlier')),
             *((f'{a}_earlier',f'{a}_current') for a in FAMILIES),
             *((f'{a}_{v}',f'{c}_{v}') for a,c in zip(FAMILIES,FAMILIES[1:]) for v in PATHS))
    result = {}
    for g,keep in selectors.items():
        idx = [i for i,r in enumerate(p['order']) if keep(r)]
        table = {}
        for band,v in rows.items():
            cur = {k:[a[i] for i in idx] for k,a in v.items()}
            table[band] = dict(metrics={k:lead.summarize(v) for k,v in cur.items()},
                paired={f'{a}__to__{b}':dict(flips=flips(cur[a],cur[b]),lead=lead.paired(cur[a],cur[b])) for a,b in pairs})
        if g!='fixed360':
            n = table['0.2']['metrics']['fixed']['n_safe']
            table['postseal_matched_fp'] = {k:{str(c):{x:r[x] for x in ('tp','fp','max_fp')}
                for c,r in ((c,best_at_count([v[i] for i in idx],int(c*n))) for c in (.10,.15,.20))}
                for k,v in peaks.items()}
        result[g] = table
        print(g,{b:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in table[b]['metrics'].items()}
                 for b in ('0.2','0.15')},flush=True)
    write(out/'result.json',dict(scope='COMPLETE_PAIRED_GAIN_TRAINING_DEVELOPMENT',groups=result,
                               raw_scored=len(p['order'])*len(new)*len(rows)))


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('out',type=Path)
    dest = a.parse_args().out.resolve()
    run(dest)
    score(dest)
