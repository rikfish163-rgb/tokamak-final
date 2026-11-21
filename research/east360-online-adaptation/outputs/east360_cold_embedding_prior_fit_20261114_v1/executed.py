"""Learn a small cold-prefix embedding prior; replay the same-parent shot adaptation."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit
from torch.nn import functional as F

from . import commissioning_temporal_pair_20261114 as pair
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_write_delay_20261112 import HISTORY
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips


embed,history,group,lag = pair.embed,pair.history,pair.group,pair.lag
layer,control = pair.feedback.layer,pair.feedback
VALUES = pair.VALUES
OLD = embed.OLD
PRIOR,FIXED,NOW,OLDER = MODES = ('cold_prior_only','cold_prior_fixed','cold_prior_current','cold_prior_earlier')
ARMS = (*OLD,*MODES)
RANK,STEPS,LR,RIDGE,TEMP = 4,64,.15,.02,.25
MIN_SEPARATION,MAX_SAFE_PEAKS = 32,8
CASES = (81406,64581,90874,70320,73405,74970,85422)


@contextmanager
def value_prior(base,weight):
    """Equivalent to changing only the observed columns of the first linear map."""
    def write_numeric(_module,args,output):
        data = embed.measured_values(args[0],values=VALUES)
        return output+F.linear(data,weight)
    h = base.embedding.register_forward_hook(write_numeric)
    try:
        yield
    finally:
        h.remove()


def pass_hidden(base,x,m,adjust):
    held = []
    hooks = (base.embedding.register_forward_pre_hook(lambda _module,a:(adjust(a[0]),)),
             base.risk_head.register_forward_pre_hook(lambda _module,a:held.append(a[0])))
    try:
        base(x,m)
    finally:
        for h in hooks:
            h.remove()
    return held[0]


def bag_indices(times,score,disruptive,down):
    if disruptive:
        ahead = np.round((down-times)*1000.,9)
        return np.flatnonzero((ahead>30.) & (ahead<=125.))
    # Sample whole-shot competing peaks, not only samples near a known alarm.
    chosen = []
    for i in np.argsort(-score,kind='stable'):
        if all(abs(i-j)>=MIN_SEPARATION for j in chosen):
            chosen.append(i)
        if len(chosen)==MAX_SAFE_PEAKS:
            break
    return np.sort(chosen)


def gradient_span(rows):
    g = F.normalize(rows.flatten(1).double(),dim=1,eps=1e-12)
    _,s,v = torch.linalg.svd(g,full_matrices=False)
    if len(s)<RANK or s[RANK-1]<=s[0]*1e-6:
        raise ValueError('insufficient independent cold gradient probes')
    return v[:RANK].to(rows.dtype).reshape(RANK,*rows.shape[1:]),s.tolist()


@dataclass
class Bag:
    shot: int
    label: bool
    times: np.ndarray
    down: float | None
    signal: dict
    track: tuple
    ends: np.ndarray
    x: torch.Tensor
    m: torch.Tensor
    phase: torch.Tensor
    correction: torch.Tensor
    offset: torch.Tensor


def graph(base,s,tr,i):
    q,h,z,beta,_,phase,correction = tr
    slot = lag.delayed_slots(q,i)[0]
    x,m = lag._windows(base,s,i)
    return x,m,phase[slot],correction[slot],z[i]-(h[i]*beta).sum(-1)


def scores(base,beta,batch):
    x,m,a,c,offset = batch
    return offset+pass_hidden(base,x,m,layer.patcher(a,c))@beta


@torch.no_grad()
def curve(base,s,tr):
    z = tr[2]
    out = []
    for i,n in control.block_indices(np.arange(len(z))):
        out.append(scores(base,tr[3],graph(base,s,tr,i))[:n].sigmoid().cpu().numpy())
    return np.concatenate(out)


def bag_graph(bag):
    return bag.x,bag.m,bag.phase,bag.correction,bag.offset


def accumulate_bags(cold,lock,args,decoder,path):
    base = args[0]
    result = []
    for sid in cold:
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('locked completed cold signal changed')
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        y,down,t = group.hdf._raw_truth(sid)
        # Re-express only the ended cold traces in the already fixed deployment
        # coordinates. The actual cold predictions were sealed before this fit.
        sig = lag.prepare_actual(raw,21,lock['mapping'])
        tr = group.parent_track(sig,args,decoder,path)
        parent = curve(base,sig,tr)
        i = bag_indices(t,parent,y,down)
        if not len(i):
            raise ValueError(f'cold shot {sid} lacks a legal supervised peak')
        result.append(Bag(sid,y,t,down,sig,tr,i,*graph(base,sig,tr,i)))
    return result


def probe_gradients(base,bags):
    found = []
    for bag in bags:
        w = bag.x.new_zeros((base.cfg.dim,embed.PATCH*len(VALUES)),requires_grad=True)
        with value_prior(base,w):
            z = scores(base,bag.track[3],bag_graph(bag))
            peak = torch.logsumexp(z/TEMP,dim=0)*TEMP
        g, = torch.autograd.grad(peak if bag.label else -peak,w)
        found.append(g)
    return torch.stack(found)


def pack(bags):
    return tuple(torch.cat(p) for p in zip(*(bag_graph(b) for b in bags),strict=True))


@torch.no_grad()
def basis_units(base,batch,basis):
    observed = []
    def measure(_m,args,_output):
        observed.append(embed.measured_values(args[0],values=VALUES))
    h = base.embedding.register_forward_hook(measure)
    try:
        scores(base,base.center.new_zeros(base.cfg.dim),batch)
    finally:
        h.remove()
    norm = torch.einsum('rde,bpe->rbpd',basis,observed[0]).flatten(2).square().sum(-1).mean(1).sqrt()
    if (norm<=1e-6).any():
        raise ValueError('cold gradient span cannot write to the selected observations')
    return norm


def fit_prior(base,bags,rows,threshold):
    basis,singular = gradient_span(rows)
    batch = pack(bags)
    units = basis_units(base,batch,basis)
    basis = basis/units[:,None,None]
    beta = bags[0].track[3]
    lengths = [len(z.ends) for z in bags]
    pos = torch.tensor([b.label for b in bags],device=beta.device)
    if not pos.any() or pos.all():
        raise ValueError('cold calibration needs both classes')
    coef = torch.nn.Parameter(basis.new_zeros(RANK))
    opt = torch.optim.Adam([coef],lr=LR)
    mark = float(logit(threshold))
    trace = []
    for j in range(STEPS):
        with value_prior(base,torch.einsum('r,rde->de',coef,basis)):
            z = scores(base,beta,batch)
        tops = torch.stack([v.max() for v in z.split(lengths)])
        pos_loss = F.softplus((mark-tops[pos])/TEMP).mean()
        safe_loss = F.softplus((tops[~pos]-mark)/TEMP).mean()
        loss = .5*(pos_loss+safe_loss)+RIDGE*coef.square().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coef],2.,error_if_nonfinite=True)
        opt.step()
        if j==0 or (j+1)%16==0:
            trace.append(dict(step=j+1,positive=float(pos_loss.detach()),safe=float(safe_loss.detach()),
                              total=float(loss.detach()),coef=coef.detach().cpu().tolist()))
    w = torch.einsum('r,rde->de',coef.detach(),basis)
    return w,dict(singular=singular,units=units.tolist(),coefficients=coef.detach().cpu().tolist(),trace=trace)


def innovation(base,s,tr,args,decoder,weights):
    q,_,z,_,prior,phase,correction = tr
    available,fields,wave = group.feedback_waveform(s,q,args[2],z.device)
    return group._estimate_innovation(base,s,q,available,phase,correction,prior,fields,
                                      wave,decoder,weights)


def history_weights(p):
    path = HISTORY/'teaching.json'
    h,done = (read(HISTORY/k) for k in ('protocol.json','seal.json'))
    if (any(h[k]!=p[k] for k in history.LOCK) or h['teaching_sha256']!=sha(path)
            or len(done['curve_sha256'])!=360):
        raise ValueError('locked original real/earlier feedback operator required')
    return torch.as_tensor(read(path)['weights'],dtype=torch.float32,device='cuda')


def fit_stage(out):
    now = monotonic()
    lock,p,old,args,decoder,path = history.inputs()
    base = args[0]
    out.mkdir()
    copyfile(__file__,out/'executed.py')
    cold = p['order'][:p['startup']]
    for i,sid in enumerate(cold,1):
        # Open their already sealed native predictions before revealing labels.
        archive = history.archived_shot(i,sid,old)
        if not np.array_equal(archive['bias_only'],archive['current_5']):
            raise ValueError('cold path should have no original dynamic correction')
    write(out/'cold_sealed.json',dict(ids=cold,source=p['source_sha256'],original_threshold=p['original']))
    allowed = set(cold)
    original_reader = group.hdf._raw_truth
    def ended_preparation(sid):
        if sid not in allowed:
            raise ValueError('only the ended, prediction-sealed preparation labels may be fitted')
        return original_reader(sid)
    with patch.object(group.hdf,'_raw_truth',side_effect=ended_preparation):
        data = accumulate_bags(cold,lock,args,decoder,path)
    truth = np.array([b.label for b in data])
    if (len(truth),int(truth.sum()))!=(20,8):
        raise ValueError('preserve the fixed cold20 composition')
    tau = p['calibration']['0.2']
    grads = probe_gradients(base,data)
    validation = {k:[] for k in ('cold_replay_parent','cold_oof_prior','cold_full_fit')}
    for hold in np.array_split(np.arange(len(data)),5):
        ix = np.setdiff1d(np.arange(len(data)),hold)
        w,_ = fit_prior(base,[data[i] for i in ix],grads[ix],tau)
        for i in hold:
            b = data[i]
            for name,weight in (('cold_replay_parent',w*0.),('cold_oof_prior',w)):
                with value_prior(base,weight):
                    z = curve(base,b.signal,b.track)
                r = score_curve(b.times,z,b.label,b.down,tau)
                validation[name].append(dict(shot=b.shot,disruptive=b.label,down_s=b.down,**r))
    w,record = fit_prior(base,data,grads,tau)
    weights = history_weights(p)
    patterns = []
    with value_prior(base,w):
        for b in data:
            z = curve(base,b.signal,b.track)
            r = score_curve(b.times,z,b.label,b.down,tau)
            validation['cold_full_fit'].append(dict(shot=b.shot,disruptive=b.label,down_s=b.down,**r))
            states,common = innovation(base,b.signal,b.track,args,decoder,weights)
            ids = np.flatnonzero(common)+1
            if len(ids):
                v = states[-2,ids].flatten(1)
                patterns.append(F.normalize(v,dim=1,eps=1e-8).mean(0))
    template = torch.stack(patterns).mean(0)
    if template.norm()<=1e-6:
        raise ValueError('ended cold traces do not support a fixed-direction control')
    template = F.normalize(template,dim=0).reshape(states.shape[-2:])
    torch.save(dict(prior=w.detach().cpu(),fixed=template.detach().cpu()),out/'prior.pt')
    vals = {k:lead.summarize(v) for k,v in validation.items()}
    write(out/'fit_result.json',dict(**{k:p[k] for k in history.LOCK},cold=cold,rank=RANK,steps=STEPS,
        learning_rate=LR,ridge=RIDGE,temperature=TEMP,values=VALUES,fitting=record,template_shots=len(patterns),
        bag_sizes=[len(b.ends) for b in data],validation=vals,per_shot=validation,
        use='only the prior and fixed template after preparation shot 20; never rescore actual cold',seconds=monotonic()-now))
    print({k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in vals.items()},flush=True)


def trial(sig,args,decoder,path,weights,archive,w,template):
    base = args[0]
    tr = group.parent_track(sig,args,decoder,path)
    q,h,z,beta,_,phase,correction = tr
    release = q+lag.H+1
    np.testing.assert_array_equal(release,archive['release'])
    with value_prior(base,w):
        states,eligible = innovation(base,sig,tr,args,decoder,weights)
        np.testing.assert_array_equal(eligible,archive['eligible'])
        state = torch.cat((template.expand(1,len(q)+1,*template.shape).clone(),states[-2:]))
        # Slot zero is before any complete observed packet for all arms.
        state[:,0] = 0.
        result = np.empty((len(MODES),len(z)),np.float32)
        spent = np.zeros_like(archive['dose'])
        maximum = 0.
        with torch.no_grad():
            for ticks,n in control.block_indices(np.arange(len(z))):
                i = ticks[:n]
                want = archive['dose'][ticks]
                j = lag.delayed_slots(q,ticks)[0]
                adjust = layer.patcher(phase[j],correction[j])
                x,m = lag._windows(base,sig,ticks)
                if want.any():
                    slot = np.searchsorted(release+history.WAIT,ticks,side='right')
                    r = np.r_[0,release+history.WAIT][slot]
                    alpha = z.new_tensor(group.handoff_fraction(r,ticks,embed.PATCH))
                    enable = torch.as_tensor(np.r_[False,eligible][slot],device=z.device)
                    prev = state[:,np.maximum(slot-1,0)]*enable[None,:,None,None]
                    view,dose,err = group.forward_hidden(base,x,m,adjust,writes=state[:,slot],
                        handoff=(prev,alpha),target_norm=z.new_tensor(want))
                    maximum = max(maximum,float(err),float(abs(dose[:n]-z.new_tensor(want[:n])).max()))
                    spent[i] = dose[:n].cpu().numpy()
                else:
                    view = pass_hidden(base,x,m,adjust)[None].expand(len(MODES),-1,-1)
                result[:,i] = (z[ticks]+((view-h[ticks])*beta).sum(-1))[:,:n].sigmoid().cpu().numpy()
    if maximum>1e-5 or not np.isfinite(result).all():
        raise ValueError(f'cold-prior replay failed to match the archived first-layer dose: {maximum}')
    return dict(zip(MODES,result,strict=True)),spent,maximum


def replay(out,fit,only):
    now = monotonic()
    lock,p,old,args,decoder,path = history.inputs()
    weight = history_weights(p)
    r = read(fit/'fit_result.json')
    if any(p[k]!=r[k] for k in history.LOCK) or p['order'][:p['startup']]!=r['cold'] or r['values']!=list(VALUES):
        raise ValueError('same frozen stream and only ended preparation supervision required')
    d = torch.load(fit/'prior.pt',map_location='cuda',weights_only=True)
    w,template = d['prior'],d['fixed']
    base = args[0]
    frozen = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},fit=str(fit),only=only,arms=ARMS,
        clock='fixed cold20 prior; at q+31 release unsupervised old/new direction, 8-sample handoff',
        interventions='same frozen head, exactly equal realized token L2 among fixed/observed/earlier',
        labels='20 completed prior records only; do not read any suffix outcome before sealing'))
    receipts,prefix,maximum = [],{},0.
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden during replay')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('HDF5 truth forbidden before replay seal'))):
        for rank,sid in enumerate(p['order'],1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,old)
            if rank<=p['startup']:
                change = dict.fromkeys(MODES,arc['bias_only'])
                dose = np.zeros_like(arc['dose'])
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('fixed EAST signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                change,dose,error = trial(sig,args,decoder,path,weight,arc,w,template)
                maximum = max(maximum,error)
                if rank in (21,120,360) or sid in CASES or only is not None:
                    n = min(650,len(dose))
                    cut,a = control.trimmed(sig,arc,n)
                    short,m,_ = trial(cut,args,decoder,path,weight,a,w,template)
                    np.testing.assert_allclose(m,dose[:n],atol=2e-6,rtol=0.)
                    for k,v in short.items():
                        np.testing.assert_allclose(v,change[k][:n],atol=3e-6,rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(v>=tau,change[k][:n]>=tau)
                    prefix[str(sid)] = dict(points=n,max_error=max(float(abs(v-change[k][:n]).max())
                                                                               for k,v in short.items()))
            name = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,**{k:arc[k] for k in OLD},**change,dose=dose)
            receipts.append(name)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(rank=rank,seconds=round(monotonic()-now,2)),flush=True)
    for k,v in base.state_dict().items():
        if not torch.equal(v.detach().cpu(),frozen[k]):
            raise ValueError(f'calibration must not change source state: {k}')
    write(out/'seal.json',dict(shots=receipts,prefix=prefix,dose_error=maximum,seconds=monotonic()-now))


def metrics(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    order = p['order']
    if p['only'] is not None or len(order)!=360 or s['shots']!=[f'{i:04d}_{j}.npz' for i,j in enumerate(order,1)]:
        raise ValueError('score only after sealing the 360 physical shot trajectories')
    old = read(history.ARCHIVE/'per_shot.json')
    tables = {c:{k:[] for k in ARMS} for c in p['calibration']}
    dose_count,hard = 0,[]
    for i,(sid,name) in enumerate(zip(order,s['shots'],strict=True)):
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            v = dict(f)
        used = v['dose']>0.
        dose_count += int(used.sum())
        for k in ARMS:
            if k in (FIXED,NOW,OLDER):
                np.testing.assert_allclose(v[k][~used],v[PRIOR][~used],rtol=0.,atol=2e-6)
            for c,tau in p['calibration'].items():
                r = dict(shot=sid,disruptive=y,down_s=d,**score_curve(t,v[k],y,d,
                            p['original'] if i<p['startup'] else tau))
                if k in OLD and r!=old[c][k][i]:
                    raise ValueError('archived reference or scoring clock changed')
                tables[c][k].append(r)
        original = tables['0.2']['current_5'][-1]
        if i>=p['startup'] and y and not original['timely']:
            ahead = np.round((d-t)*1000.,9)
            legal = used & (ahead>30.) & (ahead<=125.)
            if legal.any():
                tau = float(logit(p['calibration']['0.2']))
                gap = {k:tau-float(logit(v[k][legal].clip(1e-7,1.-1e-7).max()))
                       for k in ('bias_only',*MODES)}
                hard.append(dict(shot=sid,dose_points=int(legal.sum()),reported=original['alarm'],gap=gap))
    results = {c:{k:lead.summarize(v) for k,v in a.items()} for c,a in tables.items()}
    pairs = [('bias_only',PRIOR),('current_5',NOW),(PRIOR,NOW),(FIXED,NOW),(OLDER,NOW)]
    comp = {c:{f'{a}__to__{b}':flips(rs[a],rs[b]) for a,b in pairs} for c,rs in tables.items()}
    write(out/'per_shot.json',tables)
    write(out/'result.json',dict(metrics=results,paired=comp,hard=hard,dose_points=dose_count,
                                 raw_scores=len(order)*len(ARMS)*len(tables)))
    print({c:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in a.items()} for c,a in results.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    cli.add_argument('--prior',type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--fit',action='store_true')
    mode.add_argument('--score',action='store_true')
    mode.add_argument('--smoke-rank',action='append',type=int)
    a = cli.parse_args()
    if a.fit:
        fit_stage(a.out.resolve())
    elif a.score:
        metrics(a.out.resolve())
    elif a.prior is None:
        cli.error('--prior from ended preparation is required')
    else:
        replay(a.out.resolve(),a.prior.resolve(),a.smoke_rank)
