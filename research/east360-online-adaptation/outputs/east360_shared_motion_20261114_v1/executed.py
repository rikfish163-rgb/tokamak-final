"""Source-trained trajectory representation and an EAST shot-local update of it."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch import nn
from torch.nn import functional as F

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data, simple_train, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


history, lag, group, layer = probe.history, probe.lag, probe.group, probe.layer
WIDTH, PATCH = history.WINDOW, history.PATCH
H, BIN, K = lag.H, group.BIN, lag.H//group.BIN
CHANNELS, C = probe.COLUMNS, len(probe.COLUMNS)
FEATURES = WIDTH//PATCH+4
OFFSET = H+1+history.WAIT
READY = OFFSET+PATCH
BATCH, SEED = 128, 131
FORECAST_STEPS, RISK_STEPS, TRAIN_BATCH = 512, 512, 48
CTRL = ('source_representation', 'current_feedback', 'earlier_feedback')
ARMS = (*history.OLD, *CTRL)
CASES = (70320, 73405, 69813, 73790, 74970, 85422)


@dataclass(frozen=True)
class Motion:
    features: torch.Tensor
    anchor: torch.Tensor
    known: torch.Tensor


def waveform(knots):
    """Direct piecewise-linear offsets at q+1…q+25; past anchor has offset zero."""
    initial = F.pad(knots[:, :-1], (0,0,1,0))
    frac = torch.arange(1, BIN+1, device=knots.device, dtype=knots.dtype)/BIN
    return (initial[:,:,None] + (knots-initial)[:,:,None]*frac[None,None,:,None]).flatten(1,2)


def shared_state(knots, adjustment):
    """A temporary slope and bend for each observed channel."""
    u = torch.arange(1,K+1,device=knots.device,dtype=knots.dtype)[None,:,None]/K
    return knots + u*adjustment[:,:1] + 4*u*(1-u)*adjustment[:,1:2]


def motion_view(z, mask, scale):
    # Do not count padding as an observation or replace a missing point by a zero signal.
    tail = mask[:,-2*PATCH:]
    known = tail.any(1)
    span = torch.arange(tail.shape[1], device=z.device)[None,:,None]
    index = torch.where(tail,span,0).amax(1,keepdim=True)
    anchor = z[:,-2*PATCH:].gather(1,index).squeeze(1).masked_fill(~known,0.)
    count = mask.unflatten(1,(-1,PATCH)).sum(2)
    values = (z-anchor[:,None]).masked_fill(~mask,0.).unflatten(1,(-1,PATCH)).sum(2)
    offsets = (values/count.clamp_min(1)/scale).clamp(-8.,8.)
    # The latest scales and availability are features, not substituted measurements.
    overall = z.masked_fill(~mask,0.).sum(1)/mask.sum(1).clamp_min(1)
    features = torch.cat((offsets.transpose(1,2),
        torch.stack((anchor.clamp(-8.,8.),overall.clamp(-8.,8.),
                     tail.sum(1)/tail.shape[1],mask.sum(1)/WIDTH),dim=-1)),dim=-1)
    return Motion(features,anchor,known)


def input_view(base,x,mask,scale):
    c = list(CHANNELS)
    m = mask[...,c] & torch.isfinite(x[...,c])
    z = torch.asinh((torch.where(m,x[...,c],base.center[c])-base.center[c])/base.scale[c])
    return motion_view(z,m,scale)


def target_view(base,x,mask,observed,scale):
    c = list(CHANNELS)
    a = mask[...,c] & torch.isfinite(x[...,c]) & observed.known[:,None]
    z = torch.asinh((torch.where(a,x[...,c],base.center[c])-base.center[c])/base.scale[c])
    return ((z-observed.anchor[:,None])/scale).masked_fill(~a,0.),a


def masked_average(values, mask):
    return values.masked_fill(~mask,0.).sum(dim=(1,2))/mask.sum(dim=(1,2)).clamp_min(1)


def increment(wave):
    m = wave.unflatten(1,(K,BIN)).mean(2)
    return m-F.pad(m[:,:-1],(0,0,1,0))


def increment_mask(mask):
    v = mask.unflatten(1,(K,BIN)).all(2)
    return v & F.pad(v[:,:-1],(0,0,1,0),value=True)


def sample_loss(knots, target, mask):
    guess = waveform(knots)
    def smooth(x,y):
        return F.smooth_l1_loss(x,y,beta=.5,reduction='none')

    return .5*(masked_average(smooth(guess,target),mask) +
               masked_average(smooth(increment(guess),increment(target)),increment_mask(mask)))


def absolute_errors(guess,target,mask):
    """Per-window model/hold errors on identical future points and increments."""
    m = increment_mask(mask)
    a,b = increment(guess),increment(target)
    return torch.stack((masked_average((guess-target).abs(),mask),
        masked_average(target.abs(),mask), masked_average((a-b).abs(),m),
        masked_average(b.abs(),m)),dim=1)


class Forecaster(nn.Module):
    """Source-trained causal history -> five displacement knots for every channel."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C*FEATURES,64),nn.SiLU(),nn.Linear(64,K*C))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observed):
        return self.net(observed.features.flatten(1)).unflatten(1,(K,C)).masked_fill(~observed.known[:,None],0.)


class Carrier(nn.Module):
    """Only observed time tokens can receive the forecast-state representation."""
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(C,width,bias=False)
        nn.init.zeros_(self.linear.weight)

    def static(self, knots, known):
        active = knots.masked_fill(~known[:,None],0.)
        contents = self.linear(active)
        return F.pad(contents,(0,0,WIDTH//PATCH-K,0))


def frozen_hidden(base,x,mask,shift,transform,*,matched_dose=None):
    keep, hooks = [], []
    if transform is not None:
        hooks.append(base.embedding.register_forward_pre_hook(lambda _m,v:(transform(v[0]),)))
    def inject(_module,args):
        changed = layer.add_to_patches(args[0],shift)
        if matched_dose is not None:
            n,dose,record = matched_dose
            actual = (changed[2*n:,:-1]-changed[n:2*n,:-1].repeat(2,1,1)).reshape(2,n,-1)
            record.append(float((actual.norm(dim=-1)-dose).abs().max()))
        return (changed,*args[1:])
    hooks.append(base.encoder.layers[0].register_forward_pre_hook(inject))
    hooks.append(base.risk_head.register_forward_pre_hook(lambda _m,p:keep.append(p[0])))
    try:
        base(x,mask)
    finally:
        for h in hooks:
            h.remove()
    return keep[0]


def masking(mask):
    return mask.unflatten(1,(-1,PATCH)).any(dim=(2,3))


def match_actions(raw, budget):
    norms = raw.flatten(2).norm(dim=-1)
    dose = torch.minimum(budget,norms.amin(0)).masked_fill(~(norms>1e-8).all(0),0.)
    return raw*(dose/norms.clamp_min(1e-8))[...,None,None],dose


def available_states(q,t,eligible):
    release = q+OFFSET
    use = np.searchsorted(release,t,side='right')
    known = np.r_[False,eligible]
    older = np.maximum(use-1,0)
    blend = group.handoff_fraction(np.r_[0,release][use],t,PATCH)
    return use*known[use],older*known[older]*known[use],blend


def past_eligible(q, eligible):
    """Most recent *distinct* eligible prediction-target pair at issuance."""
    previous = np.maximum.accumulate(np.r_[-1,np.where(eligible,np.arange(len(q)),-1)])
    j = previous[np.searchsorted(q+OFFSET,q,side='right')]
    return j,eligible & (j>=0)


def future(base,signal,q,depth=H):
    index = q[:,None]+np.arange(1,depth+1)
    return tuple(torch.as_tensor(signal[k][index],device=base.center.device) for k in ('values','available'))


def learn_update(model,observed,target,mask):
    knots = model(observed)
    state = knots.new_zeros((len(knots),2,C),requires_grad=True)
    delta, = torch.autograd.grad(sample_loss(shared_state(knots,state),target,mask).sum(),state)
    return knots.detach(),(-delta).clamp(-1.,1.).detach()


def robust_scale(rows,base):
    values = []
    c = list(CHANNELS)
    center,scale = (v[c].cpu().numpy() for v in (base.center,base.scale))
    for s in rows:
        q = np.arange(WIDTH-1,int(s['valid'].sum())-H,lag.STRIDE)
        if len(q):
            raw,m = s['values'][:,c],s['available'][:,c]
            z = np.arcsinh((raw-center)/scale)
            a = m[q] & m[q+H]
            d = abs(z[q+H]-z[q])
            values.append(np.divide((d*a).sum(0),a.sum(0),out=np.full(C,np.nan),where=a.any(0)))
    width = np.nanmedian(values,axis=0)
    if not np.isfinite(width).all():
        raise ValueError('source TRAIN must cover each selected diagnostic')
    return torch.as_tensor(np.maximum(width,.025),dtype=torch.float32,device=base.center.device)


def source_batches(base,shots,scale):
    rows = []
    for s in shots:
        q = np.arange(WIDTH-1,int(s['valid'].sum())-H,lag.STRIDE)
        if not len(q):
            continue
        a,m = lag._windows(base,s,q)
        view = input_view(base,a,m,scale)
        target,mask = target_view(base,*future(base,s,q),view,scale)
        use = view.known.any(1) & mask.any(dim=(1,2))
        if use.any():
            rows.append(dict(features=view.features[use],anchor=view.anchor[use],known=view.known[use],
                target=target[use],mask=mask[use],shot=torch.full((int(use.sum()),),s['shot'],device=a.device)))
    return {name:torch.cat([r[name] for r in rows]) for name in rows[0]}


def train_forecast(data,device):
    torch.manual_seed(SEED)
    model = Forecaster().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    _,i,n = data['shot'].unique(return_inverse=True,return_counts=True)
    w = n[i].float().reciprocal()
    rows = []
    for step in range(FORECAST_STEPS):
        idx = torch.multinomial(w,TRAIN_BATCH,replacement=True)
        observed = Motion(*(data[k][idx] for k in ('features','anchor','known')))
        loss = sample_loss(model(observed),data['target'][idx],data['mask'][idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (step+1)%128==0:
            rows.append(dict(step=step+1, loss=float(loss.detach())))
    return model.eval().requires_grad_(False),rows


def risk_episodes(shots):
    """Only historical source endpoints; labels never enter the deployed inner loss."""
    pools = [[],[]]
    for i,s in enumerate(shots):
        # Multiple issue phases cover scarce source positives; EAST retains the
        # original fixed issue grid, not a grid chosen around DownTime.
        q = np.arange(WIDTH-1,len(s['values'])-READY,4)
        if s['is_disrupt']:
            t = q+READY
            ok = s['y'][t].astype(bool) & (np.round((s['down']-s['times'][t])*1e3,9)>10.)
            if ok.any():
                pools[1].append((i,q[ok]))
        elif len(q):
            pools[0].append((i,q))
    return pools


def source_episode(base,shots,pool,rng,scale):
    packets = []
    for label in (0,1):
        for i in rng.integers(len(pool[label]),size=TRAIN_BATCH//2):
            sid,ends = pool[label][i]
            s,q = shots[sid],int(rng.choice(ends))
            a,m = simple_data.window(s,q,WIDTH)
            b,n = simple_data.window(s,q+READY,WIDTH)
            packets.append((a,m,b,n,s['values'][q+1:q+H+1],s['available'][q+1:q+H+1],label))
    x,m,s,n,y,a,d = (torch.as_tensor(np.stack(v),device=base.center.device) for v in zip(*packets,strict=True))
    issued = input_view(base,x,m,scale)
    target,mask = target_view(base,y,a,issued,scale)
    return x,m,s,n,target,mask,d.float()


def train_carrier(base,model,shots,scale,threshold):
    head = Carrier(base.cfg.dim).to(base.center.device)
    opt = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    pool = risk_episodes(shots)
    rng = np.random.default_rng(SEED)
    mark,runs = float(logit(threshold)),[]
    for step in range(RISK_STEPS):
        x,m,s,n,target,mask,label = source_episode(base,shots,pool,rng,scale)
        _,delta = learn_update(model,input_view(base,x,m,scale),target,mask)
        view = input_view(base,s,n,scale)
        with torch.no_grad():
            initial = model(view)
            parent = base(s,n)['logit']
        before = head.static(initial,view.known)
        after = head.static(shared_state(initial,delta),view.known)
        change = torch.cat((before,after))*masking(n).repeat(2,1)[...,None]
        result = base.risk_head(frozen_hidden(base,s.repeat(2,1,1),n.repeat(2,1,1),change,None)).flatten()
        b,a = result.chunk(2)
        supervised = F.binary_cross_entropy_with_logits(result-mark,label.repeat(2))
        loss = supervised + .25*(b-parent).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (step+1)%128==0:
            runs.append(dict(step=step+1,loss=float(loss.detach()),
                before=float(F.binary_cross_entropy_with_logits(b.detach()-mark,label)),
                after=float(F.binary_cross_entropy_with_logits(a.detach()-mark,label))))
    return head.eval().requires_grad_(False),runs,[len(v) for v in pool]


@torch.no_grad()
def source_check(base,model,head,source_val,scale,threshold):
    samples,risks = [],{k:[] for k in ('parent','source_representation')}
    for s in source_val:
        q = np.arange(WIDTH-1,int(s['valid'].sum())-H,lag.STRIDE)
        if len(q):
            a,m = lag._windows(base,s,q)
            obs = input_view(base,a,m,scale)
            y,valid = target_view(base,*future(base,s,q),obs,scale)
            good = valid.all(dim=(1,2)).cpu().numpy()
            t = q[good]+READY
            critical = t<len(s['times'])
            j = t[critical]
            critical[critical] = (s['y'][j]>0.) & (np.round((s['down']-s['times'][j])*1e3,9)>10.) if (
                s['is_disrupt']) else False
            samples.append(dict(shot=s['shot'], issue=q[good],
                error=absolute_errors(waveform(model(obs)),y,valid).cpu().numpy()[good],
                critical=critical))
        ends = np.flatnonzero(s['valid'])
        curves = []
        for j in range(0,len(ends),BATCH):
            x,m = lag._windows(base,s,ends[j:j+BATCH])
            observed = input_view(base,x,m,scale)
            a = head.static(model(observed),observed.known)*masking(m)[...,None]
            h = frozen_hidden(base,x.repeat(2,1,1),m.repeat(2,1,1),torch.cat((torch.zeros_like(a),a)),None)
            curves.append(base.risk_head(h).flatten().sigmoid().reshape(2,len(x)).cpu().numpy())
        for key,curve in zip(risks,np.concatenate(curves,axis=-1),strict=True):
            risks[key].append(dict(shot=s['shot'],scores=curve,is_disrupt=s['is_disrupt'],
                                   down=s['down'],times=s['times'][ends]))
    groups = {k:simple_train.metrics(v,threshold) for k,v in risks.items()}
    err = {name:forecast_summary([r['error'][r['critical']] if name=='critical' else r['error'] for r in samples])
           for name in ('all','critical')}
    return dict(alarm=groups,forecast=err)


def forecast_summary(rows):
    usable = [r for r in rows if len(r)]
    if not usable:
        return dict(shots=0,windows=0,mae=None)
    return dict(shots=len(usable),windows=sum(map(len,usable)),mae=np.mean([v.mean(0) for v in usable],axis=0).tolist())


def fit(out):
    lag.configure()
    started = time.monotonic()
    base,checkpoint = load_parent(DEFAULT_PARENT)
    base.cuda()
    roles = simple_data.source_roles()
    a = io23.advanced_times()
    shots = simple_train.load_shots(roles['train'],a)
    scale = robust_scale(shots,base)
    out.mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'training_protocol.json',dict(
        scope='J-TEXT TRAIN only: learn a persistence-anchored trajectory representation',
        source_sha256=PARENT_SHA256, train_ids=roles['train'], validation_ids=roles['val'],
        channels=[io23.RAW23[k] for k in CHANNELS], seed=SEED, source_threshold=checkpoint['threshold'],
        forecast_steps=FORECAST_STEPS,risk_steps=RISK_STEPS,batch=TRAIN_BATCH,
        quality='frozen source backbone; one direct 25-point waveform, no source online evaluation',
        loss='first fit five knots by full waveform and adjacent-bin differences; then learn'
             ' 8-to-64 trajectory carrier in final five time tokens against pre/post-update timely'
             ' source risk with frozen original risk head and before-risk anchor'))
    state = state_digest(base)
    training = source_batches(base,shots,scale)
    forecaster,h = train_forecast(training,base.center.device)
    print(dict(stage='source_forecast_fit', examples=len(training['shot']),
               shots=int(training['shot'].unique().numel()), seconds=time.monotonic()-started),flush=True)
    carrier,g, support = train_carrier(base,forecaster,shots,scale,float(checkpoint['threshold']))
    with torch.no_grad():
        measured = source_check(base,forecaster,carrier,simple_train.load_shots(roles['val'],a),
                                scale,float(checkpoint['threshold']))
    if state!=state_digest(base):
        raise ValueError('original source model changed')
    with (out/'shared.pt').open('xb') as file:
        torch.save(dict(forecaster=forecaster.state_dict(),carrier=carrier.state_dict(),scale=scale,
                        source_sha256=PARENT_SHA256),file)
    write(out/'source_result.json',dict(history=dict(forecast=h,risk=g),episode_shots=support,
        validation=measured, train_examples=len(training['shot']), seconds=time.monotonic()-started))
    print(dict(stage='source_fit_sealed', support=support, forecast=measured['forecast'],
        alarms={k:(a['tp'],a['fp']) for k,a in measured['alarm'].items()}, seconds=time.monotonic()-started),flush=True)


def inputs_from_fit(out,base):
    fit = torch.load(out/'shared.pt',map_location=base.center.device,weights_only=True)
    if fit['source_sha256']!=PARENT_SHA256:
        raise ValueError('source carrier belongs to a different frozen parent')
    forecaster,carrier = Forecaster().to(base.center.device),Carrier(base.cfg.dim).to(base.center.device)
    forecaster.load_state_dict(fit['forecaster'])
    carrier.load_state_dict(fit['carrier'])
    return forecaster.eval().requires_grad_(False),carrier.eval().requires_grad_(False),fit['scale']


def shot_targets(signal,base,model,scale,q,support):
    target = base.center.new_zeros((len(q), H, C))
    pred = torch.zeros_like(target)
    mask = torch.zeros_like(target,dtype=torch.bool)
    directions = torch.zeros((len(q)+1,2,C),device=base.center.device)
    for j in range(0,len(q),BATCH):
        i = slice(j,j+BATCH)
        x,m = lag._windows(base,signal,q[i])
        view = input_view(base,x,m,scale)
        target[i],mask[i] = target_view(base,*future(base,signal,q[i]),view,scale)
        knots,delta = learn_update(model,view,target[i],mask[i])
        pred[i] = waveform(knots)
        directions[j+1:j+1+len(delta)] = delta
    eligible = support & mask.all(dim=(1,2)).cpu().numpy()
    predecessor, eligible_for_match = past_eligible(q,eligible)
    recent = directions * torch.as_tensor(np.r_[False,eligible_for_match],device=base.center.device)[:,None,None]
    older = torch.cat((directions[:1],directions[predecessor+1])) * torch.as_tensor(
        np.r_[False,eligible_for_match],device=base.center.device)[:,None,None]
    err = absolute_errors(pred,target,mask)
    return torch.stack((recent,older)),eligible,eligible_for_match,err.cpu().numpy()


@torch.no_grad()
def render(signal,base,model,carrier,scale,track,feedback,matched,cap):
    q,h,z,beta,phase,correction = track
    t = np.arange(len(signal['values']))
    pures = np.zeros((len(CTRL)+1,len(t)),np.float32)
    doses = np.zeros(len(t),np.float32)
    norm_error = 0.
    for j in range(0,len(t),BATCH):
        t0 = t[j:j+BATCH]
        x,m = lag._windows(base,signal,t0)
        observed = input_view(base,x,m,scale)
        first, prev, portion = available_states(q,t0,matched)
        portion = z.new_tensor(portion)[None,:,None,None]
        state = feedback[:,prev].lerp(feedback[:,first],portion)
        knots = model(observed)
        static = carrier.static(knots,observed.known)
        diff = torch.stack([carrier.static(shared_state(knots,v)-knots,observed.known) for v in state])
        mask = masking(m)[...,None]
        actual,expected = match_actions(diff*mask[None],z.new_tensor(cap[t0]))
        static *= mask
        inserted = torch.cat((torch.zeros_like(static)[None],static[None],static[None]+actual),dim=0)
        slots = lag.delayed_slots(q,t0)[0]
        transform = layer.patcher(phase[slots].repeat(4,1,1),correction[slots].repeat(4,1))
        actual_check = []
        out = frozen_hidden(base,x.repeat(4,1,1),m.repeat(4,1,1),inserted.flatten(0,1),transform,
                            matched_dose=(len(t0),expected,actual_check))
        hidden = out.reshape(4,len(t0),-1)
        log = z[t0]+((hidden-h[t0])*beta).sum(-1)
        logits = log.sigmoid()
        pures[:,t0] = logits.cpu().numpy()
        doses[t0] = expected.cpu().numpy()
        norm_error = max(norm_error,actual_check[0])
    pures[2:] = np.where(doses[None]>0.,pures[2:],pures[1])
    return pures,doses,norm_error


def replay(out):
    begun = time.monotonic()
    lock,p,seal,args,decoder,path = history.inputs()
    base = args[0]
    model,carrier,scale = inputs_from_fit(out,base)
    weights = [state_digest(m) for m in (*args[:2],model,carrier)]
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(scope='EAST360_PERSISTENCE_ANCHORED_SHARED_REPRESENTATION_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK},arms=ARMS,archive=str(history.ARCHIVE),
        source_fit=sha(out/'shared.pt'),archive_seal=sha(history.ARCHIVE/'seal.json'),
        feedback='direct H25 from last valid measurement, score-time slope and bend in forecast coordinates',
        earlier='own earlier correctly paired H25 loss and issue gradient; no mismatching targets',
        risk='one fixed readout over current shared encoded tokens; no added risk score or alarm adjustment',
        dose='equal recent/earlier reachable norm, not exceeding inherited equal-dose schedule',
        timing=dict(issue_stride=lag.STRIDE,join_offset=OFFSET,fully_effective_offset=READY),
        later_labels='unavailable until complete seal; no inter-shot training; cold20 unchanged'))
    files, prefixes, maxerr = {}, {}, np.zeros(4)
    tau = tuple(p['calibration'].values())
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('target outcome before seal')):
        for rank,sid in enumerate(p['order'],1):
            arc = history.archived_shot(rank,sid,seal)
            n = len(arc['bias_only'])
            curves = np.broadcast_to(arc['bias_only'],(len(CTRL),n))
            err = np.zeros((len(arc['release']),4),np.float32)
            valid,matched = (np.zeros(len(err),bool) for _ in range(2))
            doses, local = np.zeros(n,np.float32), np.zeros((0,3),np.float32)
            if rank>p['startup']:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('fixed cohort input changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                track = group.parent_track(signal,args,decoder,path)
                q,h,z,beta,_,phase,correction = track
                np.testing.assert_array_equal(q+H+1,arc['release'])
                all_feedback,valid,matched,err = shot_targets(signal,base,model,scale,q,arc['eligible'])
                got,doses,norm = render(signal,base,model,carrier,scale,
                    (q,h,z,beta,phase,correction),all_feedback,matched,arc['dose'])
                probe.check_thresholds(got[0],arc['bias_only'],tau)
                curves = got[1:]
                end = q+READY
                keep = (end<n)&matched
                selected = np.flatnonzero(keep)
                local = np.c_[end[selected],np.log(np.clip(curves[:,end[selected]],1e-6,1.-1e-6)/
                    np.clip(1.-curves[:,end[selected]],1e-6,1.-1e-6)).T].astype(np.float32)
                maxerr[:2] = np.maximum(maxerr[:2], [abs(got[0]-arc['bias_only']).max(),norm])
                if rank in (21,120,360) or sid in CASES:
                    ncut = min(n,768)
                    short = {k:v[:ncut] if k in ('values','available') else v for k,v in signal.items()}
                    q0,h0,z0,b0,_,ph0,co0 = group.parent_track(short,args,decoder,path)
                    eligible = arc['eligible'][arc['release']<ncut]
                    f0,_,m0,_ = shot_targets(short,base,model,scale,q0,eligible)
                    test,d0,_ = render(short,base,model,carrier,scale,(q0,h0,z0,b0,ph0,co0),f0,m0,arc['dose'][:ncut])
                    probe.check_thresholds(test,got[:,:ncut],tau)
                    np.testing.assert_allclose(d0,doses[:ncut],atol=1e-5,rtol=0.)
                    maxerr[2:] = np.maximum(maxerr[2:],[abs(test-got[:,:ncut]).max(),abs(d0-doses[:ncut]).max()])
                    prefixes[sid] = ncut
            key = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f,**dict(zip(CTRL,curves,strict=True)),dose=doses,
                    eligible=valid,matched=matched,forecast=err,compare_at_release=local)
            files[key] = sha(out/'shots'/key)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='sealed',rank=rank,seconds=time.monotonic()-begun),flush=True)
    if weights!=[state_digest(m) for m in (*args[:2],model,carrier)]:
        raise ValueError('persistent weights must be identical before and after replay')
    if maxerr[1]>1e-5:
        raise ValueError('unequal achieved update dose')
    write(out/'seal.json',dict(curves=files,max_errors=maxerr.tolist(),
        prefixes=prefixes,source_fit=sha(out/'shared.pt'),seconds=time.monotonic()-begun))


def verify(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    old,lock = (read(history.ARCHIVE/f'{k}.json') for k in ('per_shot','seal'))
    if (len(s['curves'])!=360 or p['archive_seal']!=sha(history.ARCHIVE/'seal.json')
            or s['source_fit']!=p['source_fit'] or sha(out/'shared.pt')!=p['source_fit']):
        raise ValueError('complete immutable run and parent required')
    rows = {band:{k:[] for k in ARMS} for band in p['calibration']}
    forecast, attribution, all_support = [],[],[]
    for rank,sid in enumerate(p['order'],1):
        key = f'{rank:04d}_{sid}.npz'
        if sha(out/'shots'/key)!=s['curves'][key]:
            raise ValueError('changed curve')
        arc = history.archived_shot(rank,sid,lock)
        disrupted,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/key,allow_pickle=False) as f:
            n = len(t)
            active = f['dose']>0.
            if np.any(f['dose']-arc['dose']>1e-5):
                raise ValueError('over budget')
            observed = f['eligible']
            matched = f['matched']
            if (matched & ~observed).any() or (observed & ~arc['eligible']).any():
                raise ValueError('timing support differs')
            clock, _, _ = available_states(arc['release']-(H+1),np.arange(n),matched)
            if (active & (clock==0)).any():
                raise ValueError('feedback was used before it arrived')
            if rank<=p['startup'] and (active.any() or not np.all(
                np.stack([f[k] for k in CTRL])==arc['bias_only'])):
                raise ValueError('cold preparation predictions must remain identical')
            for k in CTRL[1:]:
                np.testing.assert_array_equal(f[k][~active],f[CTRL[0]][~active])
            main = p['calibration']['0.2']
            for band,tau in p['calibration'].items():
                tau = tau if rank>p['startup'] else p['original']
                for k in ARMS:
                    v=dict(shot=sid,disruptive=disrupted,down_s=down,**score_curve(
                        t,(arc if k in history.OLD else f)[k],disrupted,down,tau))
                    if k in history.OLD and v!=old[band][k][rank-1]:
                        raise ValueError('archived warning does not reproduce')
                    rows[band][k].append(v)
            if rank>p['startup']:
                end = arc['release']+history.WAIT+PATCH
                ll = np.full(len(end),np.nan)
                if disrupted:
                    within = end<n
                    ll[within] = np.round((down-t[end[within]])*1e3,9)
                late = disrupted and not old['0.2']['current_5'][rank-1]['timely']
                crit = late & (ll>30.) & (ll<=125.) & observed
                forecast.append(dict(all=f['forecast'][observed],critical=f['forecast'][crit]))
                first = np.flatnonzero(f[CTRL[0]]>=main)
                previous = np.flatnonzero(arc['current_5']>=main)
                safe = not disrupted and len(previous)>0
                release = f['compare_at_release']
                all_support.append((int(observed.sum()),int(matched.sum()),int(active.sum())))
                if late or safe:
                    u = release[:,0].astype(int)
                    good = ((np.round((down-t[u])*1e3,9)>30.) &
                            (np.round((down-t[u])*1e3,9)<=125.)) if late else u<=previous[0]
                    u,score = u[good],release[good,1:]
                    u0 = np.searchsorted(end,u)
                    if not np.array_equal(end[u0],u):
                        raise ValueError('effect point does not match its H25 ground truth packet')
                    errors = f['forecast'][u0]
                    d = (score[:,1:]-score[:,:1])*(1. if late else -1.)
                    margin = logit(main)-score[:,0]
                    attribution.append(dict(shot=sid,kind='late' if late else 'false',
                        checks=len(u),first_no_update=int(first[0]) if len(first) else None,
                        feedback_skill=(errors[:,[1,3]]-errors[:,[0,2]]).tolist(),signed_delta=d.tolist(),
                        gap=margin.tolist(),endpoints=u.tolist(),nonzero_dose=active[u].tolist()))
    metrics = {band:{k:lead.summarize(v) for k,v in r.items()} for band,r in rows.items()}
    pair = {b:{f'{a}__to__{k}':dict(flips=flips(r[a],r[k]),lead=lead.paired(r[a],r[k]))
               for a in ('bias_only','current_5',CTRL[0],CTRL[2])
               for k in CTRL if a!=k} for b,r in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,paired=pair,support=np.asarray(all_support).sum(0).tolist(),
        forecast={k:forecast_summary([r[k] for r in forecast]) for k in ('all','critical')},
        raw_verified=360*len(ARMS)*len(p['calibration']),attribution=attribution))
    for b,r in metrics.items():
        print(b,{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in r.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('fit','replay','score'))
    cli.add_argument('output',type=Path)
    opts = cli.parse_args()
    {'fit':fit,'replay':replay,'score':verify}[opts.stage](opts.output.resolve())
