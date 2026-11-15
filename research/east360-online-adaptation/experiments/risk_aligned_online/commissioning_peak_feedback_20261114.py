"""Teach the direction of a shot-local write using point or warning-peak loss."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_motion_clock_dose_20261114 as dose
from . import commissioning_shared_motion_20261114 as motion
from . import io23, simple_data, simple_train, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


history, lag, layer = motion.history, motion.lag, motion.layer
BASES = ('fixed', 'bias_only', 'current_5', 'earlier_5')
STATIC = 'source_representation'
TRIALS = ('point_current', 'point_earlier', 'peak_current', 'peak_earlier')
VIEWS = (STATIC, *TRIALS)
CASES = (70320, 73405, 69813, 74970, 85422)
SOURCE_STRIDE, FIT_STEPS = 16, 256
# Source-only fit uses the already reported median legacy token norm. Replay
# uses each endpoint's individual archived budget, not this median.
SOURCE_RADIUS, TEMPERATURE = .89072, .25


def bag_ends(shot):
    if not shot['is_disrupt']:
        return np.flatnonzero(shot['valid'])
    ahead = np.round((shot['down']-shot['times'])*1e3, 9)
    return np.flatnonzero(shot['valid'] & (shot['y']>0.) & (ahead>10.))


def aggregate(z, *, peak):
    return z.amax() if peak else z


def penalty(z, threshold, disruptive):
    sign = -1. if disruptive else 1.
    return F.softplus(sign*(z-threshold)/TEMPERATURE)


def remap(g, matrix):
    flat = g.flatten(-2)
    return (flat+flat@matrix).reshape_as(g)


def token_actions(carrier, view, keep, directions, cap):
    # The displacement basis is linear; do not subtract two nearly equal
    # forecast paths when constructing a difference.
    start = directions.new_zeros((directions.shape[1], motion.K, motion.C))
    offsets = torch.stack([carrier.static(motion.shared_state(start,g),view.known)
                           for g in directions])
    return dose.fixed_radius(offsets*keep[None],cap)


@dataclass(frozen=True)
class Episode:
    shot: dict
    state: torch.Tensor
    ends: np.ndarray


def source_history(base, model, scale, s):
    """Each training episode follows a causal clock on a completed source shot."""
    n = int(s['valid'].sum())
    q = np.arange(motion.WIDTH-1, n-motion.H, SOURCE_STRIDE, dtype=np.int64)
    ready = np.zeros(len(q),bool)
    grads = base.center.new_zeros((len(q)+1,2,motion.C))
    for i in range(0,len(q),motion.BATCH):
        t = q[i:i+motion.BATCH]
        x,mask = lag._windows(base,s,t)
        v = motion.input_view(base,x,mask,scale)
        y,a = motion.target_view(base,*motion.future(base,s,t),v,scale)
        ready[i:i+len(t)] = a.all(dim=(1,2)).cpu().numpy()
        _, grads[i+1:i+len(t)+1] = motion.learn_update(model,v,y,a)
    first,last,f = motion.available_states(q,np.arange(n),ready)
    return grads[last].lerp(grads[first],grads.new_tensor(f)[:,None,None]).cpu()


class SourceScorer:
    """Same model, endpoint tensors, cap and optimizer exposure for both losses."""

    def __init__(self, base, model, carrier, scale):
        self.base, self.model, self.carrier, self.scale = base, model, carrier, scale

    @torch.no_grad()
    def batches(self, ep, ends):
        for j in range(0,len(ends),motion.BATCH):
            t = ends[j:j+motion.BATCH]
            x,m = lag._windows(self.base,ep.shot,t)
            view = motion.input_view(self.base,x,m,self.scale)
            keep = motion.masking(m)[...,None]
            static = self.carrier.static(self.model(view),view.known)*keep
            g = ep.state[t].to(x.device)
            cap = x.new_full((len(t),),SOURCE_RADIUS)
            yield x,m,view,keep,static,g,cap

    def curve(self, batch, matrices):
        x,m,view,keep,static,g,cap = batch
        mapped = torch.stack([remap(g,a) for a in matrices])
        delta, _ = token_actions(self.carrier,view,keep,mapped,cap)
        changes = torch.cat((static[None],static[None]+delta),dim=0)
        k = len(changes)
        h = motion.frozen_hidden(self.base,x.repeat(k,1,1),m.repeat(k,1,1),changes.flatten(0,1),None)
        return self.base.risk_head(h).reshape(k,len(x))

    def learn(self, ep, matrix, mark, *, peak):
        batches = list(self.batches(ep,ep.ends))
        disruptive = ep.shot['is_disrupt']
        if peak:
            with torch.no_grad():
                tops = torch.stack([aggregate(self.curve(b,[matrix])[1],peak=True) for b in batches])
            # Re-evaluate just the winning chunk with gradients. The maximum
            # was over the entire bag, including unmodified endpoints.
            z = aggregate(self.curve(batches[int(tops.argmax())],[matrix])[1],peak=True)
            v = .5*penalty(z,mark,disruptive)
            v.backward()
            return float(v.detach())
        losses = []
        for batch in batches:
            z = self.curve(batch,[matrix])[1]
            v = .5*penalty(aggregate(z,peak=False),mark,disruptive).sum()/len(ep.ends)
            v.backward()
            losses.append(float(v.detach()))
        return sum(losses)

    @torch.no_grad()
    def validation(self, episodes, matrices, threshold):
        arms = (STATIC, 'identity', 'point', 'peak')
        saved = {k:[] for k in arms}
        loss = {k:[[],[]] for k in arms}
        for ep in episodes:
            s = ep.shot
            ends = np.flatnonzero(s['valid'])
            z = torch.cat([self.curve(b,matrices) for b in self.batches(ep,ends)],dim=-1)
            is_disrupt = s['is_disrupt']
            for k,p in zip(arms,z,strict=True):
                saved[k].append(dict(shot=s['shot'],scores=p.sigmoid().cpu().numpy(),
                                     is_disrupt=is_disrupt,down=s['down'],times=s['times'][ends]))
                if len(ep.ends):
                    p = p[np.searchsorted(ends,ep.ends)]
                    v = penalty(aggregate(p,peak=True),float(logit(threshold)),is_disrupt)
                    loss[k][int(is_disrupt)].append(float(v))
        return {k:dict(metrics=simple_train.metrics(saved[k],threshold),
                       peak_loss_by_class=[float(np.mean(a)) for a in loss[k]]) for k in arms}


def fit(source,out):
    lag.configure()
    base, checkpoint = load_parent(DEFAULT_PARENT)
    base.cuda()
    model,carrier,scale = motion.inputs_from_fit(source,base)
    roles,a = simple_data.source_roles(),io23.advanced_times()
    episodes = {}
    for role in ('train','val'):
        episodes[role] = [
            Episode(s,source_history(base,model,scale,s),bag_ends(s))
            for s in simple_train.load_shots(roles[role],a)
        ]
    data = episodes['train']
    pools = [[i for i,e in enumerate(data) if int(e.shot['is_disrupt'])==c and len(e.ends)
              and bool(e.state[e.ends].any())] for c in (0,1)]
    if not all(pools):
        raise ValueError('both source classes need a timely writable bag')
    rng = np.random.default_rng(motion.SEED)
    draws = np.column_stack([rng.choice(a,FIT_STEPS,replace=True) for a in pools])
    shape = (2*motion.C,2*motion.C)
    identity = base.center.new_zeros(shape)
    matrices = [torch.nn.Parameter(identity.clone()) for _ in range(2)]
    opts = [torch.optim.Adam([m],lr=.005) for m in matrices]
    scorer = SourceScorer(base,model,carrier,scale)
    tau = float(checkpoint['threshold'])
    trace = []
    for step,ids in enumerate(draws,1):
        losses = []
        for peak,m,opt in zip((False,True),matrices,opts,strict=True):
            opt.zero_grad(set_to_none=True)
            losses.append(sum(scorer.learn(data[i],m,float(logit(tau)),peak=peak) for i in ids))
            (.005*m.square().mean()).backward()
            torch.nn.utils.clip_grad_norm_([m],1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                m.div_((m.norm()/4.).clamp_min(1.))
        if step%32==0:
            trace.append(dict(step=step,loss=losses,norms=[float(m.detach().norm()) for m in matrices]))
            print(dict(stage='source_fit',**trace[-1]),flush=True)
    val = scorer.validation(episodes['val'],[identity,*matrices],tau)
    with (out/'direction.pt').open('xb') as f:
        torch.save(torch.stack(matrices).detach().cpu(),f)
    write(out/'source_result.json',dict(threshold=tau,loops=trace,
        shots={k:roles[k] for k in ('train','val')},fit_draw_shots=[[data[i].shot['shot'] for i in r] for r in draws],
        active_bags=dict(train=list(map(len,pools)), validation=[
            sum(int(e.shot['is_disrupt'])==k and bool(e.state[e.ends].any()) for e in episodes['val'])
            for k in (0,1)]), validation=val))
    print(dict(stage='source_fit_sealed',val={k:(v['metrics']['tp'],v['metrics']['fp']) for k,v in val.items()}),
          flush=True)


@torch.no_grad()
def render(signal,base,model,carrier,scale,track,matched,states,cap,matrices):
    q,h,z,beta,phase,correction = track
    n = len(z)
    curve = np.empty((len(VIEWS),n),np.float32)
    value = np.zeros(n,np.float32)
    error = 0.
    for j in range(0,n,motion.BATCH):
        t = np.arange(j,min(j+motion.BATCH,n))
        x,m = lag._windows(base,signal,t)
        view = motion.input_view(base,x,m,scale)
        keep = motion.masking(m)[...,None]
        root = carrier.static(model(view),view.known)*keep
        a,b,f = motion.available_states(q,t,matched)
        g = states[:,b].lerp(states[:,a],z.new_tensor(f)[None,:,None,None])
        variants = torch.cat(tuple(remap(g,w) for w in matrices),dim=0)
        delta,want = token_actions(carrier,view,keep,variants,z.new_tensor(cap[t]))
        edits = torch.cat((root[None],root[None]+delta),dim=0)
        slot = lag.delayed_slots(q,t)[0]
        k = len(VIEWS)
        patcher = layer.patcher(phase[slot].repeat(k,1,1),correction[slot].repeat(k,1))
        observed = []

        def actual_write(tokens):
            tk = tokens[:,:-1].reshape(k,len(t),-1)
            observed.append(float(((tk[1:]-tk[:1]).norm(dim=-1)-want).abs().max()))

        fwd = motion.frozen_hidden(base,x.repeat(k,1,1),m.repeat(k,1,1),edits.flatten(0,1),
                                   patcher,observe_tokens=actual_write)
        p = (z[t]+((fwd.reshape(k,len(t),-1)-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        value[t] = want.cpu().numpy()
        curve[:,t] = np.where(value[t]>0,p,p[:1])
        error = max(error,observed[0])
    return curve,value,error


def predict(signal,args,decoder,path,model,carrier,scale,arc,matrices):
    base = args[0]
    q,h,z,beta,_,phase,correction = motion.group.parent_track(signal,args,decoder,path)
    np.testing.assert_array_equal(q+motion.H+1,arc['release'])
    states,observed,matched,_ = motion.shot_targets(signal,base,model,scale,q,arc['eligible'])
    tracks = q,h,z,beta,phase,correction
    return (*render(signal,base,model,carrier,scale,tracks,matched,states,arc['dose'],matrices),
            observed,matched)


def run(source,reference,out):
    begun = time.monotonic()
    lock,p,seal,args,decoder,path = history.inputs()
    base = args[0]
    s,sp = (read(source/f'{k}.json') for k in ('seal','protocol'))
    rp,rs = (read(reference/f'{k}.json') for k in ('protocol','seal'))
    if (any(p[k]!=rp[k] or p[k]!=sp[k] for k in history.LOCK)
            or len(s['curves'])!=360 or len(rs['curves'])!=360 or sha(source/'shared.pt')!=sp['source_fit']
            or rp['source_seal']!=sha(source/'seal.json')):
        raise ValueError('same sealed 360-shot reference and source carrier required')
    model,carrier,scale = motion.inputs_from_fit(source,base)
    matrices = torch.load(out/'direction.pt',map_location=base.center.device,weights_only=True)
    (out/'shots').mkdir()
    expected = (*VIEWS,*dose.DOSE[:2])
    write(out/'protocol.json',dict(scope='EAST360_SOURCE_TAUGHT_FIRST_ALARM_FEEDBACK_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK}, arms=(*BASES,*expected),
        source=str(source), reference=str(reference), source_seal=rp['source_seal'],
        direction=get_direction_spec(),
        fit='same TRAIN shots drawn in the same order; all valid safe endpoints, only labeled and >10ms positive '
            'endpoints; peak versus pointwise mean; fixed source threshold; source VAL diagnostics only',
        dose='same archived token norm and common signal-only availability across four new arms',
        outcome='no EAST outcome in prediction; cold20 unchanged; no inter-shot learning'))
    files,witness = {},{}
    max_error,max_static = 0.,0.
    tau = tuple(p['calibration'].values())
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcome during replay')),
          patch.object(motion.group.hdf,'_raw_truth',side_effect=RuntimeError('no outcome during replay'))):
        for rank,sid in enumerate(p['order'],1):
            key = f'{rank:04d}_{sid}.npz'
            arc = history.archived_shot(rank,sid,seal)
            n = len(arc['dose'])
            y = np.broadcast_to(arc['bias_only'],(len(VIEWS),n))
            radius = np.zeros(n,np.float32)
            observed = matched = np.zeros(len(arc['release']),bool)
            if rank>p['startup']:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('input file changed since the paired reference')
                signal = lag.prepare_actual(simple_data.read_shot('EAST',sid,rate_contract='nominal'),
                                            rank,lock['mapping'])
                y,radius,e,observed,matched = predict(
                    signal,args,decoder,path,model,carrier,scale,arc,matrices)
                max_error = max(max_error,e)
                earlier_file = reference/'shots'/key
                if sha(earlier_file)!=rs['curves'][key]:
                    raise ValueError('legacy-norm reference changed')
                with np.load(earlier_file,allow_pickle=False) as f:
                    motion.probe.check_thresholds(y[0],f[STATIC],tau)
                    np.testing.assert_array_equal(observed,f['observed'])
                    np.testing.assert_array_equal(matched,f['legal'])
                    max_static = max(max_static,float(abs(y[0]-f[STATIC]).max()))
                if rank in (21,120,360) or sid in CASES:
                    hits = np.flatnonzero((radius>0.) & (y[3]>=p['calibration']['0.2']))
                    cut = min(n,max(768,int(hits[0])+1) if len(hits) else 768)
                    short = {k:v[:cut] if k in ('values','available') else v for k,v in signal.items()}
                    i = arc['release']<cut
                    sub = dict(dose=arc['dose'][:cut],release=arc['release'][i],eligible=arc['eligible'][i])
                    z,d,*_ = predict(short,args,decoder,path,model,carrier,scale,sub,matrices)
                    motion.probe.check_thresholds(z,y[:,:cut],tau)
                    np.testing.assert_allclose(d,radius[:cut],atol=1e-6,rtol=0.)
                    witness[str(sid)] = dict(points=cut,max_difference=float(abs(z-y[:,:cut]).max()))
            with (out/'shots'/key).open('xb') as f:
                np.savez_compressed(f,**dict(zip(VIEWS,y,strict=True)),radius=radius,observed=observed,matched=matched)
            files[key] = sha(out/'shots'/key)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='EAST_predictions',rank=rank,seconds=time.monotonic()-begun),flush=True)
    if max_error>1e-5:
        raise ValueError('equal token write check failed')
    write(out/'seal.json',dict(curves=files,prefix=witness,max_token_norm_error=max_error,
                             max_static_error=max_static,seconds=time.monotonic()-begun))


def get_direction_spec():
    return dict(fit_steps=FIT_STEPS,source_stride=SOURCE_STRIDE,source_token_radius=SOURCE_RADIUS,
                logistic_temperature=TEMPERATURE,offset=motion.OFFSET,handoff=motion.PATCH,
                east_stride=lag.STRIDE, update='g(I+M), fixed Frobenius norm <=4; no bias or score residual')


def confirmed(times,z,down,tau,n):
    valid = times<=down+1e-9 if down is not None else np.ones(len(times),bool)
    if len(z)<n:
        return False,None
    hits = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(valid & (z>=tau),n).all(1))
    if not len(hits):
        return False,None
    return True,round(float((down-times[hits[0]+n-1])*1e3),9) if down is not None else None


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    old,os = (read(history.ARCHIVE/f'{k}.json') for k in ('per_shot','seal'))
    ref = Path(p['reference'])
    previous,rs = (read(ref/f'{k}.json') for k in ('per_shot','seal'))
    arms = p['arms']
    rows = {b:{k:[] for k in arms} for b in p['calibration']}
    confirmations = {b:{n:{k:[0,0] for k in arms} for n in (2,3,5)} for b in p['calibration']}
    critical, radii = [],[]
    support = 0
    if len(s['curves'])!=len(p['order']) or len(p['order'])!=360:
        raise ValueError('all predictions must be sealed before revealing an EAST outcome')
    for rank,sid in enumerate(p['order'],1):
        key = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/key
        if sha(path)!=s['curves'][key] or sha(ref/'shots'/key)!=rs['curves'][key]:
            raise ValueError('sealed curve changed')
        d,down,t = motion.group.hdf._raw_truth(sid)
        a = history.archived_shot(rank,sid,os)
        with np.load(path,allow_pickle=False) as w, np.load(ref/'shots'/key,allow_pickle=False) as r:
            curves = {k:(w if k in VIEWS else a if k in BASES else r)[k] for k in arms}
            radius = w['radius']
            active = radius>0.
            support += int(active.any())
            if active.any():
                radii.append(radius[active])
            slots = motion.available_states(a['release']-motion.H-1,np.arange(len(t)),w['matched'])[0]
            if ((w['observed'] & ~a['eligible']).any() or (w['matched'] & ~w['observed']).any()
                    or (active & ((slots==0) | (abs(radius-a['dose'])>1e-6))).any()
                    or (rank<=p['startup'] and active.any())):
                raise ValueError('write exceeds clock, mask, dose or preparation contract')
            for k in TRIALS:
                np.testing.assert_array_equal(curves[k][~active],curves[STATIC][~active])
            for b,theta in p['calibration'].items():
                theta = p['original'] if rank<=p['startup'] else theta
                for k,z in curves.items():
                    record = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,z,d,down,theta))
                    expected = old[b] if k in BASES else previous[b]
                    if k in expected and record!=expected[k][rank-1]:
                        raise ValueError('locked HDF5 re-score differs from archived baseline')
                    rows[b][k].append(record)
                    for n,group in confirmations[b].items():
                        alarm,m = confirmed(t,z,down,theta,n)
                        counters = group[k]
                        counters[0] += bool(d and alarm and m>30.)
                        counters[1] += bool(not d and alarm)
            if rank>p['startup'] and d and not old['0.2']['current_5'][rank-1]['timely']:
                ahead = np.round((down-t)*1e3,9)
                at = active & (ahead>30.) & (ahead<=125.)
                log_scores = {k:logit(z[at].clip(1e-6,1-1e-6)) for k,z in curves.items() if k in VIEWS}
                record = old['0.2']['current_5'][rank-1]
                theta = p['calibration']['0.2']
                base = log_scores[STATIC]
                critical.append(dict(shot=sid,reference_lead=record['lead_ms'],points=int(at.sum()),
                    gap=float(logit(theta)-base.max()) if len(base) else None,
                    max_delta={k:float((log_scores[k]-base).max()) if len(base) else None for k in TRIALS},
                    new_crossing={k:bool(((curves[k][at]>=theta)&(curves[STATIC][at]<theta)).any()) for k in TRIALS}))
    metrics = {b:{k:lead.summarize(v) for k,v in a.items()} for b,a in rows.items()}
    baselines = (STATIC,'current_5','large_current','point_current','peak_earlier')
    pairs = {b:{f'{a}__to__{k}':dict(flips=flips(v[a],v[k]),lead=lead.paired(v[a],v[k]))
                for a in baselines for k in TRIALS if a!=k} for b,v in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,confirmation=confirmations,paired=pairs,critical=critical,
        write_shots=support,write_endpoints=sum(map(len,radii)),
        token_norm_quantiles=np.quantile(np.concatenate(radii),[.1,.5,.9]).tolist(),
        raw_verified=360*len(arms)*len(p['calibration'])))
    for b,v in metrics.items():
        print(b,{k:(m['by_deadline']['30']['tp'],m['fp']) for k,m in v.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit','run','score'))
    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    ns = parser.parse_args()
    src,ref,out = ns.source.resolve(),ns.reference.resolve(),ns.output.resolve()
    if ns.phase=='fit':
        out.mkdir(parents=True)
        copyfile(__file__,out/'executed.py')
        fit(src,out)
    elif ns.phase=='run':
        run(src,ref,out)
    else:
        score(out)
