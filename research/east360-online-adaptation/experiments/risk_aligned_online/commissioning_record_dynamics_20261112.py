"""Two-part diagnostic forecast for record status and conditional motion."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_complementary_feedback_20261111 as rotation
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_frequency_feedback_20261112 as previous
from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_roughness_20261110 import coordinates
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


H, LAG = lag.H, lag.LAG
C = len(lag.CHANNELS)
SPECTRAL = (previous.TARGET_A,previous.TARGET_F)
OTHER = tuple(k for k in range(C) if k not in SPECTRAL)
N1 = lag.CHANNELS.index(io23.RAW23.index('n=1 amplitude'))
STEPS, BATCH, SEED = 320, 192, 131
DIM = H+(H-LAG)*C


def origins(n):
    return np.arange(lag.FIRST,n-H-1,lag.STRIDE,dtype=np.int64)


def record(raw):
    f = raw['values'][:,previous.RAW_F]
    known = raw['available'][:,previous.RAW_F] & np.isfinite(f)
    return known & (f>0.),known


def record_context(positive, known, q):
    # Smoothed occupancy gives a finite reference even without known samples.
    context = [positive[q],known[q]]
    for start,end in ((-15,1),(-63,1),(-31,-15)):
        i = q[:, None]+np.arange(start,end)
        context.append((positive[i].sum(1)+1.)/(known[i].sum(1)+2.))
    return np.column_stack(context).astype(np.float32)


def block_targets(z, available, positive, known, q, scale):
    i = q[:, None]+1+np.arange(H)
    v,m,b,k = z[i],available[i],positive[i],known[i]
    spans = np.lib.stride_tricks.sliding_window_view(m,LAG+1,axis=1).all(-1)
    present = np.lib.stride_tricks.sliding_window_view(b,LAG+1,axis=1).all(-1)
    spans[...,list(SPECTRAL)] &= present[:,:,None]
    y = np.tanh((v[:,LAG:]-v[:,:-LAG])/scale)
    return dict(status=b.astype(np.float32),status_observed=k,
        motion=np.where(spans,y,0.).astype(np.float32),motion_observed=spans)


def observation(raw, signal, q, args):
    center,scale,_,noise,_ = args[2]
    columns = list(lag.CHANNELS)
    z,m = coordinates(signal['values'][:,columns],signal['available'][:,columns],
                      center[columns],scale[columns])
    positive,known = record(raw)
    return record_context(positive,known,q),block_targets(z,m,positive,known,q,noise)


def shift_supported(y, supported):
    """Rotate each channel's observed sequence, leaving support and values fixed as sets."""
    t = y.shape[1]
    i = torch.arange(t,device=y.device)[None,:,None]
    order = torch.where(supported,i,i+t).argsort(1)
    n = supported.sum(1,keepdim=True)
    src = torch.where(i<n,(i+1)%n.clamp_min(1),i)
    sorted_y = y.gather(1,order)
    return torch.empty_like(y).scatter(1,order,sorted_y.gather(1,src))


def isolate(motion_grad, record_grad):
    dot = (motion_grad*record_grad).sum((1,2),keepdim=True)
    norm = record_grad.square().sum((1,2),keepdim=True)
    return motion_grad-(dot/norm.clamp_min(1e-20))*record_grad


def equal_rotation(old, candidates):
    """Matched realized angle and norm, relative to the same already-written patch."""
    a = old.flatten(1)
    b = candidates.flatten(2)
    n = a.norm(dim=1,keepdim=True)
    u = a/n.clamp_min(1e-7)
    t = b-(b*u).sum(2,keepdim=True)*u
    mag = t.norm(dim=2,keepdim=True)
    use = (n>1e-7)&(mag>(1e-3*b.norm(dim=2,keepdim=True)).clamp_min(1e-7)).all(0)
    r = rotation.RATIO
    made = (a+r*n*t/mag.clamp_min(1e-7))/(1.+r*r)**.5
    return torch.where(use,made,a).reshape_as(candidates),use.flatten()


def _mean(value, observed):
    axes = tuple(range(1,value.ndim))
    n = observed.sum(axes)
    return value.masked_fill(~observed,0.).sum(axes)/n.clamp_min(1),n>0


def losses(status_logit, motion, target, *, shifted=False):
    b,m = target['status'],target['motion']
    mask,valid = target['status_observed'],target['motion_observed']
    if shifted:
        b = shift_supported(b[:,:,None],mask[:,:,None])[:,:,0]
        m = shift_supported(m,valid)
    status,_ = _mean(F.binary_cross_entropy_with_logits(status_logit,b,reduction='none'),mask)
    error = F.smooth_l1_loss(motion,m,beta=.2,reduction='none')
    a,sa = _mean(error[...,list(SPECTRAL)],valid[...,list(SPECTRAL)])
    c,sc = _mean(error[...,list(OTHER)],valid[...,list(OTHER)])
    return status,(a+c)/(sa.int()+sc.int()).clamp_min(1)


def forecast(h, context, fitted):
    center,scale,w,b = fitted
    q = torch.cat((h,context),dim=1)
    y = ((q-center)/scale)@w.T+b
    return y[:,:H],y[:,H:].reshape(len(h),H-LAG,C)


@torch.no_grad()
def collect(args, lock, role, ids):
    base = args[0]
    device = 'EAST' if role=='cold' else 'J-TEXT'
    rows = []
    for j,sid in enumerate(ids,1):
        raw = simple_data.read_shot(device,sid,rate_contract='nominal' if role=='cold' else 'strict')
        q = origins(len(raw['values']))
        if not len(q):
            continue
        # The cold signals are diagnostic-only, in the already approved
        # suffix coordinates. No EAST shot is used to fit these readouts.
        sig = lag.prepare_actual(raw,lag.STARTUP+1,lock['mapping']) if role=='cold' else raw
        c,y = observation(raw,sig,q,args)
        h = []
        for start in range(0,len(q),256):
            x,m = lag._windows(base,sig,q[start:start+256])
            w = x.new_zeros((len(x),C,C))
            h.append(lag._features(base,x,m,w).cpu().numpy())
        rows.append(dict(h=np.concatenate(h),context=c,shot=np.full(len(q),sid),**y))
        if j%450==0 or j==len(ids):
            print(dict(stage=f'collect_{role}',shots=j),flush=True)
    return {k:np.concatenate([row[k] for row in rows]) for k in rows[0]}


def fit(rows):
    cuda = {k:torch.as_tensor(v,device='cuda') for k,v in rows.items() if k!='shot'}
    x = torch.cat((cuda['h'],cuda['context']),dim=1)
    _,i,n = np.unique(rows['shot'],return_inverse=True,return_counts=True)
    prob = 1./n[i]/len(n)
    weight = torch.as_tensor(prob[:,None],device=x.device,dtype=x.dtype)
    center = (x*weight).sum(0)
    scale = (weight*(x-center).square()).sum(0).sqrt().clamp_min(.05)
    head = torch.nn.Linear(x.shape[1],DIM,device=x.device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    opt = torch.optim.Adam(head.parameters(),lr=.015)
    rng = np.random.default_rng(SEED)
    for step in range(STEPS):
        batch = rng.choice(len(x),BATCH,p=prob)
        f = (center,scale,head.weight,head.bias)
        y,p = forecast(cuda['h'][batch],cuda['context'][batch],f)
        b,d = losses(y,p,{k:cuda[k][batch] for k in
                          ('status','motion','status_observed','motion_observed')})
        loss = (b+d).mean()+.02*head.weight.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1)%160==0:
            print(dict(stage='fit_source_readouts',step=step+1,
                       status=float(b.mean().detach()),motion=float(d.mean().detach())),flush=True)
    return center,scale,head.weight.detach(),head.bias.detach()


def _shot_average(error, usable, shots):
    by = np.unique(shots[usable])
    values = np.asarray([error[usable & (shots==s)].mean(dtype=float) for s in by])
    return dict(shots=len(by),value=float(values.mean()) if len(values) else None)


@torch.no_grad()
def diagnostics(rows, fitted):
    def to(a):
        return torch.as_tensor(a,device=fitted[0].device)

    y,p = forecast(to(rows['h']),to(rows['context']),fitted)
    st,m = to(rows['status']),to(rows['status_observed'])
    current = .05+.90*to(rows['context'][:,:1])
    recent = to(rows['context'][:,2:3])
    score = {}
    for name,prob in (('frozen_readout',y.sigmoid()),('hold_state',current),('recent_occupancy',recent)):
        prob = prob.expand_as(st).clamp(1e-6,1.-1e-6)
        nll,ok = _mean(F.binary_cross_entropy(prob,st,reduction='none'),m)
        brier,_ = _mean((prob-st).square(),m)
        score[name] = dict(status_log_loss=_shot_average(nll.cpu().numpy(),ok.cpu().numpy(),rows['shot']),
                           status_brier=_shot_average(brier.cpu().numpy(),ok.cpu().numpy(),rows['shot']))
    t,v = to(rows['motion']),to(rows['motion_observed'])
    for name,pred in (('frozen_readout',p),('zero_motion',torch.zeros_like(p))):
        err = F.smooth_l1_loss(pred,t,beta=.2,reduction='none')
        for g,cols in (('spectral',SPECTRAL),('other',OTHER)):
            e,ok = _mean(err[...,list(cols)],v[...,list(cols)])
            score.setdefault(name,{})[f'{g}_huber'] = _shot_average(e.cpu().numpy(),ok.cpu().numpy(),rows['shot'])
    return dict(shots=len(np.unique(rows['shot'])),origins=len(st),
        origins_spectral_supported=int(v[...,list(SPECTRAL)].any((1,2)).sum()),
        observed_positive_fraction=float(st[m].mean()),scores=score)


def run(out):
    lag.configure()
    began = time.monotonic()
    lock = lag.load_inputs()
    ref = Path(read(previous.REFERENCE/'protocol.json')['reference'])
    _,_,_,args,_ = eos._load(ref,lock)
    digests = [state_digest(v) for v in args[:2]]
    roles = io23.source_roles()
    out.mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_fit.py')
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('diagnostic fitting does not read outcomes')):
        train = collect(args,lock,'train',roles['train'])
        model = fit(train)
        val = collect(args,lock,'val',roles['val'])
        cold = collect(args,lock,'cold',lock['ids'][:lag.STARTUP])
    receipt = {k:diagnostics(v,model) for k,v in (('train',train),('val',val),('cold',cold))}
    if digests!=[state_digest(v) for v in args[:2]]:
        raise ValueError('source weights changed while fitting diagnostic readouts')
    with (out/'readout.npz').open('xb') as handle:
        np.savez_compressed(handle,**{k:v.cpu().numpy() for k,v in zip(('center','scale','w','b'),model,strict=True)})
    write(out/'fit.json',dict(scope='JTEXT_TRAIN_ONLY_TWO_PART_DIAGNOSTIC_FORECAST',
        parent=str(previous.REFERENCE),parent_seal=sha(previous.REFERENCE/'seal.json'),
        source_sha256=read(previous.REFERENCE/'protocol.json')['source_sha256'],
        checkpoint=sha(out/'readout.npz'),code=sha(out/'executed_fit.py'),
        eligible_train_ids=np.unique(train['shot']).tolist(),val_ids=np.unique(val['shot']).tolist(),
        cold_ids=np.unique(cold['shot']).tolist(),seed=SEED,steps=STEPS,forecast=receipt,
        seconds=time.monotonic()-began))
    print(receipt,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out.resolve())
