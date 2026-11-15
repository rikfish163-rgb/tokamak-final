"""Test whether ambiguous Mirnov jumps corrupt existing within-shot feedback."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_bias_dynamics_20261112 as bias
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


REFERENCE = io23.ROOT/'probe_outputs/commissioning_bias_dynamics_east360_20261112_v1'
RAW_F,RAW_A = [io23.RAW23.index(k) for k in ('Mir_avg_fre','Mir_avg_amp')]
TARGET_F,TARGET_A = [lag.CHANNELS.index(k) for k in (RAW_F,RAW_A)]
OLD = ('fixed',guard.PATH,'bias_only')
MATCH,FREQ,JOINT,WRONG = NEW = ('legacy_equal_dose','freq_only','freq_amp','freq_amp_wrong_time')
ARMS = (*OLD,*NEW)
PAIRS = (('bias_only',MATCH),('bias_only',FREQ),('bias_only',JOINT),(MATCH,FREQ),(MATCH,JOINT),
         (FREQ,JOINT),(WRONG,JOINT))
TOL = 4e-7


def masks(raw, q, rank):
    """A derivative interval needs an identified positive frequency throughout."""
    idx = q[:, None]+np.arange(1,lag.H+1)
    f = raw['values'][idx,RAW_F]
    positive = raw['available'][idx,RAW_F] & np.isfinite(f) & (f>0.)
    true = np.lib.stride_tricks.sliding_window_view(positive,lag.LAG+1,axis=1).all(2)
    n = true.shape[1]
    offset = 1+((rank+(q-lag.FIRST)//lag.STRIDE)%(n-1))
    wrong = true[np.arange(len(q))[:, None],(np.arange(n)-offset[:, None])%n]
    return true,wrong


def motion(y, trusted, columns=(TARGET_F,)):
    difference = (y[:,lag.LAG:]-y[:,:-lag.LAG]).tanh()
    norm = (difference.square().mean(1,keepdim=True)+.02).sqrt()
    d = difference/norm
    clean = difference[...,list(columns)].masked_fill(~trusted[:,:,None],0.)
    scale = (clean.square().sum(1,keepdim=True)/trusted.sum(1,keepdim=True).clamp_min(1)[:,:,None]+.02).sqrt()
    result = d.clone()
    result[...,list(columns)] = clean/scale
    return result


def equalize(delta):
    rms = delta.flatten(2).square().mean(2).sqrt()
    target = rms.amin(0)
    factor = target[None]/rms.clamp_min(1e-12)
    return delta*factor.reshape((*factor.shape,*(1,)*(delta.ndim-2)))


def proposals(base, signal, q, prior, target, decoder, key, weights, trusted, placebo):
    matrices, residual = [],[]
    n = len(lag.CHANNELS)
    for start in range(0,len(q),128):
        cut = slice(start,start+128)
        x,m = lag._windows(base,signal,q[cut])
        w = x.new_zeros((len(x),n,n)).requires_grad_()
        p,y,k = (torch.as_tensor(a[cut],device=x.device) for a in (prior,target,key))
        pred = lag._predict(lag._features(base,x,m,w),p,decoder)
        residual.append((y-pred.detach()).mean(1))
        t,s = (torch.as_tensor(a[cut],device=x.device) for a in (trusted,placebo))
        losses = []
        settings = ((None,()),(t,(TARGET_F,)),(t,(TARGET_F,TARGET_A)),(s,(TARGET_F,TARGET_A)))
        for valid,cols in settings:
            if valid is None:
                a,b = lag.motion(pred),lag.motion(y)
            else:
                a,b = motion(pred,valid,cols),motion(y,valid,cols)
            losses.append(teacher.grouped_error(lag.ordered_signature(a),lag.ordered_signature(b)))
        g = [torch.autograd.grad(e[:,j].sum(),w,retain_graph=i+1<len(losses)*teacher.GROUPS)[0]
             for i,(e,j) in enumerate((e,j) for e in losses for j in range(teacher.GROUPS))]
        groups = torch.stack(g,dim=1).split(teacher.GROUPS,dim=1)
        mats = torch.stack([-lag.RADIUS*guard.signed_direction(*guard.basis(v),k,weights) for v in groups])
        matrices.append(mats.detach())
    if not len(q):
        return base.center.new_empty((len(NEW),0,n,n)),base.center.new_empty((0,n))
    return torch.cat(matrices,1),torch.cat(residual)


@torch.no_grad()
def _forward(base, x, m, phase, b):
    n = len(x)
    hidden,trace = [],[]

    def patcher(_layer, inputs):
        raw = inputs[0][:n]
        anchors = torch.stack([lag.patch_input(raw,a) for a in phase])
        d = anchors-raw
        # Freeze the exact original per-channel bias; match only the phase
        # writes. Each phase delta has zero channel-wise mean, so the shared
        # bias is orthogonal to it up to floating-point rounding.
        cap = d[0].flatten(1).square().mean(1).sqrt()
        bias_delta = bias.bias_patches(anchors[0],b,cap)-anchors[0]
        matched = (raw+equalize(d))+bias_delta
        dose = (matched-raw).flatten(2).square().mean(2).sqrt()
        trace.extend((dose[0],(dose[1:]-dose[:1]).abs().amax()))
        legacy = anchors[0]+bias_delta
        return (torch.cat((legacy,matched.flatten(0,1))),)

    a = base.embedding.register_forward_pre_hook(patcher)
    c = base.risk_head.register_forward_pre_hook(lambda _m, args: hidden.append(args[0]))
    try:
        base(x.repeat(len(NEW)+1,1,1),m.repeat(len(NEW)+1,1,1))
    finally:
        c.remove()
        a.remove()
    return hidden[0].reshape(len(NEW)+1,n,-1),*trace


def predict(raw, signal, rank, args, decoder, weight, archived):
    base = args[0]
    tr,h,z,beta = guard._track(signal,args)
    q,key,prior,target = teacher._mature(tr,args[2][2],args[2][3])
    true,wrong = masks(raw,q,rank)
    mat,residual = proposals(base,signal,q,prior,target,decoder,key,weight,true,wrong)
    corrected,_ = bias.past_bias(q,residual.cpu().numpy())
    b = torch.as_tensor(corrected,device=beta.device)*torch.as_tensor(args[2][3],device=beta.device)
    slots,_ = lag.delayed_slots(q,np.arange(len(z)))
    active = np.flatnonzero(slots>0)
    scores = np.broadcast_to(archived,(len(NEW),len(z))).copy()
    h,z = (torch.as_tensor(v,device=beta.device) for v in (h,z))
    dose = np.zeros(len(z),dtype=np.float32)
    e0,e1 = 0.,0.
    for start in range(0,len(active),eos.BATCH):
        i = active[start:start+eos.BATCH]
        k = slots[i]-1
        x,m = lag._windows(base,signal,i)
        feature,rms,error = _forward(base,x,m,mat[:,k],b[k])
        p = (z[i]+((feature-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        e0 = max(e0,float(np.max(np.abs(p[0]-archived[i]))))
        e1 = max(e1,float(error))
        scores[:,i] = p[1:]
        dose[i] = rms.cpu().numpy()
    if e0>3e-6 or e1>TOL:
        raise ValueError(f'legacy replay / actual dose mismatch: {e0:g} / {e1:g}')
    return scores,q+lag.H+1,true.sum(1),dose,e0,e1


def run(out):
    lag.configure()
    began = time.monotonic()
    lock = lag.load_inputs()
    b,s = [read(REFERENCE/k) for k in ('protocol.json','seal.json')]
    ref = Path(b['reference'])
    p,_,_,args,decoder = eos._load(ref,lock)
    teacher_file = ref/'teacher.json'
    if (b['reference_seal']!=sha(ref/'seal.json') or b['order']!=lock['ids']
            or b['calibration']!=p['calibration'] or b['original']!=p['original']
            or sha(teacher_file)!=p['teacher_sha256'] or b['history']!=bias.HISTORY):
        raise ValueError('reference forecaster, path weights or bias protocol changed')
    weight = torch.as_tensor(read(teacher_file)['path'],device='cuda',dtype=torch.float32)
    before = [state_digest(v) for v in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_AMBIGUOUS_MIRNOV_FEEDBACK_CAUSAL_REPLAY',
        order=lock['ids'],startup=p['startup'],source_sha256=p['source_sha256'],
        calibration=p['calibration'],original=p['original'],arms=ARMS,
        parent=str(REFERENCE),parent_seal=sha(REFERENCE/'seal.json'),
        parent_score=sha(REFERENCE/'per_shot.json'),teacher_sha256=p['teacher_sha256'],
        executed_sha256=sha(Path(__file__)),lag=lag.LAG,horizon=lag.H,stride=lag.STRIDE,
        contract='same frozen H25 forecaster; zero-ambiguous spans are removed only from lag loss; '
                 'prior mask, transforms, frozen risk readout, threshold, bias and base all unchanged',
        control='frequency only, both coupled fields, equal-count circularly shifted shared validity; '
                'match actual total patch RMS with attenuated legacy arm, never amplify a smaller arm',
        outcomes='no outcomes until all curves seal; no persistent learning'))
    hashes,witnesses,errs = {},{},[]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            old = REFERENCE/'shots'/name
            if sha(old)!=s['curve_sha256'][name] or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('changed raw data or sealed reference')
            with np.load(old,allow_pickle=False) as f:
                t,ref_release,legacy = (f[k] for k in ('times','release','bias_only'))
            if rank<=p['startup']:
                a = np.broadcast_to(legacy,(len(NEW),len(t)))
                r,spans,dose,e0,e1 = ref_release,np.zeros(0,int),np.zeros(len(t),np.float32),0.,0.
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                np.testing.assert_array_equal(raw['times'],t)
                a,r,spans,dose,e0,e1 = predict(raw,signal,rank,args,decoder,weight,legacy)
                np.testing.assert_array_equal(r,ref_release)
                if rank in (21,120,360):
                    end = min(len(t),330)
                    x,y = ({k:v[k][:end] for k in ('values','available')} for v in (raw,signal))
                    aa,rr,ss,dd,_,_ = predict(x,y,rank,args,decoder,weight,legacy[:end])
                    np.testing.assert_array_equal(rr,r[r<end])
                    np.testing.assert_array_equal(ss,spans[r<end])
                    np.testing.assert_allclose(aa,a[:,:end],atol=3e-6,rtol=0.)
                    np.testing.assert_allclose(dd,dose[:end],atol=TOL,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(aa>=tau,a[:,:end]>=tau)
                    witnesses[str(sid)] = end
            f = out/'shots'/name
            if not np.isfinite(a).all():
                raise FloatingPointError('nonfinite proposed risk curve')
            with f.open('xb') as handle:
                np.savez_compressed(handle,release=r,valid_spans=spans,rms=dose,**dict(zip(NEW,a,strict=True)))
            hashes[name] = sha(f)
            errs.append((e0,e1))
            if rank in (21,360) or rank%60==0:
                print(dict(stage='seal_frequency_guard',rank=rank,seconds=round(time.monotonic()-began,2)),flush=True)
    if before!=[state_digest(v) for v in args[:2]]:
        raise ValueError('frozen source weights changed')
    write(out/'seal.json',dict(seconds=time.monotonic()-began,curve_sha256=hashes,
        prefix_witnesses=witnesses,max_legacy_error=max(v[0] for v in errs),max_dose_error=max(v[1] for v in errs)))


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    ref = Path(p['parent'])
    if (p['executed_sha256']!=sha(out/'executed_probe.py') or p['parent_seal']!=sha(ref/'seal.json')
            or p['parent_score']!=sha(ref/'per_shot.json') or len(s['curve_sha256'])!=360):
        raise ValueError('sealed inputs changed')
    legacy = read(ref/'per_shot.json')
    refs = read(ref/'seal.json')['curve_sha256']
    per = {band:{k:[] for k in ARMS} for band in p['calibration']}
    count = dict(blocks=0,ambiguous_blocks=0,all_blocked=0,partial_support_blocks=0,
                 active_points=0)
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        a,b = out/'shots'/name,ref/'shots'/name
        if sha(a)!=s['curve_sha256'][name] or sha(b)!=refs[name]:
            raise ValueError('changed sealed curve')
        d,down,t = hdf._raw_truth(sid)
        with np.load(a,allow_pickle=False) as new,np.load(b,allow_pickle=False) as old:
            np.testing.assert_array_equal(t,old['times'])
            np.testing.assert_array_equal(new['release'],old['release'])
            count['blocks'] += len(new['release'])
            spans = new['valid_spans']
            count['ambiguous_blocks'] += int((spans<lag.H-lag.LAG).sum())
            count['all_blocked'] += int((spans==0).sum())
            count['partial_support_blocks'] += int(((spans>0)&(spans<lag.H-lag.LAG)).sum())
            count['active_points'] += int((new['rms']>0.).sum())
            for band,tau in p['calibration'].items():
                for k in ARMS:
                    v = old[k] if k in OLD else new[k]
                    row = dict(shot=sid,disruptive=d,down_s=down,**score_curve(
                        t,v,d,down,p['original'] if rank<=p['startup'] else tau))
                    if k in OLD and row!=legacy[band][k][rank-1]:
                        raise ValueError('original scores do not match raw HDF5')
                    per[band][k].append(row)
    write(out/'per_shot.json',per)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {band:{k:timing_decomposition(v) for k,v in rows.items()} for band,rows in per.items()}
    changes = {band:{f'{a}__to__{b}':flips(r[a],r[b]) for a,b in PAIRS} for band,r in per.items()}
    write(out/'result.json',dict(status='COMPLETED_AMBIGUOUS_MIRNOV_FEEDBACK',metrics=metrics,
                                paired=changes,activity=count,raw_verified=360*len(ARMS)*len(metrics)))
    print({band:{k:(v['tp'],v['fp']) for k,v in rows.items()} for band,rows in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    a = parser.parse_args()
    run(a.out.resolve())
    score(a.out.resolve())
