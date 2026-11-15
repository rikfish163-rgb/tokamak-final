"""Retrospective direction and size control using a separately trained EAST embedding."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit

from . import commissioning_embedding_time_scope_20261114 as offline
from . import commissioning_temporal_pair_20261114 as pair
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips


embed,history,group,lag = pair.embed,pair.history,pair.group,pair.lag
layer,common = pair.feedback.layer,pair.feedback
UNMASKED = embed.ALL_AVAILABLE_VALUES
D22,D8,NATIVE = NEW = ('archival_22_matched','archival_8_matched','archival_22_unscaled')
ARMS = ('bias_only','current_5',pair.TRUE,*NEW)


def teacher_columns(w):
    measured = w.unflatten(1,(embed.PATCH,23))[:,:,list(UNMASKED)]
    same = torch.as_tensor([c in pair.VALUES for c in UNMASKED],device=w.device)
    return measured.flatten(1),measured.masked_fill(~same,0.).flatten(1)


def matched_writes(x,weights,requested):
    # Cap the actual first-layer displacement on this input, not the matrix norm.
    delta = torch.einsum('ade,bpe->abpd',weights,x)
    norm = delta.flatten(2).norm(dim=-1)
    v = delta*(requested[None]/norm.clamp_min(embed.EPS))[...,None,None]
    return v,delta[0],norm[0]


@torch.no_grad()
def project(base,values,mask,adjust,teacher,spent):
    parts,head,quality = [],[],[]
    n = len(values)
    views = 1+len(NEW)

    def inputs(_m,a):
        x = adjust(a[0][:n])
        parts.append(embed.measured_values(x,values=UNMASKED))
        return (x.repeat(views,1,1),)

    def updates(_m,_a,out):
        h = out[:n]
        matched,native,size = matched_writes(parts[0],teacher,spent)
        error = (((h[None]+matched)-h[None]).flatten(2).norm(dim=-1)-spent).abs().max()
        quality.extend([error,size])
        delta = torch.cat((matched,native[None]),dim=0)
        return torch.cat((h,out[n:]+delta.flatten(0,1)),dim=0)

    handles = (base.embedding.register_forward_pre_hook(inputs),
               base.embedding.register_forward_hook(updates),
               base.risk_head.register_forward_pre_hook(lambda _m,a:head.append(a[0])))
    try:
        base(values.repeat(views,1,1),mask.repeat(views,1,1))
    finally:
        for h in handles:
            h.remove()
    return head[0].unflatten(0,(views,n)),quality


def evaluate(signal,args,decoder,path,arc,teacher,dose):
    base = args[0]
    q,h,z,beta,_,phase,correction = group.parent_track(signal,args,decoder,path)
    x = np.flatnonzero(dose>0.)
    out = np.broadcast_to(arc['bias_only'],(len(NEW),len(dose))).copy()
    sizes = np.zeros_like(dose)
    errors = np.zeros(2)
    for i,n in common.block_indices(x):
        v,m = lag._windows(base,signal,i)
        j = lag.delayed_slots(q,i)[0]
        adjust = layer.patcher(phase[j],correction[j])
        p, (error, size) = project(base,v,m,adjust,teacher,z.new_tensor(dose[i]))
        t = (z[i]+((p-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        errors = np.maximum(errors,[float(abs(t[0,:n]-arc['bias_only'][i[:n]]).max()),float(error)])
        out[:,i[:n]] = t[1:,:n]
        sizes[i[:n]] = size[:n].cpu().numpy()
    if errors[0]>3e-6 or errors[1]>1e-5 or not np.isfinite(out).all():
        raise ValueError(f'parent or same token dose mismatch: {errors}')
    return dict(zip(NEW,out,strict=True)),sizes,errors


def seal(out,replay,only):
    now = monotonic()
    lock,p,arcseal,args,decoder,path = history.inputs()
    pp,ss = (read(replay/f'{k}.json') for k in ('protocol','seal'))
    ref = read(offline.REFERENCE/'protocol.json')
    fit_shots = set(ref['train']) | set(ref['val'])
    if (any(p[k]!=pp[k] for k in history.LOCK) or pp['only'] is not None or len(ss['shots'])!=360
            or p['order']!=ref['order'] or not fit_shots.isdisjoint(p['order'])):
        raise ValueError('requires an already sealed paired replay and out-of-stream diagnostic fit')
    deltas = torch.stack(teacher_columns(offline.models()[2]))
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},replay=str(replay),teacher=str(offline.REFERENCE),
        only=only,arms=ARMS,scope='retrospective reach diagnostic using 1917 additional archival labeled shots',
        readout='original EAST parent and thresholds; no additional weights persisted',
        dose='matched arms use exact per-endpoint realized norm of the completed pairing experiment; '
             'unscaled uses the native learned correction at the same endpoints'))
    receipts,prefix,maximum = [],{},np.zeros(2)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes forbidden before seal')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('HDF5 truth forbidden before seal'))):
        for rank,(sid,name) in enumerate(zip(p['order'],ss['shots'],strict=True),1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,arcseal)
            with np.load(replay/'shots'/name,allow_pickle=False) as f:
                dose,actual,predictor = f['dose'],f[pair.TRUE],f['bias_only']
            np.testing.assert_array_equal(predictor,arc['bias_only'])
            if dose.any():
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                curve,size,error = evaluate(sig,args,decoder,path,arc,deltas,dose)
                maximum = np.maximum(maximum,error)
                if rank in (21,120,360) or sid in pair.CASES or only is not None:
                    n = min(len(dose),650)
                    cut,view = common.trimmed(sig,arc,n)
                    replayed,native,_ = evaluate(cut,args,decoder,path,view,deltas,dose[:n])
                    np.testing.assert_allclose(native,size[:n],atol=1e-4,rtol=1e-6)
                    for k,z in replayed.items():
                        np.testing.assert_allclose(z,curve[k][:n],atol=3e-6,rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(z>=tau,curve[k][:n]>=tau)
                    prefix[str(sid)] = dict(points=n,probability=max(float(abs(z-curve[k][:n]).max())
                                                                                 for k,z in replayed.items()))
            else:
                curve,size = dict.fromkeys(NEW,predictor),np.zeros_like(dose)
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,bias_only=predictor,current_5=arc['current_5'],**{pair.TRUE:actual},
                                    **curve,dose=dose,native_size=size)
            receipts.append(name)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(rank=rank,seconds=round(monotonic()-now,2)),flush=True)
    write(out/'seal.json',dict(shots=receipts,prefix=prefix,maximum=maximum.tolist(),seconds=monotonic()-now))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    q = read(Path(p['replay'])/'per_shot.json')
    order = p['order']
    if p['only'] is not None or len(order)!=360 or s['shots']!=[f'{i:04d}_{j}.npz' for i,j in enumerate(order,1)]:
        raise ValueError('full diagnostic replay required')
    rows = {c:{k:[] for k in ARMS} for c in p['calibration']}
    receipts,hard = [],[]
    for i,(sid,name) in enumerate(zip(order,s['shots'],strict=True)):
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            v = dict(f)
        used = v['dose']>0.
        for k in ARMS:
            if k in NEW:
                np.testing.assert_array_equal(v[k][~used],v['bias_only'][~used])
            for c,tau in p['calibration'].items():
                row = dict(shot=sid,disruptive=y,down_s=d,**score_curve(t,v[k],y,d,
                            p['original'] if i<p['startup'] else tau))
                if k not in NEW and row!=q[c][k][i]:
                    raise ValueError('same-parent reference changed')
                rows[c][k].append(row)
        if used.any():
            native,actual = v['native_size'][used],v['dose'][used]
            receipts.append(dict(shot=sid,n=int(used.sum()),native_median=float(np.median(native)),
                common_median=float(np.median(actual)),native_to_common_median=float(np.median(native/actual))))
        original = rows['0.2']['current_5'][-1]
        if y and not original['timely'] and i>=p['startup']:
            lead_ms = np.round((d-t)*1e3,9)
            own = used & (lead_ms>30.) & (lead_ms<=125.)
            if own.any():
                th = p['calibration']['0.2']
                z = logit(v['bias_only'][own].clip(1e-7,1.-1e-7))
                changes = {k:float(np.max(logit(v[k][own].clip(1e-7,1.-1e-7))-z)) for k in (pair.TRUE,*NEW)}
                hard.append(dict(shot=sid,kind='late' if original['alarm'] else 'miss',support=int(own.sum()),
                    gap=float(logit(th)-z.max()),max_lift=changes,
                    already=bool((v['bias_only'][own]>=th).any()),
                    hit={k:bool((v[k][own]>=th).any()) for k in (pair.TRUE,*NEW)}))
    summary = {c:{k:lead.summarize(rs) for k,rs in a.items()} for c,a in rows.items()}
    comparison = {c:{k:flips(a['bias_only'],a[k]) for k in ARMS[1:]} for c,a in rows.items()}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=summary,paired=comparison,raw_verified=len(rows)*360*len(ARMS),
                               size_per_shot=receipts,hard=hard))
    print({c:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in a.items()} for c,a in summary.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    cli.add_argument('--replay',type=Path)
    which = cli.add_mutually_exclusive_group()
    which.add_argument('--score',action='store_true')
    which.add_argument('--smoke-rank',type=int,action='append')
    a = cli.parse_args()
    if a.score:
        score(a.out.resolve())
    elif a.replay is None:
        cli.error('--replay with sealed paired forecast is required')
    else:
        seal(a.out.resolve(),a.replay.resolve(),a.smoke_rank)
