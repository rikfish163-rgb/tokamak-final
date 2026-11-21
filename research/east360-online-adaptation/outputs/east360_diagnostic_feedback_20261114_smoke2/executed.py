"""Add observed radiation and magnetic information to the shot-local embedding step."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_latent_teacher_20261114 as latent
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot


embed, history, group, lag, layer = latent.embed, latent.history, latent.group, latent.lag, latent.layer
REF = io23.ROOT/'outputs/east360_latent_teacher_20261114_v3_value_scope'
VALUES = embed.ALL_AVAILABLE_VALUES


def channels(*ns):
    return tuple(io23.RAW23.index(n) for n in ns)


SPECTRUM = channels('Mir_avg_amp','Mir_avg_fre')
GROUPS = (channels('sxr_kurt','sxr_mean'), channels('xuv_kurt','xuv_ratio'),
          channels('CIII'), channels('Mir_VV'), SPECTRUM)
NAMES = ('SXR','AXUV','CIII','Mir_VV','spectral_positive')
OPTICS = np.array([1,1,1,0,0],bool)
MAGNETICS = ~OPTICS
MODES = ('core_current','core_earlier','optical_current','optical_earlier','optical_reversed',
         'magnetic_current','magnetic_earlier','magnetic_reversed')
ARMS = (*embed.OLD,'legacy_common',*MODES)
PAIRS = tuple((a,b) for b in ('optical_current','magnetic_current')
    for a in ('bias_only','current_5','core_current',b.replace('current','earlier'),b.replace('current','reversed')))
CASES = (90874,74970,81406,70320,85422,73743,73405,91689)
# Counterfactual CLS views are subtracted in float32. Below this norm, a
# roundoff-sized difference in two views must not create a unit direction.
MIN_CONTRAST = 1.e-4


def available_groups(signal,q):
    ix = q[:,None]+np.arange(lag.H+1)
    observed = signal['available'][ix]
    value = signal['values'][ix]
    closed = [observed[:,:,cols].all((1,2)) for cols in GROUPS]
    # This is a records-support condition, not detection of a mode or proof of
    # loss of rotation. Keep all original recorded values in the risk input.
    closed[-1] &= (value[:,:,SPECTRUM]>0.).all((1,2))
    return np.stack(closed,axis=1)


def diagnostic_views(x,m):
    core,valid = latent.teacher_views(x,m)
    v = core[0].expand(1+2*len(GROUPS),*core.shape[1:]).clone()
    for j,cols in enumerate(GROUPS):
        actual = x[:,-lag.H:,list(cols)]
        v[1+2*j,:,-lag.H:,list(cols)] = actual
        v[2+2*j,:,-lag.H:,list(cols)] = actual.flip(1)
    # Extra diagnostic masks stay at their issue-time state. Training uses
    # these extra views only for groups observed throughout the packet.
    return v,valid


def fuse(core, extra, gate):
    """Each available diagnostic group contributes a direction, not a scale."""
    n = extra.norm(dim=-1,keepdim=True)
    use = gate & (n[...,0]>MIN_CONTRAST)
    u = (extra/n.clamp_min(MIN_CONTRAST)).masked_fill(~use[...,None],0.)
    effect = u.sum(1)/use.sum(1).clamp_min(1)[:,None]
    return core+core.norm(dim=-1,keepdim=True)*effect


@torch.no_grad()
def innovations(base,signal,track,i,transform):
    x,m = lag._windows(base,signal,track[0][i]+lag.H)
    views,available = diagnostic_views(x,m)
    h = latent.encode_teacher_views(base,views,available,transform)
    return (h[1:]-h[:1]).unflatten(0,(-1,2)).permute(1,2,0,3)


def estimate(signal,args,dec,path,readout,arc):
    base = args[0]
    tr = group.parent_track(signal,args,dec,path)
    q,_,_,beta,*_ = tr
    _,obs = latent.support_examples(signal,tr)
    np.testing.assert_array_equal(q+lag.H+1,arc['release'])
    prev,common = embed.feedback_support(q,obs)
    common &= arc['eligible']
    available = available_groups(signal,q)
    same = available & available[prev.clip(min=0)] & common[:,None]
    active = np.flatnonzero(common)
    used = np.unique(np.r_[active,prev[active]])
    f = base.center.new_zeros((len(q)+1,base.cfg.dim))
    extra = f.new_zeros((2,len(q)+1,len(GROUPS),base.cfg.dim))
    r = f.new_zeros((len(MODES),len(q)+1,base.cfg.dim,embed.PATCH*len(VALUES)))

    def orthogonal(x):
        return x-latent.risk_component(x,beta)

    for start in range(0,len(used),group.BATCH):
        j = used[start:start+group.BATCH]
        x,m,transform,z = latent.origin_graph(base,signal,tr,j)
        with torch.no_grad():
            y,_ = latent.actual_targets(base,signal,q,j,transform)
            zero = z.new_zeros((1,len(j),base.cfg.dim,embed.PATCH*len(VALUES)))
            h,_ = embed.forward_embed(base,x,m,transform,zero,values=VALUES)
            prediction = h[1]@readout[0]+readout[1]
            f[j+1] = orthogonal(y-prediction)
        extra[:,j+1] = orthogonal(innovations(base,signal,tr,j,transform))
    for start in range(0,len(active),group.BATCH):
        j = active[start:start+group.BATCH]
        source = j+1
        prior = prev[j]+1
        # Same already-observed channels, origin and VJP for the additive
        # comparisons; age changes the target, not the gradient readout.
        cur,old = f[source],f[prior]
        x,m,transform,z = latent.origin_graph(base,signal,tr,j)
        d = z.new_zeros((len(j),base.cfg.dim,embed.PATCH*len(VALUES)),requires_grad=True)
        out,_ = embed.forward_embed(base,x,m,transform,d[None],values=VALUES)
        prediction = out[1]@readout[0]+readout[1]
        forces = [cur,old]
        for kind in (OPTICS,MAGNETICS):
            gate = torch.as_tensor(same[j]&kind,device=z.device)
            forces.extend((fuse(cur,extra[0,source],gate),fuse(cur,extra[0,prior],gate),
                           fuse(cur,extra[1,source],gate)))
        r[:,source] = torch.stack([torch.autograd.grad(prediction,d,a,retain_graph=k+1<len(forces))[0]
                                   for k,a in enumerate(forces)])
    projected = embed.stable_direction(r,r)
    if not torch.isfinite(projected).all():
        raise FloatingPointError('nonfinite diagnostic update')
    return tr,projected,common,same


def predict(signal,args,dec,path,readout,arc):
    tr,w,eligible,matched = estimate(signal,args,dec,path,readout,arc)
    arm,dose,err = embed.render(args[0],signal,tr,w,eligible,arc,modes=MODES,values=VALUES)
    return arm,dict(q=tr[0],eligible=eligible,matched=matched,dose=dose),err


def trimmed(signal,arc,n):
    small = {k:v[:n] for k,v in signal.items() if k in ('values','available')}
    cut = arc['release']<n
    view = {k:(v[cut] if k in ('release','eligible') else v[:n]) for k,v in arc.items()}
    return small,view


def run(out, only):
    start = monotonic()
    lock,p,archive,args,decoder,path = history.inputs()
    prior,ss = (read(REF/f'{k}.json') for k in ('protocol','seal'))
    if (any(p[k]!=prior[k] for k in history.LOCK) or len(ss['shots'])!=360 or
            prior['target_channels']!=[io23.RAW23[i] for i in embed.VALUES]):
        raise ValueError('frozen core forecast and fixed calibration required')
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    p = dict(**{k:p[k] for k in history.LOCK},only=only,arms=ARMS,ref=str(REF),
        target='conditional change in frozen CLS when one extra group replaces persistence',
        conditions=dict(groups=dict(zip(NAMES,[[io23.RAW23[i] for i in g] for g in GROUPS],strict=True)),
                        continuous_spectrum='both positive throughout issue and all 25 observations',
                        compared_with='earlier or time-reversed extra group target; same core feedback',
                        minimum_cls_contrast_norm=MIN_CONTRAST),
        state=dict(value_channels=[io23.RAW23[i] for i in VALUES],freeze='all remaining parameters',
                   reset='every shot',dose='original common first-layer shift, identical across arms'),
        clock=dict(horizon=lag.H,issue_stride=lag.STRIDE,first_release=embed.OFFSET,handoff=embed.PATCH),
        labels='no new target-domain labels; the entire replay is sealed before scoring')
    write(out/'protocol.json',p)
    base = args[0]
    with np.load(REF/'readout.npz',allow_pickle=False) as f:
        readout = latent.device_readout(base,(f['weight'],f['bias']))
    files,prefix,max_error = [],{},np.zeros(2)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no unsealed outcomes')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('no unsealed HDF5 truth'))):
        for rank,sid in enumerate(p['order'],1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,archive)
            if rank<=p['startup']:
                parent = arc['bias_only']
                arm = dict.fromkeys(('legacy_common',*MODES),parent)
                d = dict(q=np.empty(0,np.int64),eligible=np.empty(0,bool),
                         matched=np.empty((0,len(GROUPS)),bool),dose=np.zeros_like(parent))
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signal changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                arm,d,err = predict(signal,args,decoder,path,readout,arc)
                max_error = np.maximum(max_error,err)
                if rank in (21,120,360) or sid in CASES or only is not None:
                    n = min(len(raw['times']),650)
                    past,arc_past = trimmed(signal,arc,n)
                    a,short,_ = predict(past,args,decoder,path,readout,arc_past)
                    for k in ('q','eligible','matched'):
                        np.testing.assert_array_equal(short[k],d[k][:len(short[k])])
                    np.testing.assert_allclose(short['dose'],d['dose'][:n],atol=2e-6,rtol=0.)
                    for k,v in a.items():
                        np.testing.assert_allclose(v,arm[k][:n],atol=3e-6,rtol=0.)
                        for th in p['calibration'].values():
                            np.testing.assert_array_equal(v>=th,arm[k][:n]>=th)
                    prefix[str(sid)] = dict(points=n, max_probability=max(float(abs(v-arm[k][:n]).max())
                                                                        for k,v in a.items()))
            fn = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/fn).open('xb') as f:
                np.savez_compressed(f,**{k:arc[k] for k in embed.OLD},**arm,**d)
            files.append(fn)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(stage='diagnostic_feedback_sealed',rank=rank,seconds=round(monotonic()-start,2)),flush=True)
    write(out/'seal.json',dict(shots=files,prefix=prefix,error=max_error.tolist(),seconds=monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if p['only'] is not None or s['shots']!=[f'{i:04d}_{v}.npz' for i,v in enumerate(p['order'],1)]:
        raise ValueError('complete sealed stream required')
    source = read(REF/'per_shot.json')
    tab = {c:{k:[] for k in ARMS} for c in p['calibration']}
    ranking = {k:[] for k in ARMS}
    hits,summary,cold_fp,core_err = [],[],0,0.
    for r,(sid,fn) in enumerate(zip(p['order'],s['shots'],strict=True),1):
        y,down,t = group.hdf._raw_truth(sid)
        cold = r<=p['startup']
        with np.load(out/'shots'/fn,allow_pickle=False) as f, np.load(REF/'shots'/fn,allow_pickle=False) as ref:
            z = dict(f)
            np.testing.assert_array_equal(z['dose'],ref['dose'])
            same = float(abs(z['core_current']-ref['orthogonal_current']).max())
            core_err = max(core_err,same)
        common = z['dose']>0.
        arms = {k:z[k] for k in ARMS}
        for a in MODES:
            np.testing.assert_array_equal(arms[a][~common],z['bias_only'][~common])
        for cap,rows in tab.items():
            th = p['original'] if cold else p['calibration'][cap]
            for a,curve in arms.items():
                v = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,curve,y,down,th))
                if a not in MODES and v!=source[cap][a][r-1]:
                    raise ValueError('reference arm changed')
                rows[a].append(v)
        ref = tab['0.2']['bias_only'][-1]
        if cold:
            cold_fp += int(not y and ref['alarm'])
            continue
        for a,curve in arms.items():
            ranking[a].append(dict(disruptive=y,peak=peak_for_shot(t,curve,y,down)))
        summary.append(dict(shot=sid,points=int(common.sum()),windows=int(z['eligible'].sum()),
                            matched=dict(zip(NAMES,z['matched'].sum(axis=0).tolist(),strict=True))))
        old = tab['0.2']['current_5'][-1]
        if y and not old['timely']:
            ms = np.round((down-t)*1000.,9)
            at = (ms>30.) & (ms<=125.) & common
            tau = p['calibration']['0.2']
            hits.append(dict(shot=sid,kind='late' if old['alarm'] else 'never',supported=int(at.sum()),
                             near_cross={k:bool((v[at]>=tau).any()) for k,v in arms.items()}))
    scores = {c:{a:lead.summarize(v) for a,v in tables.items()} for c,tables in tab.items()}
    compared = {c:{f'{a}__to__{b}':dict(flips=flips(t[a],t[b]),timing=lead.paired(t[a],t[b]))
                   for a,b in PAIRS} for c,t in tab.items()}
    same_budget = {str(n):{k:{i:r[i] for i in ('tp','fp')} for k,v in ranking.items()
                      for r in [best_at_count(v,n-cold_fp)]} for n in (18,27,36)}
    write(out/'per_shot.json',tab)
    write(out/'result.json',dict(metrics=scores,paired=compared,hard_cases=hits,support=summary,
                                suffix_ranking_only=same_budget,raw_verified=len(ARMS)*len(tab)*len(s['shots']),
                                max_core_diff_vs_previous=core_err))
    print({c:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in m.items()} for c,m in scores.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--score',action='store_true')
    mode.add_argument('--smoke-rank',action='append',type=int)
    a = cli.parse_args()
    if a.score:
        score(a.output.resolve())
    else:
        run(a.output.resolve(),a.smoke_rank)
