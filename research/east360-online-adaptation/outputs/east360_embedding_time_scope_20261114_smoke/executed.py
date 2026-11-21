"""Locate the time-local part of a frozen archival numeric-embedding gain."""

import argparse
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_target_separability_20261114 as target
from . import io23, warning_lead_metrics as lead
from .commissioning_coincident_support_audit_20261111 import _raw_truth
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot, timely_auc


REFERENCE = io23.ROOT/'outputs/east360_target_separability_20261114_v4_value_scope'
ARCHIVED = 'all_values_fixed_head'
NULL,FULL = 'no_numeric_update','full'
WINDOW,PATCH,START,STRIDE = target.WIDTH,8,2*target.WIDTH,target.STRIDE
ARMS = (NULL,FULL,'first_level','past_level','current_departure','wrong_patch_order','early96','latest32')
PAIRS = ((NULL,FULL),('past_level',FULL),('first_level',FULL),('wrong_patch_order',FULL),
         ('latest32',FULL),('early96',FULL))
CASES = (90874,70320,73405,81406,74970,85422,73743)


def past_windows(values, available, endpoints):
    return target.windows(values,available,endpoints-WINDOW)


def channel_mean(values, available):
    return values.sum(dim=-2)/available.sum(dim=-2).clamp_min(1)


def counterfactual_values(v,m,old,old_m,first):
    """Return what the fitted *delta* sees; the source projection sees real v."""
    past = channel_mean(old,old_m)[:,None].expand_as(v).masked_fill(~m,0.)
    def reverse(x):
        return x.unflatten(1,(-1,PATCH)).flip(1).flatten(1,2)
    # Exchange positions only on paired observations. This is an involution:
    # missingness, the channel value distribution and within-patch order survive.
    wrong = torch.where(m & reverse(m),reverse(v),v)
    near = torch.arange(WINDOW,device=v.device)>=WINDOW-32
    return dict(first_level=first.expand_as(v).masked_fill(~m,0.),
                past_level=past,current_departure=v-past,wrong_patch_order=wrong,
                early96=v.masked_fill(near[:,None],0.),latest32=v.masked_fill(~near[:,None],0.))


def models():
    base,_ = target.load_parent(target.DEFAULT_PARENT)
    trained,_ = target.load_parent(target.DEFAULT_PARENT)
    target.train_scope(trained,ARCHIVED)
    state = torch.load(REFERENCE/f'{ARCHIVED}.pt',map_location='cpu',weights_only=True)['state']
    trained.load_state_dict(state)
    for name,value in base.state_dict().items():
        if name!='embedding.weight' and not torch.equal(value,state[name]):
            raise ValueError(f'archival fit changed {name} outside the numeric embedding')
    delta = trained.embedding.weight.detach()-base.embedding.weight.detach()
    permitted = torch.isin(torch.arange(delta.shape[1])%46,torch.tensor(target.NUMERIC[ARCHIVED]))
    if (base.cfg.input_view,base.cfg.window,base.cfg.patch,delta.shape[1])!=('source',WINDOW,PATCH,368):
        raise ValueError('requires the archived flat23 source-scale embedding')
    if not delta.any() or delta[:,~permitted].any():
        raise ValueError('numeric correction must exclude masks and frozen channels')
    weight = delta.unflatten(1,(PATCH,46))[:,:,:23].flatten(1)
    return (base.cuda().eval().requires_grad_(False), trained.cuda().eval().requires_grad_(False),
            weight.contiguous().cuda())


def measured(base,x,mask):
    return ((torch.where(mask,x,base.center)-base.center)/base.scale).asinh()


@torch.no_grad()
def curves(base,trained,weight,shot):
    out = []
    take = np.arange(0,len(shot['times']),STRIDE)
    x,m = shot['values'],shot['available']
    device = base.center.device
    for section in (take[take<START],take[take>=START]):
        for start in range(0,len(section),128):
            i = section[start:start+128]
            def to_device(seq):
                return tuple(torch.as_tensor(a,device=device) for a in seq)
            values,mask = to_device(target.windows(x,m,i))
            nominal = base(values,mask)['logit']
            full = trained(values,mask)['logit']
            if i[0]<START:
                each = full[None].expand(len(ARMS)-2,-1)
            else:
                a,ok = to_device(past_windows(x,m,i))
                a0,ok0 = to_device((x[:WINDOW],m[:WINDOW]))
                first = channel_mean(measured(base,a0,ok0),ok0)
                terms = counterfactual_values(measured(base,values,mask),mask,measured(base,a,ok),ok,first)
                correction = torch.stack([terms[k] for k in ARMS[2:]])
                correction = F.linear(correction.reshape(len(terms),len(i),WINDOW//PATCH,-1),weight)

                def replace(_layer,_ins,original):
                    return original+correction.flatten(0,1)

                hook = base.embedding.register_forward_hook(replace)
                try:
                    each = base(values.repeat(len(terms),1,1),mask.repeat(len(terms),1,1))['logit']
                    each = each.unflatten(0,(len(terms),len(i)))
                finally:
                    hook.remove()
            out.append(torch.cat((nominal[None],full[None],each)).sigmoid().cpu().numpy())
    return take,np.concatenate(out,axis=1)


def seal(out,only):
    now = monotonic()
    p,s,selected = (read(REFERENCE/f'{n}.json') for n in ('protocol','seal','selected'))
    if (p['order']!=s['order'] or len(s['files'])!=360 or s['score_stride']!=STRIDE or
            p['source']!=target.PARENT_SHA256 or set(p['order'])&(set(p['train'])|set(p['val']))):
        raise ValueError('requires the fixed archival model with no training overlap')
    calibration = selected['fits'][ARCHIVED]['calibration']
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(order=p['order'],arms=ARMS,ref=str(REFERENCE),only=only,
        calibration=calibration,changed_from_sample=START,stride=STRIDE,
        prior='first 128 samples or earlier disjoint 128; normalized only in fixed source coordinates',
        frozen='original signals and source projection intact; synthetic ablations of archived numeric delta',
        scope='offline supervised attribution only; no fitting, no alarm changes; all labels after seal'))
    base,model,w = models()
    witness,files,parity = {},[],0.
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes forbidden before seal')),
          patch(f'{__name__}._raw_truth',side_effect=RuntimeError('outcomes forbidden before seal'))):
        for rank,(sid,name) in enumerate(zip(p['order'],s['files'],strict=True),1):
            if only is not None and rank not in only:
                continue
            raw = target.load_signal(sid)
            take,pred = curves(base,model,w,raw)
            without_ablation = take<START
            np.testing.assert_array_equal(pred[2:,without_ablation],
                np.broadcast_to(pred[1,without_ablation],pred[2:,without_ablation].shape))
            with np.load(REFERENCE/'shots'/name,allow_pickle=False) as a:
                ref = a[ARCHIVED][take]
            parity = max(parity,float(abs(pred[1]-ref).max()))
            np.testing.assert_allclose(pred[1],ref,atol=3e-6,rtol=0.)
            for fit in calibration.values():
                np.testing.assert_array_equal(pred[1]>=fit['threshold'],ref>=fit['threshold'])
            if rank in (21,120,360) or sid in CASES or only is not None:
                n = min(650,len(raw['times']))
                cut = {k:raw[k][:n] for k in ('values','available','times')}
                j,test = curves(base,model,w,cut)
                np.testing.assert_array_equal(j,take[:len(j)])
                np.testing.assert_allclose(test,pred[:,:len(j)],atol=3e-6,rtol=0.)
                for fit in calibration.values():
                    np.testing.assert_array_equal(test>=fit['threshold'],pred[:,:len(j)]>=fit['threshold'])
                witness[str(sid)] = dict(points=n,max_probability=float(abs(test-pred[:,:len(j)]).max()))
            with (out/'shots'/name).open('xb') as h:
                np.savez_compressed(h,**dict(zip(ARMS,pred,strict=True)))
            files.append(name)
            if only is not None or rank%120==0:
                print(dict(rank=rank,seconds=round(monotonic()-now,2)),flush=True)
    write(out/'seal.json',dict(files=files,prefix=witness,archival_parity=parity,seconds=monotonic()-now))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if p['only'] is not None or len(s['files'])!=360:
        raise ValueError('requires completed fixed360')
    ref = read(REFERENCE/'per_shot.json')
    rows = {c:{k:[] for k in ARMS} for c in p['calibration']}
    peak = {k:[] for k in ARMS}
    early = {c:dict(disruptive=0,safe=0) for c in rows}
    for ix,(sid,name) in enumerate(zip(p['order'],s['files'],strict=True)):
        y,d,t = _raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            vals = {k:np.repeat(f[k],STRIDE)[:len(t)] for k in ARMS}
        for k,v in vals.items():
            peak[k].append(dict(disruptive=y,peak=peak_for_shot(t,v,y,d)))
            for c,row in rows.items():
                threshold = p['calibration'][c]['threshold']
                j = dict(shot=sid,disruptive=y,down_s=d,**score_curve(t,v,y,d,threshold))
                row[k].append(j)
                if k==FULL:
                    if j!=ref[c][ARCHIVED][ix]:
                        raise ValueError(f'archived warning changed on {sid}')
                    eligible = not y or np.round((d-t[:START])*1000.,9)>30.
                    early[c]['disruptive' if y else 'safe'] += int(((v[:START]>=threshold)&eligible).any())
    write(out/'per_shot.json',rows)
    metrics = {c:{k:lead.summarize(a) for k,a in row.items()} for c,row in rows.items()}
    ranking = {str(n):{k:{j:r[j] for j in ('tp','fp')} for k,v in peak.items()
                       for r in (best_at_count(v,n),)} for n in (18,27,36)}
    paired = {c:{f'{a}__to__{b}':flips(v[a],v[b]) for a,b in PAIRS} for c,v in rows.items()}
    write(out/'result.json',dict(metrics=metrics,early_common=early,
        pairs=paired,retrospective_equal_fp_only=ranking,auc={k:timely_auc(v) for k,v in peak.items()},
        raw_rows=len(rows)*360*len(ARMS)))
    print({c:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in table.items()}
           for c,table in metrics.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--smoke-rank',action='append',type=int)
    mode.add_argument('--score',action='store_true')
    a = cli.parse_args()
    if a.score:
        score(a.output.resolve())
    else:
        seal(a.output.resolve(),a.smoke_rank)
