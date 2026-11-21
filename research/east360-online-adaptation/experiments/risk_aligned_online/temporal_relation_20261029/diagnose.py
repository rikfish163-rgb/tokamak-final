"""Posthoc one-step task/risk alignment; never used to choose online updates."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..completion_gradient_20261028.analysis import selected_ends
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..latent_completion_20261028.model import tokens,readout
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import OrderJudge,margin,permutation


def probe(base,net,x,a,future,fa,p,slow):
    s=base.center.new_zeros(len(x),128,requires_grad=True)
    delta,eligible,tv=margin(base,net,x,a,p,s)
    g,=torch.autograd.grad(F.softplus(-delta).sum(),s,retain_graph=True)
    gt,=torch.autograd.grad(F.softplus(-tv).sum(),s)
    h,v=tokens(base,future,fa,s);z=readout(base,h,v,slow)
    gz,=torch.autograd.grad(z.sum(),s)
    change=-.05*g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        new_delta,_,_=margin(base,net,x,a,p,change)
        nh,nv=tokens(base,future,fa,change);nz=readout(base,nh,nv,slow)
    rows=[]
    for i in range(len(x)):
        norm=float(g[i].norm());tn=float(gt[i].norm())
        rows.append(dict(eligible=bool(eligible[i]),margin=float(delta[i].detach()),updated_margin=float(new_delta[i]),
            roughness_margin=float(tv[i].detach()),gradient_norm=norm,
            gradient_roughness_cosine=float(g[i]@gt[i])/(norm*tn) if norm*tn>1e-15 else None,
            future_logit_derivative=float(change[i]@gz[i]),future_logit_change=float(nz[i]-z[i].detach())))
    return rows


def summarize(rows):
    rs=[r for r in rows if r['eligible']]
    if not rs:return dict(windows=len(rows),eligible=0)
    cos=[r['gradient_roughness_cosine'] for r in rs if r['gradient_roughness_cosine'] is not None]
    by={}
    for r in rs:by.setdefault(r['shot'],[]).append(r)
    return dict(windows=len(rows),eligible=len(rs),shots=len(by),
        shot_mean_pair_accuracy=float(np.mean([np.mean([1. if r['margin']>0 else .5 if r['margin']==0 else 0. for r in sub]) for sub in by.values()])),
        shot_mean_roughness_accuracy=float(np.mean([np.mean([1. if r['roughness_margin']>0 else .5 if r['roughness_margin']==0 else 0. for r in sub]) for sub in by.values()])),
        auxiliary_margin_increased=sum(r['updated_margin']>r['margin'] for r in rs),
        future_risk_increased=sum(r['future_logit_change']>0 for r in rs),
        median_future_logit_change=float(np.median([r['future_logit_change'] for r in rs])),
        median_roughness_gradient_cosine=float(np.median(cos)) if cos else None)


def run(root):
    if (root/'diagnosis.json').exists():raise FileExistsError(root/'diagnosis.json')
    audit=read(root,'audit.json');p=read(root,'protocol.json')
    if not audit['complete'] or audit['complete_sha256']!=sha(root/'complete.json'):raise ValueError('audited360 required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,_=load_parent(DEFAULT_PARENT);base.cuda();source=Path(p['source_root']);reference=Path(p['reference'])
    if sha(source/'judge.pt')!=p['judge_sha256']:raise ValueError('auxiliary changed')
    net=OrderJudge().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(source/'judge.pt',map_location='cpu',weights_only=True)['model'])
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    cold={r['shot']:r for r in read(root,'per_shot.json')['cold']};records=[];coverage=[]
    def deny(*args,**kwargs):raise AssertionError('probe must not use current outcome')
    for sid in p['order'][20:]:
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw input changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            # Diagnostic subsampling uses file length, not an online policy.
            # Every query32 samples later is scored with a state learned ONLY
            # from its preceding observed window; no lookahead in the step.
            ends=selected_ends(len(sig['values'])-32)
            if not len(ends):raise ValueError('this fixed cohort needs a complete-window probe')
            x,a=batch(sig,ends,'cuda');fx,fa=batch(sig,ends+32,'cuda')
            perm=permutation(len(ends),np.random.default_rng(2718))
            rows=probe(base,net,x,a,fx,fa,perm,slow)
        flag,down=io23.read_outcome('EAST',sid)
        for e,r in zip(ends,rows):
            lead=float(round((down-raw['times'][e+32])*1000,9)) if flag else None
            r.update(shot=sid,observed_end=int(e),query_end=int(e+32),flag=flag,lead_ms=lead,
                timely_positive=bool(lead is not None and 30<lead<=125),cold_missed=bool(flag and not cold[sid]['timely']))
            records.append(r)
        coverage.append(dict(shot=sid,ends=ends.tolist()))
    groups={name:summarize(sub) for name,sub in (
        ('all',records),('safe',[r for r in records if not r['flag']]),
        ('timely_positive',[r for r in records if r['timely_positive']]),
        ('cold_missed_timely_positive',[r for r in records if r['timely_positive'] and r['cold_missed']]))}
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('source changed')
    result=dict(scope='POSTHOC_SINGLE_STEP_ZERO_STATE_DIAGNOSIS_NOT_AN_ONLINE_RESULT',
        stream_complete_sha256=sha(root/'complete.json'),query_offset_samples=32,coverage=coverage,groups=groups,records=records,
        code_sha256=sha(Path(__file__)),online_policy_changed=False,
        warning='uniform probes use file length; terminal labels only group after probing; no physical-causality claim')
    _durable_json(root/'diagnosis.json',result);print(groups,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
