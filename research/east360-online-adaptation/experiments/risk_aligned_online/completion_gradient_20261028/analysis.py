"""Posthoc gradient attribution, not an online policy or performance result."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..latent_completion_20261028.model import Imputer,tokens,masked,readout
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform


def selected_ends(length):
    """At most four distinct complete windows; never pad short records.

    This is diagnostic subsampling, not a condition for admitting a shot to
    the online stream. A record shorter than128 has no complete-window probe.
    """
    ends=np.arange(127,length,128,dtype=int)
    return ends[np.linspace(0,len(ends)-1,min(4,len(ends)),dtype=int)] if len(ends) else ends


def loss_parts(prediction,target,valid):
    """Exact MSE split: common token offset + deviations from it + CLS.

    Uses the SAME full-observation validity mask in both views. The dynamic
    term is temporal variation of latent error, not certified plasma dynamics.
    Empty signal-token rows contribute zero, with their CLS still counted.
    """
    if prediction.shape!=target.shape or prediction.shape[1:]!=(17,64) or valid.shape!=prediction.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('matching 17x64 latent fields and boolean token mask required')
    error=prediction-target;mask=valid[:,:-1,None]
    count=valid[:,:-1].sum(1);den=(valid.sum(1)*64).clamp_min(1)
    mean=(error[:,:-1]*mask).sum(1)/count[:,None].clamp_min(1)
    return dict(mean=mean.square().sum(1)*count/den,
                dynamic=((error[:,:-1]-mean[:,None]).square()*mask).sum((1,2))/den,
                cls=error[:,-1].square().sum(1)*valid[:,-1]/den)


def cosine(a,b):
    d=float(a.norm()*b.norm())
    return float(a@b/d) if d>1e-15 else None


def components(base,net,x,a,slow):
    # Independent temporary state per window: summing losses does not mix
    # examples, since there is no batch normalization or cross-example layer.
    state=base.center.new_zeros(len(x),128,requires_grad=True)
    h,v=tokens(base,x,a,state);xx,aa=masked(x,a)
    mh,mv=tokens(base,xx,aa,state.repeat_interleave(2,0))
    target=h.repeat_interleave(2,0);valid=v.repeat_interleave(2,0)
    pred=net(mh,mv)
    losses={k:q.reshape(-1,2).mean(1) for k,q in loss_parts(pred,target,valid).items()}
    losses['full']=sum(losses.values())
    losses['prediction_path']=sum(loss_parts(pred,target.detach(),valid).values()).reshape(-1,2).mean(1)
    losses['target_path']=sum(loss_parts(pred.detach(),target,valid).values()).reshape(-1,2).mean(1)
    rev_h=torch.cat((mh[:,:-1].flip(1),mh[:,-1:]),1)
    rev_v=torch.cat((mv[:,:-1].flip(1),mv[:,-1:]),1)
    rev=net(rev_h,rev_v)
    rev_parts=loss_parts(rev,target,valid)
    losses['reverse']=sum(rev_parts.values()).reshape(-1,2).mean(1)
    losses['reverse_dynamic']=rev_parts['dynamic'].reshape(-1,2).mean(1)
    gradients={k:torch.autograd.grad(q.sum(),state,retain_graph=True)[0].detach().cpu() for k,q in losses.items()}
    z=readout(base,h,v,slow)
    risk_gradient=torch.autograd.grad(z.sum(),state)[0].detach().cpu()
    if not all(torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('nonfinite attribution')
    torch.testing.assert_close(gradients['full'],gradients['prediction_path']+gradients['target_path'],atol=1e-7,rtol=2e-5)
    torch.testing.assert_close(gradients['full'],gradients['mean']+gradients['dynamic']+gradients['cls'],atol=1e-7,rtol=2e-5)
    losses={k:q.detach().cpu().tolist() for k,q in losses.items()};zs=z.detach().cpu().tolist()
    rows=[]
    for i in range(len(x)):
        gs={k:g[i] for k,g in gradients.items()};full=gs['full'];norm=float(full.norm())
        rows.append(dict(loss={k:q[i] for k,q in losses.items()},norm={k:float(g.norm()) for k,g in gs.items()},
            cosine_to_full={k:cosine(g,full) for k,g in gs.items()},
            fraction_along_full={k:float(g@full)/(norm*norm) if norm>1e-15 else None for k,g in gs.items()},
            normalized_step_logit_derivative={k:-.05*float(g@risk_gradient[i])/max(float(g.norm()),1e-12) for k,g in gs.items()},
            prediction_target_cosine=cosine(gs['prediction_path'],gs['target_path']),
            dynamic_reverse_cosine=cosine(gs['dynamic'],gs['reverse_dynamic']),logit=zs[i]))
    return rows


def summarize(records):
    fields=('loss','norm','cosine_to_full','fraction_along_full','normalized_step_logit_derivative')
    out=dict(n=len(records))
    for field in fields:
        out[field]={k:float(np.median([r[field][k] for r in records if r[field][k] is not None]))
                    for k in records[0][field]}
    for k in ('prediction_target_cosine','dynamic_reverse_cosine'):
        values=[r[k] for r in records if r[k] is not None];out[k]=float(np.median(values)) if values else None
    return out


def run(root,source_root,reference):
    if root.exists() or root.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree diagnostic output required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,_=load_parent(DEFAULT_PARENT);base.cuda();initial=classifier_state(base)
    sd=read(source_root,'complete.json');path=source_root/'task.pt'
    if sha(path)!=sd['checkpoints']['task']:raise ValueError('source imputer changed')
    net=Imputer().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    prior=read(reference,'protocol.json');order=read(reference,'order.json');ids=[r['shot'] for r in order]
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    if len(ids)!=360 or prior['warmup']!=20 or mapping['calibration_ids']!=ids[:20]:raise ValueError('fixed stream required')
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    root.mkdir();start=time.monotonic();records=[];coverage=[]
    protocol=dict(scope='POSTHOC_ZERO_STATE_GRADIENT_ATTRIBUTION_NOT_ONLINE_RESULT',
        order=ids,excluded_cold20=ids[:20],source_root=str(source_root),reference=str(reference),
        source_sha256=PARENT_SHA256,task_sha256=sha(path),code_sha256=sha(Path(__file__)),
        reference_hashes={k:sha(reference/k) for k in ('protocol.json','order.json','input_map.json','models/cold.pt')},
        endpoint_selection='up to four equally spaced distinct complete128-sample blocks per file; selected BEFORE outcome read',
        temporary_state='zero at each diagnostic window, NOT actual accumulated online state',
        outcome_use='posthoc validity and grouping only; never gradient construction')
    _durable_json(root/'protocol.json',protocol)
    def deny(*args,**kwargs):raise AssertionError('outcome used for selecting or differentiating windows')
    for i,sid in enumerate(ids[20:],21):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('input changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            selected=selected_ends(len(sig['values']))
            local=[]
            if len(selected):
                x,a=batch(sig,selected,base.center.device)
                for end,r in zip(selected,components(base,net,x,a,slow)):
                    local.append(dict(shot=sid,index=i,end=int(end),time=float(raw['times'][end]),**r))
            coverage.append(dict(shot=sid,index=i,samples=len(sig['values']),ends=selected.tolist()))
        flag,down=io23.read_outcome('EAST',sid)
        for r in local:
            lead=(down-r['time'])*1000 if flag else None
            r.update(disruptive=flag,lead_ms=lead,preterminal=not flag or lead>=-1e-6,
                endpoint_label=int(flag and -1e-6<=lead<=125))
        records.extend(local)
        if i%40==0:
            _durable_json(root/'partial_records.json',records)
            _durable_json(root/'coverage.json',coverage)
            print(dict(shots=i,seconds=time.monotonic()-start),flush=True)
    valid=[r for r in records if r['preterminal']]
    groups={g:summarize(sub) for g,sub in (
        ('all_preterminal',valid),('D',[r for r in valid if r['disruptive']]),
        ('S',[r for r in valid if not r['disruptive']]),
        ('positive_window',[r for r in valid if r['endpoint_label']])) if sub}
    assert_classifier_unchanged(base,initial)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('source changed')
    _durable_json(root/'records.json',records)
    _durable_json(root/'coverage.json',coverage)
    _durable_json(root/'complete.json',dict(complete=True,groups=groups,seconds=time.monotonic()-start,
        total_windows=len(records),preterminal_windows=len(valid),source_unchanged=True,
        no_online_performance_claim=True,records_sha256=sha(root/'records.json'),
        coverage_sha256=sha(root/'coverage.json'),shots_with_fewer_than_four_windows=[r for r in coverage if len(r['ends'])<4]))
    print(groups,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_root,a.reference)
