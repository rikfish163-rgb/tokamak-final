"""Post-hoc future probes only: never trains or selects a deployable step."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..feedback_gradient_20261023.geometry import lead_ms
from ..heldout_threshold_20261023.run import models_for_calibration
from ..history_tangent_20261023.learning import flat,logit
from ..matured_negative_20261023.historical_control import PRIMARY,PRIMARY_SHA,REFERENCE
from ..matured_negative_20261023.learning import adapt,matured_indices,tensors
from ..matured_negative_20261023.run import historical_replay
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import relative_loss,step


def probe_counts(rows,probe_index):
    counts=[r['first_affected_prediction'] for r in rows]
    if any(a>=b for a,b in zip(counts,counts[1:])):raise ValueError('strictly ordered update receipts required')
    before=[c for c in counts if c<=probe_index]
    return sorted(set(before[:1]+before[-2:]))


def main(output):
    if output.exists():raise FileExistsError(output)
    assert sha(PRIMARY/'completed.json')==PRIMARY_SHA
    p=READ(PRIMARY/'protocol.json');done=READ(PRIMARY/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(PRIMARY/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    rows=READ(PRIMARY/'per_shot.json')
    a={r['shot']:r for r in rows['static/calibrated']['predictions']}
    b={r['shot']:r for r in rows['matured_plus_replay/calibrated']['predictions']}
    ids=[sid for sid in a if a[sid]['timely'] and not b[sid]['timely']]
    assert ids==[64581,69768,71013,71415,73653]
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];initial=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');positive,pm=historical_replay(p['preparation_ids'],mapping)
    assert pm==p['replay_manifest'];results=[];replayed=0
    def deny(*args,**kwargs):raise AssertionError('no outcome read during prefix replay')
    for sid in ids:
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
        leads=lead_ms(s['times'],s['down']);eligible=np.flatnonzero(s['valid']&(leads>30)&(leads<=125))
        with np.load(PRIMARY/'shots'/f'{sid}.npz') as z:
            e=int(eligible[np.argmax(z['static'][eligible])]);original_peak=float(z['static'][e])
        signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
        receipts=READ(PRIMARY/'shots'/f'{sid}_updates.json')['matured_plus_replay'];selected=probe_counts(receipts,e)
        # Future window is deliberately separated and used only to diagnose the
        # effect of already-defined updates; it is not passed into adapt().
        probe=tensors(signal,[e],'cuda');fast=initial_encoder(model)
        with patch.object(io23,'read_outcome',side_effect=deny):
            for expected in receipts:
                count=expected['first_affected_prediction']
                if not selected or count>max(selected):break
                prefix={k:v[:count] for k,v in signal.items()};now=float(prefix['times'][-1])
                snapshot=None
                if count in selected:
                    ends=matured_indices(prefix['times'],event_free_through=now)
                    xn,mn=tensors(prefix,ends,'cuda')
                    neg=F.softplus(logit(model,xn,mn,fast)).mean()
                    pos=F.softplus(-logit(model,*positive,fast)).mean()
                    gn=flat(torch.autograd.grad(.5*neg,tuple(fast.values()))).detach().double()
                    gp=flat(torch.autograd.grad(.5*pos,tuple(fast.values()))).detach().double()
                    target=logit(model,*probe,fast).squeeze()
                    target_value=float(target.detach());gz=flat(torch.autograd.grad(target,tuple(fast.values()))).detach().double()
                    g=gn+gp;scale=.001/max(1.,float(g.norm()))
                    rankloss=relative_loss(logit(model,xn,mn,fast),logit(model,*positive,fast))
                    gr=torch.autograd.grad(rankloss,tuple(fast.values()))
                    candidate,actual,norm=step(fast,gr,length=expected['parameter_delta_norm'])
                    with torch.no_grad():rank_after=float(logit(model,*probe,candidate).squeeze())
                    snapshot=dict(shot=sid,count=count,probe_index=e,probe_lead_ms=float(leads[e]),
                        current_lead_ms=float(leads[count-1]),future_gap_ms=float(round((s['times'][e]-now)*1000,9)),
                        original_static_peak=original_peak,probe_logit_before=target_value,
                        negative_loss=float(neg.detach()),positive_loss=float(pos.detach()),
                        negative_linear_probe_change=float(-scale*torch.dot(gn,gz)),
                        positive_linear_probe_change=float(-scale*torch.dot(gp,gz)),
                        total_linear_probe_change=float(-scale*torch.dot(g,gz)),
                        component_gradient_cosine=float(torch.dot(gn,gp)/(gn.norm()*gp.norm())) if gn.norm()>0 and gp.norm()>0 else None,
                        rank_matched_actual_probe_change=rank_after-target_value,rank_matched_step=actual,rank_gradient_norm=norm,
                        negative_origin_lead_min=float(leads[ends].min()),negative_origin_lead_max=float(leads[ends].max()))
                new,r=adapt(model,prefix,fast,mode='matured_plus_replay',replay=positive,event_free_through=now,device='cuda')
                assert r==expected,(sid,count)
                if snapshot is not None:
                    with torch.no_grad():after=float(logit(model,*probe,new).squeeze())
                    snapshot['bce_actual_probe_change']=after-snapshot['probe_logit_before']
                    snapshot['bce_step']=r['parameter_delta_norm'];results.append(snapshot)
                fast=new;replayed+=1
    assert_unchanged(model,initial);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(scope='POSTHOC_FUTURE_PROBE_DIAGNOSIS_ONLY; NOT A CAUSAL POLICY OR NEW PERFORMANCE RESULT',
                primary_sha256=PRIMARY_SHA,code_sha256=sha(Path(__file__)),shots=ids,replayed_updates=replayed,
                selection='static peak within (30,125]ms; first and last two updates strictly preceding probe',
                results=results,source_unchanged=True)
    _durable_json(output,result)
    print(dict(shots=ids,replayed_updates=replayed,snapshots=len(results),
               negative_component_lowers_probe=sum(r['negative_linear_probe_change']<0 for r in results),
               positive_component_raises_probe=sum(r['positive_linear_probe_change']>0 for r in results),
               bce_step_lowers_probe=sum(r['bce_actual_probe_change']<0 for r in results),
               rank_step_lowers_probe=sum(r['rank_matched_actual_probe_change']<0 for r in results)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);main(parser.parse_args().output)
