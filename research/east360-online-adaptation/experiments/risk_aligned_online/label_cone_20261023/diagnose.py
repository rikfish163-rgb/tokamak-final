"""Separate stale history derivatives from finite-step nonlinear error.

Historical-loss extremes choose probes AFTER the sealed experiment. No TEST
outcomes enter this diagnostic. Fresh-tangent candidates are local probes,
not deployed stream results and not an improvement claim.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..history_tangent_20261023.learning import flat,logit,apply_step,matched_step
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,Cone,update_all,history_bce
from .run import load_geometry


def main(root):
    p,v=READ(root/'protocol.json'),READ(root/'verification.json')
    assert v['status']=='VERIFIED' and v['completed_sha256']==sha(root/'completed.json')
    scores=[]
    for sid in p['contract']['shot_ids']:
        for r in READ(root/'shots'/f'{sid}_updates.json')['signed']:
            if r['gradient_steps']:
                scores.append((sid,r['first_affected_prediction'],float(np.mean(r['history_bce_change'])),
                               float(np.max(r['history_bce_change'])),r['parent_distance']))
    selected={scores[0][:2]:'first_stream_update'}
    for column,name in ((2,'largest_mean_history_harm'),(3,'largest_single_anchor_harm'),(4,'largest_parent_distance')):
        row=max(scores,key=lambda x:x[column]);key=row[:2]
        selected[key]=selected.get(key,'')+' '+name
    torch.set_num_threads(2);model,_=load_model();original=state_cpu(model);cones,anchors,_=load_geometry()
    ref=Path(p['reference']);base=Path(READ(ref/'protocol.json')['reference'])
    mapping=READ(base/'input_map.json')
    parent_geometry=torch.load(ref/'geometry.pt',map_location='cuda',weights_only=True)
    parent_rows=parent_geometry['rows'].double();parent_z=anchors['logits'].double()
    labels=anchors['labels'].double();cases=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('diagnostic cannot read TEST outcomes')):
        for (sid,probe),selection in sorted(selected.items()):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            states={a:initial_encoder(model) for a in ARMS};stored=READ(root/'shots'/f'{sid}_updates.json')
            for index,count in enumerate(range(256,probe+1,128)):
                before=states
                states,receipts=update_all(model,signal,count,states,cones,anchors,device='cuda')
                for a in ARMS:assert receipts[a]==stored[a][index],(sid,count,a)
            fast=before['signed'];after=states['signed'];r=receipts['signed']
            actual_step=flat([fast[k].detach()-after[k].detach() for k in fast]).double()
            z=logit(model,anchors['x'],anchors['mask'],fast)
            rows=torch.stack([flat(torch.autograd.grad(z[j],tuple(fast.values()),retain_graph=True)).detach()
                              for j in range(len(labels))]).double()
            old_first_order=-((parent_z.sigmoid()-labels)[:,None]*parent_rows)@actual_step
            new_first_order=-((z.detach().double().sigmoid()-labels)[:,None]*rows)@actual_step
            actual=np.asarray(r['history_bce_change'])
            cosine=(parent_rows*rows).sum(1)/(parent_rows.norm(dim=1)*rows.norm(dim=1))
            x,m,target,tm,_=forecast_examples(signal,probe,model.cfg,'cuda')
            loss=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],target,tm)
            g=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach()
            fresh=Cone(rows,anchors['labels']);direction,solver=fresh.project(g)
            candidate=apply_step(fast,matched_step(direction,r['requested_step_norm']))
            fresh_change=(history_bce(model,anchors,candidate)-history_bce(model,anchors,fast)).cpu().numpy()
            entry=dict(shot=sid,count=probe,selection=selection.strip(),exact_prefix_update_replay=True,
                original_history_bce_change=actual.tolist(),parent_first_order=old_first_order.cpu().tolist(),
                current_first_order=new_first_order.cpu().tolist(),
                finite_step_residual=(actual-new_first_order.cpu().numpy()).tolist(),
                jacobian_cosine=cosine.cpu().tolist(),fresh_same_length_history_bce_change=fresh_change.tolist(),
                fresh_solver=solver,original_step_norm=r['parameter_delta_norm'],
                fresh_step_norm=float(flat([candidate[k].detach()-fast[k].detach() for k in fast]).double().norm()),
                parent_distance_before=float(flat([fast[k].detach()-dict(model.named_parameters())[k].detach() for k in fast]).double().norm()),
                counts=dict(actual_worse=int((actual>1e-6).sum()),old_first_order_worse=int((old_first_order>1e-6).sum()),
                    current_first_order_worse=int((new_first_order>1e-6).sum()),fresh_worse=int((fresh_change>1e-6).sum())),
                mean=dict(actual=float(actual.mean()),old_first_order=float(old_first_order.mean()),
                    current_first_order=float(new_first_order.mean()),fresh=float(fresh_change.mean())))
            cases.append(entry);print(json.dumps({k:entry[k] for k in ('shot','count','selection','counts','mean')}),flush=True)
    assert_unchanged(model,original)
    report=dict(status='POSTHOC_LOCAL_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(root/'verification.json'),code_sha256=sha(Path(__file__)),
        selection='first update and historical-loss/parameter-distance extremes, not representative random sample',
        test_outcome_access=False,cases=cases,
        scope='fresh-tangent one-step historical probes only; no fresh-tangent TEST360 result')
    _durable_json(root/'local_derivative_diagnostic.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
