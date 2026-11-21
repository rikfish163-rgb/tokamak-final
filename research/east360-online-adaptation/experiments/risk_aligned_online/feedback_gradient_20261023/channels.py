"""Post-hoc four-case attribution of change-gradient/risk-gradient inner products."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import forecast_examples,target_mask,update
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .geometry import lead_ms,vector_geometry


def inspect(model,case,mapping):
    sid,count=case['shot'],case['count']
    raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    signal=transform({k:raw[k] for k in ('values','available')},mapping)
    fast=initial_encoder(model)
    for expected in READ(REFERENCE/'fold0/shots'/f'{sid}_updates.json')['mapped_fixed']:
        n=expected['first_affected_prediction']
        if n>=count: break
        fast,r=update(model,{k:v[:n] for k,v in signal.items()},n,fast,device='cuda')
        assert r==expected
    outcome=simple_data.attach_outcome(raw,'EAST',{})
    inds=np.arange(count,min(count+128,len(signal['values'])))
    keep=outcome['valid'][inds].copy()
    if outcome['is_disrupt']: keep &= lead_ms(outcome['times'][inds],outcome['down'])>30.
    windows=[simple_data.window(signal,int(e),128) for e in inds]
    xx,mm=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    z=torch.cat([logit(model,xx[lo:lo+32],mm[lo:lo+32],fast) for lo in range(0,len(xx),32)])
    peak=z[torch.as_tensor(keep,device='cuda')].max()
    rg=torch.autograd.grad(F.binary_cross_entropy_with_logits(peak,peak.new_tensor(float(outcome['is_disrupt']))),tuple(fast.values()))
    x,m,truth,tm,_=forecast_examples({k:v[:count] for k,v in signal.items()},count,model.cfg,'cuda')
    mask=target_mask(x,m,truth,tm)
    center,scale=model.base.center,model.base.scale
    target=torch.asinh((torch.where(mask,truth,center)-center)/scale)
    flat=torch.asinh((torch.where(mask,x[:,-1:].expand_as(truth),center)-center)/scale)
    prediction=model.predict_fast(x,m,fast)['forecast']
    per_channel=((F.smooth_l1_loss(prediction,target,reduction='none')-
        F.smooth_l1_loss(prediction,flat,reduction='none')).masked_fill(~mask,0.)).sum((0,1))/mask.sum()
    full=torch.autograd.grad(per_channel.sum(),tuple(fast.values()),retain_graph=True)
    norm=float(sum(g.square().sum() for g in full).sqrt());step=.001/max(norm,1.)
    total=vector_geometry(full,rg)
    assert np.isclose(-step*total['dot'],case['candidates']['change']['first_order_change'],rtol=2e-4,atol=1e-8)
    rows=[];summed=[torch.zeros_like(g) for g in full]
    for c,name in enumerate(io23.RAW23):
        cg=torch.autograd.grad(per_channel[c],tuple(fast.values()),retain_graph=True)
        for a,b in zip(summed,cg): a.add_(b)
        geom=vector_geometry(cg,rg)
        rows.append(dict(feature=name,observed=int(mask[:,:,c].sum()),gradient_norm=geom['norm'],
            predicted_risk_loss_change=-step*geom['dot']))
    for a,b in zip(summed,full): torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-7)
    assert np.isclose(sum(r['predicted_risk_loss_change'] for r in rows),-step*total['dot'],rtol=3e-4,atol=1e-8)
    return dict(shot=sid,count=count,phase=case['phase'],
        actual_change=case['candidates']['change']['actual_risk_loss_change'],
        first_order_change=-step*total['dot'],channels=sorted(rows,key=lambda r:r['predicted_risk_loss_change'],reverse=True))


def main(root):
    output=root/'channel_attribution.json'
    if output.exists(): raise FileExistsError(output)
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    assert not (root/'INVALIDATED_DIAGNOSTIC.md').exists()
    for name,digest in done['artifact_sha256'].items(): assert sha(root/name)==digest,name
    all_rows=[r for sid in p['shot_ids'] for r in READ(root/'shots'/f'{sid}.json')]
    selected=[]
    for phase in ('safe','actionable_30_125'):
        rs=[r for r in all_rows if r['status']=='MEASURED' and r['phase']==phase and not r['alarm_already_emitted']]
        selected.extend([min(rs,key=lambda r:r['candidates']['change']['actual_risk_loss_change']),
                         max(rs,key=lambda r:r['candidates']['change']['actual_risk_loss_change'])])
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');cases=[inspect(model,r,mapping) for r in selected]
    assert_unchanged(model,before)
    result=dict(status='POSTHOC_FOUR_CASE_GRADIENT_ATTRIBUTION',completed_sha256=sha(root/'completed.json'),
        code_sha256=sha(Path(__file__)),selection='largest beneficial/adverse one-step change per safe and timely-disruption group; no alarm emitted',
        cases=cases,note='Mathematical one-step parameter-gradient contribution, not physical causality or a feature-removal recommendation.')
    _durable_json(output,result)
    print([dict(shot=r['shot'],count=r['count'],actual_change=r['actual_change'],
                harmful=r['channels'][:3],helpful=r['channels'][-3:]) for r in cases],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
