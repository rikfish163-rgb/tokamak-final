"""POST-HOC ONLY: decompose lost 63226 alarm, never an online selection policy.

The future probe is selected from the sealed STATIC curve. It is used solely
to measure consequences, not passed to adapt_arrived or used to accept steps.
Channel derivatives are model sensitivities, NOT physical causal effects.
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
from ..deepcoral_prefix_20261022.learning import adapt_arrived,forecast_examples
from ..deepcoral_prefix_20261022.run import assert_unchanged,state_cpu
from ..history_tangent_20261023.learning import flat,logit
from ..paired_feedback_20261022.model import PairedFeedback,initial_encoder
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

READ=lambda p:json.loads(p.read_text())


def channel_losses(model,prediction,truth,available):
    if prediction.shape!=truth.shape or available.shape!=truth.shape or available.dtype!=torch.bool:
        raise ValueError('matching full forecast targets and masks required')
    observed=available & torch.isfinite(truth)
    clean=torch.where(observed,truth.detach(),model.base.center)
    target=torch.asinh((clean-model.base.center)/model.base.scale)
    error=F.smooth_l1_loss(prediction,target,reduction='none').masked_fill(~observed,0.)
    return error.sum((0,1))/observed.sum().clamp_min(1)


def run(root):
    output=root/'posthoc_63226.json'
    if output.exists():raise FileExistsError(output)
    p,verification=READ(root/'protocol.json'),READ(root/'verification.json')
    assert verification['status']=='VERIFIED' and verification['completed_sha256']==sha(root/'completed.json')
    done=READ(root/'completed.json');folder=root/'fold0'/'raw_bce';path=folder/'shots'/'63226.npz'
    for item in (folder/'parent.pt',folder/'input_map.json',path,folder/'shots'/'63226_updates.json'):
        assert sha(item)==done['artifact_sha256'][str(item.relative_to(root))]
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda()
    model.load_state_dict(torch.load(folder/'parent.pt',map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
    parent=state_cpu(model);mapping=READ(folder/'input_map.json')
    def no_outcome(*args,**kwargs):raise AssertionError('no outcomes inside replay or gradient decomposition')
    with patch.object(io23,'read_outcome',side_effect=no_outcome):
        shot=apply_input_contract(simple_data.read_shot('EAST',63226,rate_contract='nominal'))
        signal=transform({k:shot[k] for k in ('values','available')},mapping)
        with np.load(path,allow_pickle=False) as trace:
            probe=int(np.flatnonzero(trace['static']>=p['threshold'])[0])
            expected_static=float(trace['static'][probe]);expected_intra=float(trace['intra'][probe])
        x,m=[torch.as_tensor(v[None],device='cuda') for v in simple_data.window(signal,probe,128)]
        fast=initial_encoder(model);steps=[];saved=[];old_receipts=READ(folder/'shots'/'63226_updates.json')
        with torch.no_grad():initial=float(logit(model,x,m,fast).sigmoid())
        assert abs(initial-expected_static)<2e-6
        for count in range(256,probe+1,128):
            prefix={k:v[:count] for k,v in signal.items()}
            with torch.no_grad():before=float(logit(model,x,m,fast))
            prior=fast
            fast,receipt=adapt_arrived(model,prefix,count,fast,device='cuda')
            assert receipt==old_receipts[len(steps)]
            with torch.no_grad():after=float(logit(model,x,m,fast))
            steps.append(dict(count=count,probe_logit_before=before,probe_logit_after=after,delta_logit=after-before,
                              forecast_loss_before=receipt['loss_before'],forecast_loss_after=receipt['loss_after']))
            saved.append(prior)
        with torch.no_grad():final=float(logit(model,x,m,fast).sigmoid())
        assert abs(final-expected_intra)<2e-6
        indices=sorted(range(len(steps)),key=lambda i:steps[i]['delta_logit'])[:3]
        decomposition=[]
        for index in indices:
            state=saved[index];count=steps[index]['count'];prefix={k:v[:count] for k,v in signal.items()}
            xx,mm,truth,available,ends=forecast_examples(prefix,count,model.cfg,'cuda')
            prediction=model.predict_fast(xx,mm,state)['forecast']
            parts=channel_losses(model,prediction,truth,available)
            original=model.base.forecast_loss(prediction,truth,available)
            torch.testing.assert_close(parts.sum(),original,rtol=1e-6,atol=1e-7)
            gs=torch.autograd.grad(original,tuple(state.values()),retain_graph=True)
            g=flat(gs).detach()
            norm=sum(v.square().sum() for v in gs).sqrt();factor=float((.001/norm.clamp_min(1.)).detach())
            probe_z=logit(model,x,m,state)
            probe_gradient=flat(torch.autograd.grad(probe_z.sum(),tuple(state.values()))).detach()
            contributions=[];sum_gradient=torch.zeros_like(g)
            for channel,name in enumerate(io23.RAW23):
                gc=flat(torch.autograd.grad(parts[channel],tuple(state.values()),retain_graph=True)).detach()
                sum_gradient+=gc
                contributions.append(dict(channel=name,loss_component=float(parts[channel].detach()),
                    gradient_norm=float(gc.double().norm()),linear_probe_delta=-factor*float(torch.dot(gc.double(),probe_gradient.double()))))
            relative=float((sum_gradient-g).double().norm()/g.double().norm().clamp_min(1e-12))
            assert relative<1e-5
            decomposition.append(dict(**steps[index],probe_end=probe,forecast_origins=ends,gradient_sum_relative_error=relative,
                linear_probe_delta=sum(c['linear_probe_delta'] for c in contributions),channels=contributions))
        assert_unchanged(model,parent)
    # Outcome is read only after numerical replay/decomposition, to describe
    # lead time and the mismatch with the existing125ms point label.
    labeled=simple_data.attach_outcome(shot,'EAST',{})
    assert labeled['is_disrupt']
    report=dict(status='POSTHOC_DIAGNOSTIC',script_sha256=sha(Path(__file__)),
        test_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_preparation_factorial_diagnostic.py'),
        main_completed_sha256=sha(root/'completed.json'),main_verification_sha256=sha(root/'verification.json'),
        shot=63226,condition='raw_bce',probe_end=probe,probe_time=float(shot['times'][probe]),
        lead_ms=round(1000*(labeled['down']-shot['times'][probe]),9),point_training_label=int(labeled['y'][probe]),
        source_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,static_probability=initial,intra_probability=final,
        complete_update_receipts_matched_until_probe=len(steps),steps=steps,largest_decreases=decomposition,
        scope='selected future probe used ONLY for posthoc attribution; not available to online policy; not physical causation')
    _durable_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('steps','largest_decreases')}),flush=True)
    for d in decomposition:
        print(json.dumps(dict(count=d['count'],actual=d['delta_logit'],linear=d['linear_probe_delta'],
            channels=sorted(d['channels'],key=lambda r:r['linear_probe_delta'])[:5])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
