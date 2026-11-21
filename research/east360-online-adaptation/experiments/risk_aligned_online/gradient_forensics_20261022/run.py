"""POSTHOC gradient forensics; target outcomes never enter a deployed policy.

Every changed decision in the completed six-arm experiment is inspected.
Local derivatives describe only the last update before the critical endpoint,
not an additive causal decomposition of all accumulated within-shot updates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..deepcoral_prefix_20261022.learning import forecast_examples, adapt_arrived
from ..warmstart_alignment_20261022.alignment import transform


def channel_losses(prediction,truth,available,center,scale):
    if prediction.shape!=truth.shape or available.shape!=truth.shape or available.dtype!=torch.bool:
        raise ValueError('matching full horizon tensors required')
    mask=available & torch.isfinite(truth)
    target=torch.asinh((torch.where(mask,truth.detach(),center)-center)/scale)
    losses=F.smooth_l1_loss(prediction,target,reduction='none').masked_fill(~mask,0.)
    return losses.sum((0,1))/mask.sum().clamp_min(1)


def gradient_geometry(auxiliary,risk,effective_step):
    dot=sum((a.double()*b.double()).sum() for a,b in zip(auxiliary,risk))
    an=sum(a.double().square().sum() for a in auxiliary).sqrt()
    rn=sum(b.double().square().sum() for b in risk).sqrt()
    return dict(dot=float(dot),aux_norm=float(an),risk_norm=float(rn),
                cosine=float(dot/(an*rn)) if an>0 and rn>0 else None,
                predicted_risk_loss_change=-float(effective_step)*float(dot))


def inspect_case(model,signal,outcome,critical,*,device):
    count=critical//128*128
    if count<256: raise AssertionError('changed decision precedes first update')
    fast=initial_encoder(model); previous_updates=0
    for arrived in range(256,count,128):
        fast,_=adapt_arrived(model,signal,arrived,fast,device=device); previous_updates+=1
    x,m,target,tm,origins=forecast_examples(signal,count,model.cfg,device)
    forecast=model.predict_fast(x,m,fast)['forecast']
    losses=channel_losses(forecast,target,tm,model.base.center,model.base.scale)
    total=model.base.forecast_loss(forecast,target,tm)
    torch.testing.assert_close(losses.sum(),total)
    gradients=torch.autograd.grad(total,tuple(fast.values()),retain_graph=True)
    effective_step=.001/max(1.,float(sum(g.square().sum() for g in gradients).sqrt()))
    qx,qm=[torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,critical,model.cfg.window)]
    label=float(outcome['y'][critical])
    z=model.predict_fast(qx,qm,fast)['logit']
    risk=F.binary_cross_entropy_with_logits(z,torch.full_like(z,label))
    risk_grad=torch.autograd.grad(risk,tuple(fast.values()))
    summed=[torch.zeros_like(g) for g in gradients]; channels=[]
    for c,name in enumerate(io23.RAW23):
        g=torch.autograd.grad(losses[c],tuple(fast.values()),retain_graph=c<22)
        for a,b in zip(summed,g): a.add_(b)
        channels.append(dict(name=name,forecast_loss=float(losses[c].detach()),
            observed_targets=int(tm[:,:,c].sum()),**gradient_geometry(g,risk_grad,effective_step)))
    decomposition_error=max(float((a-b).abs().max()) for a,b in zip(summed,gradients))
    for a,b in zip(summed,gradients): torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)
    updated,receipt=adapt_arrived(model,signal,count,fast,device=device)
    after=model.predict_fast(qx,qm,updated)['logit']
    risk_after=F.binary_cross_entropy_with_logits(after,torch.full_like(after,label))
    original=model.predict_fast(qx,qm,initial_encoder(model))['logit']
    lead=float((outcome['down']-outcome['times'][critical])*1000) if outcome['is_disrupt'] else None
    return dict(critical_index=critical,critical_time=float(outcome['times'][critical]),
        lead_ms=lead,endpoint_label125ms=label,
        warning_metric_and_endpoint_label_differ=bool(outcome['is_disrupt'] and lead>125),
        inspected_last_update=count,previous_updates=previous_updates,
        static_probability=float(original.sigmoid().detach()),
        before_last_update_probability=float(z.sigmoid().detach()),
        after_last_update_probability=float(after.sigmoid().detach()),
        risk_loss_before=float(risk.detach()),risk_loss_after=float(risk_after.detach()),
        actual_risk_loss_change=float((risk_after-risk).detach()),
        total_geometry=gradient_geometry(gradients,risk_grad,effective_step),
        decomposition_max_abs_error=decomposition_error,channels=channels,update_receipt=receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    verified=json.loads((args.run/'verification.json').read_text()); assert verified['status']=='VERIFIED'
    protocol=json.loads((args.run/'protocol.json').read_text()); threshold=protocol['threshold']
    full=json.loads((args.run/'per_shot.json').read_text())
    mapping=json.loads((args.run/'density_map.json').read_text())
    prep=json.loads((args.run/'preparation_receipts.json').read_text())
    torch.set_num_threads(2); results=[]
    for arm in ('density','replay','coral'):
        base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda()
        path=args.run/f'{arm}.pt';assert sha(path)==prep[arm]['checkpoint_sha256']
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model']);model.set_stage('frozen')
        changes=verified['paired_comparisons'][f'{arm}_static_to_intra']
        for kind,ids in changes.items():
            for sid in ids:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                outcome=simple_data.attach_outcome(s,'EAST',{})
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                with np.load(args.run/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                    lane=f'{arm}_intra' if kind in ('rescued_timely','added_fp') else f'{arm}_static'
                    indices=np.flatnonzero(outcome['valid'] & (trace[lane]>=threshold))
                    if not len(indices): raise AssertionError('selected change has no critical alarm')
                    critical=int(indices[0]); saved=float(trace[f'{arm}_intra'][critical])
                result=inspect_case(model,signal,outcome,critical,device='cuda')
                if abs(result['after_last_update_probability']-saved)>2e-6:
                    raise AssertionError(f'local state does not replay saved score: {sid}')
                result.update(arm=arm,shot=sid,change=kind,is_disrupt=outcome['is_disrupt'],saved_probability=saved)
                results.append(result)
                print(json.dumps(dict(arm=arm,shot=sid,change=kind,
                    cosine=result['total_geometry']['cosine'],risk_change=result['actual_risk_loss_change'])),flush=True)
    report=dict(scope='POSTHOC known-outcome derivative analysis; never a deployment loss or policy',
        case_selection='all shot-level decisions changed by intra in the prior completed six-arm experiment',
        source_run=str(args.run),verification_sha256=sha(args.run/'verification.json'),
        code_sha256=sha(__file__),cases=results,
        limitation='last-update local derivatives, not causal attribution of accumulated changes; y125 differs from unlimited warning lead')
    _durable_json(args.output,report)


if __name__=='__main__': main()
